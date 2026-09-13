"""Gymnasium environment wrapping a real game window.

The environment is *real-time*: the game keeps running whether or not we call
``step()``, so unlike a simulator the step rate is governed by wall-clock
pacing (``target_fps``) and every observation is whatever is on screen right
now. Consequences for RL:

* Only one environment instance can exist per machine (it owns the keyboard).
* Episodes are non-deterministic; use frame stacking and small learning rates.
* Slow policies degrade the agent's reaction time, so keep networks small.

Observation (``gymnasium.spaces.Dict``):
    ``image``: ``uint8`` array ``(frame_stack, H, W)`` of grayscale frames.
    ``audio``: ``float32`` array ``(n_mels, n_frames)`` log-mel spectrogram in [0, 1].

Action (``gymnasium.spaces.Discrete``): index into ``ControlConfig.actions``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import CONFIG, Config
from controls import GameController
from observer import GameObserver
from perception import AudioCapture, ScreenCapture

logger = logging.getLogger(__name__)


class GameEnv(gym.Env):
    """Multi-modal (vision + audio) environment for a windowed game.

    All collaborators can be injected, which lets tests run the environment
    against synthetic frames and a dry-run controller without a game or a
    display attached.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": CONFIG.screen.target_fps}

    def __init__(
        self,
        config: Config = CONFIG,
        *,
        screen: ScreenCapture | None = None,
        audio: AudioCapture | None = None,
        controller: GameController | None = None,
        observer: GameObserver | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.render_mode = render_mode

        self.screen = screen or ScreenCapture(config.screen)
        self.audio = audio or AudioCapture(config.audio)
        self.controller = controller or GameController(config.controls)
        self.observer = observer or GameObserver(config.reward)

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=self.screen.observation_shape, dtype=np.uint8),
                "audio": spaces.Box(low=0.0, high=1.0, shape=self.audio.observation_shape, dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(self.controller.n_actions)

        self._frame_period = 1.0 / config.screen.target_fps
        self._max_steps = config.train.max_episode_steps
        self._step_count = 0
        self._episode_return = 0.0
        self._last_step_time = 0.0

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Restart the game and return the first observation.

        ``seed`` is accepted for API compatibility but has no effect: a real
        game cannot be seeded from outside.
        """
        super().reset(seed=seed)
        self._step_count = 0
        self._episode_return = 0.0

        self.controller.restart()

        image = self.screen.reset()
        audio = self.audio.reset()
        # The observer needs the *full-resolution* frame to read UI text.
        self.observer.reset(self.screen.last_raw_frame)

        self._last_step_time = time.perf_counter()
        return {"image": image, "audio": audio}, {"score": 0}

    def step(self, action: int):
        """Act, wait one frame period, sense, and score the new frame."""
        action = int(action)
        self.controller.execute(action)
        self._pace()

        image = self.screen.observe()
        audio = self.audio.observe()
        signal = self.observer.evaluate(self.screen.last_raw_frame)

        self._step_count += 1
        self._episode_return += signal.reward
        truncated = self._step_count >= self._max_steps
        terminated = signal.terminated

        info = {
            **signal.info,
            "action": self.controller.action_name(action),
            "step": self._step_count,
            "episode_return": self._episode_return,
        }
        if terminated or truncated:
            logger.info(
                "Episode end: steps=%d return=%.1f score=%s reason=%s",
                self._step_count,
                self._episode_return,
                info.get("score"),
                "game_over" if terminated else "time_limit",
            )
        return {"image": image, "audio": audio}, float(signal.reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array" and self.screen.last_raw_frame is not None:
            return self.screen.last_raw_frame[:, :, ::-1].copy()  # BGR -> RGB
        return None

    def close(self) -> None:
        for component in (self.controller, self.audio, self.screen):
            try:
                component.close()
            except Exception:  # pragma: no cover
                logger.exception("Error closing %s", type(component).__name__)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _pace(self) -> None:
        """Sleep so consecutive steps are spaced by at least one frame period.

        This both gives the game time to react to the injected input and keeps
        the effective decision frequency stable when the policy's inference
        time fluctuates.
        """
        elapsed = time.perf_counter() - self._last_step_time
        remaining = self._frame_period - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_step_time = time.perf_counter()


def make_env(config: Config = CONFIG, **kwargs: Any) -> GameEnv:
    """Factory used by ``train.py``/``play.py`` (and handy for ``gym.make``-style code)."""
    return GameEnv(config, **kwargs)
