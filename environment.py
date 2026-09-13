"""Gymnasium environment that connects perception, controls, and rewards."""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import CONFIG, GameConfig
from controls import GameController
from observer import GameStateObserver
from perception import AudioCapture, ScreenCapture


class WindowsGameEnv(gym.Env[dict[str, np.ndarray], int]):
    """A real-time environment for a windowed game running on Windows."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(
        self,
        config: GameConfig = CONFIG,
        *,
        screen: ScreenCapture | None = None,
        audio: AudioCapture | None = None,
        controller: GameController | None = None,
        observer: GameStateObserver | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.screen = screen or ScreenCapture(config)
        self.audio = audio or AudioCapture(config)
        self.controller = controller or GameController(config)
        self.observer = observer or GameStateObserver(config)

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        config.frame_stack,
                        config.image_height,
                        config.image_width,
                    ),
                    dtype=np.uint8,
                ),
                "audio": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(config.audio_feature_size,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Discrete(self.controller.action_count)
        self._episode_steps = 0
        self._last_raw_frame: np.ndarray | None = None
        self._closed = False

    def _observe(self, *, reset_stack: bool = False) -> dict[str, np.ndarray]:
        raw = self.screen.capture_raw()
        self._last_raw_frame = raw
        image = (
            self.screen.reset_stack(raw)
            if reset_stack
            else self.screen.update_stack(raw)
        )
        audio = self.audio.capture_features()
        return {"image": image, "audio": audio}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Restart the game and initialize the frame stack."""
        super().reset(seed=seed)
        del options
        self._closed = False
        self.controller.restart()
        observation = self._observe(reset_stack=True)
        assert self._last_raw_frame is not None
        self.observer.reset(self._last_raw_frame)
        self._episode_steps = 0
        return observation, {"episode_step": 0}

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one action and evaluate the resulting game state."""
        started_at = time.perf_counter()
        self.controller.execute(int(action))
        observation = self._observe()
        assert self._last_raw_frame is not None
        result = self.observer.evaluate(self._last_raw_frame)
        self._episode_steps += 1
        truncated = self._episode_steps >= self.config.max_episode_steps

        elapsed = time.perf_counter() - started_at
        remaining = self.config.frame_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

        info: dict[str, Any] = dict(result.info)
        info.update(
            {
                "episode_step": self._episode_steps,
                "step_seconds": time.perf_counter() - started_at,
            }
        )
        return observation, result.reward, result.terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self._last_raw_frame is None:
            return None
        # Gymnasium's rgb_array convention differs from OpenCV's BGR.
        return self._last_raw_frame[:, :, ::-1].copy()

    def close(self) -> None:
        if self._closed:
            return
        self.controller.release_all()
        self.screen.close()
        self._closed = True
