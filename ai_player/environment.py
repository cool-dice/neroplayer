"""The Gymnasium environment: one agent decision per captured frame.

Timeline of a single ``step``:

1. Send the action to the OS (``controls``).
2. Sleep until the next frame slot so the agent runs at a fixed decision rate.
3. Capture the screen and the most recent audio window (``perception``).
4. Score the frame for reward/termination (``observer``).
5. Return ``(obs, reward, terminated, truncated, info)``.

Holding the decision rate constant is not cosmetic: frame stacking encodes
velocity as a pixel delta *per step*, so a wobbling step duration makes the
same state look different and slows learning down considerably.

Every collaborator is injectable, which is how ``--mock`` and the tests run the
identical control flow without Windows, a display or audio hardware.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import AppConfig
from .controls import ActionController, build_controller
from .observer import GameObserver
from .perception import (
    AudioFeatureExtractor,
    AudioSource,
    FrameProcessor,
    FrameSource,
    FrameStack,
    ScreenCapture,
    build_audio_source,
)


class GameEnv(gym.Env):
    """Screen-and-sound driven environment for a windowed game.

    Observation is a ``Dict`` space so that image and audio keep their own
    natural shapes and can be fed to different network branches:

    * ``frames``: ``(frame_stack, 84, 84)`` uint8, channels-first for the CNN.
    * ``audio``:  ``(n_mels, n_frames)`` float32 log-mel spectrogram in [0, 1]
      (present only when audio capture is enabled).

    Action space is ``Discrete(len(control.action_keys))``, with one entry
    reserved for "do nothing".
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 30,
    }

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        frame_source: FrameSource | None = None,
        audio_source: AudioSource | None = None,
        controller: ActionController | None = None,
        observer: GameObserver | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or AppConfig()
        self.render_mode = render_mode

        self._frames = frame_source or ScreenCapture(self.config.capture)
        self._audio = audio_source or build_audio_source(self.config.audio)
        self._controller = controller or build_controller(self.config.control)
        self._observer = observer or GameObserver(self.config.reward)
        # Idempotent: build_audio_source already started the recorder it built,
        # but an injected source may not be running yet.
        self._audio.start()

        self._processor = FrameProcessor(self.config.vision)
        self._stack = FrameStack(self.config.vision)
        self._audio_features = AudioFeatureExtractor(self.config.audio)
        self._audio_enabled = self.config.audio.enabled

        obs_spaces: dict[str, spaces.Space] = {
            "frames": spaces.Box(low=0, high=255, shape=self.config.vision.observation_shape, dtype=np.uint8)
        }
        if self._audio_enabled:
            obs_spaces["audio"] = spaces.Box(
                low=0.0, high=1.0, shape=self.config.audio.observation_shape, dtype=np.float32
            )
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Discrete(self.config.control.n_actions)

        self._frame_period = 1.0 / self.config.env.target_fps if self.config.env.target_fps else 0.0
        self._next_frame_at = 0.0
        self._last_frame: np.ndarray | None = None
        self._last_info: dict[str, Any] = {}
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._episode_index = 0
        self._window_name = "ai-player"
        self._window_open = False

    @property
    def last_frame(self) -> np.ndarray | None:
        """The most recently captured frame, in BGR, before preprocessing."""
        return None if self._last_frame is None else self._last_frame.copy()

    # -- gymnasium API -------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Restart the round and return the first observation."""
        super().reset(seed=seed)
        env_cfg = self.config.env

        # A captured window cannot be reseeded, but a simulated one can: the
        # hook is what makes mock runs reproducible.
        if seed is not None:
            seed_backend = getattr(self._frames, "seed", None)
            if callable(seed_backend):
                seed_backend(seed)

        self._controller.release_all()
        self._observer.reset()
        self._controller.restart()
        if env_cfg.reset_delay:
            time.sleep(env_cfg.reset_delay)

        frame = self._wait_for_playable_frame()
        self._stack.reset(self._processor.process(frame))
        # A couple of throwaway frames let start-of-round animations finish, so
        # the stack holds live gameplay rather than a menu.
        for _ in range(max(0, env_cfg.warmup_frames)):
            frame = self._capture_frame()
            self._stack.push(self._processor.process(frame))

        self._episode_steps = 0
        self._episode_reward = 0.0
        self._episode_index += 1
        self._next_frame_at = time.perf_counter()
        self._last_frame = frame

        info = {"episode_index": self._episode_index}
        return self._observation(), info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._controller.act(int(action))
        frame = self._capture_frame()
        self._last_frame = frame

        self._stack.push(self._processor.process(frame))
        verdict = self._observer.evaluate(frame)

        self._episode_steps += 1
        self._episode_reward += verdict.reward
        truncated = self._episode_steps >= self.config.env.max_episode_steps

        if verdict.terminated or truncated:
            # Never leave a key held across an episode boundary; the game would
            # receive phantom input during the restart sequence.
            self._controller.release_all()

        info: dict[str, Any] = {
            "score": verdict.score,
            "points": verdict.points,
            "episode_points": self._observer.episode_points,
            "episode_steps": self._episode_steps,
            "episode_reward": self._episode_reward,
            "game_over_confidence": verdict.game_over_confidence,
            "frame_diff": verdict.frame_diff,
            "is_idle": verdict.is_idle,
            "action": int(action),
        }
        self._last_info = info
        if self.render_mode == "human":
            self.render()
        return self._observation(), float(verdict.reward), bool(verdict.terminated), truncated, info

    def _draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """Overlay telemetry and action info onto the frame for human preview."""
        out = frame.copy()
        h, w = out.shape[:2]

        # Draw ROI boxes
        boxes = (
            (self.config.reward.score_region, (0, 255, 0)),
            (self.config.reward.game_over_region, (0, 0, 255)),
        )
        for region, colour in boxes:
            x1 = max(0, min(region.left, w))
            y1 = max(0, min(region.top, h))
            x2 = max(x1, min(region.left + region.width, w))
            y2 = max(y1, min(region.top + region.height, h))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(out, (x1, y1), (x2, y2), colour, 1)

        # Draw semi-transparent telemetry bar at the top or bottom
        bar_height = 80
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

        action_idx = self._last_info.get("action")
        keys = self.config.control.action_keys
        if action_idx is not None and 0 <= action_idx < len(keys):
            key_name = keys[action_idx]
            action_text = f"ACTION: [{key_name if key_name is not None else 'IDLE/NONE'}]"
        else:
            action_text = "ACTION: --"

        ep_rew = self._last_info.get("episode_reward", 0.0)
        ep_step = self._last_info.get("episode_steps", 0)
        points = self._last_info.get("episode_points", 0.0)
        frame_diff = self._last_info.get("frame_diff", 0.0)
        is_idle = self._last_info.get("is_idle", False)
        go_conf = self._last_info.get("game_over_confidence", 0.0)

        # Status text lines
        status_color = (0, 0, 255) if is_idle else (0, 255, 0)
        motion_status = "STAGNANT / IDLE" if is_idle else "ACTIVE MOTION"

        cv2.putText(
            out,
            f"EP #{self._episode_index} | STEP: {ep_step:4d} | REWARD: {ep_rew:+6.1f} | PTS: {points:.0f}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"{action_text} | DIFF: {frame_diff * 100:4.1f}% [{motion_status}]",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"GAME OVER CONF: {go_conf:4.2f} | FPS TARGET: {self.config.env.target_fps:.1f}",
            (10, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return out

    def render_hud(self) -> np.ndarray | None:
        """Return the current frame with the HUD telemetry overlay drawn on it."""
        if self._last_frame is None:
            return None
        return self._draw_hud(self._last_frame)

    def render(self) -> np.ndarray | None:
        if self._last_frame is None:
            return None
        hud_frame = self._draw_hud(self._last_frame)
        if self.render_mode == "rgb_array":
            return cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
        if self.render_mode == "human":  # pragma: no cover - needs a display
            cv2.imshow(self._window_name, hud_frame)
            cv2.waitKey(1)
            self._window_open = True
        return None

    def close(self) -> None:
        self._controller.release_all()
        self._audio.stop()
        self._frames.close()
        if self._window_open:  # pragma: no cover - needs a display
            cv2.destroyWindow(self._window_name)
            self._window_open = False

    # -- internals -----------------------------------------------------------
    def _observation(self) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {"frames": self._stack.observation}
        if self._audio_enabled:
            obs["audio"] = self._audio_features.extract(self._audio.read_window())
        return obs

    def _capture_frame(self) -> np.ndarray:
        """Grab the next frame, holding the configured decision rate."""
        if self._frame_period:
            remaining = self._next_frame_at - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
                self._next_frame_at += self._frame_period
            else:
                # We fell behind (slow game, GC pause): resynchronise instead of
                # trying to catch up with a burst of zero-length steps.
                self._next_frame_at = time.perf_counter() + self._frame_period
        return self._frames.grab()

    def _wait_for_playable_frame(self) -> np.ndarray:
        """Block until the game-over screen clears, or the timeout expires."""
        deadline = time.perf_counter() + self.config.env.reset_timeout
        frame = self._frames.grab()
        while self._observer.is_game_over(frame):
            if time.perf_counter() >= deadline:
                # Re-tap restart once; some games need the input twice (e.g. a
                # confirmation prompt) and we would otherwise start an episode
                # on a dead screen.
                self._controller.restart()
                break
            time.sleep(0.05)
            frame = self._frames.grab()
        return frame


def make_env(
    config: AppConfig | None = None,
    *,
    mock: bool = False,
    dry_run: bool = False,
    render_mode: str | None = None,
    seed: int | None = None,
) -> GameEnv:
    """Build a ``GameEnv``.

    ``mock=True`` swaps in the bundled simulated game (no OS integration at
    all). ``dry_run=True`` keeps real screen capture but discards key presses,
    which is the safe way to verify capture regions and reward detection
    against a live game.
    """
    if mock:
        from .mock_game import build_mock_backends, mock_config

        config = mock_config(config)
        _game, frames, audio, controller = build_mock_backends(config, seed=seed)
        return GameEnv(
            config,
            frame_source=frames,
            audio_source=audio,
            controller=controller,
            render_mode=render_mode,
        )

    config = config or AppConfig()
    return GameEnv(
        config,
        controller=build_controller(config.control, dry_run=dry_run),
        render_mode=render_mode,
    )
