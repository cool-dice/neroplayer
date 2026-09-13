"""A tiny simulated arcade game, rendered with OpenCV.

Its only purpose is to exercise the *real* pipeline without Windows, a display
or a sound card: the environment still captures frames, still reads the score
box with OCR/pixel checks, still detects the game-over banner, and still sends
keys through a controller. Only the backends are swapped.

The game itself: you are a square in one of three lanes, obstacles scroll in
from the right, ArrowUp/ArrowDown change lane, and you score a point for every
obstacle you survive. On collision it paints the same solid game-over banner
that ``ColorGameOverDetector`` looks for.

Run ``python -m ai_player.mock_game`` to dump a short video of random play.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import cv2
import numpy as np

from .config import AppConfig, Region
from .perception import AudioConfig, AudioFeatureExtractor

CANVAS_WIDTH = 320
CANVAS_HEIGHT = 240
N_LANES = 3

BG_COLOR = (24, 20, 20)
PLAYER_COLOR = (90, 220, 120)
OBSTACLE_COLOR = (60, 90, 235)
HUD_COLOR = (255, 255, 255)
# Must match RewardConfig.game_over_color_bgr for the colour detector to fire.
GAME_OVER_COLOR = (32, 32, 160)


@dataclass
class _Obstacle:
    x: float
    lane: int
    scored: bool = False


class MockArcadeGame:
    """Deterministic-with-seed lane dodger driven one tick at a time."""

    def __init__(
        self,
        width: int = CANVAS_WIDTH,
        height: int = CANVAS_HEIGHT,
        *,
        seed: int | None = None,
        sample_rate: int = 16_000,
        ticks_per_second: float = 10.0,
    ) -> None:
        self.width = width
        self.height = height
        self._rng = np.random.default_rng(seed)
        self._sample_rate = sample_rate
        self._samples_per_tick = max(1, int(sample_rate / max(ticks_per_second, 1e-6)))

        self.player_size = max(10, height // 12)
        self.obstacle_size = self.player_size
        self.player_x = int(width * 0.18)
        self._lane_y = [int(height * (0.3 + 0.2 * i)) for i in range(N_LANES)]

        self._audio = np.zeros(sample_rate, dtype=np.float32)
        self._phase = 0.0
        self.reset()

    # -- state ---------------------------------------------------------------
    def seed(self, seed: int | None) -> None:
        """Reseed and restart, so a given seed always replays identically."""
        self._rng = np.random.default_rng(seed)
        self._phase = 0.0
        self.reset()

    def reset(self) -> None:
        self.lane = N_LANES // 2
        self.score = 0
        self.ticks = 0
        self.game_over = False
        self._obstacles: list[_Obstacle] = []
        self._spawn_cooldown = 6
        self._pending_key: str | None = None
        self._events: list[tuple[str, int]] = []
        self._audio[:] = 0.0

    @property
    def speed(self) -> float:
        """Obstacle speed in px/tick, ramping up with the score."""
        return self.width * (0.055 + 0.0015 * min(self.score, 20))

    # -- input ---------------------------------------------------------------
    def press(self, key: str | None) -> None:
        """Queue a key for the next tick (mirrors a real tap between frames)."""
        if key is not None:
            self._pending_key = key.lower()

    def restart(self) -> None:
        self.reset()
        self._events.append(("restart", self.ticks))

    # -- simulation ----------------------------------------------------------
    def tick(self) -> None:
        """Advance one frame. Frozen once the game is over, until restart()."""
        key, self._pending_key = self._pending_key, None
        if self.game_over:
            self._synthesize_audio()
            return

        if key == "up":
            self.lane = max(0, self.lane - 1)
        elif key == "down":
            self.lane = min(N_LANES - 1, self.lane + 1)

        self.ticks += 1
        self._advance_obstacles()
        self._spawn()
        self._check_collision()
        self._synthesize_audio()

    def _advance_obstacles(self) -> None:
        speed = self.speed
        for obstacle in self._obstacles:
            obstacle.x -= speed
            if not obstacle.scored and obstacle.x < self.player_x - self.player_size:
                obstacle.scored = True
                self.score += 1
                self._events.append(("score", self.ticks))
        self._obstacles = [o for o in self._obstacles if o.x > -self.obstacle_size]

    def _spawn(self) -> None:
        self._spawn_cooldown -= 1
        if self._spawn_cooldown > 0:
            return
        # Block at most N_LANES - 1 lanes so a perfect player never dies.
        blocked = self._rng.choice(N_LANES, size=self._rng.integers(1, N_LANES), replace=False)
        for lane in np.atleast_1d(blocked):
            self._obstacles.append(_Obstacle(x=float(self.width), lane=int(lane)))
        self._spawn_cooldown = int(self._rng.integers(5, 9))

    def _check_collision(self) -> None:
        reach = (self.player_size + self.obstacle_size) / 2
        for obstacle in self._obstacles:
            if obstacle.lane == self.lane and abs(obstacle.x - self.player_x) < reach:
                self.game_over = True
                self._events.append(("crash", self.ticks))
                return

    # -- rendering -----------------------------------------------------------
    def render(self) -> np.ndarray:
        """Draw the current state as a BGR uint8 frame."""
        frame = np.full((self.height, self.width, 3), BG_COLOR, dtype=np.uint8)
        for y in self._lane_y:
            cv2.line(frame, (0, y), (self.width, y), (40, 38, 38), 1)

        half = self.player_size // 2
        for obstacle in self._obstacles:
            cx, cy = int(obstacle.x), self._lane_y[obstacle.lane]
            cv2.rectangle(
                frame,
                (cx - half, cy - half),
                (cx + half, cy + half),
                OBSTACLE_COLOR,
                thickness=-1,
            )
        py = self._lane_y[self.lane]
        cv2.rectangle(
            frame,
            (self.player_x - half, py - half),
            (self.player_x + half, py + half),
            PLAYER_COLOR,
            thickness=-1,
        )
        self._draw_hud(frame)
        if self.game_over:
            self._draw_game_over(frame)
        return frame

    def _draw_hud(self, frame: np.ndarray) -> None:
        # Black plate behind white digits: exactly what the score reader wants.
        cv2.rectangle(frame, (0, 0), (self.width, 26), (0, 0, 0), thickness=-1)
        cv2.putText(
            frame,
            f"{self.score}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            HUD_COLOR,
            2,
            cv2.LINE_AA,
        )

    def _draw_game_over(self, frame: np.ndarray) -> None:
        region = mock_game_over_region(self.width, self.height)
        cv2.rectangle(
            frame,
            (region.left, region.top),
            (region.left + region.width, region.top + region.height),
            GAME_OVER_COLOR,
            thickness=-1,
        )
        cv2.putText(
            frame,
            "GAME OVER",
            (region.left + 12, region.top + region.height // 2 + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # -- audio ---------------------------------------------------------------
    def _synthesize_audio(self) -> None:
        """Append one tick of fake game audio to the ring buffer.

        A quiet engine hum, a bright blip when scoring and broadband noise on
        death -- enough structure that the audio branch of the network sees a
        real, event-correlated signal.
        """
        n = self._samples_per_tick
        t = (np.arange(n, dtype=np.float32) + self._phase) / self._sample_rate
        self._phase += n
        chunk = 0.05 * np.sin(2 * np.pi * 120.0 * t).astype(np.float32)
        recent = {name for name, tick in self._events if tick >= self.ticks - 1}
        if "score" in recent:
            envelope = np.exp(-8.0 * np.linspace(0, 1, n, dtype=np.float32))
            chunk += 0.4 * envelope * np.sin(2 * np.pi * 880.0 * t).astype(np.float32)
        if "crash" in recent:
            chunk += 0.5 * self._rng.standard_normal(n).astype(np.float32)
        self._audio = np.concatenate([self._audio, np.clip(chunk, -1.0, 1.0)])[
            -self._sample_rate :
        ]
        self._events = [(name, tick) for name, tick in self._events if tick >= self.ticks - 2]

    def audio_window(self, n_samples: int) -> np.ndarray:
        if n_samples >= self._audio.size:
            return np.pad(self._audio, (n_samples - self._audio.size, 0))
        return self._audio[-n_samples:].copy()


def mock_game_over_region(width: int = CANVAS_WIDTH, height: int = CANVAS_HEIGHT) -> Region:
    """Centre banner rectangle, shared by the renderer and the detector."""
    w, h = int(width * 0.6), int(height * 0.22)
    return Region(left=(width - w) // 2, top=(height - h) // 2, width=w, height=h)


# ---------------------------------------------------------------------------
# Backends that make the mock game look like a captured window
# ---------------------------------------------------------------------------
class MockFrameSource:
    """Frame source that advances the simulation once per capture.

    Ticking on grab keeps the game in lockstep with the agent's decisions, so
    dry runs are deterministic and as fast as the CPU allows.
    """

    def __init__(self, game: MockArcadeGame) -> None:
        self._game = game

    def grab(self) -> np.ndarray:
        self._game.tick()
        return self._game.render()

    def seed(self, seed: int | None) -> None:
        """Optional hook honoured by ``GameEnv.reset(seed=...)``."""
        self._game.seed(seed)

    def close(self) -> None:
        return None


class MockAudioSource:
    def __init__(self, game: MockArcadeGame, audio: AudioConfig) -> None:
        self._game = game
        self._audio = audio

    def start(self) -> None:
        return None

    def read_window(self) -> np.ndarray:
        return self._game.audio_window(self._audio.window_samples)

    def stop(self) -> None:
        return None


class MockController:
    """Sends the agent's key presses straight into the simulation."""

    def __init__(self, game: MockArcadeGame, action_keys: list[str | None]) -> None:
        self._game = game
        self._action_keys = action_keys

    def act(self, action: int) -> None:
        if not 0 <= int(action) < len(self._action_keys):
            raise ValueError(f"Action {action} outside 0..{len(self._action_keys) - 1}")
        self._game.press(self._action_keys[int(action)])

    def restart(self) -> None:
        self._game.restart()

    def release_all(self) -> None:
        return None


def mock_config(base: AppConfig | None = None) -> AppConfig:
    """Copy ``base`` and point every region at the mock canvas."""
    config = copy.deepcopy(base) if base is not None else AppConfig()
    config.capture.region = Region(left=0, top=0, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
    config.reward.game_over_region = mock_game_over_region()
    config.reward.game_over_template = None
    config.reward.game_over_color_bgr = GAME_OVER_COLOR
    config.reward.score_region = Region(left=0, top=0, width=120, height=26)
    # The simulation advances on capture, so real-time pacing only slows tests.
    config.env.target_fps = 0.0
    config.env.reset_delay = 0.0
    return config


def build_mock_backends(
    config: AppConfig, *, seed: int | None = None
) -> tuple[MockArcadeGame, MockFrameSource, MockAudioSource, MockController]:
    """Create a game plus the three backends the environment expects."""
    game = MockArcadeGame(
        width=config.capture.region.width,
        height=config.capture.region.height,
        seed=seed,
        sample_rate=config.audio.sample_rate,
        ticks_per_second=config.env.target_fps or 10.0,
    )
    return (
        game,
        MockFrameSource(game),
        MockAudioSource(game, config.audio),
        MockController(game, config.control.action_keys),
    )


def _demo(path: str = "mock_game_demo.mp4", n_frames: int = 300) -> None:
    """Record random play, a quick sanity check that rendering works."""
    config = mock_config()
    game, frames, audio, controller = build_mock_backends(config, seed=0)
    features = AudioFeatureExtractor(config.audio)
    rng = np.random.default_rng(0)
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (game.width, game.height)
    )
    for _ in range(n_frames):
        controller.act(int(rng.integers(0, config.control.n_actions)))
        frame = frames.grab()
        features.extract(audio.read_window())
        writer.write(frame)
        if game.game_over:
            controller.restart()
    writer.release()
    print(f"wrote {path}")


if __name__ == "__main__":  # pragma: no cover
    _demo()
