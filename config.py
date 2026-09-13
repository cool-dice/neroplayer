"""Central configuration for the game-playing RL agent.

Every tunable that depends on *your* game (where the window sits on screen,
which keys it listens to, where the score is drawn) lives here so the other
modules stay game-agnostic. Values are grouped into small frozen dataclasses
and exposed through the ``CONFIG`` singleton.

Coordinates are in physical screen pixels with the origin at the top-left of
the primary monitor. Use ``python tools/calibrate.py`` (or any screenshot
tool) to find the right numbers for your game window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
TEMPLATES_DIR = ASSETS_DIR / "templates"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


@dataclass(frozen=True)
class Region:
    """A rectangular screen region in absolute pixel coordinates."""

    left: int
    top: int
    width: int
    height: int

    def as_mss(self) -> dict[str, int]:
        """Return the region in the dict format expected by ``mss.grab``."""
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def as_slice(self) -> tuple[slice, slice]:
        """Return ``(rows, cols)`` slices for indexing a NumPy frame."""
        return slice(self.top, self.top + self.height), slice(self.left, self.left + self.width)


@dataclass(frozen=True)
class ScreenConfig:
    """Screen capture and image preprocessing settings."""

    # Full monitor size, used only for sanity checks / calibration tooling.
    monitor_width: int = 1920
    monitor_height: int = 1080

    # The game viewport we feed to the agent. Default: a 800x600 window
    # roughly centred on a 1080p monitor.
    capture_region: Region = Region(left=560, top=240, width=800, height=600)

    # Agent-facing observation: downscaled grayscale frames (Atari-style).
    frame_width: int = 84
    frame_height: int = 84
    frame_stack: int = 4  # consecutive frames stacked to encode motion

    # Control loop pacing. Each env.step() waits for at most one frame period
    # so the game has time to react to the injected input.
    target_fps: int = 15


@dataclass(frozen=True)
class AudioConfig:
    """System loopback audio capture and spectrogram settings."""

    enabled: bool = True
    sample_rate: int = 22_050
    channels: int = 1
    # Length of the trailing audio window converted into a spectrogram on
    # every step. Short windows keep the observation responsive.
    window_seconds: float = 0.25
    n_mels: int = 32
    n_fft: int = 1024
    hop_length: int = 512
    # Decibel range used to normalise the log-mel spectrogram into [0, 1].
    min_db: float = -80.0
    max_db: float = 0.0

    @property
    def window_samples(self) -> int:
        return int(self.sample_rate * self.window_seconds)

    @property
    def n_frames(self) -> int:
        """Number of STFT frames librosa produces for one window (center=True)."""
        return 1 + self.window_samples // self.hop_length


@dataclass(frozen=True)
class ControlConfig:
    """Discrete action -> keyboard mapping consumed by ``controls.py``.

    ``None`` means "do nothing" (a no-op action is important so the agent can
    learn *when not to act*). Key names follow the pydirectinput/pyautogui
    convention, e.g. ``"up"``, ``"space"``, ``"a"``.
    """

    actions: tuple[str | None, ...] = ("up", "down", None)
    action_names: tuple[str, ...] = ("ArrowUp", "ArrowDown", "NoOp")
    # How long a key is held per action. Too short and some games miss the
    # input; too long and the agent's reaction time suffers.
    key_hold_seconds: float = 0.05
    # Key sequence sent by env.reset() to restart the game after a game over.
    restart_keys: tuple[str, ...] = ("space",)
    # Pause after restarting so the "new game" frame is on screen before we
    # capture the first observation.
    restart_delay_seconds: float = 1.0
    # Where the mouse is parked before a run so the game window has focus.
    focus_click: tuple[int, int] | None = (960, 540)


@dataclass(frozen=True)
class RewardConfig:
    """Everything ``observer.py`` needs to turn pixels into rewards."""

    # --- Game-over detection ---------------------------------------------
    # Preferred: template matching against a cropped screenshot of the
    # game-over banner. Coordinates are relative to the *capture region*.
    game_over_template: Path = TEMPLATES_DIR / "game_over.png"
    game_over_match_threshold: float = 0.80
    # Fallback: mean BGR colour of a small region that only takes this colour
    # on the game-over screen (e.g. a red banner). Relative to capture region.
    game_over_region: Region = Region(left=300, top=250, width=200, height=100)
    game_over_color_bgr: tuple[int, int, int] = (40, 40, 200)
    game_over_color_tolerance: int = 40
    # Require N consecutive positive detections to avoid one-frame glitches.
    game_over_confirm_frames: int = 2

    # --- Score detection -------------------------------------------------
    # Region (relative to capture region) where the score digits are drawn.
    score_region: Region = Region(left=650, top=10, width=140, height=40)
    # Use pytesseract OCR if it is installed *and* a tesseract binary exists;
    # otherwise fall back to a pixel-change detector on ``score_region``.
    use_ocr: bool = True
    # Fraction of pixels in score_region that must change (after binarising)
    # for the fallback detector to declare "score changed".
    score_change_pixel_fraction: float = 0.02

    # --- Reward shaping ----------------------------------------------------
    reward_score_increase: float = 10.0  # per detected point change / per point with OCR
    reward_alive: float = 0.1  # small per-step bonus for surviving
    reward_game_over: float = -100.0
    # Cap the per-step OCR reward so a misread ("12" -> "1200") cannot blow up
    # the value function.
    max_score_delta: int = 100


@dataclass(frozen=True)
class TrainConfig:
    """Hyper-parameters and paths for ``train.py``."""

    algorithm: str = "ppo"  # "ppo" or "dqn"
    total_timesteps: int = 200_000
    # Truncate an episode after this many steps so a game with no game-over
    # condition still produces bounded rollouts.
    max_episode_steps: int = 2_000
    checkpoint_every_steps: int = 5_000
    model_dir: Path = MODELS_DIR
    log_dir: Path = LOGS_DIR
    model_name: str = "game_agent"
    seed: int = 42
    device: str = "auto"

    # PPO hyper-parameters tuned for a single, slow, real-time environment.
    ppo: dict = field(
        default_factory=lambda: {
            "n_steps": 512,
            "batch_size": 64,
            "n_epochs": 4,
            "learning_rate": 2.5e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.1,
            "ent_coef": 0.01,
        }
    )
    dqn: dict = field(
        default_factory=lambda: {
            "learning_rate": 1e-4,
            "buffer_size": 50_000,
            "learning_starts": 1_000,
            "batch_size": 32,
            "train_freq": 4,
            "target_update_interval": 1_000,
            "exploration_fraction": 0.2,
            "exploration_final_eps": 0.05,
        }
    )


@dataclass(frozen=True)
class Config:
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    controls: ControlConfig = field(default_factory=ControlConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CONFIG = Config()
