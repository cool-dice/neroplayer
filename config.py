"""Central configuration for the game-playing agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Region:
    """A rectangular region in screen coordinates."""

    left: int
    top: int
    width: int
    height: int

    def as_mss_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class GameConfig:
    """Runtime settings. Adjust regions and keys for the target game."""

    capture_region: Region = Region(left=100, top=100, width=800, height=600)
    image_width: int = 84
    image_height: int = 84
    frame_stack: int = 4
    target_fps: int = 15

    audio_sample_rate: int = 22_050
    audio_channels: int = 2
    audio_duration_seconds: float = 0.05
    audio_mel_bands: int = 32

    # Coordinates below are relative to capture_region, not the desktop.
    score_region: Region = Region(left=620, top=20, width=150, height=60)
    game_over_region: Region = Region(left=250, top=200, width=300, height=120)
    game_over_template: Path = Path("assets/game_over.png")
    template_match_threshold: float = 0.82

    # Used when no template exists. Set this to a distinctive game-over color.
    game_over_bgr: tuple[int, int, int] = (40, 40, 220)
    game_over_color_tolerance: int = 25
    game_over_min_pixel_ratio: float = 0.20

    score_change_threshold: float = 10.0
    score_reward: float = 10.0
    living_reward: float = 0.01
    game_over_penalty: float = -100.0
    max_episode_steps: int = 10_000

    action_keys: tuple[str | None, ...] = ("up", "down", None)
    action_hold_seconds: float = 0.035
    restart_keys: tuple[str, ...] = ("enter",)
    restart_wait_seconds: float = 1.0
    random_seed: int = 42
    model_path: Path = Path("models/game_agent")
    log_dir: Path = Path("runs")

    # Optional options passed to keyboard handling may be extended by callers.
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def frame_interval(self) -> float:
        return 1.0 / self.target_fps

    @property
    def audio_feature_size(self) -> int:
        # Mean and standard deviation for each mel band.
        return self.audio_mel_bands * 2


CONFIG = GameConfig()
