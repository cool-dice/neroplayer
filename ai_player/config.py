"""Central configuration for the AI player.

Every tunable lives here as a frozen-ish dataclass so that the perception,
control, reward and training layers all agree on shapes and timings. The whole
tree can be serialised to JSON, which is what ``calibrate.py`` writes after you
pick the game window on your own machine.

Coordinates are in physical screen pixels. On Windows with display scaling
enabled (125%/150%), make sure Python is DPI-aware or capture a region you
verified with ``calibrate.py`` rather than one you measured by eye.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
TEMPLATES_DIR = ASSETS_DIR / "templates"
CAPTURES_DIR = ASSETS_DIR / "captures"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


@dataclass
class Region:
    """An axis-aligned rectangle in pixels.

    Used both for the capture region (absolute desktop coordinates) and for
    sub-regions such as the score box, which are relative to the captured
    frame's own origin. ``perception.crop_region`` slices frames with it.
    """

    left: int
    top: int
    width: int
    height: int

    def as_mss_monitor(self) -> dict[str, int]:
        """Return the dict shape that ``mss.mss().grab()`` expects."""
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class CaptureConfig:
    """Where on the desktop the game lives.

    The static ``region`` is a fallback (and the initial guess written by
    calibrate). At runtime ``follow_foreground`` (default) re-targets capture
    at the live OS window so you can alt-tab between games without re-running
    calibrate. ``window_title`` pins a substring/regex when you want the agent
    to keep capturing a game even after you click away from it.
    """

    # Defaults describe a 800x600 window parked in the top-left corner. Used
    # only until a live window is found (or when follow-foreground is off).
    region: Region = field(default_factory=lambda: Region(left=0, top=0, width=800, height=600))
    # mss monitor index used by calibrate.py for the full-desktop screenshot.
    monitor_index: int = 1
    follow_foreground: bool = True
    window_title: str = ""
    # Extra title substrings to treat as non-games (merged with the built-in list).
    exclude_titles: list[str] = field(default_factory=list)
    min_width: int = 200
    min_height: int = 150
    # How often to re-query the OS for window geometry / foreground identity.
    poll_interval: float = 0.25


@dataclass
class VisionConfig:
    """Image preprocessing, mirroring the classic Atari DQN pipeline or modern backbones."""

    width: int = 84
    height: int = 84
    grayscale: bool = True
    # Number of consecutive frames handed to the CNN. Stacking is what gives a
    # feed-forward policy access to velocity/direction information.
    frame_stack: int = 4
    # Feature extractor backbone architecture: 'nature_cnn', 'convnext', 'resnet'
    backbone: str = "nature_cnn"

    @property
    def rgb(self) -> bool:
        return not self.grayscale

    @property
    def channels(self) -> int:
        return self.frame_stack if self.grayscale else self.frame_stack * 3

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """Channels-first shape, which is what SB3's CNN extractors want."""
        return (self.channels, self.height, self.width)


@dataclass
class AudioConfig:
    """System loopback capture and mel-spectrogram feature extraction."""

    enabled: bool = True
    sample_rate: int = 16_000
    # Length of the rolling audio window converted into one observation.
    window_seconds: float = 0.5
    n_mels: int = 32
    n_fft: int = 512
    hop_length: int = 256
    # Frames are pulled from the loopback device in chunks this large.
    block_size: int = 1024
    top_db: float = 80.0

    @property
    def window_samples(self) -> int:
        return int(self.sample_rate * self.window_seconds)

    @property
    def n_frames(self) -> int:
        """Spectrogram columns for one window (librosa with ``center=False``)."""
        return 1 + max(0, self.window_samples - self.n_fft) // self.hop_length

    @property
    def observation_shape(self) -> tuple[int, int]:
        return (self.n_mels, self.n_frames)


@dataclass
class ControlConfig:
    """Mapping from discrete agent actions to keyboard and mouse actions."""

    # Index == action id. ``None`` is an explicit no-op, which the agent needs in
    # order to *not* act -- without it every step would perturb the game.
    # Supports single keys ('d'), combos ('d+space' or ['d', 'space']),
    # mouse actions ('mouse_left', 'mouse_right', 'aim_left', etc.),
    # compound keyboard+mouse ('w+mouse_left', ['w', 'mouse_left']), and None.
    action_keys: list[Any] = field(default_factory=lambda: ["up", "down", None])
    # When auto_combos=True, generates all simultaneous combinations up to max_combo_size
    # from available single keys, so the agent can learn and discover which combos work.
    auto_combos: bool = False
    max_combo_size: int = 2
    # Keys tapped on reset to start/restart a round (e.g. ["r"], ["enter"], ["space"]).
    restart_keys: list[Any] = field(default_factory=lambda: ["space"])
    # How long a key stays down for a single action. Keep it under one frame
    # period so actions do not bleed into the next step.
    tap_duration: float = 0.02
    # Hold the key for the whole step instead of tapping it. Useful for games
    # where movement is continuous while a key is down.
    hold_keys: bool = False
    # Mouse controller settings
    mouse_enabled: bool = False
    mouse_mode: str = "relative"  # "relative" for 3D FPS aiming, "absolute" for RTS cursor clicking
    mouse_sensitivity: float = 1.0
    aim_step: int = 15  # discrete pixel step for relative camera turns in discrete action mode
    # pydirectinput inserts a global pause after every call; we manage our own
    # timing, so disable it.
    pause_between_calls: float = 0.0
    fail_safe: bool = False

    @property
    def n_actions(self) -> int:
        return len(self.action_keys)


@dataclass
class RewardConfig:
    """Reward shaping plus the vision heuristics the critic relies on."""

    # Small positive drip per surviving step. Dense enough to bootstrap PPO
    # before the agent ever discovers how to score.
    step_reward: float = 0.1
    # Multiplier applied to each detected point gain.
    score_reward: float = 10.0
    # Terminal penalty. Large and negative so death dominates the return.
    game_over_penalty: float = -100.0
    # Ignore absurd OCR jumps (misreads such as 7 -> 771).
    max_score_delta: int = 50

    # --- Game-over detection -------------------------------------------------
    # Mode for detecting game over: "auto", "template", "color", "text".
    game_over_mode: str = "auto"
    # Keywords searched by autonomous text detection and OCR.
    game_over_keywords: list[str] = field(
        default_factory=lambda: [
            "GAME OVER",
            "CONTINUE",
            "YOU DIED",
            "DEFEAT",
            "MISSION FAILED",
            "TRY AGAIN",
        ]
    )
    # Detection confidence threshold [0.0, 1.0] for autonomous game-over detector.
    game_over_threshold: float = 0.65
    # Region (relative to the capture region) inspected for the game-over cue.
    game_over_region: Region = field(default_factory=lambda: Region(left=200, top=200, width=400, height=200))
    # Path to a grayscale template crop. When present, cv2.matchTemplate is
    # used; otherwise we fall back to the mean-colour check below.
    game_over_template: str | None = None
    template_threshold: float = 0.85
    # Colour of the game-over banner, and how far a pixel may sit from it on
    # *every* channel to still count as part of the banner.
    game_over_color_bgr: tuple[int, int, int] = (32, 32, 160)
    color_tolerance: float = 45.0
    # Fraction of the region that must match that colour. Well above what any
    # sprite drifting through the region can cover, and comfortably below a
    # solid banner with text on it.
    color_coverage: float = 0.6
    # Consecutive positive detections required before terminating. Debounces
    # single-frame flashes and screen transitions.
    detection_patience: int = 2

    # --- Score detection -----------------------------------------------------
    score_region: Region = field(default_factory=lambda: Region(left=8, top=8, width=180, height=40))
    # "ocr" reads the number with Tesseract; "pixel" only detects that the score
    # box changed (and rewards the change). "auto" prefers OCR when available.
    score_mode: str = "auto"
    # Binarisation cut-off for both the OCR and the pixel-signature readers.
    score_threshold: int = 140
    # Fraction of the score box's *ink* (thresholded glyph) pixels that must
    # change for the pixel detector to count one point. Measured against the
    # glyphs rather than the whole crop, so the value does not depend on how
    # generously the score box was drawn. Anything from ~0.01 to ~0.1 behaves
    # identically on the bundled game.
    pixel_change_ratio: float = 0.05

    # --- Stagnation / Idle detection -----------------------------------------
    # When consecutive frames have high similarity (e.g. > 90% identical / low delta),
    # the scene is standing still (player is idling/stuck).
    # idle_diff_threshold: normalized mean absolute pixel change below which
    # the frame is considered stagnant/idle (e.g. 0.05 = 95% similarity).
    idle_diff_threshold: float = 0.05
    # Penalty applied when the scene is standing still (negative float, e.g. -0.1).
    idle_penalty: float = 0.0
    # Bonus reward applied when the scene is actively moving/progressing.
    movement_reward: float = 0.0


@dataclass
class EnvConfig:
    """Episode pacing and reset behaviour."""

    # Agent decisions per second. The env sleeps to hold this rate so that the
    # policy sees a consistent time step (crucial for frame-stacked velocity).
    target_fps: float = 10.0
    # Truncate runaway episodes so the learner keeps seeing fresh starts.
    max_episode_steps: int = 3_000
    # Grace period after tapping the restart keys, letting the game redraw.
    reset_delay: float = 1.0
    # Maximum time spent waiting for the game-over screen to clear on reset.
    reset_timeout: float = 10.0
    # Extra settle time before the first observation of a new episode.
    warmup_frames: int = 2


@dataclass
class TrainConfig:
    """Stable-Baselines3 hyper-parameters."""

    algo: str = "ppo"
    total_timesteps: int = 200_000
    learning_rate: float = 2.5e-4
    # Short rollouts: real-time games generate samples slowly, so update often.
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # DQN-only knobs.
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    train_freq: int = 4
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.2
    # Divide rewards by their running standard deviation before the update.
    # The reward weights are hand-picked magnitudes (+0.1 vs -100), and PPO
    # shares one feature extractor between the actor and the critic, so an
    # un-normalised value loss in the tens dominates the policy gradient.
    # On-policy only; it is ignored for DQN, whose replay buffer would mix
    # transitions scaled by different running statistics.
    normalize_reward: bool = True
    clip_reward: float = 10.0
    # Fused image+audio feature width produced by MultiModalExtractor.
    features_dim: int = 512
    checkpoint_every: int = 10_000
    seed: int | None = None
    device: str = "auto"


@dataclass
class TrackerConfig:
    """Settings for Controllability Probe and Dynamic Entity Tracker."""

    enabled: bool = False
    auto_probe: bool = False
    probe_actions: int = 6  # number of test probe action steps to run
    history_len: int = 30
    min_entity_area: int = 16
    max_entity_area: int = 15000
    collision_distance_threshold: float = 24.0
    # Global motion compensation: threshold above which camera scrolling is detected
    scroll_threshold: float = 25.0
    # Max simultaneous detected entities before throttling as a scene change
    max_active_entities: int = 15


@dataclass
class HUDConfig:
    """Settings for Autonomous HUD Detection & Cognitive VLM Supervisor."""

    auto_detect: bool = True
    use_vlm: bool = False
    vlm_endpoint: str = "http://localhost:11434/api/generate"  # Ollama / OpenAI-compatible
    vlm_model: str = "qwen2.5-vl"
    vlm_timeout: float = 3.0
    # Longest image side (px) sent to the VLM; larger frames are downscaled and
    # any boxes the model returns are mapped back to full resolution.
    vlm_max_image_dim: int = 640
    # Asynchronous VLM Sentinel & Cognitive Game Supervisor
    vlm_sentinel_interval: float = 2.0
    # A sentinel verdict older than this (seconds) is ignored for terminal
    # decisions such as game-over short-circuits and menu navigation.
    vlm_max_age: float = 6.0
    # Minimum sentinel confidence for a game-over verdict to end the episode
    # without waiting for ``reward.detection_patience``.
    vlm_game_over_confidence: float = 0.75
    # Menu navigation injects input on its own; opt in explicitly.
    auto_menu_nav: bool = False
    # Minimum seconds between automatic restart taps triggered by a menu verdict.
    menu_nav_cooldown: float = 1.5
    # Semantic Cognitive Observation Space for RL
    cognitive_obs: bool = False
    cognitive_dim: int = 8
    # ``lives`` from the VLM is normalised against this for the cognitive vector.
    cognitive_max_lives: float = 5.0


@dataclass
class AppConfig:
    """Root config object passed around the whole project."""

    capture: CaptureConfig = field(default_factory=CaptureConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    hud: HUDConfig = field(default_factory=HUDConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        return _build(cls, data)

    @classmethod
    def load(cls, path: str | Path) -> AppConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# Field name -> dataclass, used to rebuild the tree from JSON. Resolving by name
# rather than by annotation is necessary because `from __future__ import
# annotations` turns every field type into a string.
_NESTED_FIELDS: dict[str, type] = {
    "capture": CaptureConfig,
    "vision": VisionConfig,
    "audio": AudioConfig,
    "control": ControlConfig,
    "reward": RewardConfig,
    "env": EnvConfig,
    "train": TrainConfig,
    "tracker": TrackerConfig,
    "hud": HUDConfig,
    "region": Region,
    "game_over_region": Region,
    "score_region": Region,
}


def _build(cls: type, data: Any) -> Any:
    """Rebuild nested dataclasses from plain JSON dicts, ignoring stale keys."""
    if not is_dataclass(cls) or not isinstance(data, dict):
        return data
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        sub_cls = _NESTED_FIELDS.get(f.name)
        if sub_cls is not None and isinstance(value, dict):
            value = _build(sub_cls, value)
        kwargs[f.name] = value
    if cls is RewardConfig and isinstance(kwargs.get("game_over_color_bgr"), list):
        kwargs["game_over_color_bgr"] = tuple(kwargs["game_over_color_bgr"])
    return cls(**kwargs)


def default_config() -> AppConfig:
    return AppConfig()


def load_config(path: str | Path | None) -> AppConfig:
    """Load a JSON config, or return defaults when no path is given."""
    if path is None:
        return default_config()
    return AppConfig.load(path)
