"""The critic: derive reward and termination from the raw game frame.

A game running behind a window exposes no API, so the reward function has to be
*inferred from pixels*. This module implements that inference:

* **Termination** -- ``cv2.matchTemplate`` against a saved crop of the
  game-over screen, or a mean-colour check on a fixed region when no template
  is available. Detections are debounced over consecutive frames because a
  single-frame false positive would end the episode and poison the return.
* **Scoring** -- OCR of the score box with Tesseract when installed, otherwise
  a pixel-signature detector that rewards *changes* in the score box without
  needing to read the digits.

Reward shaping used below:

    r = step_reward                      (survival drip, dense signal)
      + score_reward * points_gained     (sparse task reward)
      + game_over_penalty  if terminal   (dominant negative)

The survival drip is what lets PPO make progress before it has ever scored;
the terminal penalty is large enough that the value function learns to avoid
death rather than farm the drip next to a hazard.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from .config import RewardConfig
from .perception import BGRFrame, crop_region

_DIGITS = re.compile(r"\d+")


# ---------------------------------------------------------------------------
# Game-over detection
# ---------------------------------------------------------------------------
@runtime_checkable
class GameOverDetector(Protocol):
    def detect(self, frame: BGRFrame) -> tuple[bool, float]:
        """Return ``(is_game_over, confidence)`` for one frame."""
        ...


class TemplateGameOverDetector:
    """Normalised cross-correlation against a saved game-over crop.

    Robust to minor scaling/anti-aliasing differences, and tolerant of the
    background changing around the banner.
    """

    def __init__(self, template_path: str | Path, reward: RewardConfig) -> None:
        path = Path(template_path)
        template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"Could not read game-over template: {path}")
        self._template = template
        self._reward = reward

    def detect(self, frame: BGRFrame) -> tuple[bool, float]:
        search = crop_region(frame, self._reward.game_over_region)
        if search.size == 0:
            return False, 0.0
        gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY) if search.ndim == 3 else search
        th, tw = self._template.shape[:2]
        if gray.shape[0] < th or gray.shape[1] < tw:
            # Region smaller than the template: compare the template shrunk to
            # fit rather than silently reporting "no game over" forever.
            gray = cv2.resize(gray, (max(tw, gray.shape[1]), max(th, gray.shape[0])))
        result = cv2.matchTemplate(gray, self._template, cv2.TM_CCOEFF_NORMED)
        score = float(result.max())
        return score >= self._reward.template_threshold, score


class ColorGameOverDetector:
    """Mean-colour check on a fixed region.

    Cheapest possible detector: most games dim the playfield or show a solid
    banner on death, which moves the mean BGR of the centre region far enough
    to be unambiguous.
    """

    def __init__(self, reward: RewardConfig) -> None:
        self._reward = reward
        self._target = np.asarray(reward.game_over_color_bgr, dtype=np.float32)

    def detect(self, frame: BGRFrame) -> tuple[bool, float]:
        patch = crop_region(frame, self._reward.game_over_region)
        if patch.size == 0:
            return False, 0.0
        mean = patch.reshape(-1, patch.shape[-1]).mean(axis=0).astype(np.float32)
        distance = float(np.abs(mean - self._target).mean())
        tolerance = max(self._reward.color_tolerance, 1e-6)
        # Map distance onto a 0..1 confidence so logs are comparable with the
        # template detector's correlation score.
        confidence = max(0.0, 1.0 - distance / tolerance)
        return distance <= self._reward.color_tolerance, confidence


def build_game_over_detector(reward: RewardConfig) -> GameOverDetector:
    """Prefer template matching; fall back to the colour heuristic."""
    if reward.game_over_template:
        path = Path(reward.game_over_template)
        if path.exists():
            return TemplateGameOverDetector(path, reward)
        warnings.warn(
            f"Game-over template {path} not found; using the mean-colour detector.",
            RuntimeWarning,
            stacklevel=2,
        )
    return ColorGameOverDetector(reward)


# ---------------------------------------------------------------------------
# Score detection
# ---------------------------------------------------------------------------
@runtime_checkable
class ScoreSignal(Protocol):
    def reset(self) -> None: ...

    def update(self, frame: BGRFrame) -> tuple[float, int | None]:
        """Return ``(points_gained, absolute_score_or_None)``."""
        ...


def _binarise_score_box(frame: BGRFrame, reward: RewardConfig) -> np.ndarray:
    """Crop the score box and threshold it to a clean black-on-white mask."""
    patch = crop_region(frame, reward.score_region)
    if patch.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    _, mask = cv2.threshold(gray, reward.score_threshold, 255, cv2.THRESH_BINARY)
    return mask


class OcrScoreSignal:
    """Read the score with Tesseract and reward the increase.

    Only forward jumps up to ``max_score_delta`` are rewarded: a misread such as
    ``7 -> 771`` would otherwise hand the agent a huge bogus reward, and a
    counter reset to 0 must not be punished as a negative reward.
    """

    def __init__(self, reward: RewardConfig) -> None:
        self._reward = reward
        self._last_score: int | None = None
        import pytesseract  # imported eagerly so build_score_signal can fall back

        self._pytesseract = pytesseract
        # Single-line, digits only: dramatically more accurate than free-form
        # OCR on a HUD counter.
        self._tess_config = "--psm 7 -c tessedit_char_whitelist=0123456789"

    def reset(self) -> None:
        self._last_score = None

    def update(self, frame: BGRFrame) -> tuple[float, int | None]:
        score = self._read_score(frame)
        if score is None:
            return 0.0, self._last_score
        previous, self._last_score = self._last_score, score
        if previous is None:
            return 0.0, score
        delta = score - previous
        if 0 < delta <= self._reward.max_score_delta:
            return float(delta), score
        return 0.0, score

    def _read_score(self, frame: BGRFrame) -> int | None:
        mask = _binarise_score_box(frame, self._reward)
        if mask.size <= 1:
            return None
        # Tesseract expects dark glyphs on a light background, and is far more
        # reliable when the text is at least ~30 px tall.
        mask = cv2.bitwise_not(mask)
        if mask.shape[0] < 32:
            scale = 32 / mask.shape[0]
            mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        try:
            text = self._pytesseract.image_to_string(mask, config=self._tess_config)
        except Exception:
            return None
        match = _DIGITS.search(text.replace(" ", ""))
        return int(match.group()) if match else None


class PixelChangeScoreSignal:
    """Dependency-free score detector: reward *any* change in the score box.

    We cannot know how many points were awarded, so each detected change counts
    as one point. That is enough for the policy gradient -- the agent only needs
    the event to be correlated with good behaviour.
    """

    def __init__(self, reward: RewardConfig) -> None:
        self._reward = reward
        self._last_mask: np.ndarray | None = None

    def reset(self) -> None:
        self._last_mask = None

    def update(self, frame: BGRFrame) -> tuple[float, int | None]:
        mask = _binarise_score_box(frame, self._reward)
        previous, self._last_mask = self._last_mask, mask
        if previous is None or previous.shape != mask.shape or mask.size <= 1:
            return 0.0, None
        changed_ratio = float(np.count_nonzero(previous != mask)) / mask.size
        points = 1.0 if changed_ratio >= self._reward.pixel_change_ratio else 0.0
        return points, None


def build_score_signal(reward: RewardConfig) -> ScoreSignal:
    """Resolve ``reward.score_mode`` into a concrete detector.

    ``auto`` tries OCR and degrades to the pixel detector when Tesseract (the
    Python wrapper *or* the binary) is missing.
    """
    mode = reward.score_mode.lower()
    if mode == "pixel":
        return PixelChangeScoreSignal(reward)
    if mode in {"ocr", "auto"}:
        try:
            signal = OcrScoreSignal(reward)
            import pytesseract

            pytesseract.get_tesseract_version()  # raises if the binary is absent
            return signal
        except Exception as exc:
            if mode == "ocr":
                raise RuntimeError(
                    f"score_mode='ocr' requires pytesseract and the Tesseract binary on PATH: {exc}"
                ) from exc
            warnings.warn(
                f"Tesseract unavailable ({exc}); using the pixel-change score detector.",
                RuntimeWarning,
                stacklevel=2,
            )
            return PixelChangeScoreSignal(reward)
    raise ValueError(f"Unknown score_mode {reward.score_mode!r}; use auto/ocr/pixel.")


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------
@dataclass
class FrameVerdict:
    """Everything the environment needs to know about one frame."""

    reward: float
    terminated: bool
    points: float = 0.0
    score: int | None = None
    game_over_confidence: float = 0.0
    info: dict[str, float] = field(default_factory=dict)


class GameObserver:
    """Scores frames: reward shaping plus terminal detection.

    Stateful by design -- it tracks the previous score and a detection streak
    across the episode, so call :meth:`reset` at the start of each one.
    """

    def __init__(
        self,
        reward: RewardConfig,
        *,
        game_over_detector: GameOverDetector | None = None,
        score_signal: ScoreSignal | None = None,
    ) -> None:
        self._reward = reward
        self._game_over = game_over_detector or build_game_over_detector(reward)
        self._score = score_signal or build_score_signal(reward)
        self._streak = 0
        self._episode_points = 0.0

    @property
    def episode_points(self) -> float:
        return self._episode_points

    def reset(self) -> None:
        self._streak = 0
        self._episode_points = 0.0
        self._score.reset()

    def is_game_over(self, frame: BGRFrame) -> bool:
        """Single-frame, non-debounced check. Used while waiting for a restart."""
        return self._game_over.detect(frame)[0]

    def evaluate(self, frame: BGRFrame) -> FrameVerdict:
        """Turn a frame into reward + termination."""
        detected, confidence = self._game_over.detect(frame)
        self._streak = self._streak + 1 if detected else 0
        terminated = self._streak >= max(1, self._reward.detection_patience)

        if terminated:
            # No score reward on the terminal frame: the game-over overlay
            # usually covers the HUD and would produce a spurious reading.
            return FrameVerdict(
                reward=self._reward.game_over_penalty,
                terminated=True,
                game_over_confidence=confidence,
                info={"game_over": 1.0},
            )

        points, score = self._score.update(frame)
        self._episode_points += points
        reward = self._reward.step_reward + self._reward.score_reward * points
        return FrameVerdict(
            reward=float(reward),
            terminated=False,
            points=points,
            score=score,
            game_over_confidence=confidence,
            info={"game_over": 0.0, "points": points},
        )
