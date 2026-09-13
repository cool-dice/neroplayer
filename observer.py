"""Observer: the reward function. Reads game state off the screen with OpenCV.

The RL agent never sees the game's memory, so *all* reward signal must be
recovered from pixels:

* **Game over** – template matching (``cv2.matchTemplate``) against a cropped
  screenshot of the game-over banner, falling back to a mean-colour check on
  a small region when no template is available. Detection must persist for
  ``game_over_confirm_frames`` consecutive frames to filter out flicker.
* **Score** – OCR of the score region with ``pytesseract`` when a Tesseract
  binary is installed (reward proportional to the score delta), otherwise a
  pixel-change detector on the score region (reward per "the digits changed"
  event). Both are robust to the score wrapping/resetting: negative deltas are
  ignored.

Rewards are shaped as: ``+reward_alive`` every step, ``+reward_score_increase``
per point, ``reward_game_over`` (large negative) on termination.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from config import RewardConfig

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """What the observer reports for one frame."""

    reward: float
    terminated: bool
    info: dict[str, Any] = field(default_factory=dict)


class GameObserver:
    """Turns a full-resolution BGR frame of the capture region into a :class:`Signal`."""

    def __init__(self, cfg: RewardConfig) -> None:
        self.cfg = cfg
        self._template = self._load_template()
        self._ocr = self._init_ocr() if cfg.use_ocr else None
        self._score: int = 0
        self._prev_score_bits: np.ndarray | None = None
        self._game_over_streak = 0

        mode = "template" if self._template is not None else "colour"
        logger.info(
            "GameObserver ready: game-over via %s matching, score via %s",
            mode,
            "OCR" if self._ocr else "pixel-change",
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def reset(self, frame: np.ndarray) -> None:
        """Snapshot the score region of the first frame of an episode."""
        self._game_over_streak = 0
        self._prev_score_bits = self._binarise_score(frame)
        self._score = self._read_score(frame) if self._ocr else 0

    def evaluate(self, frame: np.ndarray) -> Signal:
        """Compute reward and termination for the newest frame."""
        game_over, match_score = self.detect_game_over(frame)
        if game_over:
            return Signal(
                reward=self.cfg.reward_game_over,
                terminated=True,
                info={"game_over": True, "match_score": match_score, "score": self._score},
            )

        points = self.detect_score_change(frame)
        reward = self.cfg.reward_alive + points * self.cfg.reward_score_increase
        return Signal(
            reward=reward,
            terminated=False,
            info={"game_over": False, "match_score": match_score, "points": points, "score": self._score},
        )

    # ------------------------------------------------------------------ #
    # Game over
    # ------------------------------------------------------------------ #
    def detect_game_over(self, frame: np.ndarray) -> tuple[bool, float]:
        """Return ``(confirmed, confidence)`` for the game-over screen."""
        if self._template is not None:
            hit, confidence = self._match_template(frame)
        else:
            hit, confidence = self._match_colour(frame)

        self._game_over_streak = self._game_over_streak + 1 if hit else 0
        confirmed = self._game_over_streak >= self.cfg.game_over_confirm_frames
        return confirmed, confidence

    def _match_template(self, frame: np.ndarray) -> tuple[bool, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        th, tw = self._template.shape[:2]
        if gray.shape[0] < th or gray.shape[1] < tw:
            logger.warning("Game-over template larger than frame; disabling template matching")
            self._template = None
            return self._match_colour(frame)
        result = cv2.matchTemplate(gray, self._template, cv2.TM_CCOEFF_NORMED)
        confidence = float(result.max())
        return confidence >= self.cfg.game_over_match_threshold, confidence

    def _match_colour(self, frame: np.ndarray) -> tuple[bool, float]:
        rows, cols = self.cfg.game_over_region.as_slice()
        roi = frame[rows, cols]
        if roi.size == 0:
            return False, 0.0
        mean_bgr = roi.reshape(-1, 3).mean(axis=0)
        distance = float(np.abs(mean_bgr - np.asarray(self.cfg.game_over_color_bgr)).max())
        # Express as a 0..1 confidence so the info dict is comparable across modes.
        confidence = max(0.0, 1.0 - distance / 255.0)
        return distance <= self.cfg.game_over_color_tolerance, confidence

    def _load_template(self) -> np.ndarray | None:
        path = self.cfg.game_over_template
        if not path.is_file():
            logger.info("No game-over template at %s; using colour check", path)
            return None
        template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            logger.warning("Could not read template %s; using colour check", path)
        return template

    # ------------------------------------------------------------------ #
    # Score
    # ------------------------------------------------------------------ #
    def detect_score_change(self, frame: np.ndarray) -> int:
        """Return the number of points gained since the previous frame (>= 0)."""
        if self._ocr:
            new_score = self._read_score(frame)
            if new_score is None:
                return 0
            delta = new_score - self._score
            if delta < 0:
                # Score reset (new game) or misread: never punish here, the
                # game-over detector owns negative outcomes.
                self._score = new_score
                return 0
            delta = min(delta, self.cfg.max_score_delta)
            self._score = new_score
            return delta

        bits = self._binarise_score(frame)
        if self._prev_score_bits is None or bits.shape != self._prev_score_bits.shape:
            self._prev_score_bits = bits
            return 0
        changed = float(np.mean(bits != self._prev_score_bits))
        self._prev_score_bits = bits
        if changed >= self.cfg.score_change_pixel_fraction:
            self._score += 1
            return 1
        return 0

    def _score_roi(self, frame: np.ndarray) -> np.ndarray:
        rows, cols = self.cfg.score_region.as_slice()
        return frame[rows, cols]

    def _binarise_score(self, frame: np.ndarray) -> np.ndarray:
        """Otsu-threshold the score region so anti-aliasing noise is ignored."""
        roi = self._score_roi(frame)
        if roi.size == 0:
            return np.zeros((0, 0), dtype=bool)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, bits = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return bits.astype(bool)

    def _read_score(self, frame: np.ndarray) -> int | None:
        """OCR the score region; returns ``None`` when no digits are recognised."""
        roi = self._score_roi(frame)
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Tesseract likes large, high-contrast glyphs: upscale 3x and binarise.
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if binary.mean() < 127:  # ensure dark text on light background
            binary = cv2.bitwise_not(binary)
        try:
            text = self._ocr.image_to_string(
                binary, config="--psm 7 -c tessedit_char_whitelist=0123456789"
            )
        except Exception:  # pragma: no cover - external binary
            logger.exception("OCR failed; disabling OCR for this run")
            self._ocr = None
            return None
        digits = re.sub(r"\D", "", text)
        return int(digits) if digits else None

    @staticmethod
    def _init_ocr():
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            return pytesseract
        except Exception as exc:
            logger.info("pytesseract/tesseract unavailable (%s); using pixel-change score detector", exc)
            return None
