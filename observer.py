"""Visual reward and terminal-state detection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import GameConfig, Region


@dataclass(frozen=True)
class ObserverResult:
    reward: float
    terminated: bool
    info: dict[str, float | bool | str]


class GameStateObserver:
    """Infer score changes and game-over events from screen pixels."""

    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self._previous_score_roi: np.ndarray | None = None
        self._score_change_active = False
        self._template = self._load_template()

    def _load_template(self) -> np.ndarray | None:
        path = self.config.game_over_template
        if not path.is_file():
            return None
        template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise ValueError(f"Could not decode game-over template: {path}")
        return template

    @staticmethod
    def _crop(frame: np.ndarray, region: Region) -> np.ndarray:
        height, width = frame.shape[:2]
        x1 = max(0, min(region.left, width))
        y1 = max(0, min(region.top, height))
        x2 = max(x1, min(region.left + region.width, width))
        y2 = max(y1, min(region.top + region.height, height))
        if x1 == x2 or y1 == y2:
            raise ValueError(f"Region {region} does not overlap the captured frame.")
        return frame[y1:y2, x1:x2]

    def reset(self, frame: np.ndarray) -> None:
        score_roi = self._crop(frame, self.config.score_region)
        self._previous_score_roi = cv2.cvtColor(score_roi, cv2.COLOR_BGR2GRAY)
        self._score_change_active = False

    def _detect_game_over(self, frame: np.ndarray) -> tuple[bool, float, str]:
        roi = self._crop(frame, self.config.game_over_region)
        if self._template is not None:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            template_height, template_width = self._template.shape
            if gray.shape[0] < template_height or gray.shape[1] < template_width:
                return False, 0.0, "template"
            matches = cv2.matchTemplate(gray, self._template, cv2.TM_CCOEFF_NORMED)
            confidence = float(matches.max())
            return (
                confidence >= self.config.template_match_threshold,
                confidence,
                "template",
            )

        target = np.asarray(self.config.game_over_bgr, dtype=np.int16)
        difference = np.abs(roi.astype(np.int16) - target)
        matching_pixels = np.all(difference <= self.config.game_over_color_tolerance, axis=2)
        ratio = float(matching_pixels.mean())
        return ratio >= self.config.game_over_min_pixel_ratio, ratio, "color"

    def _detect_score_change(self, frame: np.ndarray) -> tuple[bool, float]:
        current = cv2.cvtColor(
            self._crop(frame, self.config.score_region),
            cv2.COLOR_BGR2GRAY,
        )
        if self._previous_score_roi is None:
            self._previous_score_roi = current
            return False, 0.0
        if current.shape != self._previous_score_roi.shape:
            self._previous_score_roi = current
            return False, 0.0

        delta = float(cv2.absdiff(current, self._previous_score_roi).mean())
        changed = delta >= self.config.score_change_threshold
        # Reward only on the leading edge so animated score pixels do not farm reward.
        awarded = changed and not self._score_change_active
        self._score_change_active = changed
        self._previous_score_roi = current
        return awarded, delta

    def evaluate(self, frame: np.ndarray) -> ObserverResult:
        """Return reward and terminal state inferred from a fresh BGR frame."""
        game_over, confidence, detector = self._detect_game_over(frame)
        if game_over:
            return ObserverResult(
                reward=self.config.game_over_penalty,
                terminated=True,
                info={
                    "game_over": True,
                    "game_over_confidence": confidence,
                    "game_over_detector": detector,
                    "score_changed": False,
                },
            )

        score_changed, score_delta = self._detect_score_change(frame)
        reward = self.config.score_reward if score_changed else self.config.living_reward
        return ObserverResult(
            reward=reward,
            terminated=False,
            info={
                "game_over": False,
                "game_over_confidence": confidence,
                "game_over_detector": detector,
                "score_changed": score_changed,
                "score_pixel_delta": score_delta,
            },
        )
