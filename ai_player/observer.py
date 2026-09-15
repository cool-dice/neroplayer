"""The critic: derive reward and termination from the raw game frame.

A game running behind a window exposes no API, so the reward function has to be
*inferred from pixels*. This module implements that inference:

* **Termination** -- ``cv2.matchTemplate`` against a saved crop of the
  game-over screen, or a banner-colour coverage check on a fixed region when no
  template is available. Detections are debounced over consecutive frames
  because a single-frame false positive would end the episode and poison the
  return.
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

import base64
import json
import re
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import cv2
import numpy as np

from .config import HUDConfig, Region, RewardConfig
from .perception import BGRFrame, crop_region

if TYPE_CHECKING:
    from .cognitive import CognitiveSupervisor

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
    """Normalised matching against a saved game-over crop.

    Robust to minor scaling/anti-aliasing differences, and tolerant of the
    background changing around the banner.

    Correlation (``TM_CCOEFF_NORMED``) is the right measure for a banner with
    text or artwork in it, but it is undefined for a template with no variance:
    both terms of the correlation coefficient vanish and OpenCV returns 0.0 for
    a *perfect* match just as it does for a total mismatch. A user who
    calibrates against a plain colour block would get a detector that can never
    fire, silently disabling termination. For such templates we switch to
    squared-difference matching, which compares absolute intensities and so
    still separates the two cases.
    """

    _FLAT_TEMPLATE_STD = 1e-3

    def __init__(self, template_path: str | Path, reward: RewardConfig) -> None:
        path = Path(template_path)
        template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"Could not read game-over template: {path}")
        self._template = template
        self._reward = reward
        self._uniform = float(template.std()) < self._FLAT_TEMPLATE_STD

    def detect(self, frame: BGRFrame) -> tuple[bool, float]:
        search = crop_region(frame, self._reward.game_over_region)
        if search.size == 0:
            return False, 0.0
        gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY) if search.ndim == 3 else search
        th, tw = self._template.shape[:2]
        if gray.shape[0] < th or gray.shape[1] < tw:
            # matchTemplate requires the search image to be at least as large as
            # the template. Stretching the region keeps detection working after
            # a window resize instead of silently never matching again.
            gray = cv2.resize(gray, (max(tw, gray.shape[1]), max(th, gray.shape[0])))
        if self._uniform:
            result = cv2.matchTemplate(gray, self._template, cv2.TM_SQDIFF_NORMED)
            score = 1.0 - float(result.min())
        else:
            result = cv2.matchTemplate(gray, self._template, cv2.TM_CCOEFF_NORMED)
            score = float(result.max())
        return score >= self._reward.template_threshold, score


class ColorGameOverDetector:
    """Solid-colour coverage check on a fixed region.

    Most games paint a banner or dim the playfield on death, so the test is
    "what fraction of this region is the game-over colour?".

    Deliberately *not* a mean-colour test. Averaging the region first, then
    averaging the per-channel distances, lets one badly mismatched channel be
    diluted by two close ones: on the bundled game a handful of red obstacles
    drifting through the region pulled the mean close enough to the dark-red
    banner to fire on 7% of ordinary gameplay frames, ending episodes that were
    still alive and charging the agent the full death penalty. Counting pixels
    that match on *every* channel cannot be fooled that way, because sprites
    never cover most of the region.
    """

    def __init__(self, reward: RewardConfig) -> None:
        self._reward = reward
        self._target = np.asarray(reward.game_over_color_bgr, dtype=np.int16)

    def detect(self, frame: BGRFrame) -> tuple[bool, float]:
        patch = crop_region(frame, self._reward.game_over_region)
        if patch.size == 0:
            return False, 0.0
        distance = np.abs(patch.astype(np.int16) - self._target)
        matching = distance.max(axis=2) <= self._reward.color_tolerance
        # Coverage doubles as the confidence score, which keeps logs comparable
        # with the template detector's correlation value.
        coverage = float(matching.mean())
        return coverage >= self._reward.color_coverage, coverage


class AutonomousGameOverDetector:
    """Multi-strategy zero-shot autonomous game over detection.

    Combines:
    a) Template matching when a template is present on disk.
    b) Fast OCR keyword detection looking for banners like "GAME OVER", "CONTINUE", etc.,
       including countdown patterns (e.g. "CONTINUE 9", "RETRY ?").
    c) Standalone CV edge correlation against synthetic banner masks and morphological
       horizontal glyph clustering (works with zero Tesseract binary dependencies).
    d) Fade / blackout / dimming detection on stationary center screen.
    e) Color fallback if game_over_color_bgr is explicitly matched.
    f) CognitiveSupervisor (VLM sentinel) confidence fusion when linked.
    """

    def __init__(
        self,
        reward: RewardConfig,
        *,
        supervisor: CognitiveSupervisor | None = None,
    ) -> None:
        self._reward = reward
        self._supervisor = supervisor
        self._keywords = [k.upper().strip() for k in reward.game_over_keywords if k.strip()]
        self._threshold = reward.game_over_threshold
        self._target_color = np.asarray(reward.game_over_color_bgr, dtype=np.int16)

        # Template detector if file exists
        self._template_detector: TemplateGameOverDetector | None = None
        if reward.game_over_template:
            tpl_path = Path(reward.game_over_template)
            if tpl_path.exists():
                try:
                    self._template_detector = TemplateGameOverDetector(tpl_path, reward)
                except Exception:
                    self._template_detector = None

        # Check pytesseract availability
        self._has_tesseract = False
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            self._pytesseract = pytesseract
            self._has_tesseract = True
        except Exception:
            self._has_tesseract = False

        # Previous frame for fade/blackout stationarity
        self._last_center_gray: np.ndarray | None = None

    def reset(self) -> None:
        """Reset stateful frame history."""
        self._last_center_gray = None

    def _extract_center_roi(self, frame: BGRFrame) -> np.ndarray:
        h, w = frame.shape[:2]
        # Inspect central 70% of screen
        y1, y2 = int(h * 0.15), int(h * 0.85)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        return frame[y1:y2, x1:x2]

    def _ocr_detect(self, roi: np.ndarray) -> tuple[float, str]:
        if not self._has_tesseract or not self._keywords:
            return 0.0, ""

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        inv = cv2.bitwise_not(thresh)

        # High-contrast color difference (e.g. red/yellow banner text on dark background)
        chroma_img = None
        if roi.ndim == 3:
            b, g, r = roi[:, :, 0].astype(float), roi[:, :, 1].astype(float), roi[:, :, 2].astype(float)
            chroma = np.maximum(np.abs(r - g), np.abs(r - b))
            otsu = cv2.THRESH_BINARY | cv2.THRESH_OTSU
            _, chroma_img = cv2.threshold(chroma.astype(np.uint8), 0, 255, otsu)

        candidates = [thresh, inv]
        if chroma_img is not None:
            candidates.append(chroma_img)

        config = "--psm 11"
        continue_countdown_pattern = re.compile(
            r"(CONTINUE|RETRY|COUNTDOWN)\s*[:?]?\s*([0-9])", re.IGNORECASE
        )
        for img in candidates:
            try:
                txt = self._pytesseract.image_to_string(img, config=config).upper()
                for kw in self._keywords:
                    if kw in txt:
                        return 0.95, kw
                # Check for countdown patterns near continue / retry
                m = continue_countdown_pattern.search(txt)
                if m:
                    return 0.95, m.group(0)
            except Exception:
                continue

        return 0.0, ""

    def _edge_and_contour_detect(self, roi: np.ndarray) -> tuple[float, str]:
        rh, rw = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

        # 1. Morphological horizontal glyph clustering (look for prominent text banner box)
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        abs_grad = cv2.convertScaleAbs(grad_x)
        _, thresh = cv2.threshold(abs_grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_boxes: list[tuple[int, int, int, int]] = []
        contour_conf = 0.0
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            ar = bw / float(bh) if bh > 0 else 0.0
            if 2.0 <= ar <= 15.0 and 10 <= bh <= rh * 0.6 and bw >= rw * 0.25:
                patch = thresh[by : by + bh, bx : bx + bw]
                density = float(np.count_nonzero(patch)) / float(bw * bh)
                cy = by + bh / 2.0
                center_dist = abs(cy - rh / 2.0) / (rh / 2.0)
                if 0.12 <= density <= 0.60 and center_dist <= 0.6:
                    candidate_boxes.append((bx, by, bw, bh))
                    pos_score = max(0.0, 1.0 - 0.3 * center_dist)
                    width_score = min(1.0, bw / (rw * 0.40))
                    c_score = pos_score * width_score * 0.85
                    if c_score > contour_conf:
                        contour_conf = c_score

        # 2. Multi-scale Canny edge correlation against synthetic text masks
        canny_screen = cv2.Canny(gray, 50, 150)
        blur_screen = cv2.GaussianBlur(canny_screen.astype(np.float32), (5, 5), 0)

        best_score = 0.0
        best_match = ""

        test_keywords = [
            kw for kw in self._keywords if kw in ("GAME OVER", "CONTINUE", "RETRY", "YOU DIED", "DEFEAT")
        ]
        if not test_keywords and self._keywords:
            test_keywords = self._keywords[:3]
        if "CONTINUE" not in test_keywords:
            test_keywords.append("CONTINUE")
        # Also include common countdown test strings
        test_keywords.extend(["CONTINUE 9", "RETRY 9", "CONTINUE?"])

        for kw in test_keywords:
            (base_w, _), _ = cv2.getTextSize(kw, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            if base_w <= 0:
                continue
            for fraction in (0.35, 0.50, 0.65, 0.80):
                target_w = int(rw * fraction)
                scale = target_w / float(base_w)
                if scale < 0.25 or scale > 2.5:
                    continue
                (tw, th), bl = cv2.getTextSize(kw, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
                if th + bl + 4 >= rh or tw + 4 >= rw:
                    continue
                banner = np.zeros((th + bl + 6, tw + 6), dtype=np.uint8)
                cv2.putText(banner, kw, (3, th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 2)
                cb = cv2.Canny(banner, 50, 150)
                blur_banner = cv2.GaussianBlur(cb.astype(np.float32), (5, 5), 0)
                res = cv2.matchTemplate(blur_screen, blur_banner, cv2.TM_CCOEFF_NORMED)
                score = float(res.max())
                if score > best_score:
                    best_score = score
                    best_match = kw

        # Only count high edge correlation when there is a corresponding candidate text box
        # or when correlation is exceptionally high (> 0.70)
        edge_conf = 0.0
        if candidate_boxes:
            if best_score >= 0.45:
                edge_conf = min(1.0, 0.75 + (best_score - 0.45) * 1.5)
            elif best_score >= 0.35:
                edge_conf = 0.50 + (best_score - 0.35) * 2.5
        elif best_score >= 0.70:
            edge_conf = min(1.0, 0.75 + (best_score - 0.70) * 1.5)

        combined = max(edge_conf, contour_conf)
        if edge_conf >= 0.5 and contour_conf >= 0.5:
            combined = min(1.0, max(edge_conf, contour_conf) + 0.15)

        return combined, best_match

    def _dimming_detect(self, roi: np.ndarray) -> float:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        mean_intensity = float(gray.mean())

        diff = 1.0
        if self._last_center_gray is not None and self._last_center_gray.shape == gray.shape:
            diff = float(cv2.absdiff(gray, self._last_center_gray).mean()) / 255.0
        self._last_center_gray = gray

        # Death blackout: screen brightness < 15 and stationary (diff < 0.02)
        if mean_intensity < 15.0 and diff < 0.02:
            return 0.85
        if mean_intensity < 8.0:
            return 0.70
        return 0.0

    def _color_detect(self, frame: BGRFrame) -> float:
        patch = crop_region(frame, self._reward.game_over_region)
        if patch.size == 0:
            return 0.0
        distance = np.abs(patch.astype(np.int16) - self._target_color)
        matching = distance.max(axis=2) <= self._reward.color_tolerance
        coverage = float(matching.mean())
        if coverage >= self._reward.color_coverage:
            return min(1.0, coverage)
        return 0.0

    def detect(self, frame: BGRFrame) -> tuple[bool, float]:
        confidences: list[float] = []

        # 1. Template matching if configured
        if self._template_detector is not None:
            t_detected, t_conf = self._template_detector.detect(frame)
            if t_detected:
                return True, t_conf
            confidences.append(t_conf)

        roi = self._extract_center_roi(frame)
        if roi.size == 0:
            return False, 0.0

        # 2. OCR detection
        ocr_conf, _ = self._ocr_detect(roi)
        if ocr_conf >= 0.9:
            return True, ocr_conf
        confidences.append(ocr_conf)

        # 3. CV edge correlation & morphological contour clustering
        cv_conf, _ = self._edge_and_contour_detect(roi)
        confidences.append(cv_conf)

        # 4. Color fallback
        col_conf = self._color_detect(frame)
        if col_conf >= self._reward.color_coverage:
            return True, col_conf
        confidences.append(col_conf)

        # 5. Screen dimming / fade detection
        dim_conf = self._dimming_detect(roi)
        confidences.append(dim_conf)

        # 6. Cognitive supervisor (VLM sentinel) state
        if self._supervisor is not None:
            cstate = self._supervisor.current_state
            if (cstate.is_game_over or cstate.state in ("game_over", "defeat")) and cstate.confidence >= 0.7:
                return True, 1.0
            if cstate.is_game_over or cstate.state in ("game_over", "defeat"):
                # High confidence from cognitive VLM supervisor
                vlm_conf = max(0.85, cstate.confidence)
                confidences.append(vlm_conf)

        final_conf = max(confidences) if confidences else 0.0
        return final_conf >= self._threshold, final_conf


def build_game_over_detector(
    reward: RewardConfig,
    *,
    supervisor: CognitiveSupervisor | None = None,
) -> GameOverDetector:
    """Build GameOverDetector according to reward.game_over_mode and configuration."""
    mode = getattr(reward, "game_over_mode", "auto").lower()

    if mode == "template":
        if not reward.game_over_template:
            raise ValueError("game_over_mode is 'template' but game_over_template is not specified.")
        path = Path(reward.game_over_template)
        return TemplateGameOverDetector(path, reward)

    if mode == "color":
        return ColorGameOverDetector(reward)

    if mode == "text":
        return AutonomousGameOverDetector(reward, supervisor=supervisor)

    # mode == "auto" (default)
    if reward.game_over_template:
        path = Path(reward.game_over_template)
        if path.exists():
            return TemplateGameOverDetector(path, reward)
        warnings.warn(
            f"Game-over template {path} not found; falling back to autonomous game-over detector.",
            RuntimeWarning,
            stacklevel=2,
        )
        return AutonomousGameOverDetector(reward, supervisor=supervisor)

    return AutonomousGameOverDetector(reward, supervisor=supervisor)


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
    """Crop the score box and threshold it so the digits are always white.

    HUDs come in both polarities -- light digits on a dark playfield, dark
    digits on a light panel -- and everything downstream (the OCR inversion,
    the ink-relative change ratio) assumes the glyphs are the non-zero pixels.
    Digits never fill most of their box, so whichever class is in the minority
    is the ink; normalising here means callers never have to care.
    """
    patch = crop_region(frame, reward.score_region)
    if patch.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    _, mask = cv2.threshold(gray, reward.score_threshold, 255, cv2.THRESH_BINARY)
    if np.count_nonzero(mask) * 2 > mask.size:
        mask = cv2.bitwise_not(mask)
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
        # The mask always carries white glyphs; Tesseract expects dark glyphs on
        # a light background, and is far more reliable when the text is at least
        # ~30 px tall.
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

    We cannot know how many points were awarded, so each changed frame counts
    as one point. That is enough for the policy gradient -- the agent only needs
    the event to be correlated with good behaviour -- but it does undercount
    when a game awards several points in the same frame. Install Tesseract if
    the exact delta matters.
    """

    def __init__(self, reward: RewardConfig) -> None:
        self._reward = reward
        self._last_mask: np.ndarray | None = None
        # "ink" compares against the glyph area, "box" against the whole crop.
        self._denominator = "ink"

    def reset(self) -> None:
        self._last_mask = None

    def update(self, frame: BGRFrame) -> tuple[float, int | None]:
        mask = _binarise_score_box(frame, self._reward)
        previous, self._last_mask = self._last_mask, mask
        if previous is None or previous.shape != mask.shape or mask.size <= 1:
            return 0.0, None
        changed = float(np.count_nonzero(previous != mask))
        # Measuring change against the digits rather than the whole crop keeps
        # the threshold meaningful: in a generously sized score box, a 5 -> 6
        # repaint moves well under 1% of the pixels but a large share of the
        # ink, so a box-relative threshold silently drops most increments.
        if self._denominator == "ink":
            area = float(np.count_nonzero(previous | mask))
        else:
            area = float(mask.size)
        ratio = changed / max(area, 1.0)
        points = 1.0 if ratio >= self._reward.pixel_change_ratio else 0.0
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
# Autonomous HUD Detection & VLM Observer Interface
# ---------------------------------------------------------------------------
@dataclass
class DetectedHUDRegions:
    """Estimated bounding regions for core game HUD elements."""

    score_region: Region | None = None
    lives_region: Region | None = None
    game_over_region: Region | None = None
    confidence: float = 0.0
    method: str = "none"


class VLMObserverInterface:
    """Interacts with local OpenAI-compatible or Ollama Vision-Language Models (e.g. Qwen2.5-VL, MiniCPM)."""

    def __init__(self, hud_config: HUDConfig) -> None:
        self.config = hud_config

    def detect_hud(self, frame: BGRFrame) -> DetectedHUDRegions | None:
        """Query local VLM to identify bounding boxes of Score, Lives/HP, and Game Over."""
        if not self.config.use_vlm:
            return None

        h, w = frame.shape[:2]
        # Encode frame as JPEG base64
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            return None
        b64_image = base64.b64encode(buffer).decode("utf-8")

        prompt = (
            f"You are analyzing a video game frame of resolution {w}x{h}. "
            "Identify the pixel bounding box for the following HUD elements if present: "
            "1. score (number counting points/score) "
            "2. lives (remaining lives or health bar) "
            "3. game_over (area where GAME OVER banner appears). "
            'Return ONLY valid JSON: {"score": [x,y,w,h], "lives": [x,y,w,h], "game_over": [x,y,w,h]}'
        )

        # Format payload depending on endpoint (OpenAI chat completions vs Ollama generate)
        is_openai = "chat/completions" in self.config.vlm_endpoint or "/v1/" in self.config.vlm_endpoint
        if is_openai:
            payload_dict = {
                "model": self.config.vlm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                            },
                        ],
                    }
                ],
                "stream": False,
                "temperature": 0.1,
            }
        else:
            payload_dict = {
                "model": self.config.vlm_model,
                "prompt": prompt,
                "images": [b64_image],
                "stream": False,
                "format": "json",
            }

        try:
            payload = json.dumps(payload_dict).encode("utf-8")

            req = urllib.request.Request(
                self.config.vlm_endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.config.vlm_timeout) as response:
                res_data = json.loads(response.read().decode("utf-8"))

            content = ""
            if "choices" in res_data and len(res_data["choices"]) > 0:
                choice = res_data["choices"][0]
                content = choice.get("message", {}).get("content", "")
            elif "response" in res_data:
                content = res_data["response"]
            elif "content" in res_data:
                content = res_data["content"]

            # Handle possible markdown formatting (e.g. ```json ... ```)
            if isinstance(content, str):
                cleaned = content.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    cleaned = "\n".join(lines).strip()
                # Find first '{' and last '}'
                start_brace = cleaned.find("{")
                end_brace = cleaned.rfind("}")
                if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                    cleaned = cleaned[start_brace : end_brace + 1]
                parsed = json.loads(cleaned)
            else:
                parsed = content

            score_box = parsed.get("score")
            lives_box = parsed.get("lives")
            go_box = parsed.get("game_over")

            def make_reg(box: list[int] | None) -> Region | None:
                if box and len(box) == 4 and box[2] > 0 and box[3] > 0:
                    return Region(left=int(box[0]), top=int(box[1]), width=int(box[2]), height=int(box[3]))
                return None

            return DetectedHUDRegions(
                score_region=make_reg(score_box),
                lives_region=make_reg(lives_box),
                game_over_region=make_reg(go_box),
                confidence=0.85,
                method="vlm",
            )
        except Exception as exc:
            warnings.warn(
                f"VLM HUD detection request failed ({exc}); falling back to heuristic detector.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None


class AutonomousHUDDetector:
    """Zero-shot autonomous HUD extraction.

    Uses high-speed visual saliency, edge contrast, and OCR/glyph clustering in typical
    HUD bands (top-left, top-right, bottom), with optional VLM integration.
    """

    def __init__(self, hud_config: HUDConfig | None = None) -> None:
        self.config = hud_config or HUDConfig()
        self.vlm = VLMObserverInterface(self.config)

    def detect_hud(self, frame: BGRFrame) -> DetectedHUDRegions:
        """Perform zero-shot HUD detection with VLM or heuristic fallback."""
        if self.config.use_vlm:
            vlm_res = self.vlm.detect_hud(frame)
            if vlm_res is not None and vlm_res.score_region is not None:
                return vlm_res

        return self._heuristic_detect(frame)

    def _heuristic_detect(self, frame: BGRFrame) -> DetectedHUDRegions:
        """Heuristic HUD detector running locally with zero external network dependencies."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        # Typical HUD regions are located in top band (0 to 18% of height)
        # or bottom band (88% to 100% of height)
        top_band_h = max(20, int(h * 0.18))
        top_band = gray[:top_band_h, :]

        # Look for text-like contours or high-contrast digit clusters in top band
        # 1. Edge & gradient analysis
        grad_x = cv2.Sobel(top_band, cv2.CV_16S, 1, 0, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        _, thresh = cv2.threshold(abs_grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Close horizontally to merge character glyphs into words/numbers
        morph_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_k)

        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_regions: list[tuple[int, int, int, int]] = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Filter contours matching typical HUD digit/label ratios
            if 10 <= cw <= int(w * 0.4) and 8 <= ch <= top_band_h:
                candidate_regions.append((x, y, cw, ch))

        score_reg: Region | None = None
        lives_reg: Region | None = None

        if candidate_regions:
            # Sort left to right
            candidate_regions.sort(key=lambda r: r[0])
            # The first prominent candidate in top-left or top-right is usually score
            first = candidate_regions[0]
            # Add small padding
            pad = 2
            score_reg = Region(
                left=max(0, first[0] - pad),
                top=max(0, first[1] - pad),
                width=min(w - first[0], first[2] + 2 * pad),
                height=min(h - first[1], first[3] + 2 * pad),
            )
            if len(candidate_regions) > 1:
                second = candidate_regions[-1]
                lives_reg = Region(
                    left=max(0, second[0] - pad),
                    top=max(0, second[1] - pad),
                    width=min(w - second[0], second[2] + 2 * pad),
                    height=min(h - second[1], second[3] + 2 * pad),
                )
        else:
            # Default fallback: top-left corner band for score
            score_reg = Region(left=8, top=4, width=min(180, w // 2), height=min(36, h // 5))

        # Default fallback for game-over: centered banner
        banner_w = int(w * 0.6)
        banner_h = int(h * 0.25)
        go_reg = Region(
            left=(w - banner_w) // 2,
            top=(h - banner_h) // 2,
            width=banner_w,
            height=banner_h,
        )

        return DetectedHUDRegions(
            score_region=score_reg,
            lives_region=lives_reg,
            game_over_region=go_reg,
            confidence=0.75 if candidate_regions else 0.5,
            method="heuristic",
        )


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
    frame_diff: float = 0.0
    is_idle: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class GameObserver:
    """Scores frames: reward shaping plus terminal detection.

    Stateful by design -- it tracks the previous score and a detection streak
    across the episode, so call :meth:`reset` at the start of each one.
    """

    def __init__(
        self,
        reward: RewardConfig,
        *,
        hud_config: HUDConfig | None = None,
        supervisor: CognitiveSupervisor | None = None,
        game_over_detector: GameOverDetector | None = None,
        score_signal: ScoreSignal | None = None,
    ) -> None:
        self._reward = reward
        self._hud_config = hud_config
        self._supervisor = supervisor
        self._hud_detector = (
            AutonomousHUDDetector(hud_config) if hud_config and hud_config.auto_detect else None
        )
        self._detected_hud: DetectedHUDRegions | None = None
        self._game_over = game_over_detector or build_game_over_detector(reward, supervisor=supervisor)
        self._score = score_signal or build_score_signal(reward)
        self._streak = 0
        self._episode_points = 0.0
        self._last_frame_gray: np.ndarray | None = None

    @property
    def supervisor(self) -> CognitiveSupervisor | None:
        return self._supervisor

    @property
    def episode_points(self) -> float:
        return self._episode_points

    @property
    def detected_hud(self) -> DetectedHUDRegions | None:
        return self._detected_hud

    def update_dynamic_hud(self, regions: dict[str, list[int]]) -> None:
        """Dynamically update score/game-over regions and detectors from dynamic HUD boxes."""
        if not regions:
            return

        # Update supervisor if attached
        if self._supervisor is not None:
            self._supervisor.update_dynamic_hud(regions)

        # Update score region if valid
        score_box = regions.get("score")
        if score_box and len(score_box) == 4 and score_box[2] > 0 and score_box[3] > 0:
            self._reward.score_region = Region(
                left=int(score_box[0]),
                top=int(score_box[1]),
                width=int(score_box[2]),
                height=int(score_box[3]),
            )
            self._score = build_score_signal(self._reward)

        # Update game_over region if valid and template not explicitly configured
        go_box = regions.get("game_over")
        if (
            go_box
            and len(go_box) == 4
            and go_box[2] > 0
            and go_box[3] > 0
            and not self._reward.game_over_template
        ):
            self._reward.game_over_region = Region(
                left=int(go_box[0]),
                top=int(go_box[1]),
                width=int(go_box[2]),
                height=int(go_box[3]),
            )
            self._game_over = build_game_over_detector(self._reward, supervisor=self._supervisor)

    def auto_configure_hud(self, frame: BGRFrame) -> DetectedHUDRegions | None:
        """Autonomously detect and populate HUD regions if not explicitly pinned."""
        if self._hud_detector is None:
            return None
        self._detected_hud = self._hud_detector.detect_hud(frame)
        if self._detected_hud.score_region is not None:
            self._reward.score_region = self._detected_hud.score_region
            self._score = build_score_signal(self._reward)
        if self._detected_hud.game_over_region is not None and not self._reward.game_over_template:
            self._reward.game_over_region = self._detected_hud.game_over_region
            self._game_over = build_game_over_detector(self._reward, supervisor=self._supervisor)
        return self._detected_hud

    def reset(self) -> None:
        self._streak = 0
        self._episode_points = 0.0
        self._last_frame_gray = None
        self._score.reset()
        if hasattr(self._game_over, "reset"):
            self._game_over.reset()

    def is_game_over(self, frame: BGRFrame) -> bool:
        """Single-frame, non-debounced check. Used while waiting for a restart."""
        return self._game_over.detect(frame)[0]

    def _compute_frame_diff(self, frame: BGRFrame) -> float:
        """Calculate normalized mean absolute pixel change against the previous frame [0.0, 1.0]."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self._last_frame_gray is None or self._last_frame_gray.shape != gray.shape:
            self._last_frame_gray = gray
            return 1.0  # first frame is considered new/active
        diff = float(cv2.absdiff(gray, self._last_frame_gray).mean()) / 255.0
        self._last_frame_gray = gray
        return diff

    def evaluate(self, frame: BGRFrame) -> FrameVerdict:
        """Turn a frame into reward + termination."""
        detected, confidence = self._game_over.detect(frame)
        self._streak = self._streak + 1 if detected else 0
        terminated = self._streak >= max(1, self._reward.detection_patience)

        # Instant VLM game-over: if sentinel confirms game over, bypass detection patience
        if not terminated and self._supervisor is not None:
            cstate = self._supervisor.current_state
            if (cstate.is_game_over or cstate.state in ("game_over", "defeat")) and cstate.confidence >= 0.75:
                terminated = True
                confidence = max(confidence, cstate.confidence)

        frame_diff = self._compute_frame_diff(frame)
        is_idle = frame_diff < self._reward.idle_diff_threshold

        if terminated:
            # No score reward on the terminal frame: the game-over overlay
            # usually covers the HUD and would produce a spurious reading.
            vlm_info: dict[str, Any] = {}
            if self._supervisor is not None:
                cstate = self._supervisor.current_state
                vlm_info = {
                    "vlm_state": cstate.state,
                    "vlm_is_game_over": 1.0 if cstate.is_game_over else 0.0,
                    "vlm_conf": cstate.confidence,
                }
            return FrameVerdict(
                reward=self._reward.game_over_penalty,
                terminated=True,
                game_over_confidence=confidence,
                frame_diff=frame_diff,
                is_idle=is_idle,
                info={
                    "game_over": 1.0,
                    "frame_diff": frame_diff,
                    "is_idle": 1.0 if is_idle else 0.0,
                    **vlm_info,
                },
            )

        points, score = self._score.update(frame)
        self._episode_points += points

        # Base step reward + points reward
        reward = self._reward.step_reward + self._reward.score_reward * points

        # Stagnation penalty or movement reward based on visual motion
        if is_idle:
            reward += self._reward.idle_penalty
        else:
            reward += self._reward.movement_reward

        vlm_info: dict[str, Any] = {}
        if self._supervisor is not None:
            cstate = self._supervisor.current_state
            vlm_info = {
                "vlm_state": cstate.state,
                "vlm_is_game_over": 1.0 if cstate.is_game_over else 0.0,
                "vlm_conf": cstate.confidence,
            }

        return FrameVerdict(
            reward=float(reward),
            terminated=False,
            points=points,
            score=score,
            game_over_confidence=confidence,
            frame_diff=frame_diff,
            is_idle=is_idle,
            info={
                "game_over": 0.0,
                "points": points,
                "frame_diff": frame_diff,
                "is_idle": 1.0 if is_idle else 0.0,
                **vlm_info,
            },
        )
