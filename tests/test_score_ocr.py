"""OCR score-reading logic, exercised without the Tesseract binary.

``pytesseract`` is installed but the binary may not be, so these tests stub the
OCR call and focus on the part that matters for reward correctness: turning a
sequence of (possibly misread) numbers into trustworthy point gains.
"""

from __future__ import annotations

import pytest

from ai_player.mock_game import MockArcadeGame, mock_config
from ai_player.observer import OcrScoreSignal

pytest.importorskip("pytesseract")


class _StubTesseract:
    """Returns queued strings, mimicking ``pytesseract.image_to_string``."""

    def __init__(self, readings: list[str]) -> None:
        self._readings = list(readings)

    def image_to_string(self, image, config=""):
        return self._readings.pop(0) if self._readings else ""


@pytest.fixture
def frame():
    return MockArcadeGame(seed=0).render()


def signal_with(readings: list[str]) -> OcrScoreSignal:
    signal = OcrScoreSignal(mock_config().reward)
    signal._pytesseract = _StubTesseract(readings)
    return signal


def test_first_reading_only_primes_the_baseline(frame):
    signal = signal_with(["12"])

    assert signal.update(frame) == (0.0, 12)


def test_increase_is_rewarded_by_its_delta(frame):
    signal = signal_with(["10", "13"])
    signal.update(frame)

    points, score = signal.update(frame)

    assert (points, score) == (3.0, 13)


def test_implausible_jump_is_ignored_but_tracked(frame):
    signal = signal_with(["7", "771"])
    signal.update(frame)

    points, score = signal.update(frame)

    # A misread must not hand out a huge reward, yet the new baseline is kept
    # so the next genuine increment is still measured correctly.
    assert points == 0.0
    assert score == 771


def test_counter_reset_is_not_a_negative_reward(frame):
    signal = signal_with(["25", "0"])
    signal.update(frame)

    points, score = signal.update(frame)

    assert points == 0.0
    assert score == 0


def test_unreadable_box_keeps_the_last_score(frame):
    signal = signal_with(["4", "", "6"])
    signal.update(frame)

    blank_points, blank_score = signal.update(frame)
    points, score = signal.update(frame)

    assert (blank_points, blank_score) == (0.0, 4)
    assert (points, score) == (2.0, 6)


def test_reset_clears_the_baseline(frame):
    signal = signal_with(["30", "31"])
    signal.update(frame)
    signal.reset()

    assert signal.update(frame) == (0.0, 31)


def test_ocr_failure_does_not_propagate(frame):
    signal = signal_with([])

    class _Exploding:
        def image_to_string(self, image, config=""):
            raise RuntimeError("tesseract crashed")

    signal._pytesseract = _Exploding()

    assert signal.update(frame) == (0.0, None)
