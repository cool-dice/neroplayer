"""Tests for calibrate.py utility modes."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

import calibrate as calib_script
from ai_player.config import AppConfig


def test_calibrate_parse_args_profile():
    args = calib_script.parse_args(["--profile"])
    assert args.profile is True
    assert args.check is False
    assert args.test_keys is False


def test_calibrate_run_profile(monkeypatch, capsys):
    config = AppConfig()
    config.audio.enabled = False

    dummy_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    monkeypatch.setattr(calib_script, "grab", lambda mon: dummy_frame)

    # Mock cv2 to avoid actual display window operations during test
    monkeypatch.setattr(calib_script.cv2, "imshow", MagicMock())
    monkeypatch.setattr(calib_script.cv2, "waitKey", MagicMock(return_value=0))
    monkeypatch.setattr(calib_script.cv2, "destroyWindow", MagicMock())

    status = calib_script.run_profile(config, n_samples=3)
    assert status == 0
    out = capsys.readouterr().out
    assert "PIPELINE LATENCY PROFILER" in out
    assert "Screen Capture (mss)" in out
    assert "Vision Proc (84x84)" in out
    assert "MAX THEORETICAL FPS" in out
