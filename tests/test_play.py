"""Rollout script: model loading, episode stats and video recording."""

from __future__ import annotations

import numpy as np
import pytest

import play as play_script
import train as train_script
from ai_player.config import AppConfig


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    """Train once for a handful of steps and reuse the saved model."""
    root = tmp_path_factory.mktemp("play")
    models_dir, logs_dir = train_script.MODELS_DIR, train_script.LOGS_DIR
    train_script.MODELS_DIR, train_script.LOGS_DIR = root / "models", root / "logs"
    try:
        train_script.main(["--mock", "--timesteps", "48", "--run-name", "tiny", "--no-audio"])
    finally:
        train_script.MODELS_DIR, train_script.LOGS_DIR = models_dir, logs_dir
    return root / "models" / "tiny" / "final.zip"


def test_recorder_sizes_itself_from_the_first_frame(tmp_path):
    path = tmp_path / "clip.mp4"
    recorder = play_script.Recorder(path, fps=10.0)

    recorder.write(np.zeros((240, 320, 3), dtype=np.uint8))
    recorder.write(np.full((240, 320, 3), 255, dtype=np.uint8))
    recorder.close()

    # A writer built with the wrong frame size silently produces a stub file.
    assert path.exists() and path.stat().st_size > 1_000


def test_recorder_without_a_path_is_a_no_op():
    recorder = play_script.Recorder(None, fps=10.0)

    recorder.write(np.zeros((10, 10, 3), dtype=np.uint8))
    recorder.close()


def test_play_runs_episodes_and_records(trained_model, tmp_path, capsys):
    clip = tmp_path / "rollout.mp4"

    status = play_script.main(
        [str(trained_model), "--mock", "--episodes", "2", "--record", str(clip), "--deterministic"]
    )

    assert status == 0
    assert clip.exists() and clip.stat().st_size > 1_000
    out = capsys.readouterr().out
    assert "episode 1" in out and "episode 2" in out
    assert "mean return" in out


def test_play_prefers_the_config_saved_next_to_the_model(trained_model):
    args = play_script.parse_args([str(trained_model), "--mock"])

    config = play_script.resolve_config(args)

    # train.py wrote config.json beside the model, with audio disabled.
    assert config.audio.enabled is False


def test_play_falls_back_to_defaults_without_a_config(tmp_path):
    args = play_script.parse_args([str(tmp_path / "nowhere" / "model.zip"), "--mock"])

    assert play_script.resolve_config(args) == AppConfig()
