from dataclasses import replace

import numpy as np

from config import CONFIG, Region
from observer import GameStateObserver


def test_score_change_is_rewarded_once(tmp_path):
    config = replace(
        CONFIG,
        score_region=Region(0, 0, 4, 4),
        game_over_region=Region(4, 4, 4, 4),
        game_over_template=tmp_path / "missing.png",
        score_change_threshold=5.0,
    )
    observer = GameStateObserver(config)
    baseline = np.zeros((8, 8, 3), dtype=np.uint8)
    observer.reset(baseline)

    scored = baseline.copy()
    scored[:4, :4] = 255
    first = observer.evaluate(scored)
    second = observer.evaluate(scored)

    assert first.reward == config.score_reward
    assert first.info["score_changed"] is True
    assert second.reward == config.living_reward
    assert second.info["score_changed"] is False


def test_color_game_over_terminates_with_penalty(tmp_path):
    config = replace(
        CONFIG,
        score_region=Region(0, 0, 2, 2),
        game_over_region=Region(2, 2, 4, 4),
        game_over_template=tmp_path / "missing.png",
        game_over_bgr=(10, 20, 30),
        game_over_min_pixel_ratio=0.5,
        game_over_color_tolerance=0,
    )
    observer = GameStateObserver(config)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    observer.reset(frame)
    frame[2:6, 2:6] = (10, 20, 30)

    result = observer.evaluate(frame)

    assert result.terminated is True
    assert result.reward == config.game_over_penalty
    assert result.info["game_over_detector"] == "color"
