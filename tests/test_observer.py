from dataclasses import replace

import numpy as np

from config import CONFIG, Region
from observer import GameObserver


def make_config(tmp_path, **overrides):
    return replace(
        CONFIG.reward,
        game_over_template=tmp_path / "missing.png",
        use_ocr=False,
        **overrides,
    )


def test_score_change_is_rewarded_once(tmp_path):
    cfg = make_config(
        tmp_path,
        score_region=Region(0, 0, 4, 4),
        game_over_region=Region(4, 4, 4, 4),
    )
    observer = GameObserver(cfg)
    baseline = np.zeros((8, 8, 3), dtype=np.uint8)
    observer.reset(baseline)

    scored = baseline.copy()
    scored[:4, :4] = 255
    first = observer.evaluate(scored)
    second = observer.evaluate(scored)

    assert first.reward == cfg.reward_alive + cfg.reward_score_increase
    assert first.info["points"] == 1
    assert second.reward == cfg.reward_alive
    assert second.info["points"] == 0
    assert second.info["score"] == 1


def test_color_game_over_requires_confirmation_then_terminates(tmp_path):
    cfg = make_config(
        tmp_path,
        score_region=Region(0, 0, 2, 2),
        game_over_region=Region(2, 2, 4, 4),
        game_over_color_bgr=(10, 20, 30),
        game_over_color_tolerance=0,
        game_over_confirm_frames=2,
    )
    observer = GameObserver(cfg)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    observer.reset(frame)
    frame[2:6, 2:6] = (10, 20, 30)

    first = observer.evaluate(frame)
    second = observer.evaluate(frame)

    # A single positive frame is treated as flicker; the second confirms it.
    assert first.terminated is False
    assert second.terminated is True
    assert second.reward == cfg.reward_game_over
    assert second.info["game_over"] is True


def test_template_game_over_detection(tmp_path):
    import cv2

    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    cv2.putText(frame, "GAME OVER", (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    template_path = tmp_path / "game_over.png"
    cv2.imwrite(str(template_path), frame[10:30, 0:50])

    cfg = replace(
        CONFIG.reward,
        game_over_template=template_path,
        use_ocr=False,
        score_region=Region(0, 0, 2, 2),
        game_over_confirm_frames=1,
    )
    observer = GameObserver(cfg)
    observer.reset(np.zeros_like(frame))

    blank = observer.evaluate(np.zeros_like(frame))
    banner = observer.evaluate(frame)

    assert blank.terminated is False
    assert banner.terminated is True
    assert banner.info["match_score"] >= cfg.game_over_match_threshold
