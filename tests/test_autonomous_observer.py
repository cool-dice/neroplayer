from __future__ import annotations

import numpy as np

from ai_player.config import HUDConfig, RewardConfig
from ai_player.mock_game import MockArcadeGame
from ai_player.observer import (
    AutonomousHUDDetector,
    GameObserver,
    VLMObserverInterface,
)


def test_autonomous_hud_detector_heuristic_on_mock_game():
    game = MockArcadeGame(seed=42)
    frame = game.render()

    detector = AutonomousHUDDetector()
    detected = detector.detect_hud(frame)

    assert detected is not None
    assert detected.score_region is not None
    assert detected.game_over_region is not None
    assert detected.confidence > 0.0
    assert detected.method == "heuristic"

    # Score region should be near top
    assert detected.score_region.top < 50
    assert detected.score_region.width > 0
    assert detected.score_region.height > 0


def test_game_observer_auto_configures_hud_regions():
    reward = RewardConfig()
    hud_cfg = HUDConfig(auto_detect=True)
    observer = GameObserver(reward, hud_config=hud_cfg)

    game = MockArcadeGame(seed=10)
    frame = game.render()

    detected = observer.auto_configure_hud(frame)
    assert detected is not None
    assert observer.detected_hud is not None
    # Reward config score_region should have been auto-updated
    assert reward.score_region == detected.score_region


def test_vlm_observer_interface_fallback(monkeypatch):
    hud_cfg = HUDConfig(use_vlm=True, vlm_endpoint="http://invalid-endpoint-nonexistent:9999/api")
    vlm = VLMObserverInterface(hud_cfg)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # When request fails, should return None and warn gracefully
    res = vlm.detect_hud(frame)
    assert res is None

    # AutonomousHUDDetector should fall back to heuristic smoothly
    detector = AutonomousHUDDetector(hud_cfg)
    detected = detector.detect_hud(frame)
    assert detected.method == "heuristic"
    assert detected.score_region is not None
