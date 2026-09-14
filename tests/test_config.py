from __future__ import annotations

from ai_player.config import AppConfig, Region, load_config
from ai_player.mock_game import mock_config


def test_defaults_are_internally_consistent():
    config = AppConfig()

    assert config.vision.observation_shape == (4, 84, 84)
    assert config.audio.observation_shape == (config.audio.n_mels, config.audio.n_frames)
    assert config.control.n_actions == 3
    # A "do nothing" action must exist, or every step perturbs the game.
    assert None in config.control.action_keys


def test_region_converts_to_an_mss_monitor():
    region = Region(left=100, top=50, width=800, height=600)

    assert region.as_mss_monitor() == {"left": 100, "top": 50, "width": 800, "height": 600}


def test_config_survives_a_json_round_trip(tmp_path):
    config = mock_config()
    config.capture.region = Region(left=12, top=34, width=640, height=480)
    config.reward.game_over_penalty = -50.0
    config.reward.idle_penalty = -0.2
    config.reward.movement_reward = 0.05
    config.reward.idle_diff_threshold = 0.1
    config.train.algo = "dqn"

    restored = load_config(config.save(tmp_path / "config.json"))

    assert restored.capture.region == config.capture.region
    assert restored.reward.game_over_region == config.reward.game_over_region
    assert restored.reward.game_over_color_bgr == config.reward.game_over_color_bgr
    assert restored.reward.game_over_penalty == -50.0
    assert restored.reward.idle_penalty == -0.2
    assert restored.reward.movement_reward == 0.05
    assert restored.reward.idle_diff_threshold == 0.1
    assert restored.control.auto_combos is False
    assert restored.control.max_combo_size == 2
    assert restored.train.algo == "dqn"
    assert restored.vision.backbone == "nature_cnn"
    assert restored.tracker.enabled is False
    assert restored.hud.auto_detect is False


def test_vision_config_rgb_and_shapes():
    cfg = AppConfig()
    cfg.vision.grayscale = False
    cfg.vision.width = 256
    cfg.vision.height = 256
    assert cfg.vision.rgb is True
    assert cfg.vision.channels == 12
    assert cfg.vision.observation_shape == (12, 256, 256)



def test_unknown_keys_are_ignored(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"vision": {"width": 64, "height": 64, "legacy_flag": true}}')

    config = load_config(path)

    assert config.vision.width == 64
    assert config.vision.observation_shape == (4, 64, 64)


def test_load_config_without_a_path_returns_defaults():
    assert load_config(None) == AppConfig()
