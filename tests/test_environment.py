from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from ai_player.config import AppConfig
from ai_player.controls import NullController
from ai_player.environment import GameEnv, make_env
from ai_player.mock_game import build_mock_backends, mock_config


@pytest.fixture
def env() -> GameEnv:
    environment = make_env(mock=True, seed=7)
    yield environment
    environment.close()


def test_env_passes_the_gymnasium_api_checker(env):
    # Skips render-mode checks because the mock env has no display backend.
    check_env(env, skip_render_check=True)


def test_observation_matches_the_declared_space(env):
    obs, info = env.reset(seed=1)

    assert env.observation_space.contains(obs)
    assert obs["frames"].shape == env.config.vision.observation_shape
    assert obs["audio"].shape == env.config.audio.observation_shape
    assert info["episode_index"] == 1


def test_step_returns_the_five_tuple_and_info(env):
    env.reset(seed=1)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert terminated is False and truncated is False
    assert {"score", "points", "episode_steps", "episode_reward"} <= set(info)


def test_audio_key_disappears_when_audio_is_disabled():
    config = AppConfig()
    config.audio.enabled = False
    env = make_env(config, mock=True, seed=0)
    try:
        obs, _ = env.reset()
        assert set(obs) == {"frames"}
        assert "audio" not in env.observation_space.spaces
    finally:
        env.close()


def test_crashing_terminates_with_the_penalty(env):
    env.reset(seed=2)
    reward = 0.0
    terminated = False
    # Action 2 is the no-op, so the agent never dodges and always crashes.
    for _ in range(env.config.env.max_episode_steps):
        _obs, reward, terminated, _truncated, _info = env.step(2)
        if terminated:
            break

    assert terminated is True
    assert reward == pytest.approx(env.config.reward.game_over_penalty)


def test_reset_clears_the_game_over_screen(env):
    env.reset(seed=3)
    for _ in range(200):
        if env.step(2)[2]:
            break

    obs, info = env.reset()

    assert env.observation_space.contains(obs)
    assert info["episode_index"] == 2
    # A fresh episode must not immediately report termination.
    assert env.step(0)[2] is False


def test_episode_truncates_at_the_step_limit():
    config = mock_config()
    config.env.max_episode_steps = 5
    game, frames, audio, controller = build_mock_backends(config, seed=4)
    # A game that never ends isolates truncation from termination.
    game._check_collision = lambda: None
    env = GameEnv(config, frame_source=frames, audio_source=audio, controller=controller)
    try:
        env.reset()
        truncated = False
        for _ in range(5):
            _obs, _reward, terminated, truncated, _info = env.step(2)
        assert terminated is False
        assert truncated is True
    finally:
        env.close()


def test_actions_reach_the_controller():
    config = mock_config()
    controller = NullController(config.control)
    _game, frames, audio, _mock_controller = build_mock_backends(config, seed=5)
    env = GameEnv(config, frame_source=frames, audio_source=audio, controller=controller)
    try:
        env.reset()
        env.step(0)
        env.step(1)
    finally:
        env.close()

    assert controller.history == [0, 1]
    assert controller.restarts == 1


def test_frame_stack_encodes_motion(env):
    obs, _ = env.reset(seed=6)
    assert np.all(obs["frames"][0] == obs["frames"][-1])

    for _ in range(3):
        obs, *_ = env.step(2)

    assert not np.all(obs["frames"][0] == obs["frames"][-1])


def test_render_returns_rgb_frames():
    env = make_env(mock=True, render_mode="rgb_array", seed=8)
    try:
        env.reset()
        env.step(0)
        frame = env.render()
        hud = env.render_hud()
    finally:
        env.close()

    region = env.config.capture.region
    assert frame.shape == (region.height, region.width, 3)
    assert hud.shape == (region.height, region.width, 3)


def test_env_tracks_real_fps_and_renders_hud_details():
    config = mock_config()
    config.control.action_keys = ["up", "down", "up+down", None]
    env = make_env(config, mock=True, render_mode="rgb_array", seed=10)
    try:
        env.reset()
        for action in range(4):
            _obs, _rew, _term, _trunc, info = env.step(action)
            assert "action" in info
            assert info["action"] == action
            assert env.last_info["real_fps"] >= 0.0

        hud = env.render_hud()
        assert hud is not None
        assert hud.shape == (config.capture.region.height, config.capture.region.width, 3)
    finally:
        env.close()


def test_env_tracks_latency_profiler():
    config = mock_config()
    env = make_env(config, mock=True, render_mode="rgb_array", seed=11)
    try:
        env.reset()
        _obs, _rew, _term, _trunc, _info = env.step(0)
        assert "latency_ms" in env.last_info
        lat = env.last_info["latency_ms"]
        assert "grab" in lat
        assert "act" in lat
        assert "proc" in lat
        assert "obs" in lat
        assert "render" in lat
        assert isinstance(lat["grab"], float)
        assert isinstance(lat["act"], float)
    finally:
        env.close()


def test_env_exposes_player_bbox_and_entities():
    config = mock_config()
    config.tracker.enabled = True
    config.tracker.auto_probe = True
    env = make_env(config, mock=True, render_mode="rgb_array", seed=12)
    try:
        _obs, _info = env.reset()
        _obs, _rew, _term, _trunc, step_info = env.step(0)
        assert "entities" in step_info
        assert "collisions" in step_info
        assert isinstance(step_info["entities"], list)
        assert isinstance(step_info["collisions"], list)
        hud = env.render_hud()
        assert hud is not None
    finally:
        env.close()


def test_env_renders_rgb_hud_inset():
    config = mock_config()
    config.vision.grayscale = False
    config.vision.width = 128
    config.vision.height = 128
    env = make_env(config, mock=True, render_mode="rgb_array", seed=13)
    try:
        env.reset()
        env.step(0)
        hud = env.render_hud()
        assert hud is not None
        assert hud.shape == (config.capture.region.height, config.capture.region.width, 3)
    finally:
        env.close()


def test_env_with_mouse_actions_in_telemetry_and_step():
    config = mock_config()
    config.control.mouse_enabled = True
    config.control.action_keys = [
        None,
        "up",
        "mouse_left",
        "aim_right",
        "up+mouse_left",
    ]
    env = make_env(config, mock=True, render_mode="rgb_array", seed=14)
    try:
        env.reset()
        # Step with mouse click action
        _obs, _rew, _term, _trunc, info_click = env.step(2)
        assert "mouse" in info_click
        assert info_click["mouse"]["clicks"] == ["left"]

        # Step with mouse aim action
        _obs, _rew, _term, _trunc, info_aim = env.step(3)
        assert info_aim["mouse"]["aim"] == (config.control.aim_step, 0)

        # Step with hybrid key + mouse
        _obs, _rew, _term, _trunc, info_hybrid = env.step(4)
        assert info_hybrid["mouse"]["clicks"] == ["left"]

        # Render HUD should succeed without error and include telemetry
        hud = env.render_hud()
        assert hud is not None
        assert hud.shape == (config.capture.region.height, config.capture.region.width, 3)
    finally:
        env.close()


def test_env_auto_menu_navigation_and_vlm_telemetry():
    from ai_player.cognitive import CognitiveState, CognitiveSupervisor

    class MockSupervisor(CognitiveSupervisor):
        def __init__(self, cstate: CognitiveState):
            super().__init__()
            self._state = cstate

        @property
        def current_state(self) -> CognitiveState:
            return self._state

    config = mock_config()
    config.hud.auto_menu_nav = True

    mock_sup = MockSupervisor(
        CognitiveState(
            state="menu",
            is_game_over=False,
            confidence=0.95,
            suggested_action="press_start",
            description="Title menu screen",
        )
    )

    controller = NullController(config.control)
    _game, frames, audio, _mock_controller = build_mock_backends(config, seed=15)
    env = GameEnv(
        config,
        frame_source=frames,
        audio_source=audio,
        controller=controller,
        supervisor=mock_sup,
        render_mode="rgb_array",
    )
    try:
        env.reset()
        restarts_before = controller.restarts
        _obs, _rew, _term, _trunc, step_info = env.step(0)

        # Because state == "menu" and auto_menu_nav is True, controller.restart() should be triggered
        assert controller.restarts > restarts_before
        assert step_info.get("vlm_state") == "menu"
        assert step_info.get("vlm_goal") == "press_start"
        assert step_info.get("vlm_desc") == "Title menu screen"

        # Check HUD rendering displays VLM status
        hud = env.render_hud()
        assert hud is not None
    finally:
        env.close()


def test_env_cognitive_obs_vector_and_dynamic_hud():
    from ai_player.cognitive import CognitiveState, CognitiveSupervisor

    class MockSupervisor(CognitiveSupervisor):
        def __init__(self, cstate: CognitiveState):
            super().__init__()
            self._state = cstate

        @property
        def current_state(self) -> CognitiveState:
            return self._state

    config = mock_config()
    config.hud.cognitive_obs = True
    config.hud.cognitive_dim = 8

    mock_sup = MockSupervisor(
        CognitiveState(
            state="gameplay",
            is_game_over=False,
            confidence=0.92,
            suggested_action="up",
            description="Active play",
            hud_layout={
                "score": [25, 20, 110, 30],
                "hp": [25, 60, 90, 20],
                "ammo": [25, 85, 70, 15],
            },
            vital_stats={
                "hp_ratio": 0.75,
                "ammo_ratio": 0.40,
                "lives": 3.0,
                "danger_level": 0.15,
            },
        )
    )

    controller = NullController(config.control)
    _game, frames, audio, _mock_controller = build_mock_backends(config, seed=16)
    env = GameEnv(
        config,
        frame_source=frames,
        audio_source=audio,
        controller=controller,
        supervisor=mock_sup,
        render_mode="rgb_array",
    )
    try:
        assert "cognitive" in env.observation_space.spaces
        obs, _info = env.reset()
        assert "cognitive" in obs
        assert obs["cognitive"].shape == (8,)

        # Check vector values: [state_code, hp, ammo, lives, danger, action_code, conf, game_over]
        cog_vec = obs["cognitive"]
        assert cog_vec[0] == 0.0  # gameplay
        assert abs(cog_vec[1] - 0.75) < 1e-4  # hp_ratio
        assert abs(cog_vec[2] - 0.40) < 1e-4  # ammo_ratio
        assert abs(cog_vec[3] - 0.60) < 1e-4  # lives_ratio (3 / 5)
        assert abs(cog_vec[4] - 0.15) < 1e-4  # danger_level
        assert cog_vec[5] > 0.0  # suggested_action "up" mapped
        assert abs(cog_vec[6] - 0.92) < 1e-4  # vlm_conf
        assert cog_vec[7] == 0.0  # is_game_over

        # Step and check HUD rendering with dynamic boxes and vital stats
        obs, _rew, _term, _trunc, _step_info = env.step(0)
        hud = env.render_hud()
        assert hud is not None
        assert hud.shape == (config.capture.region.height, config.capture.region.width, 3)

        # Check dynamic HUD regions updated in observer
        assert env._observer._reward.score_region.left == 25
        assert env._observer._reward.score_region.top == 20
    finally:
        env.close()


def test_env_instant_vlm_game_over_termination():
    from ai_player.cognitive import CognitiveState, CognitiveSupervisor

    class MockGameOverSupervisor(CognitiveSupervisor):
        def __init__(self):
            super().__init__()
            self._state = CognitiveState(
                state="game_over",
                is_game_over=True,
                confidence=0.95,
                description="Instant death",
            )

        @property
        def current_state(self) -> CognitiveState:
            return self._state

    config = mock_config()
    # Detection patience is 2 by default
    assert config.reward.detection_patience >= 2
    mock_sup = MockGameOverSupervisor()

    controller = NullController(config.control)
    _game, frames, audio, _mock_controller = build_mock_backends(config, seed=17)
    env = GameEnv(
        config,
        frame_source=frames,
        audio_source=audio,
        controller=controller,
        supervisor=mock_sup,
    )
    try:
        env.reset()
        _obs, rew, term, _trunc, _info = env.step(0)
        # Should terminate immediately on step 1 due to VLM sentinel bypass
        assert term is True
        assert rew == pytest.approx(config.reward.game_over_penalty)
    finally:
        env.close()






