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
            assert "real_fps" in info
            assert isinstance(info["real_fps"], float)
            assert "action" in info
            assert info["action"] == action

        hud = env.render_hud()
        assert hud is not None
        assert hud.shape == (config.capture.region.height, config.capture.region.width, 3)
    finally:
        env.close()


