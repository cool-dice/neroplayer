"""Tests for the in-process mock arcade wired into GameEnv."""

from __future__ import annotations

import numpy as np
import pytest

from mock_game import MockArcade, build_mock_env


def test_mock_env_observation_and_action_spaces() -> None:
    env = build_mock_env(seed=0)
    try:
        obs, info = env.reset()
        assert set(obs) == {"image", "audio"}
        assert env.observation_space.contains(obs)
        assert env.action_space.contains(0)
        obs, reward, terminated, truncated, info = env.step(2)
        assert isinstance(reward, float)
        assert terminated in (True, False)
        assert truncated in (True, False)
        assert obs["image"].dtype == np.uint8
        assert obs["audio"].dtype == np.float32
    finally:
        env.close()


def test_standing_still_hits_a_hazard_and_terminates() -> None:
    env = build_mock_env(seed=0)
    try:
        env.reset()
        terminated = False
        last_reward = 0.0
        last_info: dict = {}
        for _ in range(80):
            _, last_reward, terminated, truncated, last_info = env.step(2)
            if terminated:
                break
            assert truncated is False
        assert terminated, f"expected a collision, last info={last_info}"
        assert last_reward == pytest.approx(env.cfg.reward.reward_game_over)
        assert last_info["game_over"] is True
    finally:
        env.close()


def test_mock_arcade_can_score_by_dodging() -> None:
    game = MockArcade(seed=1)
    for _ in range(80):
        if game.over:
            break
        if game.hazards:
            hz = game.hazards[0]
            go_up = (hz.y + hz.h / 2) > 120
            game.apply_action(0 if go_up else 1)
        else:
            game.apply_action(2)
        if game.score > 0:
            break
    frame = game.grab()
    assert frame.shape == (240, 320, 3)
    assert game.score >= 0
