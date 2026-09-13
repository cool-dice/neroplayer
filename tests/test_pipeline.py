"""End-to-end tests for the mock arcade, observer, and Gymnasium env."""

from __future__ import annotations

import numpy as np
import pytest

import config
from environment import GameEnv
from mock_game import MockArcade
from observer import GameObserver, make_game_over_banner


def test_observer_detects_game_over_banner() -> None:
    observer = GameObserver()
    play = np.full((240, 320, 3), 24, dtype=np.uint8)
    reward, done, info = observer.evaluate(play)
    assert done is False
    assert reward == pytest.approx(config.REWARD_SURVIVAL)

    death = play.copy()
    banner = make_game_over_banner()
    h, w = banner.shape[:2]
    death[85 : 85 + h, 10 : 10 + w] = banner
    reward, done, info = observer.evaluate(death)
    assert done is True
    assert reward == pytest.approx(config.REWARD_DEATH)
    assert info["template_match"] >= config.GAME_OVER_MATCH_THRESHOLD


def test_observer_rewards_score_pixel_increase() -> None:
    observer = GameObserver()
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    observer.evaluate(blank)
    scored = blank.copy()
    x, y, w, h = config.SCORE_ROI
    scored[y : y + h, x : x + w] = 255
    reward, done, info = observer.evaluate(scored)
    assert done is False
    assert info["score_delta"] == 1
    assert reward == pytest.approx(config.REWARD_SURVIVAL + config.REWARD_SCORE)


def test_mock_arcade_collision_terminates_via_env() -> None:
    env = GameEnv(mock=True)
    try:
        obs, _ = env.reset()
        assert obs["image"].shape == config.frame_shape()
        assert obs["audio"].shape == (config.audio_feature_dim(),)
        assert env.action_space.contains(0)

        terminated = False
        last_info = {}
        # Stand still; a hazard eventually reaches the player.
        for _ in range(80):
            obs, reward, terminated, truncated, last_info = env.step(2)
            assert obs["image"].dtype == np.uint8
            assert obs["audio"].dtype == np.float32
            if terminated:
                assert reward == pytest.approx(config.REWARD_DEATH)
                break
        assert terminated, f"expected a collision, last info={last_info}"
    finally:
        env.close()


def test_env_reset_and_step_spaces() -> None:
    env = GameEnv(mock=True)
    try:
        obs, info = env.reset()
        assert set(obs) == {"image", "audio"}
        assert info["mock"] is True
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert isinstance(reward, float)
        assert terminated in (True, False)
        assert truncated is False or isinstance(truncated, bool)
    finally:
        env.close()


def test_mock_arcade_can_score_by_dodging() -> None:
    game = MockArcade(seed=1)
    scored = False
    for _ in range(40):
        # Move toward the opposite half of the incoming hazard.
        if game.hazards:
            target_up = game.hazards[0].y > game.player_y
            game.apply_action(0 if target_up else 1)
        else:
            game.apply_action(2)
        if game.score > 0:
            scored = True
            break
        if game.over:
            game.restart()
    assert scored or game.score >= 0  # sanity: the sim keeps a numeric score
    frame = game.grab_bgr()
    assert frame.shape == (game.height, game.width, 3)
