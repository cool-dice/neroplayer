"""End-to-end tests of the Gymnasium environment with a synthetic game.

No display, audio device or keyboard is touched: frames come from
``FakeFrameSource``, audio capture is disabled (zeros) and the controller runs
in dry-run mode.
"""

from dataclasses import replace

import numpy as np
import pytest

from config import CONFIG, AudioConfig, Region, ScreenConfig, TrainConfig
from controls import GameController
from environment import GameEnv
from observer import GameObserver
from perception import AudioCapture, ScreenCapture

FRAME_H, FRAME_W = 32, 48
SCORE_REGION = Region(left=0, top=0, width=8, height=8)
GAME_OVER_REGION = Region(left=16, top=16, width=8, height=8)
GAME_OVER_BGR = (10, 20, 30)


class FakeFrameSource:
    """Scripted game: the score flips every ``score_every`` grabs and the
    game-over colour appears from grab ``game_over_at`` onwards."""

    def __init__(self, *, score_every: int = 0, game_over_at: int | None = None) -> None:
        self.grabs = 0
        self.score_every = score_every
        self.game_over_at = game_over_at
        self.closed = False

    def grab(self) -> np.ndarray:
        self.grabs += 1
        frame = np.full((FRAME_H, FRAME_W, 3), 128, dtype=np.uint8)
        rows, cols = SCORE_REGION.as_slice()
        if self.score_every and (self.grabs // self.score_every) % 2:
            frame[rows, cols] = 255
        else:
            frame[rows, cols] = 0
        if self.game_over_at is not None and self.grabs >= self.game_over_at:
            rows, cols = GAME_OVER_REGION.as_slice()
            frame[rows, cols] = GAME_OVER_BGR
        return frame

    def close(self) -> None:
        self.closed = True


def make_env(tmp_path, source: FakeFrameSource, max_episode_steps: int = 50, frame_size: int = 8) -> GameEnv:
    screen_cfg = replace(
        CONFIG.screen, frame_width=frame_size, frame_height=frame_size, frame_stack=2, target_fps=10_000
    )
    reward_cfg = replace(
        CONFIG.reward,
        game_over_template=tmp_path / "missing.png",
        use_ocr=False,
        score_region=SCORE_REGION,
        game_over_region=GAME_OVER_REGION,
        game_over_color_bgr=GAME_OVER_BGR,
        game_over_color_tolerance=0,
        game_over_confirm_frames=2,
    )
    controls_cfg = replace(CONFIG.controls, key_hold_seconds=0.0, restart_delay_seconds=0.0)
    config = replace(
        CONFIG,
        screen=screen_cfg,
        reward=reward_cfg,
        controls=controls_cfg,
        audio=AudioConfig(enabled=False),
        train=replace(CONFIG.train, max_episode_steps=max_episode_steps),
    )
    return GameEnv(
        config,
        screen=ScreenCapture(config.screen, source=source),
        audio=AudioCapture(config.audio),
        controller=GameController(config.controls, dry_run=True),
        observer=GameObserver(config.reward),
    )


def test_reset_and_step_produce_valid_observations(tmp_path):
    source = FakeFrameSource()
    env = make_env(tmp_path, source)

    obs, info = env.reset(seed=7)
    assert env.observation_space.contains(obs)
    assert obs["image"].shape == (2, 8, 8)
    assert obs["audio"].shape == env.audio.observation_shape
    assert info == {"score": 0}

    obs, reward, terminated, truncated, info = env.step(2)
    assert env.observation_space.contains(obs)
    assert reward == pytest.approx(env.cfg.reward.reward_alive)
    assert terminated is False and truncated is False
    assert info["action"] == "NoOp"
    assert info["step"] == 1

    env.close()
    assert source.closed is True


def test_truncation_at_max_episode_steps(tmp_path):
    env = make_env(tmp_path, FakeFrameSource(), max_episode_steps=3)
    env.reset()
    results = [env.step(0)[3] for _ in range(3)]
    assert results == [False, False, True]
    env.close()


def test_score_and_game_over_rewards(tmp_path):
    # reset() grabs once; score toggles on grabs 3, 5, ...; game over from grab 6.
    env = make_env(tmp_path, FakeFrameSource(score_every=3, game_over_at=6))
    env.reset()
    alive = env.cfg.reward.reward_alive
    scored = alive + env.cfg.reward.reward_score_increase

    rewards, terminated = [], False
    while not terminated:
        _, reward, terminated, truncated, info = env.step(1)
        rewards.append(reward)
        assert not truncated

    # grab2: nothing, grab3: score flips, grab4/5: nothing (5 flips back -> also a change),
    # grab6: game over seen once (unconfirmed), grab7: confirmed.
    assert rewards[0] == pytest.approx(alive)
    assert rewards[1] == pytest.approx(scored)
    assert rewards[-1] == pytest.approx(env.cfg.reward.reward_game_over)
    assert info["game_over"] is True
    assert len(rewards) == 6
    env.close()


def test_passes_stable_baselines3_env_checker(tmp_path):
    from stable_baselines3.common.env_checker import check_env

    env = make_env(tmp_path, FakeFrameSource())
    check_env(env, warn=True, skip_render_check=True)
    env.close()


def test_ppo_multi_input_policy_learns_a_few_steps(tmp_path):
    from stable_baselines3 import PPO

    # NatureCNN needs at least 36x36 input, so use the production frame size here.
    env = make_env(
        tmp_path, FakeFrameSource(score_every=2, game_over_at=20), max_episode_steps=16, frame_size=84
    )
    model = PPO("MultiInputPolicy", env, n_steps=16, batch_size=8, n_epochs=1, device="cpu", verbose=0)
    model.learn(total_timesteps=32)
    action, _ = model.predict(env.reset()[0], deterministic=True)
    assert env.action_space.contains(int(action))
    env.close()


def test_screen_and_train_config_are_consistent():
    assert isinstance(CONFIG.screen, ScreenConfig)
    assert isinstance(CONFIG.train, TrainConfig)
    assert len(CONFIG.controls.actions) == len(CONFIG.controls.action_names)
