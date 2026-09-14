"""End-to-end checks: the policy network and the training entry point.

These run a handful of real gradient steps against the simulated game, which is
what catches shape/dtype mismatches between the Dict observation space and the
multi-modal extractor.
"""

from __future__ import annotations

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import train as train_script
from ai_player.environment import make_env
from ai_player.policies import MultiModalExtractor


def test_extractor_fuses_both_modalities():
    env = make_env(mock=True, seed=0)
    try:
        extractor = MultiModalExtractor(env.observation_space, features_dim=64)
        obs, _ = env.reset()
        batch = {key: torch.as_tensor(value[None, ...]) for key, value in obs.items()}
        features = extractor(batch)
    finally:
        env.close()

    assert features.shape == (1, 64)
    assert torch.isfinite(features).all()
    assert extractor.audio_net is not None


def test_extractor_works_without_audio():
    from ai_player.config import AppConfig

    config = AppConfig()
    config.audio.enabled = False
    env = make_env(config, mock=True, seed=0)
    try:
        extractor = MultiModalExtractor(env.observation_space, features_dim=32)
        obs, _ = env.reset()
        features = extractor({"frames": torch.as_tensor(obs["frames"][None, ...])})
    finally:
        env.close()

    assert extractor.audio_net is None
    assert features.shape == (1, 32)


def test_ppo_learns_for_a_few_steps():
    config_env = make_env(mock=True, seed=1)
    vec_env = DummyVecEnv([lambda: Monitor(config_env)])
    args = train_script.parse_args(["--mock", "--timesteps", "64"])
    config = train_script.build_config(args)
    config.train.n_steps = 32
    config.train.batch_size = 16
    config.train.features_dim = 64

    model = train_script.build_model(config, vec_env, tensorboard_log=None)
    model.learn(total_timesteps=64)
    action, _ = model.predict(vec_env.reset(), deterministic=True)

    assert model.num_timesteps >= 64
    assert vec_env.action_space.contains(int(np.asarray(action).reshape(-1)[0]))
    vec_env.close()


def test_dqn_builds_on_the_dict_observation_space():
    env = make_env(mock=True, seed=2)
    vec_env = DummyVecEnv([lambda: Monitor(env)])
    args = train_script.parse_args(["--mock", "--algo", "dqn", "--timesteps", "32"])
    config = train_script.build_config(args)
    config.train.buffer_size = 256
    config.train.learning_starts = 8
    config.train.batch_size = 8
    config.train.features_dim = 32

    model = train_script.build_model(config, vec_env, tensorboard_log=None)
    model.learn(total_timesteps=32)

    assert model.num_timesteps >= 32
    vec_env.close()


def test_cli_overrides_reach_the_config():
    args = train_script.parse_args(
        ["--mock", "--algo", "dqn", "--timesteps", "1234", "--fps", "20", "--no-audio", "--seed", "5"]
    )

    config = train_script.build_config(args)

    assert config.train.algo == "dqn"
    assert config.train.total_timesteps == 1234
    assert config.env.target_fps == 20
    assert config.audio.enabled is False
    assert config.train.seed == 5


def test_reward_normalisation_wraps_ppo_only():
    from stable_baselines3.common.vec_env import VecNormalize

    env = make_env(mock=True, seed=3)
    vec_env = DummyVecEnv([lambda: Monitor(env)])
    config = train_script.build_config(train_script.parse_args(["--mock"]))

    wrapped = train_script.wrap_reward_normalisation(config, vec_env, None)
    assert isinstance(wrapped, VecNormalize)
    assert wrapped.norm_obs is False  # the policy must see raw pixels

    config.train.algo = "dqn"  # off-policy: a replay buffer would mix scales
    assert train_script.wrap_reward_normalisation(config, vec_env, None) is vec_env

    config.train.algo = "ppo"
    config.train.normalize_reward = False
    assert train_script.wrap_reward_normalisation(config, vec_env, None) is vec_env
    vec_env.close()


def test_training_saves_reward_statistics_for_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(train_script, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train_script, "LOGS_DIR", tmp_path / "logs")

    train_script.main(["--mock", "--timesteps", "48", "--run-name", "norm", "--no-audio"])
    stats = tmp_path / "models" / "norm" / "vec_normalize.pkl"
    assert stats.exists()

    # Resuming must pick the statistics back up instead of restarting them.
    status = train_script.main(
        [
            "--mock",
            "--timesteps",
            "48",
            "--run-name",
            "norm-resumed",
            "--no-audio",
            "--resume",
            str(tmp_path / "models" / "norm" / "final.zip"),
        ]
    )

    assert status == 0


def test_check_env_validates_the_environment():
    assert train_script.main(["--mock", "--check-env"]) == 0


def test_build_config_accepts_idle_and_movement_flags():
    args = train_script.parse_args(
        ["--idle-penalty", "-0.25", "--movement-reward", "0.1", "--idle-diff-threshold", "0.08"]
    )
    cfg = train_script.build_config(args)
    assert cfg.reward.idle_penalty == -0.25
    assert cfg.reward.movement_reward == 0.1
    assert cfg.reward.idle_diff_threshold == 0.08


def test_build_config_accepts_auto_combos_flag():
    args = train_script.parse_args(["--auto-combos", "--max-combo-size", "3"])
    cfg = train_script.build_config(args)
    assert cfg.control.auto_combos is True
    assert cfg.control.max_combo_size == 3


def test_build_config_accepts_architecture_and_autonomous_flags():
    args = train_script.parse_args([
        "--backbone", "convnext",
        "--rgb",
        "--resolution", "256",
        "--features-dim", "1024",
        "--batch-size", "128",
        "--auto-hud",
        "--auto-tracker",
    ])
    cfg = train_script.build_config(args)
    assert cfg.vision.backbone == "convnext"
    assert cfg.vision.grayscale is False
    assert cfg.vision.rgb is True
    assert cfg.vision.width == 256
    assert cfg.vision.height == 256
    assert cfg.train.features_dim == 1024
    assert cfg.train.batch_size == 128
    assert cfg.hud.auto_detect is True
    assert cfg.tracker.enabled is True



def test_console_stats_callback(capsys):
    env = make_env(mock=True, seed=20)
    vec_env = DummyVecEnv([lambda: Monitor(env)])
    cb = train_script.ConsoleStatsCallback(check_freq=1)
    cb.init_callback(None)
    dummy_info = {
        "episode_reward": 5.0,
        "episode_index": 1,
        "latency_ms": {"act": 1.0, "grab": 5.0},
    }
    cb.locals = {"infos": [dummy_info]}
    cb.num_timesteps = 10
    cb.n_calls = 1
    assert cb._on_step() is True
    out = capsys.readouterr().out
    assert "Step     10" in out
    assert "Latency: grab= 5.0ms | act= 1.0ms" in out
    vec_env.close()


def test_main_runs_a_tiny_training_loop(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train_script, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train_script, "LOGS_DIR", tmp_path / "logs")

    status = train_script.main(["--mock", "--timesteps", "48", "--run-name", "smoke", "--no-audio"])

    assert status == 0
    assert (tmp_path / "models" / "smoke" / "final.zip").exists()
    assert (tmp_path / "models" / "smoke" / "config.json").exists()
