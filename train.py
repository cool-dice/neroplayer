"""Train a PPO agent against the live Windows game environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from config import CONFIG
from environment import WindowsGameEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--device", default="auto", help="PyTorch device (auto/cpu/cuda)")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Optional Stable-Baselines3 .zip checkpoint to resume.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=25_000)
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    CONFIG.model_path.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.log_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(WindowsGameEnv(CONFIG))
    checkpoint = CheckpointCallback(
        save_freq=max(1, args.checkpoint_every),
        save_path=str(CONFIG.model_path.parent / "checkpoints"),
        name_prefix="ppo_game_agent",
    )

    if args.resume:
        model = PPO.load(args.resume, env=env, device=args.device)
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=2.5e-4,
            n_steps=512,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            tensorboard_log=str(CONFIG.log_dir),
            device=args.device,
            seed=CONFIG.random_seed,
            verbose=1,
        )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint,
            reset_num_timesteps=not bool(args.resume),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("Training interrupted; saving the latest policy.")
    finally:
        model.save(CONFIG.model_path)
        env.close()
        print(f"Model saved to {CONFIG.model_path}.zip")


if __name__ == "__main__":
    train(parse_args())
