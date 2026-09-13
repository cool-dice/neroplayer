"""Run a trained PPO checkpoint against the live game without learning."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from config import CONFIG
from environment import WindowsGameEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Model .zip, e.g. models/game_agent.zip")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = Monitor(WindowsGameEnv(CONFIG))
    model = PPO.load(args.model, env=env)
    try:
        for episode in range(1, args.episodes + 1):
            observation, _ = env.reset()
            total_reward, steps, terminated, truncated = 0.0, 0, False, False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=args.deterministic)
                observation, reward, terminated, truncated, info = env.step(int(action))
                total_reward += float(reward)
                steps += 1
            print(
                f"episode {episode}: reward={total_reward:.1f} "
                f"steps={steps} info={info}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
