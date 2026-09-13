"""Run a trained agent against the game without learning.

Usage::

    python play.py                          # models/game_agent.zip, 5 episodes
    python play.py --model models/checkpoints/game_agent_20000_steps.zip --episodes 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from stable_baselines3 import DQN, PPO

from config import CONFIG
from train import build_env, countdown

logger = logging.getLogger("play")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=CONFIG.train.model_dir / f"{CONFIG.train.model_name}.zip")
    parser.add_argument("--algo", choices=("ppo", "dqn"), default=CONFIG.train.algorithm)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of argmax")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.model.is_file():
        logger.error("Model not found: %s (train one with train.py first)", args.model)
        return 1

    env = build_env(CONFIG, dry_run=args.dry_run, audio_enabled=not args.no_audio)
    model = (PPO if args.algo == "ppo" else DQN).load(args.model, env=env)

    countdown(3)
    try:
        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset()
            done, total, steps = False, 0.0, 0
            while not done:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                steps += 1
                done = terminated or truncated
            logger.info("Episode %d: return=%.1f steps=%d score=%s", episode, total, steps, info.get("score"))
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
