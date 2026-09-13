"""Train a PPO (or DQN) agent on the real game window.

Usage::

    python train.py                       # PPO, defaults from config.py
    python train.py --algo dqn --timesteps 50000
    python train.py --resume models/game_agent.zip
    python train.py --check-env           # validate the env against the Gym API and exit
    python train.py --dry-run --no-audio  # exercise the loop without sending input

The policy is Stable-Baselines3's ``MultiInputPolicy``: the ``image`` entry of
the Dict observation is routed through a NatureCNN and the ``audio``
spectrogram is flattened into an MLP branch; both embeddings are concatenated
before the actor/critic heads. Checkpoints are written periodically and the
final model is saved on completion *or* when you hit Ctrl+C.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from config import CONFIG, AudioConfig, Config
from controls import GameController
from environment import GameEnv
from perception import AudioCapture

logger = logging.getLogger("train")

ALGORITHMS: dict[str, type[BaseAlgorithm]] = {"ppo": PPO, "dqn": DQN}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--algo", choices=ALGORITHMS, default=CONFIG.train.algorithm)
    parser.add_argument("--timesteps", type=int, default=CONFIG.train.total_timesteps)
    parser.add_argument("--model-name", default=CONFIG.train.model_name)
    parser.add_argument("--resume", type=Path, help="Path to a saved .zip model to continue training")
    parser.add_argument("--check-env", action="store_true", help="Run SB3's env checker and exit")
    parser.add_argument("--dry-run", action="store_true", help="Log key presses instead of sending them")
    parser.add_argument("--no-audio", action="store_true", help="Disable loopback audio capture")
    parser.add_argument("--countdown", type=int, default=3, help="Seconds to focus the game before training")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def build_env(config: Config, *, dry_run: bool, audio_enabled: bool) -> Monitor:
    """Create the game environment wrapped in ``Monitor`` for episode stats."""
    audio_cfg = config.audio if audio_enabled else AudioConfig(enabled=False)
    env = GameEnv(
        config,
        audio=AudioCapture(audio_cfg),
        controller=GameController(config.controls, dry_run=dry_run),
    )
    config.train.log_dir.mkdir(parents=True, exist_ok=True)
    return Monitor(env, filename=str(config.train.log_dir / "monitor"))


def build_model(algo: str, env: Monitor, config: Config, resume: Path | None) -> BaseAlgorithm:
    """Instantiate a fresh model or load one from disk to continue training."""
    cls = ALGORITHMS[algo]
    if resume is not None:
        logger.info("Resuming from %s", resume)
        return cls.load(resume, env=env, device=config.train.device)

    hyper = config.train.ppo if algo == "ppo" else config.train.dqn
    return cls(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=config.train.seed,
        device=config.train.device,
        tensorboard_log=str(config.train.log_dir / "tensorboard"),
        **hyper,
    )


def countdown(seconds: int) -> None:
    """Give the user time to click into the game window before inputs start."""
    for remaining in range(seconds, 0, -1):
        print(f"Focus the game window... starting in {remaining}s", flush=True)
        time.sleep(1.0)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = CONFIG

    env = build_env(config, dry_run=args.dry_run, audio_enabled=not args.no_audio)

    if args.check_env:
        from stable_baselines3.common.env_checker import check_env

        check_env(env.unwrapped, warn=True)
        env.close()
        print("Environment passes the Stable-Baselines3 checker.")
        return 0

    model = build_model(args.algo, env, config, args.resume)

    config.train.model_dir.mkdir(parents=True, exist_ok=True)
    final_path = config.train.model_dir / f"{args.model_name}.zip"
    checkpoint_cb = CheckpointCallback(
        save_freq=config.train.checkpoint_every_steps,
        save_path=str(config.train.model_dir / "checkpoints"),
        name_prefix=args.model_name,
        save_replay_buffer=False,
    )

    countdown(args.countdown)
    logger.info("Training %s for %d timesteps", args.algo.upper(), args.timesteps)
    status = 0
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_cb,
            reset_num_timesteps=args.resume is None,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; saving model before exit")
        status = 130
    finally:
        model.save(final_path)
        logger.info("Model saved to %s", final_path)
        env.close()
    return status


if __name__ == "__main__":
    sys.exit(main())
