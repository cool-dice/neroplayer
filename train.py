"""Train the AI player with Stable-Baselines3.

Typical use:

    # Smoke-test the whole pipeline against the bundled simulated game
    python train.py --mock --timesteps 20000

    # Train on a real window (edit/generate config.json with calibrate.py first)
    python train.py --config config.json --timesteps 500000 --countdown 5

The model is saved on normal completion *and* on Ctrl+C, so interrupting a run
never loses the policy. Checkpoints land in ``models/<run>/checkpoints``.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from types import FrameType

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ai_player.config import LOGS_DIR, MODELS_DIR, AppConfig, load_config
from ai_player.environment import make_env
from ai_player.policies import MultiModalExtractor

ALGOS = {"ppo": PPO, "dqn": DQN}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=None, help="JSON config from calibrate.py")
    parser.add_argument("--algo", choices=sorted(ALGOS), default=None, help="RL algorithm")
    parser.add_argument("--timesteps", type=int, default=None, help="Total environment steps to train for")
    parser.add_argument("--run-name", default=None, help="Name for the models/ and logs/ subfolders")
    parser.add_argument("--resume", type=Path, default=None, help="Path to a .zip model to continue training")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto | cpu | cuda")
    parser.add_argument("--fps", type=float, default=None, help="Override the agent decision rate")
    parser.add_argument("--mock", action="store_true", help="Train against the bundled simulated game")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Capture the real screen but swallow key presses (safe calibration run)",
    )
    parser.add_argument("--no-audio", action="store_true", help="Disable audio capture and the audio branch")
    parser.add_argument("--render", action="store_true", help="Show the captured frames in a window")
    parser.add_argument(
        "--countdown",
        type=int,
        default=0,
        help="Seconds to wait before starting, so you can focus the game window",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> AppConfig:
    """Load the config file and apply command-line overrides."""
    config = load_config(args.config)
    if args.algo:
        config.train.algo = args.algo
    if args.timesteps is not None:
        config.train.total_timesteps = args.timesteps
    if args.seed is not None:
        config.train.seed = args.seed
    if args.device:
        config.train.device = args.device
    if args.fps is not None:
        config.env.target_fps = args.fps
    if args.no_audio:
        config.audio.enabled = False
    return config


def build_model(config: AppConfig, env: DummyVecEnv, tensorboard_log: Path | None):
    """Instantiate PPO or DQN with the multi-modal extractor."""
    train = config.train
    policy_kwargs = {
        "features_extractor_class": MultiModalExtractor,
        "features_extractor_kwargs": {"features_dim": train.features_dim},
    }
    common = {
        "policy": "MultiInputPolicy",  # required for Dict observation spaces
        "env": env,
        "learning_rate": train.learning_rate,
        "gamma": train.gamma,
        "seed": train.seed,
        "device": train.device,
        "tensorboard_log": str(tensorboard_log) if tensorboard_log else None,
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
    }

    if train.algo == "ppo":
        policy_kwargs["net_arch"] = {"pi": [256], "vf": [256]}
        return PPO(
            **common,
            n_steps=train.n_steps,
            batch_size=train.batch_size,
            n_epochs=train.n_epochs,
            gae_lambda=train.gae_lambda,
            clip_range=train.clip_range,
            ent_coef=train.ent_coef,
            vf_coef=train.vf_coef,
            max_grad_norm=train.max_grad_norm,
        )

    policy_kwargs["net_arch"] = [256]
    return DQN(
        **common,
        buffer_size=train.buffer_size,
        batch_size=train.batch_size,
        learning_starts=train.learning_starts,
        train_freq=train.train_freq,
        target_update_interval=train.target_update_interval,
        exploration_fraction=train.exploration_fraction,
    )


def install_interrupt_handler() -> None:
    """Turn the second Ctrl+C into a hard exit.

    The first one propagates as ``KeyboardInterrupt`` so ``main`` can save the
    model; a second one means the user wants out now.
    """
    original = signal.getsignal(signal.SIGINT)

    def handler(signum: int, frame: FrameType | None) -> None:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\nStopping after the current step; press Ctrl+C again to force quit.", flush=True)
        if callable(original):
            original(signum, frame)
        else:  # pragma: no cover - depends on the host runtime
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handler)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)

    run_name = args.run_name or f"{config.train.algo}_{time.strftime('%Y%m%d-%H%M%S')}"
    model_dir = MODELS_DIR / run_name
    log_dir = LOGS_DIR / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    config.save(model_dir / "config.json")

    if args.countdown > 0:
        for remaining in range(args.countdown, 0, -1):
            print(f"Starting in {remaining}s -- focus the game window...", end="\r", flush=True)
            time.sleep(1.0)
        print()

    render_mode = "human" if args.render else None
    env = make_env(
        config,
        mock=args.mock,
        dry_run=args.dry_run,
        render_mode=render_mode,
        seed=config.train.seed,
    )
    # Monitor records episode return/length, which is what shows up as
    # rollout/ep_rew_mean in the logs -- the number to watch while training.
    vec_env = DummyVecEnv([lambda: Monitor(env, filename=str(log_dir / "monitor.csv"))])

    if args.resume:
        algo_cls = ALGOS[config.train.algo]
        print(f"Resuming {config.train.algo.upper()} from {args.resume}")
        model = algo_cls.load(args.resume, env=vec_env, device=config.train.device)
    else:
        model = build_model(config, vec_env, log_dir)

    checkpoint = CheckpointCallback(
        save_freq=max(1, config.train.checkpoint_every),
        save_path=str(model_dir / "checkpoints"),
        name_prefix=config.train.algo,
    )

    install_interrupt_handler()
    status = 0
    try:
        model.learn(
            total_timesteps=config.train.total_timesteps,
            callback=checkpoint,
            reset_num_timesteps=args.resume is None,
            tb_log_name=run_name,
            progress_bar=False,
        )
        model.save(model_dir / "final")
        print(f"Training finished. Saved {model_dir / 'final.zip'}")
    except KeyboardInterrupt:
        model.save(model_dir / "interrupted")
        print(f"\nInterrupted. Saved {model_dir / 'interrupted.zip'}")
        status = 130
    finally:
        vec_env.close()
    return status


if __name__ == "__main__":
    sys.exit(main())
