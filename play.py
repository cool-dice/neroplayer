"""Run a trained policy and report how it does.

    python play.py models/ppo_20260913-0119/final.zip --mock --episodes 5
    python play.py models/ppo_.../final.zip --config config.json --countdown 5

``--record out.mp4`` writes the captured frames to a video file, which is the
easiest way to see what the agent actually saw when it died.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from stable_baselines3 import DQN, PPO

from ai_player.config import AppConfig, load_config
from ai_player.environment import make_env

ALGOS = {"ppo": PPO, "dqn": DQN}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", type=Path, help="Path to a saved .zip model")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config (defaults to the one saved next to the model)",
    )
    parser.add_argument("--algo", choices=sorted(ALGOS), default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--mock", action="store_true", help="Play the bundled simulated game")
    parser.add_argument("--deterministic", action="store_true", help="Take argmax actions")
    parser.add_argument("--render", action="store_true", help="Show captured frames in a window")
    parser.add_argument("--record", type=Path, default=None, help="Write captured frames to this .mp4")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override the agent decision rate (ignored by --mock, which runs unpaced)",
    )
    parser.add_argument("--countdown", type=int, default=0, help="Seconds to wait before starting")
    return parser.parse_args(argv)


class Recorder:
    """Optional video writer, sized from the first frame it is given.

    Taking the size from the frame rather than the config avoids a silent
    mismatch: ``--mock`` replaces the capture region, so the configured
    dimensions are not necessarily the ones being captured.
    """

    def __init__(self, path: Path | None, fps: float) -> None:
        self._path = path
        self._fps = fps
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame: np.ndarray | None) -> None:
        if self._path is None or frame is None:
            return
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self._path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self._fps,
                (width, height),
            )
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(f"Recorded {self._path}")


def resolve_config(args: argparse.Namespace) -> AppConfig:
    """Prefer an explicit --config, else the config saved beside the model."""
    if args.config is not None:
        config = load_config(args.config)
    else:
        sibling = args.model.parent / "config.json"
        if sibling.exists():
            print(f"Using config {sibling}")
            config = AppConfig.load(sibling)
        else:
            config = AppConfig()
    if args.algo:
        config.train.algo = args.algo
    if args.fps is not None:
        config.env.target_fps = args.fps
    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = resolve_config(args)

    if args.countdown > 0:
        for remaining in range(args.countdown, 0, -1):
            print(f"Starting in {remaining}s -- focus the game window...", end="\r", flush=True)
            time.sleep(1.0)
        print()

    env = make_env(config, mock=args.mock, render_mode="human" if args.render else None, seed=0)
    model = ALGOS[config.train.algo].load(args.model, device=config.train.device)

    recorder = Recorder(args.record, env.config.env.target_fps or 10.0)

    returns: list[float] = []
    lengths: list[int] = []
    points: list[float] = []
    try:
        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset()
            total, steps, done = 0.0, 0, False
            info: dict = {}
            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total += reward
                steps += 1
                done = terminated or truncated
                recorder.write(env.last_frame)
            returns.append(total)
            lengths.append(steps)
            points.append(float(info.get("episode_points", 0.0)))
            print(f"episode {episode}: return={total:8.2f}  steps={steps:5d}  points={points[-1]:.0f}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        recorder.close()
        env.close()

    if returns:
        print(
            f"\n{len(returns)} episode(s): "
            f"mean return {statistics.fmean(returns):.2f}, "
            f"mean length {statistics.fmean(lengths):.1f} steps, "
            f"mean points {statistics.fmean(points):.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
