"""Interactive calibration: point the agent at your game window.

Run this on the machine the game runs on, with the game visible:

    python calibrate.py --output config.json

You will be asked to drag three rectangles (confirm each with Enter, cancel
with c):

1. the game window itself -- everything the agent sees;
2. the score box -- as tight as possible around the digits, which is what makes
   OCR reliable;
3. the region that shows the game-over screen.

Finally it can capture a game-over template: put the game into its game-over
state, then let the countdown expire. Template matching is noticeably more
robust than the mean-colour fallback, so it is worth the extra 10 seconds.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ai_player.config import ASSETS_DIR, Region, load_config

WINDOW = "calibrate -- drag a box, Enter to accept, c to cancel"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, default=Path("config.json"))
    parser.add_argument("--config", type=Path, default=None, help="Start from an existing config")
    parser.add_argument("--monitor", type=int, default=1, help="mss monitor index (1 = primary)")
    parser.add_argument(
        "--template-delay",
        type=int,
        default=10,
        help="Seconds to wait before grabbing the game-over template",
    )
    parser.add_argument("--skip-template", action="store_true")
    return parser.parse_args(argv)


def grab(monitor: dict[str, int]) -> np.ndarray:
    import mss

    with mss.mss() as sct:
        return np.ascontiguousarray(np.asarray(sct.grab(monitor), dtype=np.uint8)[:, :, :3])


def select_region(image: np.ndarray, prompt: str) -> Region | None:
    """Let the user drag a rectangle; returns ``None`` if they cancel."""
    print(f"\n{prompt}")
    display = image.copy()
    cv2.putText(display, prompt[:60], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    x, y, w, h = cv2.selectROI(WINDOW, display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(WINDOW)
    if w == 0 or h == 0:
        return None
    return Region(left=int(x), top=int(y), width=int(w), height=int(h))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    try:
        import mss

        with mss.mss() as sct:
            monitors = sct.monitors
        if not 0 <= args.monitor < len(monitors):
            print(f"Monitor {args.monitor} not found; available: 0..{len(monitors) - 1}")
            return 2
        monitor = monitors[args.monitor]
    except Exception as exc:
        print(f"Screen capture unavailable: {exc}")
        return 2

    desktop = grab(monitor)
    print(f"Captured monitor {args.monitor}: {desktop.shape[1]}x{desktop.shape[0]}")

    try:
        region = select_region(desktop, "1/3  Drag a box around the GAME WINDOW")
    except cv2.error as exc:
        print(
            "OpenCV could not open a window for region selection "
            f"({exc}). Install the GUI build with `pip install opencv-python` "
            "and run this on the machine with the display."
        )
        return 2
    if region is None:
        print("Cancelled.")
        return 1
    # selectROI works in monitor-local coordinates; mss needs desktop-absolute.
    config.capture.monitor_index = args.monitor
    config.capture.region = Region(
        left=monitor["left"] + region.left,
        top=monitor["top"] + region.top,
        width=region.width,
        height=region.height,
    )

    game_frame = grab(config.capture.region.as_mss_monitor())
    score = select_region(game_frame, "2/3  Drag a tight box around the SCORE digits")
    if score is not None:
        config.reward.score_region = score
    game_over = select_region(game_frame, "3/3  Drag a box where GAME OVER appears")
    if game_over is not None:
        config.reward.game_over_region = game_over

    if not args.skip_template:
        print(
            f"\nPut the game into its GAME OVER state. Capturing in "
            f"{args.template_delay}s..."
        )
        for remaining in range(args.template_delay, 0, -1):
            print(f"  {remaining}s ", end="\r", flush=True)
            time.sleep(1.0)
        over_frame = grab(config.capture.region.as_mss_monitor())
        patch = config.reward.game_over_region.crop(over_frame)
        if patch.size == 0:
            print("\nGame-over region is empty; skipping the template.")
        else:
            ASSETS_DIR.mkdir(parents=True, exist_ok=True)
            template_path = ASSETS_DIR / "game_over.png"
            cv2.imwrite(str(template_path), cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
            config.reward.game_over_template = str(template_path)
            # Record the banner's mean colour too, so the fallback detector is
            # calibrated even when the template is later removed.
            mean = patch.reshape(-1, 3).mean(axis=0)
            config.reward.game_over_color_bgr = tuple(round(float(v)) for v in mean)
            print(f"\nSaved template {template_path} (mean BGR {config.reward.game_over_color_bgr})")

    saved = config.save(args.output)
    print(f"\nWrote {saved}")
    print(f"  capture region   : {config.capture.region}")
    print(f"  score region     : {config.reward.score_region}")
    print(f"  game-over region : {config.reward.game_over_region}")
    print("\nSanity-check it without touching the game:")
    print(f"  python train.py --config {saved} --dry-run --timesteps 2000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
