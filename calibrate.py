"""Point the agent at your game window, then verify what it sees.

Run this on the machine the game runs on, with the game visible.

    python calibrate.py --output config.json    # pick the regions (default)
    python calibrate.py --check                 # verify an existing config
    python calibrate.py --live                  # watch the 84x84 agent view

**Pick** (default) asks you to drag three rectangles (confirm each with Enter,
cancel with c): the game window, the score digits, and the area where the
game-over screen appears. It then offers to capture a game-over template.

**Check** needs no GUI: it saves an annotated screenshot to
``assets/captures/calibration.png`` and prints what the observer currently
reads, so you can confirm the regions are right before training.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ai_player.config import CAPTURES_DIR, TEMPLATES_DIR, AppConfig, Region, load_config
from ai_player.observer import build_game_over_detector, build_score_signal
from ai_player.perception import FrameProcessor, crop_region

WINDOW = "calibrate -- drag a box, Enter to accept, c to cancel"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify the current config: annotated screenshot + observer readings",
    )
    mode.add_argument(
        "--live", action="store_true", help="Preview the downscaled agent view (needs a display)"
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
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds to wait before --check")
    return parser.parse_args(argv)


def grab(monitor: dict[str, int]) -> np.ndarray:
    """Screenshot one region, reporting capture failures in plain language."""
    try:
        import mss

        # mss >= 10 renamed the entry point to MSS and deprecated the old name.
        with getattr(mss, "MSS", mss.mss)() as sct:
            raw = sct.grab(monitor)
    except Exception as exc:
        raise RuntimeError(
            f"Screen capture failed ({exc}). Calibration has to run on the "
            "machine showing the game, in a normal desktop session."
        ) from exc
    return np.ascontiguousarray(np.asarray(raw, dtype=np.uint8)[:, :, :3])


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


def annotate(frame: np.ndarray, config: AppConfig) -> np.ndarray:
    """Draw the score and game-over regions onto a captured frame."""
    out = frame.copy()
    boxes = (
        (config.reward.score_region, (0, 255, 0), "score_region"),
        (config.reward.game_over_region, (0, 0, 255), "game_over_region"),
    )
    for region, colour, label in boxes:
        top_left = (region.left, region.top)
        bottom_right = (region.left + region.width, region.top + region.height)
        cv2.rectangle(out, top_left, bottom_right, colour, 2)
        cv2.putText(
            out,
            label,
            (top_left[0], max(12, top_left[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            1,
        )
    return out


def resolve_monitor(index: int) -> dict[str, int] | None:
    try:
        import mss

        with getattr(mss, "MSS", mss.mss)() as sct:
            monitors = sct.monitors
    except Exception as exc:
        print(f"Screen capture unavailable: {exc}")
        return None
    if not 0 <= index < len(monitors):
        print(f"Monitor {index} not found; available: 0..{len(monitors) - 1}")
        return None
    return monitors[index]


def run_check(args: argparse.Namespace, config: AppConfig) -> int:
    """Capture one frame and report exactly what the observer makes of it."""
    print(f"Capturing in {args.delay:.0f}s -- bring the game to the front...")
    time.sleep(args.delay)
    frame = grab(config.capture.region.as_mss_monitor())

    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CAPTURES_DIR / "calibration.png"
    cv2.imwrite(str(out_path), annotate(frame, config))
    cv2.imwrite(str(CAPTURES_DIR / "score_box.png"), crop_region(frame, config.reward.score_region))
    print(f"Captured {frame.shape[1]}x{frame.shape[0]} -> {out_path}")
    print(f"Score box crop        -> {CAPTURES_DIR / 'score_box.png'}")

    detected, confidence = build_game_over_detector(config.reward).detect(frame)
    print(f"Game over detected    : {detected}  (confidence {confidence:.3f})")

    signal = build_score_signal(config.reward)
    signal.update(frame)  # first call only primes the baseline
    time.sleep(1.0)
    points, score = signal.update(grab(config.capture.region.as_mss_monitor()))
    print(f"Score reader          : {type(signal).__name__}")
    print(f"Score read            : {score if score is not None else 'n/a'}")
    print(f"Points in the last 1s : {points}")
    print(
        "\nIf the boxes in the screenshot do not line up, re-run `python calibrate.py` and drag them again."
    )
    return 0


def run_live(config: AppConfig) -> int:
    """Show the 84x84 grayscale observation the CNN actually receives."""
    processor = FrameProcessor(config.vision)
    period = 1.0 / config.env.target_fps if config.env.target_fps else 0.0
    print("Press q in the preview window to quit.")
    try:
        while True:
            started = time.perf_counter()
            small = processor.process(grab(config.capture.region.as_mss_monitor()))
            preview = cv2.resize(small, (336, 336), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("agent view (84x84, upscaled)", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(max(0.0, period - (time.perf_counter() - started)))
    except cv2.error as exc:
        print(f"OpenCV could not open a preview window: {exc}")
        return 2
    finally:
        cv2.destroyAllWindows()
    return 0


def run_pick(args: argparse.Namespace, config: AppConfig) -> int:
    """Interactive three-rectangle flow, then an optional template capture."""
    monitor = resolve_monitor(args.monitor)
    if monitor is None:
        return 2

    desktop = grab(monitor)
    print(f"Captured monitor {args.monitor}: {desktop.shape[1]}x{desktop.shape[0]}")

    try:
        region = select_region(desktop, "1/3  Drag a box around the GAME WINDOW")
    except cv2.error as exc:
        print(
            f"OpenCV could not open a window for region selection ({exc}). "
            "Install the GUI build with `pip install opencv-python`, run this on "
            "the machine with the display, or edit config.json by hand and "
            "verify it with `python calibrate.py --check`."
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
        capture_template(args, config)

    saved = config.save(args.output)
    print(f"\nWrote {saved}")
    print(f"  capture region   : {config.capture.region}")
    print(f"  score region     : {config.reward.score_region}")
    print(f"  game-over region : {config.reward.game_over_region}")
    print("\nVerify what the agent sees:")
    print(f"  python calibrate.py --config {saved} --check")
    print("Then sanity-check the reward signal without touching the game:")
    print(f"  python train.py --config {saved} --dry-run --timesteps 2000")
    return 0


def capture_template(args: argparse.Namespace, config: AppConfig) -> None:
    """Crop the game-over region into a template for ``cv2.matchTemplate``."""
    print(f"\nPut the game into its GAME OVER state. Capturing in {args.template_delay}s...")
    for remaining in range(args.template_delay, 0, -1):
        print(f"  {remaining}s ", end="\r", flush=True)
        time.sleep(1.0)

    over_frame = grab(config.capture.region.as_mss_monitor())
    patch = crop_region(over_frame, config.reward.game_over_region)
    if patch.size == 0:
        print("\nGame-over region is empty; skipping the template.")
        return

    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    template_path = TEMPLATES_DIR / "game_over.png"
    cv2.imwrite(str(template_path), cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
    config.reward.game_over_template = str(template_path)
    # Record the banner's mean colour too, so the fallback detector stays
    # calibrated even if the template is later deleted.
    mean = patch.reshape(-1, 3).mean(axis=0)
    config.reward.game_over_color_bgr = tuple(round(float(v)) for v in mean)
    print(f"\nSaved template {template_path} (mean BGR {config.reward.game_over_color_bgr})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    try:
        if args.check:
            return run_check(args, config)
        if args.live:
            return run_live(config)
        return run_pick(args, config)
    except RuntimeError as exc:
        print(exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
