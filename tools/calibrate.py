"""Calibration helper: screenshot the capture region with all ROIs drawn on it.

Run this while the game is visible, then open ``assets/captures/calibration.png``
and adjust ``ScreenConfig.capture_region``, ``RewardConfig.score_region`` and
``RewardConfig.game_over_region`` in ``config.py`` until the boxes line up.

    python tools/calibrate.py                 # annotated screenshot
    python tools/calibrate.py --template      # also crop game_over_region into
                                              # assets/templates/game_over.png
                                              # (run this on the game-over screen)
    python tools/calibrate.py --live          # preview the 84x84 agent view at target FPS
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ASSETS_DIR, CONFIG  # noqa: E402
from perception import ScreenCapture  # noqa: E402

CAPTURES_DIR = ASSETS_DIR / "captures"


def annotate(frame):
    out = frame.copy()
    for region, colour, label in (
        (CONFIG.reward.score_region, (0, 255, 0), "score_region"),
        (CONFIG.reward.game_over_region, (0, 0, 255), "game_over_region"),
    ):
        p1 = (region.left, region.top)
        p2 = (region.left + region.width, region.top + region.height)
        cv2.rectangle(out, p1, p2, colour, 2)
        cv2.putText(out, label, (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", action="store_true", help="Save game_over_region crop as the template")
    parser.add_argument("--live", action="store_true", help="Show the downscaled agent view in a window")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds to wait before capturing")
    args = parser.parse_args()

    print(f"Capturing in {args.delay:.0f}s - bring the game to the front...")
    time.sleep(args.delay)

    screen = ScreenCapture(CONFIG.screen)
    frame = screen.grab_raw()

    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CAPTURES_DIR / "calibration.png"
    cv2.imwrite(str(out_path), annotate(frame))
    print(f"Saved annotated capture to {out_path}")

    if args.template:
        rows, cols = CONFIG.reward.game_over_region.as_slice()
        CONFIG.reward.game_over_template.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(CONFIG.reward.game_over_template), frame[rows, cols])
        print(f"Saved game-over template to {CONFIG.reward.game_over_template}")

    if args.live:
        print("Press q in the preview window to quit.")
        period = 1.0 / CONFIG.screen.target_fps
        while True:
            t0 = time.perf_counter()
            small = screen.preprocess(screen.grab_raw())
            cv2.imshow("agent view (84x84 upscaled)", cv2.resize(small, (336, 336), interpolation=cv2.INTER_NEAREST))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
        cv2.destroyAllWindows()

    screen.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
