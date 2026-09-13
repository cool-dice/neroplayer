#!/usr/bin/env python3
"""Dump a screenshot of ``config.CAPTURE_REGION`` with score / game-over ROIs drawn.

Use this after pointing CAPTURE_REGION at your game window so you can
crop ``assets/game_over.png`` from a real death screen and tune SCORE_ROI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import config
from observer import ensure_game_over_template


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview the configured capture region.")
    p.add_argument("--out", default="assets/preview.png")
    p.add_argument("--full-screen", action="store_true", help="Ignore CAPTURE_REGION and grab monitor 1.")
    return p.parse_args()


def grab(full_screen: bool) -> np.ndarray:
    import mss

    with mss.mss() as sct:
        if full_screen:
            shot = sct.grab(sct.monitors[1])
        else:
            shot = sct.grab(config.CAPTURE_REGION)
        return cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)


def draw_rois(frame: np.ndarray) -> np.ndarray:
    annotated = frame.copy()
    x, y, w, h = config.SCORE_ROI
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(annotated, "SCORE_ROI", (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if config.GAME_OVER_ROI is not None:
        gx, gy, gw, gh = config.GAME_OVER_ROI
        cv2.rectangle(annotated, (gx, gy), (gx + gw, gy + gh), (0, 0, 255), 2)
        cv2.putText(annotated, "GAME_OVER_ROI", (gx, max(12, gy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return annotated


def main() -> None:
    args = parse_args()
    ensure_game_over_template()
    frame = grab(args.full_screen)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    annotated = draw_rois(frame)
    roi_path = out.with_name(out.stem + "_rois.png")
    cv2.imwrite(str(roi_path), annotated)
    print(f"Wrote {out}  ({frame.shape[1]}x{frame.shape[0]})")
    print(f"Wrote {roi_path} with SCORE_ROI (green) and GAME_OVER_ROI (red)")
    print("Crop a tight GAME OVER sprite from the preview and save it as assets/game_over.png")


if __name__ == "__main__":
    main()
