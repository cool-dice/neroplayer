"""Point the agent at your game window, then verify what it sees.

Run this on the machine the game runs on, with the game visible.

    python calibrate.py --output config.json    # pick the regions (default)
    python calibrate.py --check                 # verify an existing config
    python calibrate.py --live                  # watch the 84x84 agent view

**Pick** (default) locks onto the foreground game window so you can alt-tab
between titles later without repeating this step. Pass ``--static-region`` for
the original three-rectangle flow (game / score / game-over). Score and
game-over boxes are optional: cancel them and the autonomous HUD detector
fills them in at train time.

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
from ai_player.controls import build_controller, format_action, parse_action_keys, validate_action_keys
from ai_player.observer import AutonomousHUDDetector, build_game_over_detector, build_score_signal
from ai_player.perception import FrameProcessor, crop_region
from ai_player.tracker import ControllabilityProbe

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
    mode.add_argument(
        "--test-keys",
        action="store_true",
        help="Validate all configured actions/keys and test pressing them in sequence",
    )
    mode.add_argument(
        "--profile",
        action="store_true",
        help="Benchmark pipeline latencies (capture, process, audio, render) to diagnose FPS bottleneck",
    )
    mode.add_argument(
        "--auto-hud",
        action="store_true",
        help="Run autonomous zero-shot HUD detection and display detected bounding boxes",
    )
    mode.add_argument(
        "--probe-avatar",
        action="store_true",
        help="Run controllability probe to auto-detect the player avatar BBox",
    )
    parser.add_argument("--output", type=Path, default=Path("config.json"))
    parser.add_argument("--config", type=Path, default=None, help="Start from an existing config")
    parser.add_argument("--monitor", type=int, default=1, help="mss monitor index (1 = primary)")
    parser.add_argument(
        "--static-region",
        action="store_true",
        help="Pin a desktop rectangle instead of following the live game window",
    )
    parser.add_argument(
        "--window-title",
        default="",
        help="Only follow windows whose title contains this substring (or matches ^regex$)",
    )
    parser.add_argument(
        "--follow-foreground",
        action="store_true",
        default=None,
        help="Follow the focused game window (default for pick mode)",
    )
    parser.add_argument(
        "--no-follow-foreground",
        action="store_true",
        help="Disable live window following; capture the saved rectangle only",
    )
    parser.add_argument(
        "--template-delay",
        type=int,
        default=10,
        help="Seconds to wait before grabbing the game-over template",
    )
    parser.add_argument("--skip-template", action="store_true")
    parser.add_argument(
        "--game-over-mode",
        choices=["auto", "template", "color", "text"],
        default=None,
        help="Game over detection mode (auto, template, color, text)",
    )
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds to wait before --check")
    parser.add_argument(
        "--auto-combos",
        action="store_true",
        help="Generate combinations for key verification and testing",
    )
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


def grab_game(config: AppConfig) -> np.ndarray:
    """Grab whatever the agent would see: live window when follow is on, else the saved region."""
    from ai_player.perception import ScreenCapture

    with ScreenCapture(config.capture) as cap:
        return cap.grab()


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
    frame = grab_game(config)

    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CAPTURES_DIR / "calibration.png"
    cv2.imwrite(str(out_path), annotate(frame, config))
    cv2.imwrite(str(CAPTURES_DIR / "score_box.png"), crop_region(frame, config.reward.score_region))
    print(f"Captured {frame.shape[1]}x{frame.shape[0]} -> {out_path}")
    print(f"Score box crop        -> {CAPTURES_DIR / 'score_box.png'}")

    go_detector = build_game_over_detector(config.reward)
    detected, confidence = go_detector.detect(frame)
    print(f"Game over detector    : {type(go_detector).__name__} (mode: {config.reward.game_over_mode})")
    print(f"Game over detected    : {detected}  (confidence {confidence:.3f})")

    # Check frame difference against a second frame 0.5s later
    time.sleep(0.5)
    second_frame = grab_game(config)
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)
    motion_diff = float(cv2.absdiff(gray1, gray2).mean()) / 255.0
    is_idle = motion_diff < config.reward.idle_diff_threshold
    thresh_pct = config.reward.idle_diff_threshold * 100.0
    print(f"Motion diff (0.5s)    : {motion_diff * 100:.2f}% (Threshold: {thresh_pct:.1f}%)")
    print(f"Motion state          : {'STAGNANT / IDLE' if is_idle else 'ACTIVE MOTION'}")

    signal = build_score_signal(config.reward)
    signal.update(frame)  # first call only primes the baseline
    time.sleep(1.0)
    points, score = signal.update(grab_game(config))
    print(f"Score reader          : {type(signal).__name__}")
    print(f"Score read            : {score if score is not None else 'n/a'}")
    print(f"Points in the last 1s : {points}")

    actions = validate_action_keys(config.control)
    print(f"\nConfigured actions ({len(actions)} total):")
    for a in actions:
        status = "OK" if a["valid"] else f"WARN: {', '.join(a['warnings'])}"
        keys_str = str(a["keys"] or "None")
        print(f"  [{a['index']}] {a['formatted']:<18} -> {keys_str:<15} [{status}]")

    print(
        "\nIf the boxes do not line up, leave follow-foreground on and let auto-HUD "
        "retarget them, or re-run `python calibrate.py --static-region`."
    )
    return 0


def run_test_keys(config: AppConfig) -> int:
    """Validate all configured action keys and simulate keypresses in sequence."""
    actions = validate_action_keys(config.control)
    print("\n" + "=" * 60)
    print("CONFIGURED ACTIONS & KEY VERIFICATION:")
    print("=" * 60)
    for a in actions:
        status = "OK" if a["valid"] else f"WARN: {', '.join(a['warnings'])}"
        keys_str = str(a["keys"] or "None")
        print(f"  [{a['index']}] {a['formatted']:<20} -> {keys_str:<15} [{status}]")
    print("=" * 60)

    try:
        controller = build_controller(config.control, dry_run=False)
    except Exception as exc:
        print(f"\nNote: Live input testing requires Windows and pydirectinput ({exc}).")
        return 0

    print("\nStarting live key test in 3 seconds -- focus your game window!")
    for sec in range(3, 0, -1):
        print(f"  {sec}...", flush=True)
        time.sleep(1.0)

    print("\nTesting actions in sequence (observe your character):")
    for a in actions:
        keys = a["keys"]
        if not keys:
            print(f"  Action {a['index']}: IDLE (wait 1.0s)...")
            time.sleep(1.0)
            continue
        print(f"  Action {a['index']}: Executing [{a['formatted']}]...")
        controller.act(a["index"])
        time.sleep(1.0)

    controller.release_all()
    print("\nKey testing complete! All configured actions executed.")
    return 0


def run_profile(config: AppConfig, n_samples: int = 50) -> int:
    """Benchmark each pipeline stage in isolation and report exact milliseconds."""
    print("\n" + "=" * 65)
    print(f"PIPELINE LATENCY PROFILER ({n_samples} frames sample)")
    print("=" * 65)

    # 1. Screen capture benchmark
    monitor = config.capture.region.as_mss_monitor()
    print(f"1. Testing screen capture region ({monitor['width']}x{monitor['height']})...")
    times_grab: list[float] = []
    frames: list[np.ndarray] = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        f = grab(monitor)
        t1 = time.perf_counter()
        times_grab.append((t1 - t0) * 1000.0)
        if len(frames) < 10:
            frames.append(f)
    avg_grab = sum(times_grab) / len(times_grab)
    min_grab = min(times_grab)
    max_grab = max(times_grab)
    print(f"   -> mss grab: avg = {avg_grab:5.2f} ms  (min {min_grab:.2f}, max {max_grab:.2f})")

    # 2. Vision preprocessing benchmark
    processor = FrameProcessor(config.vision)
    test_frame = frames[0]
    times_proc: list[float] = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        _ = processor.process(test_frame)
        t1 = time.perf_counter()
        times_proc.append((t1 - t0) * 1000.0)
    avg_proc = sum(times_proc) / len(times_proc)
    print(f"   -> FrameProcessor resize/grayscale: avg = {avg_proc:5.2f} ms")

    # 3. Audio benchmark (if enabled)
    avg_audio = 0.0
    if config.audio.enabled:
        print("3. Testing Audio feature extraction (FFT / mel-spectrogram)...")
        from ai_player.perception import AudioFeatureExtractor

        extractor = AudioFeatureExtractor(config.audio)
        dummy_audio = np.random.uniform(-0.1, 0.1, config.audio.window_samples).astype(np.float32)
        times_audio: list[float] = []
        for _ in range(n_samples):
            t0 = time.perf_counter()
            _ = extractor.extract(dummy_audio)
            t1 = time.perf_counter()
            times_audio.append((t1 - t0) * 1000.0)
        avg_audio = sum(times_audio) / len(times_audio)
        print(f"   -> librosa mel-spectrogram on CPU: avg = {avg_audio:5.2f} ms")
    else:
        print("3. Audio: DISABLED (--no-audio) -> 0.0 ms")

    # 4. Input action latency
    avg_act = 0.0
    print("4. Testing Input controller call overhead...")
    try:
        controller = build_controller(config.control, dry_run=False)
        # Check an active action (e.g. index 1 if available)
        test_act = 1 if len(config.control.action_keys) > 1 else 0
        times_act: list[float] = []
        for _ in range(min(5, n_samples)):
            t0 = time.perf_counter()
            controller.act(test_act)
            t1 = time.perf_counter()
            times_act.append((t1 - t0) * 1000.0)
        controller.release_all()
        avg_act = sum(times_act) / len(times_act)
        print(f"   -> input controller overhead: avg = {avg_act:5.2f} ms")
    except Exception as exc:
        print(f"   -> input controller skipped ({exc})")

    # 5. OpenCV imshow / waitKey render overhead benchmark
    print("5. Testing OpenCV render overhead (imshow + waitKey(1))...")
    times_rnd: list[float] = []
    try:
        dummy_display = frames[0].copy()
        cv2.imshow("latency_test", dummy_display)
        cv2.waitKey(1)
        for _ in range(n_samples):
            t0 = time.perf_counter()
            cv2.imshow("latency_test", dummy_display)
            cv2.waitKey(1)
            t1 = time.perf_counter()
            times_rnd.append((t1 - t0) * 1000.0)
        cv2.destroyWindow("latency_test")
        avg_rnd = sum(times_rnd) / len(times_rnd)
        print(f"   -> cv2.imshow + waitKey(1): avg = {avg_rnd:5.2f} ms")
    except Exception as exc:
        avg_rnd = 0.0
        print(f"   -> OpenCV display skipped ({exc})")

    # Summary and calculation
    total_ms = avg_grab + avg_proc + avg_audio + avg_act + avg_rnd
    max_possible_fps = 1000.0 / max(0.1, total_ms)
    print("=" * 65)
    print("LATENCY BREAKDOWN:")
    print(f"   Screen Capture (mss)   : {avg_grab:6.2f} ms")
    print(f"   Vision Proc (84x84)    : {avg_proc:6.2f} ms")
    print(f"   Audio Extraction       : {avg_audio:6.2f} ms")
    print(f"   Keyboard Action        : {avg_act:6.2f} ms")
    print(f"   OpenCV Window/Render   : {avg_rnd:6.2f} ms")
    print("   --------------------------------------")
    print(f"   TOTAL ESTIMATED STEP   : {total_ms:6.2f} ms")
    print(f"   MAX THEORETICAL FPS    : {max_possible_fps:6.1f} FPS")
    print("=" * 65)

    if avg_rnd > 12.0:
        print(">> BOTTLENECK DETECTED: cv2.waitKey(1) on Windows takes ~15-16ms (timer quantization).")
        print("   Recommendation: run without `--render` for maximum training speed.")
    if avg_act > 15.0:
        print(">> BOTTLENECK DETECTED: Keyboard tap_duration or pydirectinput overhead is >15ms.")
        print("   Recommendation: lower `tap_duration` in config.json or use `hold_keys: true`.")
    if avg_audio > 15.0:
        print(">> BOTTLENECK DETECTED: Audio extraction is taking significant CPU time.")
        print("   Recommendation: run with `--no-audio` to recover full FPS.")
    if avg_grab > 35.0:
        print(">> BOTTLENECK DETECTED: Screen capture is slow (mss taking >35ms).")
        print("   Recommendation: reduce game window resolution or capture region size.")
    print()
    return 0


def run_live(config: AppConfig) -> int:
    """Show the 84x84 grayscale observation the CNN actually receives."""
    processor = FrameProcessor(config.vision)
    period = 1.0 / config.env.target_fps if config.env.target_fps else 0.0
    print("Press q in the preview window to quit.")
    from ai_player.perception import ScreenCapture

    try:
        with ScreenCapture(config.capture) as cap:
            while True:
                started = time.perf_counter()
                small = processor.process(cap.grab())
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


def apply_capture_flags(args: argparse.Namespace, config: AppConfig) -> None:
    """Apply --follow-foreground / --window-title / --static-region to ``config``."""
    if getattr(args, "window_title", ""):
        config.capture.window_title = args.window_title
    if getattr(args, "no_follow_foreground", False) or getattr(args, "static_region", False):
        config.capture.follow_foreground = False
    elif getattr(args, "follow_foreground", None) or getattr(args, "window_title", ""):
        config.capture.follow_foreground = True


def run_pick_follow(args: argparse.Namespace, config: AppConfig) -> int:
    """Write a config that follows the live game window instead of a dragged box."""
    from ai_player.window_target import WindowLocator

    config.capture.follow_foreground = True
    config.capture.monitor_index = args.monitor
    locator = WindowLocator()
    target = locator.locate(config.capture)
    if target is not None:
        config.capture.region = target.region
        print(f"Following '{target.title}'  {target.region.width}x{target.region.height} at "
              f"({target.region.left},{target.region.top})")
        print("Alt-tab to another game during train/play and capture will retarget.")
    else:
        print(
            "No eligible game window yet. follow_foreground is on: the agent will "
            "lock onto whichever game you focus when training starts."
        )
        print(f"Fallback rectangle: {config.capture.region}")

    print("\nOptional: drag HUD boxes on a live grab (cancel with c to skip; auto-HUD fills them in).")
    try:
        game_frame = grab_game(config)
    except RuntimeError as exc:
        print(exc)
        saved = config.save(args.output)
        print(f"\nWrote {saved} without a live grab.")
        return 0
    try:
        score = select_region(game_frame, "Optional  Drag a tight box around the SCORE digits")
    except cv2.error:
        score = None
    if score is not None:
        config.reward.score_region = score
    try:
        game_over = select_region(game_frame, "Optional  Drag a box where GAME OVER appears")
    except cv2.error:
        game_over = None
    if game_over is not None:
        config.reward.game_over_region = game_over

    saved = config.save(args.output)
    print(f"\nWrote {saved}")
    print(f"  follow_foreground: {config.capture.follow_foreground}")
    print(f"  window_title     : {config.capture.window_title or '(any)'}")
    print(f"  capture region   : {config.capture.region}  (fallback / last lock)")
    print(f"  score region     : {config.reward.score_region}")
    print(f"  game-over region : {config.reward.game_over_region}")
    print("\nVerify what the agent sees:")
    print(f"  python calibrate.py --config {saved} --check")
    print("Then train; alt-tab between games whenever you like:")
    print(f"  python train.py --config {saved} --timesteps 500000 --render")
    return 0


def run_pick(args: argparse.Namespace, config: AppConfig) -> int:
    """Lock onto a live window (default) or pin a static desktop rectangle."""
    apply_capture_flags(args, config)
    if not args.static_region:
        return run_pick_follow(args, config)

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
    config.capture.follow_foreground = False
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

    over_frame = grab_game(config)
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


def run_auto_hud(args: argparse.Namespace, config: AppConfig) -> int:
    """Run autonomous zero-shot HUD detection on game frame and save/display result."""
    print(f"Capturing game frame in {args.delay:.0f}s...")
    time.sleep(args.delay)
    frame = grab_game(config)

    detector = AutonomousHUDDetector(config.hud)
    detected = detector.detect_hud(frame)

    print("\n" + "=" * 60)
    print("AUTONOMOUS HUD DETECTION RESULT:")
    print("=" * 60)
    print(f"  Method     : {detected.method.upper()}")
    print(f"  Confidence : {detected.confidence:.2f}")
    print(f"  Score ROI  : {detected.score_region}")
    print(f"  Lives ROI  : {detected.lives_region}")
    print(f"  Game Over  : {detected.game_over_region}")
    print("=" * 60)

    if detected.score_region:
        config.reward.score_region = detected.score_region
    if detected.game_over_region:
        config.reward.game_over_region = detected.game_over_region

    go_detector = build_game_over_detector(config.reward)
    go_detected, go_conf = go_detector.detect(frame)
    print(f"  Game Over Detector : {type(go_detector).__name__} (mode: {config.reward.game_over_mode})")
    print(f"  Game Over Check    : {go_detected} (confidence: {go_conf:.3f})")
    print("=" * 60)

    out_img = annotate(frame, config)
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CAPTURES_DIR / "auto_hud_detected.png"
    cv2.imwrite(str(out_path), out_img)
    print(f"\nSaved preview annotation to {out_path}")

    if args.output:
        saved = config.save(args.output)
        print(f"Updated configuration written to {saved}")
    return 0


def run_probe_avatar(args: argparse.Namespace, config: AppConfig) -> int:
    """Run controllability probe to discover the player avatar bounding box."""
    print("\nStarting Controllability Probe in 3 seconds -- focus game window!")
    for sec in range(3, 0, -1):
        print(f"  {sec}...", flush=True)
        time.sleep(1.0)

    probe = ControllabilityProbe(config.tracker)

    frame0 = grab_game(config)
    probe.record_probe_step("idle", frame0)

    try:
        controller = build_controller(config.control, dry_run=False)
    except Exception as exc:
        print(f"Input controller unavailable ({exc}); probe requires live controls.")
        return 1

    actions = config.control.action_keys
    for a_idx, raw_k in enumerate(actions):
        keys = parse_action_keys(raw_k)
        if not keys:
            continue
        act_name = format_action(raw_k)
        print(f"  Probing action [{act_name}]...")
        controller.act(a_idx)
        time.sleep(0.06)
        f = grab_game(config)
        probe.record_probe_step(act_name, f)
        time.sleep(0.04)

    controller.release_all()
    print("Analyzing differential optical flow...")
    bbox = probe.analyze_probes()

    print("\n" + "=" * 60)
    print("CONTROLLABILITY PROBE RESULT:")
    print("=" * 60)
    if bbox is not None:
        print(f"  Player Avatar BBox : x={bbox.x}, y={bbox.y}, w={bbox.w}, h={bbox.h}")
        print(f"  Confidence         : {probe.confidence:.2f}")
        # Draw box on last frame
        annotated = frame0.copy()
        cv2.rectangle(annotated, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), (0, 255, 255), 2)
        cv2.putText(
            annotated, "PLAYER AVATAR", (bbox.x, max(14, bbox.y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
        )
        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(CAPTURES_DIR / "avatar_probe.png"), annotated)
        print(f"  Preview written to {CAPTURES_DIR / 'avatar_probe.png'}")
    else:
        print("  Player Avatar      : Inconclusive (insufficient differential flow detected)")
    print("=" * 60)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.auto_combos:
        config.control.auto_combos = True
    if args.game_over_mode:
        config.reward.game_over_mode = args.game_over_mode
    apply_capture_flags(args, config)
    try:
        if args.check:
            return run_check(args, config)
        if args.live:
            return run_live(config)
        if args.test_keys:
            return run_test_keys(config)
        if args.profile:
            return run_profile(config)
        if args.auto_hud:
            return run_auto_hud(args, config)
        if args.probe_avatar:
            return run_probe_avatar(args, config)
        return run_pick(args, config)
    except RuntimeError as exc:
        print(exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
