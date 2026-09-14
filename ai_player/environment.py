"""The Gymnasium environment: one agent decision per captured frame.

Timeline of a single ``step``:

1. Send the action to the OS (``controls``).
2. Sleep until the next frame slot so the agent runs at a fixed decision rate.
3. Capture the screen and the most recent audio window (``perception``).
4. Score the frame for reward/termination (``observer``).
5. Return ``(obs, reward, terminated, truncated, info)``.

Holding the decision rate constant is not cosmetic: frame stacking encodes
velocity as a pixel delta *per step*, so a wobbling step duration makes the
same state look different and slows learning down considerably.

Every collaborator is injectable, which is how ``--mock`` and the tests run the
identical control flow without Windows, a display or audio hardware.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, ClassVar

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import AppConfig
from .controls import ActionController, build_controller, format_action, get_effective_action_keys
from .observer import GameObserver
from .perception import (
    AudioFeatureExtractor,
    AudioSource,
    FrameProcessor,
    FrameSource,
    FrameStack,
    ScreenCapture,
    build_audio_source,
)
from .tracker import ControllabilityProbe, EntityKind, EntityTracker


class GameEnv(gym.Env):
    """Screen-and-sound driven environment for a windowed game.

    Observation is a ``Dict`` space so that image and audio keep their own
    natural shapes and can be fed to different network branches:

    * ``frames``: ``(frame_stack, 84, 84)`` uint8, channels-first for the CNN.
    * ``audio``:  ``(n_mels, n_frames)`` float32 log-mel spectrogram in [0, 1]
      (present only when audio capture is enabled).

    Action space is ``Discrete(len(control.action_keys))``, with one entry
    reserved for "do nothing".
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 30,
    }

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        frame_source: FrameSource | None = None,
        audio_source: AudioSource | None = None,
        controller: ActionController | None = None,
        observer: GameObserver | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or AppConfig()
        self.render_mode = render_mode

        self._frames = frame_source or ScreenCapture(self.config.capture)
        self._audio = audio_source or build_audio_source(self.config.audio)
        self._controller = controller or build_controller(self.config.control)
        self._observer = observer or GameObserver(self.config.reward, hud_config=self.config.hud)
        # Idempotent: build_audio_source already started the recorder it built,
        # but an injected source may not be running yet.
        self._audio.start()

        self._probe = ControllabilityProbe(self.config.tracker)
        self._entity_tracker = EntityTracker(self.config.tracker)
        self._probed_avatar = False

        self._processor = FrameProcessor(self.config.vision)
        self._stack = FrameStack(self.config.vision)
        self._audio_features = AudioFeatureExtractor(self.config.audio)
        self._audio_enabled = self.config.audio.enabled

        obs_spaces: dict[str, spaces.Space] = {
            "frames": spaces.Box(low=0, high=255, shape=self.config.vision.observation_shape, dtype=np.uint8)
        }
        if self._audio_enabled:
            obs_spaces["audio"] = spaces.Box(
                low=0.0, high=1.0, shape=self.config.audio.observation_shape, dtype=np.float32
            )
        self.observation_space = spaces.Dict(obs_spaces)
        self._action_keys = get_effective_action_keys(self.config.control)
        self.action_space = spaces.Discrete(len(self._action_keys))

        self._frame_period = 1.0 / self.config.env.target_fps if self.config.env.target_fps else 0.0
        self._next_frame_at = 0.0
        self._last_frame: np.ndarray | None = None
        self._last_info: dict[str, Any] = {}
        self._last_step_time = 0.0
        self._fps_history: deque[float] = deque(maxlen=30)
        self._real_fps = 0.0
        # Latency breakdown (rolling ms)
        self._prof_act: deque[float] = deque(maxlen=30)
        self._prof_grab: deque[float] = deque(maxlen=30)
        self._prof_proc: deque[float] = deque(maxlen=30)
        self._prof_obs: deque[float] = deque(maxlen=30)
        self._prof_render: deque[float] = deque(maxlen=30)
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._episode_index = 0
        self._window_name = "ai-player"
        self._window_open = False

    @property
    def last_frame(self) -> np.ndarray | None:
        """The most recently captured frame, in BGR, before preprocessing."""
        return None if self._last_frame is None else self._last_frame.copy()

    @property
    def last_info(self) -> dict[str, Any]:
        """Telemetry and metrics from the most recent step or reset."""
        return dict(self._last_info)

    # -- gymnasium API -------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Restart the round and return the first observation."""
        super().reset(seed=seed)
        env_cfg = self.config.env

        # A captured window cannot be reseeded, but a simulated one can: the
        # hook is what makes mock runs reproducible.
        if seed is not None:
            seed_backend = getattr(self._frames, "seed", None)
            if callable(seed_backend):
                seed_backend(seed)

        self._controller.release_all()
        self._observer.reset()
        self._controller.restart()
        if env_cfg.reset_delay:
            time.sleep(env_cfg.reset_delay)

        frame = self._wait_for_playable_frame()

        # Autonomous HUD auto-configuration on first playable frame if enabled
        if self.config.hud.auto_detect and self._observer.detected_hud is None:
            self._observer.auto_configure_hud(frame)

        # Autonomous Controllability Probe for avatar detection on first episode
        if self.config.tracker.enabled and self.config.tracker.auto_probe and not self._probed_avatar:
            self._run_controllability_probe(frame)
            self._probed_avatar = True

        self._stack.reset(self._processor.process(frame))
        # A couple of throwaway frames let start-of-round animations finish, so
        # the stack holds live gameplay rather than a menu.
        for _ in range(max(0, env_cfg.warmup_frames)):
            frame = self._capture_frame()
            self._stack.push(self._processor.process(frame))
            if self.config.tracker.enabled and self._probe.player_bbox is not None:
                self._probe.update_player_location(frame)

        self._episode_steps = 0
        self._episode_reward = 0.0
        self._episode_index += 1
        self._last_step_time = 0.0
        self._fps_history.clear()
        self._real_fps = 0.0
        self._next_frame_at = time.perf_counter()
        self._last_frame = frame

        info = {"episode_index": self._episode_index}
        return self._observation(), info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        now = time.perf_counter()
        if self._last_step_time > 0.0:
            dt = now - self._last_step_time
            if dt > 0.0:
                self._fps_history.append(dt)
                self._real_fps = len(self._fps_history) / sum(self._fps_history)
        self._last_step_time = now

        t0 = time.perf_counter()
        self._controller.act(int(action))
        t1 = time.perf_counter()
        self._prof_act.append((t1 - t0) * 1000.0)

        frame = self._capture_frame()
        self._last_frame = frame
        t2 = time.perf_counter()
        self._prof_grab.append((t2 - t1) * 1000.0)

        self._stack.push(self._processor.process(frame))
        verdict = self._observer.evaluate(frame)
        t3 = time.perf_counter()
        self._prof_proc.append((t3 - t2) * 1000.0)

        self._episode_steps += 1
        self._episode_reward += verdict.reward
        truncated = self._episode_steps >= self.config.env.max_episode_steps

        if verdict.terminated or truncated:
            # Never leave a key held across an episode boundary; the game would
            # receive phantom input during the restart sequence.
            self._controller.release_all()

        t_obs0 = time.perf_counter()
        obs = self._observation()
        t_obs1 = time.perf_counter()
        self._prof_obs.append((t_obs1 - t_obs0) * 1000.0)

        ms_act = sum(self._prof_act) / max(1, len(self._prof_act))
        ms_grab = sum(self._prof_grab) / max(1, len(self._prof_grab))
        ms_proc = sum(self._prof_proc) / max(1, len(self._prof_proc))
        ms_obs = sum(self._prof_obs) / max(1, len(self._prof_obs))
        ms_render = sum(self._prof_render) / max(1, len(self._prof_render))

        # Entity Tracking and Causal Collision Analysis
        tracked_entities: list[dict[str, Any]] = []
        collision_events: list[dict[str, Any]] = []
        player_bbox_dict: dict[str, Any] | None = None

        if self.config.tracker.enabled:
            # Update player location
            pbox = self._probe.update_player_location(frame)
            if pbox is not None:
                player_bbox_dict = {"x": pbox.x, "y": pbox.y, "w": pbox.w, "h": pbox.h}

            # Update entity tracker
            entities = self._entity_tracker.update(
                frame,
                pbox,
                score_region=self.config.reward.score_region,
                game_over_region=self.config.reward.game_over_region,
            )
            collision_events = self._entity_tracker.check_collisions(
                pbox,
                points_gained=verdict.points,
                terminated=verdict.terminated,
            )
            for e in entities:
                tracked_entities.append({
                    "id": e.entity_id,
                    "kind": e.kind.value,
                    "bbox": [e.bbox.x, e.bbox.y, e.bbox.w, e.bbox.h],
                    "vx": e.velocity[0],
                    "vy": e.velocity[1],
                    "conf": e.confidence,
                })

        info: dict[str, Any] = {
            "score": verdict.score,
            "points": verdict.points,
            "episode_points": self._observer.episode_points,
            "episode_steps": self._episode_steps,
            "episode_reward": self._episode_reward,
            "game_over_confidence": verdict.game_over_confidence,
            "frame_diff": verdict.frame_diff,
            "is_idle": verdict.is_idle,
            "action": int(action),
            "player_bbox": player_bbox_dict,
            "entities": tracked_entities,
            "collisions": collision_events,
        }
        # Benchmarking telemetry for HUD, diagnostics and logging
        self._last_info = {
            **info,
            "real_fps": self._real_fps,
            "latency_ms": {
                "act": ms_act,
                "grab": ms_grab,
                "proc": ms_proc,
                "obs": ms_obs,
                "render": ms_render,
            },
        }

        if self.render_mode == "human":
            tr0 = time.perf_counter()
            self.render()
            tr1 = time.perf_counter()
            self._prof_render.append((tr1 - tr0) * 1000.0)

        return obs, float(verdict.reward), bool(verdict.terminated), truncated, info

    def _draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """Overlay telemetry, action palette, and CNN vision inset for human preview."""
        out = frame.copy()
        h, w = out.shape[:2]

        # Draw ROI boxes
        boxes = [
            (self.config.reward.score_region, (0, 255, 0)),
            (self.config.reward.game_over_region, (0, 0, 255)),
        ]
        for region, colour in boxes:
            x1 = max(0, min(region.left, w))
            y1 = max(0, min(region.top, h))
            x2 = max(x1, min(region.left + region.width, w))
            y2 = max(y1, min(region.top + region.height, h))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(out, (x1, y1), (x2, y2), colour, 1)

        # Draw detected player avatar BBox and dynamic entities
        if self.config.tracker.enabled:
            # Draw Player Avatar
            p_bbox = self._last_info.get("player_bbox")
            if p_bbox:
                px = p_bbox["x"]
                py = p_bbox["y"]
                pw = p_bbox["w"]
                ph = p_bbox["h"]
                cv2.rectangle(out, (px, py), (px + pw, py + ph), (0, 255, 255), 2)
                cv2.putText(
                    out, "PLAYER", (px, max(12, py - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA
                )

            # Draw tracked foreground entities and threat vectors
            entities = self._last_info.get("entities", [])
            for ent in entities:
                eb = ent.get("bbox", [0, 0, 0, 0])
                ex, ey, ew, eh = eb[0], eb[1], eb[2], eb[3]
                ekind = ent.get("kind", "unknown")
                vx = ent.get("vx", 0.0)
                vy = ent.get("vy", 0.0)

                if ekind == EntityKind.PROJECTILE.value:
                    e_color = (0, 69, 255)  # Orange-red
                    label = "PROJ"
                elif ekind == EntityKind.THREAT.value:
                    e_color = (0, 0, 255)  # Red
                    label = "THREAT"
                elif ekind == EntityKind.COLLECTIBLE.value:
                    e_color = (255, 215, 0)  # Gold/Yellow
                    label = "BONUS"
                else:
                    e_color = (200, 200, 200)
                    label = "OBJ"

                cv2.rectangle(out, (ex, ey), (ex + ew, ey + eh), e_color, 1)
                cv2.putText(
                    out, f"{label}#{ent.get('id')}", (ex, max(10, ey - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, e_color, 1, cv2.LINE_AA
                )

                # Draw velocity/threat motion vector
                cx = ex + ew // 2
                cy = ey + eh // 2
                tip_x = int(cx + vx * 3.0)
                tip_y = int(cy + vy * 3.0)
                cv2.arrowedLine(out, (cx, cy), (tip_x, tip_y), e_color, 1, tipLength=0.3)

        # Draw semi-transparent telemetry bar at the top
        bar_height = min(112, max(44, h // 3))
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

        action_idx = self._last_info.get("action")
        keys = self._action_keys
        if action_idx is not None and 0 <= action_idx < len(keys):
            action_name = format_action(keys[action_idx])
            action_text = f"ACTIVE [{action_idx}]: [{action_name}]"
        else:
            action_text = "ACTIVE: --"

        ep_rew = self._last_info.get("episode_reward", 0.0)
        ep_step = self._last_info.get("episode_steps", 0)
        points = self._last_info.get("episode_points", 0.0)
        frame_diff = self._last_info.get("frame_diff", 0.0)
        is_idle = self._last_info.get("is_idle", False)
        go_conf = self._last_info.get("game_over_confidence", 0.0)

        status_color = (0, 0, 255) if is_idle else (0, 255, 0)
        motion_status = "STAGNANT / IDLE" if is_idle else "ACTIVE MOTION"

        target_fps = self.config.env.target_fps
        fps_text = (
            f"REAL FPS: {self._real_fps:4.1f} (target {target_fps:.1f})"
            if target_fps > 0
            else f"REAL FPS: {self._real_fps:4.1f} (UNPACED)"
        )

        # Line 1: Episode stats & Real FPS
        line1 = (
            f"EP #{self._episode_index} | STEP: {ep_step:4d} | "
            f"REW: {ep_rew:+6.1f} | PTS: {points:.0f} | {fps_text}"
        )
        cv2.putText(
            out,
            line1,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Line 2: Active action & Motion Status
        cv2.putText(
            out,
            f"{action_text} | DIFF: {frame_diff * 100:4.1f}% [{motion_status}]",
            (10, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            status_color,
            1,
            cv2.LINE_AA,
        )

        # Line 3: Configured Action Palette (or preview window if many actions)
        chips = []
        # If there are many actions (e.g. combinatorial), show a window around the active action
        if len(keys) <= 7:
            palette_subset = list(enumerate(keys))
            prefix = "ACTIONS: "
        else:
            cur = action_idx if action_idx is not None else 0
            start_i = max(0, min(cur - 2, len(keys) - 5))
            end_i = min(len(keys), start_i + 5)
            palette_subset = list(enumerate(keys))[start_i:end_i]
            prefix = f"ACTIONS ({len(keys)} total): "

        for i, raw_k in palette_subset:
            name = format_action(raw_k)
            if i == action_idx:
                chips.append(f">> [{i}:{name}] <<")
            else:
                chips.append(f"[{i}:{name}]")
        chips_str = " ".join(chips)
        cv2.putText(
            out,
            f"{prefix}{chips_str}",
            (10, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Line 4: Latency breakdown profiler
        lat = self._last_info.get("latency_ms", {})
        l_act = lat.get("act", 0.0)
        l_grab = lat.get("grab", 0.0)
        l_proc = lat.get("proc", 0.0)
        l_obs = lat.get("obs", 0.0)
        l_rnd = lat.get("render", 0.0)
        lat_text = (
            f"LATENCY(ms): grab={l_grab:4.1f} | act={l_act:4.1f} | "
            f"proc={l_proc:4.1f} | audio/obs={l_obs:4.1f} | render={l_rnd:4.1f}"
        )
        cv2.putText(
            out,
            lat_text,
            (10, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 180, 50),
            1,
            cv2.LINE_AA,
        )

        # Line 5: Game Over confidence & Region key
        cv2.putText(
            out,
            f"GAME OVER CONF: {go_conf:4.2f} | BOXES: [GREEN=SCORE, RED=OVER]",
            (10, 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        # Picture-in-Picture: Visualizing the Neural Network's actual 84x84 observation
        try:
            obs = self._stack.observation  # shape: (frame_stack, 84, 84)
            if obs is not None and obs.ndim == 3 and obs.shape[0] > 0:
                inset_w = 168 if w >= 450 else (84 if w >= 220 else 0)
                inset_h = inset_w
                header_h = 18
                if inset_w > 0 and h >= inset_h + header_h + bar_height + 20:
                    latest_gray = obs[-1]
                    scaled_cnn = cv2.resize(
                        latest_gray, (inset_w, inset_h), interpolation=cv2.INTER_NEAREST
                    )
                    cnn_bgr = cv2.cvtColor(scaled_cnn, cv2.COLOR_GRAY2BGR)

                    x_start = w - inset_w - 10
                    y_start = h - inset_h - 10

                    # Inset background and border
                    cv2.rectangle(
                        out,
                        (x_start - 2, y_start - header_h - 2),
                        (x_start + inset_w + 2, y_start + inset_h + 2),
                        (0, 255, 255),
                        1,
                    )
                    cv2.rectangle(
                        out,
                        (x_start, y_start - header_h),
                        (x_start + inset_w, y_start),
                        (30, 30, 30),
                        -1,
                    )
                    cv2.putText(
                        out,
                        "CNN INPUT (84x84)",
                        (x_start + 6, y_start - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    out[y_start : y_start + inset_h, x_start : x_start + inset_w] = cnn_bgr

                    # If vertical room permits, show the 4-frame temporal filmstrip
                    film_h = 32
                    film_w = inset_w // 4
                    if h >= inset_h + header_h + film_h + bar_height + 30 and film_w > 0:
                        y_film = y_start - header_h - film_h - 6
                        cv2.rectangle(
                            out,
                            (x_start - 2, y_film - 2),
                            (x_start + inset_w + 2, y_film + film_h + 2),
                            (100, 100, 100),
                            1,
                        )
                        n_chips = min(4, obs.shape[0])
                        for i in range(n_chips):
                            chip = cv2.resize(
                                obs[i], (film_w, film_h), interpolation=cv2.INTER_NEAREST
                            )
                            chip_bgr = cv2.cvtColor(chip, cv2.COLOR_GRAY2BGR)
                            cx = x_start + i * film_w
                            out[y_film : y_film + film_h, cx : cx + film_w] = chip_bgr
                            cv2.rectangle(
                                out,
                                (cx, y_film),
                                (cx + film_w, y_film + film_h),
                                (60, 60, 60),
                                1,
                            )
                            cv2.putText(
                                out,
                                f"t{i - n_chips + 1}",
                                (cx + 2, y_film + film_h - 3),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.30,
                                (0, 255, 0),
                                1,
                            )
        except Exception:
            pass

        return out

    def render_hud(self) -> np.ndarray | None:
        """Return the current frame with the HUD telemetry overlay drawn on it."""
        if self._last_frame is None:
            return None
        return self._draw_hud(self._last_frame)

    def render(self) -> np.ndarray | None:
        if self._last_frame is None:
            return None
        hud_frame = self._draw_hud(self._last_frame)
        if self.render_mode == "rgb_array":
            return cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
        if self.render_mode == "human":  # pragma: no cover - needs a display
            cv2.imshow(self._window_name, hud_frame)
            cv2.waitKey(1)
            self._window_open = True
        return None

    def close(self) -> None:
        self._controller.release_all()
        self._audio.stop()
        self._frames.close()
        if self._window_open:  # pragma: no cover - needs a display
            cv2.destroyWindow(self._window_name)
            self._window_open = False

    # -- internals -----------------------------------------------------------
    def _observation(self) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {"frames": self._stack.observation}
        if self._audio_enabled:
            obs["audio"] = self._audio_features.extract(self._audio.read_window())
        return obs

    def _capture_frame(self) -> np.ndarray:
        """Grab the next frame, holding the configured decision rate."""
        if self._frame_period:
            remaining = self._next_frame_at - time.perf_counter()
            if remaining > 0:
                # Use busy-wait spinloop for remaining fractions of milliseconds
                # because time.sleep() on Windows quantizes to 15.6ms!
                target = self._next_frame_at
                if remaining > 0.002:
                    time.sleep(remaining - 0.002)
                while time.perf_counter() < target:
                    pass
                self._next_frame_at += self._frame_period
            else:
                # We fell behind (slow game, GC pause): resynchronise instead of
                # trying to catch up with a burst of zero-length steps.
                self._next_frame_at = time.perf_counter() + self._frame_period
        return self._frames.grab()

    def _wait_for_playable_frame(self) -> np.ndarray:
        """Block until the game-over screen clears, or the timeout expires."""
        deadline = time.perf_counter() + self.config.env.reset_timeout
        frame = self._frames.grab()
        while self._observer.is_game_over(frame):
            if time.perf_counter() >= deadline:
                # Re-tap restart once; some games need the input twice (e.g. a
                # confirmation prompt) and we would otherwise start an episode
                # on a dead screen.
                self._controller.restart()
                break
            time.sleep(0.05)
            frame = self._frames.grab()
        return frame

    def _run_controllability_probe(self, frame: np.ndarray) -> None:
        """Inject test actions to probe screen motion and locate the player avatar."""
        self._probe.record_probe_step("idle", frame)
        # Probe up to configured probe_actions with available action keys
        actions_to_probe = [a for a in range(len(self._action_keys)) if self._action_keys[a] is not None]
        if not actions_to_probe:
            return

        probe_steps = min(len(actions_to_probe), self.config.tracker.probe_actions)
        for i in range(probe_steps):
            act_idx = actions_to_probe[i % len(actions_to_probe)]
            act_name = format_action(self._action_keys[act_idx])
            self._controller.act(act_idx)
            time.sleep(0.05)
            step_frame = self._capture_frame()
            self._probe.record_probe_step(act_name, step_frame)

        self._controller.release_all()
        self._probe.analyze_probes()


def make_env(
    config: AppConfig | None = None,
    *,
    mock: bool = False,
    dry_run: bool = False,
    render_mode: str | None = None,
    seed: int | None = None,
) -> GameEnv:
    """Build a ``GameEnv``.

    ``mock=True`` swaps in the bundled simulated game (no OS integration at
    all). ``dry_run=True`` keeps real screen capture but discards key presses,
    which is the safe way to verify capture regions and reward detection
    against a live game.
    """
    if mock:
        from .mock_game import build_mock_backends, mock_config

        config = mock_config(config)
        _game, frames, audio, controller = build_mock_backends(config, seed=seed)
        return GameEnv(
            config,
            frame_source=frames,
            audio_source=audio,
            controller=controller,
            render_mode=render_mode,
        )

    config = config or AppConfig()
    return GameEnv(
        config,
        controller=build_controller(config.control, dry_run=dry_run),
        render_mode=render_mode,
    )
