"""Cognitive Game Supervisor & Asynchronous VLM Sentinel.

Runs a dedicated background daemon worker thread to continuously analyze gameplay
frames using Vision-Language Models (VLM) such as Qwen2.5-VL / MiniCPM via Ollama
or OpenAI-compatible endpoints (LM Studio, vLLM, etc.).

Provides:
1. High-level Game State Detection: 'gameplay', 'game_over', 'menu', 'cutscene', 'loading'.
2. Game Over sentinel to guide/override heuristic detectors.
3. Tactical Guidance: suggested actions (e.g. 'press_start', 'press_space', 'dodge_left').
4. Descriptive scene commentary for telemetry and HUD display.
5. Non-blocking thread-safe design ensuring no impact on 30-60 FPS gameplay loop.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any

from .config import HUDConfig
from .perception import BGRFrame
from .vlm_client import parse_vlm_json, query_vlm, rescale_boxes

__all__ = [
    "AsyncVLMSentinel",
    "CognitiveState",
    "CognitiveSupervisor",
    "parse_vlm_json",
]

logger = logging.getLogger(__name__)

# Upper bound for the exponential backoff applied after consecutive endpoint failures.
_MAX_BACKOFF_SECONDS = 30.0
# Longest single wait inside the worker so ``stop()`` is always honoured promptly.
_IDLE_WAIT_SECONDS = 0.2

_VALID_STATES = frozenset({"gameplay", "game_over", "menu", "cutscene", "loading"})

_SENTINEL_PROMPT = (
    "Analyze this video game screenshot. Return a JSON object with exactly these fields:\n"
    '1. "state": string, one of ["gameplay", "game_over", "menu", "cutscene", "loading"]\n'
    '2. "game_over": boolean (true if player died or game over screen is shown, false otherwise)\n'
    '3. "confidence": float between 0.0 and 1.0\n'
    '4. "suggested_action": string or null (e.g. "press_start", "fire", "restart")\n'
    '5. "description": short string (max 15 words) describing what is happening\n'
    '6. "hud_layout": optional dict of bounding boxes [x, y, w, h] for HUD elements '
    'e.g. {"score": [x,y,w,h], "hp": [x,y,w,h], "ammo": [x,y,w,h], "game_over": [x,y,w,h]}\n'
    '7. "vital_stats": optional dict of gameplay statistics e.g. '
    '{"hp_ratio": 0.0-1.0, "ammo_ratio": 0.0-1.0, "lives": int/float, "danger_level": 0.0-1.0}\n\n'
    'Respond ONLY with valid JSON. Example: {"state": "gameplay", "game_over": false, '
    '"confidence": 0.95, "suggested_action": "fire", "description": "Space battle", '
    '"hud_layout": {"score": [10, 10, 100, 30], "hp": [10, 50, 120, 20]}, '
    '"vital_stats": {"hp_ratio": 0.85, "ammo_ratio": 0.5, "lives": 3.0, "danger_level": 0.2}}'
)


@dataclass
class CognitiveState:
    """Snapshot of VLM cognitive assessment at a point in time."""

    state: str = "gameplay"  # "gameplay" | "game_over" | "menu" | "cutscene" | "loading"
    is_game_over: bool = False
    confidence: float = 0.0
    suggested_action: str | None = None
    description: str = ""
    hud_layout: dict[str, list[int]] = field(default_factory=dict)
    vital_stats: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    raw_response: dict[str, Any] = field(default_factory=dict)
    # False for the placeholder state held before the first successful VLM
    # query (and after ``CognitiveSupervisor.reset``); such states are never
    # "fresh" regardless of their timestamp.
    valid: bool = False

    @property
    def age(self) -> float:
        """Seconds elapsed since this verdict was produced."""
        return max(0.0, time.time() - self.timestamp)

    def is_fresh(self, max_age: float) -> bool:
        """True if this is a real VLM verdict no older than ``max_age`` seconds."""
        return self.valid and self.age <= max_age

    @property
    def signals_game_over(self) -> bool:
        """True if the model reported game over via either the flag or the state label."""
        return self.is_game_over or self.state in ("game_over", "defeat")


class CognitiveSupervisor:
    """Asynchronous background VLM sentinel and cognitive game supervisor.

    Periodically queries the configured VLM endpoint in a background daemon thread
    without blocking the primary agent decision loop.

    Threading model: everything read by the game loop (``_current_state``,
    ``_dynamic_hud_layout``, ``_latest_frame``, counters) is guarded by
    ``_lock``. Pacing bookkeeping (``_next_query_at``, ``_backoff``) is touched
    only by the worker thread and needs no lock.
    """

    def __init__(
        self,
        config: HUDConfig | None = None,
        *,
        interval: float | None = None,
    ) -> None:
        self.config = config or HUDConfig()
        self.interval = float(interval if interval is not None else self.config.vlm_sentinel_interval)

        self._lock = threading.Lock()
        self._latest_frame: BGRFrame | None = None
        self._new_frame_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        # Guarded by ``_lock``.
        self._current_state = CognitiveState()
        self._dynamic_hud_layout: dict[str, list[int]] = {}
        self._query_count = 0
        self._consecutive_failures = 0

        # Worker-thread only (monotonic clock).
        self._next_query_at = 0.0
        self._backoff = self.interval

    @property
    def is_running(self) -> bool:
        """True if the background worker thread is active."""
        return self._worker_thread is not None and self._worker_thread.is_alive()

    @property
    def current_state(self) -> CognitiveState:
        """Return a copy of the latest thread-safe cognitive state."""
        with self._lock:
            src = self._current_state
            return replace(
                src,
                hud_layout=dict(src.hud_layout),
                vital_stats=dict(src.vital_stats),
                raw_response=dict(src.raw_response),
            )

    @property
    def dynamic_hud_layout(self) -> dict[str, list[int]]:
        """Return the latest dynamic HUD layout bounding boxes (full-frame coordinates)."""
        with self._lock:
            return dict(self._dynamic_hud_layout)

    def update_dynamic_hud(self, regions: dict[str, list[int]]) -> None:
        """Store dynamic HUD layout regions."""
        with self._lock:
            self._dynamic_hud_layout.update(regions)

    @property
    def is_game_over(self) -> bool:
        """Convenience property for game over flag."""
        with self._lock:
            return self._current_state.is_game_over

    @property
    def suggested_action(self) -> str | None:
        """Convenience property for suggested action."""
        with self._lock:
            return self._current_state.suggested_action

    @property
    def description(self) -> str:
        """Convenience property for scene description."""
        with self._lock:
            return self._current_state.description

    @property
    def query_count(self) -> int:
        """Number of successful VLM queries since construction."""
        with self._lock:
            return self._query_count

    @property
    def failure_count(self) -> int:
        """Length of the current streak of consecutive failed queries (0 after a success)."""
        with self._lock:
            return self._consecutive_failures

    def start(self) -> None:
        """Start the background daemon sentinel thread."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="AsyncVLMSentinel",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the background sentinel thread cleanly."""
        self._stop_event.set()
        # Wake a worker blocked waiting for its first frame.
        self._new_frame_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)
            self._worker_thread = None

    def reset(self) -> None:
        """Forget the last verdict and frame at an episode boundary.

        Without this the terminal game-over screenshot would be re-queried (and
        re-confirmed) for the whole of the next ``reset()``, stalling it until
        ``env.reset_timeout`` and terminating the next episode on its first step.
        The worker thread keeps running; it simply waits for the next frame.
        """
        with self._lock:
            self._current_state = CognitiveState()
            self._latest_frame = None

    def update_frame(self, frame: BGRFrame | None) -> None:
        """Pass the latest gameplay frame to the sentinel (non-blocking)."""
        if frame is None or frame.size == 0:
            return
        with self._lock:
            self._latest_frame = frame.copy()
        self._new_frame_event.set()

    def _worker_loop(self) -> None:
        """Background daemon polling loop.

        Starts at most one query per ``interval`` seconds (stretched by the
        failure backoff), and only when a frame is available; with no frame it
        blocks on ``_new_frame_event`` instead of spinning.
        """
        while not self._stop_event.is_set():
            if not self.config.use_vlm:
                self._stop_event.wait(_IDLE_WAIT_SECONDS)
                continue

            remaining = self._next_query_at - time.monotonic()
            if remaining > 0:
                self._stop_event.wait(min(remaining, _IDLE_WAIT_SECONDS))
                continue

            # Clear before reading so a frame published in between is not missed:
            # update_frame stores the frame first and sets the event afterwards.
            self._new_frame_event.clear()
            with self._lock:
                frame = self._latest_frame
            if frame is None:
                self._new_frame_event.wait(_IDLE_WAIT_SECONDS)
                continue

            started = time.monotonic()
            try:
                state = self._query_vlm(frame)
            except Exception as exc:
                self._on_query_failure(exc, started)
            else:
                self._on_query_success(state, started)

    def _on_query_success(self, state: CognitiveState, started: float) -> None:
        with self._lock:
            self._current_state = state
            if state.hud_layout:
                self._dynamic_hud_layout.update(state.hud_layout)
            self._query_count += 1
            recovered_after = self._consecutive_failures
            self._consecutive_failures = 0
        if recovered_after:
            logger.info("AsyncVLMSentinel recovered after %d failed queries", recovered_after)
        self._backoff = self.interval
        self._next_query_at = started + self.interval

    def _on_query_failure(self, exc: BaseException, started: float) -> None:
        with self._lock:
            self._consecutive_failures += 1
            streak = self._consecutive_failures
        if streak == 1:
            logger.warning("AsyncVLMSentinel query failed (%s); backing off", exc)
        else:
            logger.debug("AsyncVLMSentinel query failed again (%d in a row): %s", streak, exc)
        self._next_query_at = started + self._backoff
        self._backoff = min(self._backoff * 2.0, _MAX_BACKOFF_SECONDS)

    def _query_vlm(self, frame: BGRFrame) -> CognitiveState:
        """Synchronously query the VLM endpoint with the given frame.

        Network, encoding and decoding errors propagate to the caller.
        """
        parsed, scale = query_vlm(
            self.config,
            frame,
            _SENTINEL_PROMPT,
            max_dim=self.config.vlm_max_image_dim,
        )
        return self._state_from_response(parsed, scale)

    @staticmethod
    def _state_from_response(parsed: dict[str, Any], scale: float) -> CognitiveState:
        """Normalise a parsed model reply into a ``CognitiveState`` in full-frame coordinates."""
        state_str = str(parsed.get("state", "gameplay")).lower().strip()
        if state_str not in _VALID_STATES:
            state_str = "gameplay"

        raw_go = parsed.get("game_over")
        if isinstance(raw_go, bool):
            is_go = raw_go
        elif isinstance(raw_go, str):
            is_go = raw_go.lower() in ("true", "1", "yes")
        else:
            is_go = state_str == "game_over"

        # An absent/garbled confidence must never clear the game-over threshold.
        try:
            conf = float(parsed.get("confidence", 0.0))
            conf = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            conf = 0.0

        action = parsed.get("suggested_action")
        if action is not None:
            action = str(action).strip()
            if action.lower() in ("null", "none", ""):
                action = None

        desc = str(parsed.get("description", "")).strip()

        hud_layout: dict[str, list[int]] = {}
        raw_hud = parsed.get("hud_layout")
        if isinstance(raw_hud, dict):
            for k, box in raw_hud.items():
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    with contextlib.suppress(ValueError, TypeError):
                        hud_layout[str(k).lower()] = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        hud_layout = rescale_boxes(hud_layout, scale)

        vital_stats: dict[str, float] = {}
        raw_vitals = parsed.get("vital_stats")
        if isinstance(raw_vitals, dict):
            for k, val in raw_vitals.items():
                with contextlib.suppress(ValueError, TypeError):
                    vital_stats[str(k).lower()] = float(val)

        return CognitiveState(
            state=state_str,
            is_game_over=is_go,
            confidence=conf,
            suggested_action=action,
            description=desc,
            hud_layout=hud_layout,
            vital_stats=vital_stats,
            timestamp=time.time(),
            raw_response=parsed,
            valid=True,
        )


# Alias for backwards compatibility / alternate naming
AsyncVLMSentinel = CognitiveSupervisor
