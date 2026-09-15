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

import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass, field
from typing import Any

import cv2

from .config import HUDConfig
from .perception import BGRFrame

logger = logging.getLogger(__name__)


@dataclass
class CognitiveState:
    """Snapshot of VLM cognitive assessment at a point in time."""

    state: str = "gameplay"  # "gameplay" | "game_over" | "menu" | "cutscene" | "loading"
    is_game_over: bool = False
    confidence: float = 0.0
    suggested_action: str | None = None
    description: str = ""
    timestamp: float = field(default_factory=time.time)
    raw_response: dict[str, Any] = field(default_factory=dict)


def parse_vlm_json(content: str | dict[str, Any]) -> dict[str, Any]:
    """Parse JSON from raw VLM response string, handling markdown fences and surrounding text."""
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return {}

    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start_brace = cleaned.find("{")
    end_brace = cleaned.rfind("}")
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        cleaned = cleaned[start_brace : end_brace + 1]

    try:
        return json.loads(cleaned)
    except Exception:
        return {}


class CognitiveSupervisor:
    """Asynchronous background VLM sentinel and cognitive game supervisor.

    Periodically queries the configured VLM endpoint in a background daemon thread
    without blocking the primary agent decision loop.
    """

    def __init__(
        self,
        config: HUDConfig | None = None,
        *,
        interval: float | None = None,
        auto_menu_nav: bool | None = None,
        guidance_enabled: bool | None = None,
    ) -> None:
        self.config = config or HUDConfig()
        self.interval = (
            interval
            if interval is not None
            else getattr(self.config, "vlm_sentinel_interval", 2.0)
        )
        self.auto_menu_nav = (
            auto_menu_nav
            if auto_menu_nav is not None
            else getattr(self.config, "auto_menu_nav", True)
        )
        self.guidance_enabled = (
            guidance_enabled
            if guidance_enabled is not None
            else getattr(self.config, "guidance_enabled", True)
        )

        self._lock = threading.Lock()
        self._latest_frame: BGRFrame | None = None
        self._new_frame_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        # Thread-safe telemetry state
        self._current_state = CognitiveState()
        self._query_count = 0
        self._last_query_time = 0.0

    @property
    def is_running(self) -> bool:
        """True if the background worker thread is active."""
        return self._worker_thread is not None and self._worker_thread.is_alive()

    @property
    def current_state(self) -> CognitiveState:
        """Return a copy of the latest thread-safe cognitive state."""
        with self._lock:
            return CognitiveState(
                state=self._current_state.state,
                is_game_over=self._current_state.is_game_over,
                confidence=self._current_state.confidence,
                suggested_action=self._current_state.suggested_action,
                description=self._current_state.description,
                timestamp=self._current_state.timestamp,
                raw_response=dict(self._current_state.raw_response),
            )

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
        self._new_frame_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)
            self._worker_thread = None

    def update_frame(self, frame: BGRFrame | None) -> None:
        """Pass the latest gameplay frame to the sentinel (non-blocking)."""
        if frame is None or frame.size == 0:
            return
        with self._lock:
            self._latest_frame = frame.copy()
        self._new_frame_event.set()

    def _worker_loop(self) -> None:
        """Background daemon polling loop."""
        while not self._stop_event.is_set():
            # Wait for next interval or new frame
            now = time.time()
            elapsed = now - self._last_query_time
            if elapsed < self.interval:
                sleep_time = min(self.interval - elapsed, 0.2)
                if self._stop_event.wait(sleep_time):
                    break

            # Grab current latest frame
            frame_to_process: BGRFrame | None = None
            with self._lock:
                if self._latest_frame is not None:
                    frame_to_process = self._latest_frame.copy()

            if frame_to_process is None or not self.config.use_vlm:
                # If VLM is disabled or no frame, idle brief period
                if self._stop_event.wait(0.2):
                    break
                continue

            try:
                state = self._query_vlm(frame_to_process)
                with self._lock:
                    self._current_state = state
                    self._last_query_time = time.time()
                    self._query_count += 1
            except Exception as exc:
                warnings.warn(
                    f"AsyncVLMSentinel query failed ({exc})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._last_query_time = time.time()

    def _query_vlm(self, frame: BGRFrame) -> CognitiveState:
        """Synchronously query the VLM endpoint with the given frame."""
        h, w = frame.shape[:2]
        # Downscale frame if overly large to reduce HTTP latency and tokens
        target_frame = frame
        if max(h, w) > 640:
            scale = 640.0 / max(h, w)
            target_frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        success, buffer = cv2.imencode(".jpg", target_frame)
        if not success:
            return self.current_state
        b64_image = base64.b64encode(buffer).decode("utf-8")

        prompt = (
            "Analyze this video game screenshot. Return a JSON object with exactly these fields:\n"
            '1. "state": string, one of ["gameplay", "game_over", "menu", "cutscene", "loading"]\n'
            '2. "game_over": boolean (true if player died or game over screen is shown, false otherwise)\n'
            '3. "confidence": float between 0.0 and 1.0\n'
            '4. "suggested_action": string or null (e.g. "press_start", "fire", "restart")\n'
            '5. "description": short string (max 15 words) describing what is happening\n\n'
            'Respond ONLY with valid JSON. Example: {"state": "gameplay", "game_over": false, '
            '"confidence": 0.95, "suggested_action": "fire", "description": "Space battle"}'
        )

        is_openai = "chat/completions" in self.config.vlm_endpoint or "/v1/" in self.config.vlm_endpoint
        if is_openai:
            payload_dict = {
                "model": self.config.vlm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                            },
                        ],
                    }
                ],
                "stream": False,
                "temperature": 0.1,
            }
        else:
            payload_dict = {
                "model": self.config.vlm_model,
                "prompt": prompt,
                "images": [b64_image],
                "stream": False,
                "format": "json",
            }

        payload = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            self.config.vlm_endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.config.vlm_timeout) as response:
            res_data = json.loads(response.read().decode("utf-8"))

        content = ""
        if "choices" in res_data and len(res_data["choices"]) > 0:
            choice = res_data["choices"][0]
            content = choice.get("message", {}).get("content", "")
        elif "response" in res_data:
            content = res_data["response"]
        elif "content" in res_data:
            content = res_data["content"]

        parsed = parse_vlm_json(content)
        state_str = str(parsed.get("state", "gameplay")).lower().strip()
        valid_states = {"gameplay", "game_over", "menu", "cutscene", "loading"}
        if state_str not in valid_states:
            state_str = "gameplay"

        raw_go = parsed.get("game_over")
        if isinstance(raw_go, bool):
            is_go = raw_go
        elif isinstance(raw_go, str):
            is_go = raw_go.lower() in ("true", "1", "yes")
        else:
            is_go = (state_str == "game_over")

        try:
            conf = float(parsed.get("confidence", 0.8))
            conf = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            conf = 0.8

        action = parsed.get("suggested_action")
        if action is not None:
            action = str(action).strip()
            if action.lower() in ("null", "none", ""):
                action = None

        desc = str(parsed.get("description", "")).strip()

        return CognitiveState(
            state=state_str,
            is_game_over=is_go,
            confidence=conf,
            suggested_action=action,
            description=desc,
            timestamp=time.time(),
            raw_response=parsed,
        )


# Alias for backwards compatibility / alternate naming
AsyncVLMSentinel = CognitiveSupervisor
