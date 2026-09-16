from __future__ import annotations

import json
import time

import numpy as np

from ai_player.cognitive import (
    AsyncVLMSentinel,
    CognitiveState,
    CognitiveSupervisor,
    parse_vlm_json,
)
from ai_player.config import HUDConfig


def test_parse_vlm_json_plain():
    raw = (
        '{"state": "gameplay", "game_over": false, "confidence": 0.9, '
        '"suggested_action": "fire", "description": "fighting"}'
    )
    parsed = parse_vlm_json(raw)
    assert parsed["state"] == "gameplay"
    assert parsed["game_over"] is False
    assert parsed["confidence"] == 0.9


def test_parse_vlm_json_with_markdown_fences():
    raw = """```json
    {
        "state": "game_over",
        "game_over": true,
        "confidence": 0.98,
        "suggested_action": "press_space",
        "description": "Game Over Continue 9s countdown"
    }
    ```"""
    parsed = parse_vlm_json(raw)
    assert parsed["state"] == "game_over"
    assert parsed["game_over"] is True
    assert parsed["confidence"] == 0.98
    assert parsed["suggested_action"] == "press_space"
    assert "Continue" in parsed["description"]


def test_parse_vlm_json_surrounding_garbage():
    raw = (
        'Here is the analysis: {"state": "menu", "game_over": false, '
        '"suggested_action": "press_start"} Hope this helps!'
    )
    parsed = parse_vlm_json(raw)
    assert parsed["state"] == "menu"
    assert parsed["suggested_action"] == "press_start"


def test_parse_vlm_json_invalid():
    assert parse_vlm_json("No json here") == {}
    assert parse_vlm_json(123) == {}
    assert parse_vlm_json({"already": "dict"}) == {"already": "dict"}


def test_cognitive_supervisor_init_and_properties():
    cfg = HUDConfig(vlm_sentinel_interval=1.5)
    supervisor = CognitiveSupervisor(cfg)

    assert supervisor.interval == 1.5
    assert CognitiveSupervisor(cfg, interval=0.25).interval == 0.25
    assert supervisor.is_running is False
    assert supervisor.query_count == 0
    assert supervisor.failure_count == 0

    cstate = supervisor.current_state
    assert isinstance(cstate, CognitiveState)
    assert cstate.state == "gameplay"
    assert cstate.is_game_over is False
    assert cstate.valid is False
    assert cstate.is_fresh(1e9) is False
    assert supervisor.is_game_over is False
    assert supervisor.suggested_action is None
    assert supervisor.description == ""


def test_cognitive_supervisor_start_stop():
    cfg = HUDConfig(use_vlm=False)
    supervisor = CognitiveSupervisor(cfg, interval=0.1)

    supervisor.start()
    assert supervisor.is_running is True

    # Calling start again is idempotent
    supervisor.start()
    assert supervisor.is_running is True

    supervisor.stop(timeout=1.0)
    assert supervisor.is_running is False

    # Calling stop again is idempotent
    supervisor.stop(timeout=1.0)
    assert supervisor.is_running is False


def test_cognitive_supervisor_query_mock_openai(monkeypatch):
    cfg = HUDConfig(
        use_vlm=True,
        vlm_endpoint="http://localhost:1234/v1/chat/completions",
        vlm_model="qwen-vl",
        vlm_sentinel_interval=0.05,
    )
    supervisor = CognitiveSupervisor(cfg)

    captured_requests = []

    class MockResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            content = json.dumps({
                "state": "menu",
                "game_over": False,
                "confidence": 0.92,
                "suggested_action": "press_start",
                "description": "Main title screen with Start prompt",
            })
            res = {"choices": [{"message": {"content": content}}]}
            return json.dumps(res).encode("utf-8")

    def mock_urlopen(req, timeout=None):
        captured_requests.append(req)
        return MockResponse()

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    supervisor.update_frame(frame)
    supervisor.start()

    # Wait for sentinel query to happen
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if supervisor.current_state.state == "menu":
            break
        time.sleep(0.02)

    supervisor.stop(timeout=1.0)

    assert supervisor.current_state.state == "menu"
    assert supervisor.current_state.suggested_action == "press_start"
    assert supervisor.current_state.confidence == 0.92
    assert "Main title screen" in supervisor.current_state.description
    assert len(captured_requests) > 0


def test_cognitive_supervisor_query_mock_ollama(monkeypatch):
    cfg = HUDConfig(
        use_vlm=True,
        vlm_endpoint="http://localhost:11434/api/generate",
        vlm_model="qwen2.5-vl",
        vlm_sentinel_interval=0.05,
    )
    supervisor = CognitiveSupervisor(cfg)

    class MockResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            content = json.dumps({
                "state": "game_over",
                "game_over": True,
                "confidence": 0.99,
                "suggested_action": "restart",
                "description": "Game Over Continue 9",
            })
            return json.dumps({"response": content}).encode("utf-8")

    def mock_urlopen(req, timeout=None):
        return MockResponse()

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    supervisor.update_frame(frame)
    supervisor.start()

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if supervisor.is_game_over:
            break
        time.sleep(0.02)

    supervisor.stop(timeout=1.0)

    assert supervisor.is_game_over is True
    assert supervisor.current_state.state == "game_over"
    assert supervisor.suggested_action == "restart"


def test_async_vlm_sentinel_alias():
    assert AsyncVLMSentinel is CognitiveSupervisor


def test_cognitive_supervisor_dynamic_hud_and_vital_stats(monkeypatch):
    cfg = HUDConfig(
        use_vlm=True,
        vlm_endpoint="http://localhost:11434/api/generate",
        vlm_model="qwen2.5-vl",
        vlm_sentinel_interval=0.05,
    )
    supervisor = CognitiveSupervisor(cfg)

    class MockResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            content = json.dumps({
                "state": "gameplay",
                "game_over": False,
                "confidence": 0.95,
                "suggested_action": "fire",
                "description": "Intense side-scrolling shooter level",
                "hud_layout": {
                    "score": [20, 15, 120, 30],
                    "hp": [20, 50, 100, 25],
                    "ammo": [20, 80, 80, 20],
                    "game_over": [200, 150, 400, 100],
                },
                "vital_stats": {
                    "hp_ratio": 0.85,
                    "ammo_ratio": 0.60,
                    "lives": 3.0,
                    "danger_level": 0.25,
                },
            })
            return json.dumps({"response": content}).encode("utf-8")

    def mock_urlopen(req, timeout=None):
        return MockResponse()

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    supervisor.update_frame(frame)
    supervisor.start()

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if supervisor.current_state.hud_layout:
            break
        time.sleep(0.02)

    supervisor.stop(timeout=1.0)

    st = supervisor.current_state
    assert st.hud_layout["score"] == [20, 15, 120, 30]
    assert st.hud_layout["hp"] == [20, 50, 100, 25]
    assert st.vital_stats["hp_ratio"] == 0.85
    assert st.vital_stats["ammo_ratio"] == 0.60
    assert st.vital_stats["lives"] == 3.0
    assert st.vital_stats["danger_level"] == 0.25

    # Test update_dynamic_hud method
    supervisor.update_dynamic_hud({"score": [30, 25, 110, 35]})
    assert supervisor.dynamic_hud_layout["score"] == [30, 25, 110, 35]


class _ScriptedEndpoint:
    """Mock ``urllib.request.urlopen`` returning Ollama-style bodies and counting calls."""

    def __init__(self, body: dict | None = None, *, fail: bool = False):
        self.body = body or {"state": "gameplay", "game_over": False, "confidence": 0.9}
        self.fail = fail
        self.calls = 0

    def __call__(self, req, timeout=None):
        self.calls += 1
        if self.fail:
            raise OSError("endpoint down")
        payload = json.dumps({"response": json.dumps(self.body)}).encode("utf-8")

        class _Resp:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def read(self_inner):
                return payload

        return _Resp()


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_sentinel_honours_polling_interval(monkeypatch):
    endpoint = _ScriptedEndpoint()
    monkeypatch.setattr("urllib.request.urlopen", endpoint)
    supervisor = CognitiveSupervisor(HUDConfig(use_vlm=True), interval=0.5)
    supervisor.update_frame(np.zeros((64, 64, 3), dtype=np.uint8))
    supervisor.start()
    try:
        assert _wait_until(lambda: supervisor.query_count >= 1)
        # Keep publishing frames: new frames must not trigger extra queries inside the interval.
        deadline = time.time() + 1.2
        while time.time() < deadline:
            supervisor.update_frame(np.zeros((64, 64, 3), dtype=np.uint8))
            time.sleep(0.02)
    finally:
        supervisor.stop(timeout=1.0)
    assert 1 <= endpoint.calls <= 3, endpoint.calls


def test_sentinel_waits_for_a_frame_instead_of_querying():
    supervisor = CognitiveSupervisor(HUDConfig(use_vlm=True), interval=0.01)
    endpoint = _ScriptedEndpoint()
    import urllib.request

    original = urllib.request.urlopen
    urllib.request.urlopen = endpoint
    try:
        supervisor.start()
        time.sleep(0.2)
        assert endpoint.calls == 0
        supervisor.update_frame(np.zeros((32, 32, 3), dtype=np.uint8))
        assert _wait_until(lambda: endpoint.calls >= 1)
    finally:
        supervisor.stop(timeout=1.0)
        urllib.request.urlopen = original


def test_sentinel_rescales_hud_layout_to_full_frame(monkeypatch):
    endpoint = _ScriptedEndpoint(
        {
            "state": "gameplay",
            "game_over": False,
            "confidence": 0.9,
            "hud_layout": {"score": [10, 10, 100, 30], "game_over": [50, 80, 200, 60]},
        }
    )
    monkeypatch.setattr("urllib.request.urlopen", endpoint)
    supervisor = CognitiveSupervisor(HUDConfig(use_vlm=True, vlm_max_image_dim=640), interval=0.01)
    supervisor.update_frame(np.zeros((720, 1280, 3), dtype=np.uint8))
    supervisor.start()
    try:
        assert _wait_until(lambda: bool(supervisor.current_state.hud_layout))
    finally:
        supervisor.stop(timeout=1.0)
    assert supervisor.current_state.hud_layout == {
        "score": [20, 20, 200, 60],
        "game_over": [100, 160, 400, 120],
    }
    assert supervisor.dynamic_hud_layout["score"] == [20, 20, 200, 60]


def test_sentinel_state_validity_confidence_default_and_reset(monkeypatch):
    endpoint = _ScriptedEndpoint({"state": "defeat", "game_over": "yes"})  # no confidence field
    monkeypatch.setattr("urllib.request.urlopen", endpoint)
    supervisor = CognitiveSupervisor(HUDConfig(use_vlm=True), interval=0.01)
    supervisor.update_frame(np.zeros((32, 32, 3), dtype=np.uint8))
    supervisor.start()
    try:
        assert _wait_until(lambda: supervisor.current_state.valid)
    finally:
        supervisor.stop(timeout=1.0)

    st = supervisor.current_state
    assert st.valid is True
    assert st.is_fresh(6.0) is True
    assert st.confidence == 0.0
    # "defeat" is not a canonical state label, but the flag still signals game over.
    assert st.state == "gameplay"
    assert st.is_game_over is True
    assert st.signals_game_over is True
    assert CognitiveState(state="defeat").signals_game_over is True
    assert CognitiveState(state="gameplay").signals_game_over is False

    supervisor.reset()
    after = supervisor.current_state
    assert after.valid is False
    assert after.is_fresh(6.0) is False
    assert after.is_game_over is False
    assert supervisor.is_game_over is False


def test_sentinel_survives_endpoint_failures_and_recovers(monkeypatch):
    endpoint = _ScriptedEndpoint(fail=True)
    monkeypatch.setattr("urllib.request.urlopen", endpoint)
    supervisor = CognitiveSupervisor(HUDConfig(use_vlm=True), interval=0.01)
    supervisor.update_frame(np.zeros((32, 32, 3), dtype=np.uint8))
    supervisor.start()
    try:
        assert _wait_until(lambda: supervisor.failure_count >= 1)
        assert supervisor.is_running is True
        assert supervisor.current_state.valid is False

        endpoint.fail = False
        assert _wait_until(lambda: supervisor.query_count >= 1, timeout=3.0)
    finally:
        supervisor.stop(timeout=1.0)
    assert supervisor.failure_count == 0
    assert supervisor.current_state.valid is True
