from __future__ import annotations

from unittest.mock import MagicMock

from ai_player.config import ControlConfig
from ai_player.controls import (
    KeyboardController,
    NullController,
    format_action,
    get_effective_action_keys,
    parse_action_keys,
    validate_action_keys,
)


def test_parse_action_keys():
    assert parse_action_keys(None) == ()
    assert parse_action_keys("d") == ("d",)
    assert parse_action_keys("  space  ") == ("space",)
    assert parse_action_keys("d+space") == ("d", "space")
    assert parse_action_keys("d + space + j") == ("d", "space", "j")
    assert parse_action_keys(["d", "space"]) == ("d", "space")
    assert parse_action_keys(["d+space", "j"]) == ("d", "space", "j")
    assert parse_action_keys(["", None, "k"]) == ("k",)


def test_format_action():
    assert format_action(None) == "IDLE/NONE"
    assert format_action("d") == "d"
    assert format_action("d+space") == "d + space"
    assert format_action(["d", "space"]) == "d + space"
    assert format_action([]) == "IDLE/NONE"


def test_validate_action_keys():
    control = ControlConfig(action_keys=["d", "d+space", ["w", "j"], None])
    results = validate_action_keys(control)
    assert len(results) == 4
    assert results[0]["keys"] == ("d",)
    assert results[1]["keys"] == ("d", "space")
    assert results[1]["formatted"] == "d + space"
    assert results[2]["keys"] == ("w", "j")
    assert results[3]["keys"] == ()
    assert results[3]["formatted"] == "IDLE/NONE"


def test_keyboard_controller_tap_combo(monkeypatch):
    mock_pdi = MagicMock()
    mock_pdi.PAUSE = 0.0
    mock_pdi.FAILSAFE = False

    monkeypatch.setattr(KeyboardController, "_load_backend", lambda self, ctrl: mock_pdi)

    control = ControlConfig(action_keys=[None, "d", ["d", "space"]], tap_duration=0.001)
    controller = KeyboardController(control)

    # Action 0: None (no key events)
    controller.act(0)
    assert mock_pdi.keyDown.call_count == 0
    assert mock_pdi.keyUp.call_count == 0

    # Action 2: ["d", "space"] simultaneous tap
    controller.act(2)
    assert mock_pdi.keyDown.call_count == 2
    mock_pdi.keyDown.assert_any_call("d")
    mock_pdi.keyDown.assert_any_call("space")
    assert mock_pdi.keyUp.call_count == 2
    mock_pdi.keyUp.assert_any_call("space")
    mock_pdi.keyUp.assert_any_call("d")


def test_keyboard_controller_hold_combo(monkeypatch):
    mock_pdi = MagicMock()
    monkeypatch.setattr(KeyboardController, "_load_backend", lambda self, ctrl: mock_pdi)

    control = ControlConfig(
        action_keys=[None, "d", ["d", "space"], "s"],
        hold_keys=True,
    )
    controller = KeyboardController(control)

    # Press combo ["d", "space"]
    controller.act(2)
    assert controller._held_keys == {"d", "space"}
    mock_pdi.keyDown.assert_any_call("d")
    mock_pdi.keyDown.assert_any_call("space")

    # Switch to single key "s": "d" and "space" must be released, "s" pressed
    mock_pdi.reset_mock()
    controller.act(3)
    assert controller._held_keys == {"s"}
    mock_pdi.keyUp.assert_any_call("d")
    mock_pdi.keyUp.assert_any_call("space")
    mock_pdi.keyDown.assert_called_once_with("s")

    # Release all
    mock_pdi.reset_mock()
    controller.release_all()
    assert controller._held_keys == set()
    mock_pdi.keyUp.assert_any_call("s")


def test_null_controller():
    control = ControlConfig()
    controller = NullController(control)
    controller.act(0)
    controller.act(1)
    controller.restart()
    controller.release_all()
    assert controller.history == [0, 1]
    assert controller.restarts == 1


def test_auto_combos_generation():
    control = ControlConfig(action_keys=["d", "space", "j"], auto_combos=True, max_combo_size=2)
    combos = get_effective_action_keys(control)
    # Expected: None (IDLE), ['d'], ['space'], ['j'], ['d', 'space'], ['d', 'j'], ['space', 'j']
    assert None in combos
    assert ["d"] in combos
    assert ["space"] in combos
    assert ["j"] in combos
    assert ["d", "space"] in combos
    assert ["d", "j"] in combos
    assert ["space", "j"] in combos
    assert len(combos) == 1 + 3 + 3  # 7 actions

    val = validate_action_keys(control)
    assert len(val) == 7
    assert val[0]["formatted"] == "IDLE/NONE"

