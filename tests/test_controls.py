from __future__ import annotations

from unittest.mock import MagicMock

from ai_player.config import ControlConfig
from ai_player.controls import (
    KeyboardController,
    MouseController,
    NullController,
    UnifiedInputController,
    build_controller,
    format_action,
    get_effective_action_keys,
    is_mouse_action,
    parse_action_keys,
    validate_action_keys,
)
from ai_player.mock_game import MockArcadeGame, MockController


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


def test_mouse_action_parsing_and_formatting():
    # Parsing single mouse actions
    assert parse_action_keys("mouse_left") == ("mouse_left",)
    assert parse_action_keys("mouse_right") == ("mouse_right",)
    assert parse_action_keys("mouse_middle") == ("mouse_middle",)
    assert parse_action_keys("aim_left") == ("aim_left",)
    assert parse_action_keys("aim_right") == ("aim_right",)
    assert parse_action_keys("aim_up") == ("aim_up",)
    assert parse_action_keys("aim_down") == ("aim_down",)

    # Parsing compound keyboard + mouse actions
    assert parse_action_keys("w+mouse_left") == ("w", "mouse_left")
    assert parse_action_keys(["w", "mouse_left"]) == ("w", "mouse_left")
    assert parse_action_keys("shift+w+mouse_left") == ("shift", "w", "mouse_left")
    assert parse_action_keys(["a", "aim_left"]) == ("a", "aim_left")

    # is_mouse_action checks
    assert is_mouse_action("mouse_left") is True
    assert is_mouse_action("mouse_right") is True
    assert is_mouse_action("mouse_middle") is True
    assert is_mouse_action("click") is True
    assert is_mouse_action("right_click") is True
    assert is_mouse_action("fire") is True
    assert is_mouse_action("aim_up") is True
    assert is_mouse_action("aim_down") is True
    assert is_mouse_action("aim_left") is True
    assert is_mouse_action("aim_right") is True
    assert is_mouse_action("w") is False
    assert is_mouse_action("space") is False

    # Formatting
    assert format_action("mouse_left") == "MOUSE_L"
    assert format_action("click") == "MOUSE_L"
    assert format_action("fire") == "MOUSE_L"
    assert format_action("mouse_right") == "MOUSE_R"
    assert format_action("mouse_middle") == "MOUSE_M"
    assert format_action("aim_left") == "AIM_LEFT"
    assert format_action("aim_right") == "AIM_RIGHT"
    assert format_action("aim_up") == "AIM_UP"
    assert format_action("aim_down") == "AIM_DOWN"
    assert format_action("w+mouse_left") == "w + MOUSE_L"
    assert format_action(["w", "mouse_left"]) == "w + MOUSE_L"


def test_mouse_controller_fallback_recording():
    control = ControlConfig(mouse_sensitivity=2.0, aim_step=20)
    mouse = MouseController(control)

    # Relative movement with sensitivity scaling (2.0)
    mouse.move_rel(10, -5)
    assert len(mouse.history) == 1
    assert mouse.history[0] == {"type": "move_rel", "dx": 20, "dy": -10}

    # Absolute movement
    mouse.move_to(100, 200)
    assert mouse.history[1] == {"type": "move_to", "x": 100, "y": 200}

    # Clicks
    mouse.click("left")
    assert mouse.history[2] == {"type": "click", "button": "left"}
    mouse.click("right")
    assert mouse.history[3] == {"type": "click", "button": "right"}

    # Press, release, release_all
    mouse.press("left")
    assert "left" in mouse._held_buttons
    assert mouse.history[4] == {"type": "press", "button": "left"}
    mouse.release("left")
    assert "left" not in mouse._held_buttons
    assert mouse.history[5] == {"type": "release", "button": "left"}

    mouse.press("right")
    mouse.press("middle")
    assert mouse._held_buttons == {"right", "middle"}
    mouse.release_all()
    assert mouse._held_buttons == set()
    assert mouse.history[-1] == {"type": "release_all"}


def test_mouse_controller_with_mock_backend():
    control = ControlConfig(mouse_sensitivity=1.0)
    mouse = MouseController(control)
    mock_pdi = MagicMock()
    mouse._pdi = mock_pdi

    mouse.move_rel(15, -10)
    mock_pdi.moveRel.assert_called_once_with(15, -10, relative=True)

    mouse.move_to(300, 400)
    mock_pdi.moveTo.assert_called_once_with(300, 400)

    mouse.click("left")
    mock_pdi.click.assert_called_once_with(button="left")

    mouse.press("right")
    mock_pdi.mouseDown.assert_called_once_with(button="right")

    mouse.release("right")
    mock_pdi.mouseUp.assert_called_once_with(button="right")

    mouse.press("left")
    mouse.press("right")
    mock_pdi.reset_mock()
    mouse.release_all()
    assert mock_pdi.mouseUp.call_count == 2
    mock_pdi.mouseUp.assert_any_call(button="left")
    mock_pdi.mouseUp.assert_any_call(button="right")


def test_unified_input_controller_tap_mode(monkeypatch):
    mock_pdi = MagicMock()
    monkeypatch.setattr(KeyboardController, "_load_backend", lambda self, ctrl: mock_pdi)

    control = ControlConfig(
        action_keys=[
            None,
            "w",
            "mouse_left",
            "aim_left",
            "w+mouse_left",
            ["s", "aim_right", "mouse_right"],
        ],
        tap_duration=0.001,
        aim_step=15,
        mouse_sensitivity=1.0,
    )
    unified = UnifiedInputController(control)
    # Action 0: None
    unified.act(0)
    assert mock_pdi.keyDown.call_count == 0
    assert len(unified.mouse.history) == 0

    # Action 1: "w" (keyboard only)
    mock_pdi.reset_mock()
    unified.act(1)
    mock_pdi.keyDown.assert_called_once_with("w")
    mock_pdi.keyUp.assert_called_once_with("w")
    assert len(unified.mouse.history) == 0

    # Action 2: "mouse_left" (mouse click only)
    mock_pdi.reset_mock()
    unified.mouse.history.clear()
    unified.act(2)
    assert mock_pdi.keyDown.call_count == 0
    # In tap mode, press then release is executed
    assert len(unified.mouse.history) == 2
    assert unified.mouse.history[0] == {"type": "press", "button": "left"}
    assert unified.mouse.history[1] == {"type": "release", "button": "left"}

    # Action 3: "aim_left" (relative mouse move: dx = -15, dy = 0)
    unified.mouse.history.clear()
    unified.act(3)
    assert len(unified.mouse.history) == 1
    assert unified.mouse.history[0] == {"type": "move_rel", "dx": -15, "dy": 0}

    # Action 4: "w+mouse_left" (hybrid tap)
    mock_pdi.reset_mock()
    unified.mouse.history.clear()
    unified.act(4)
    mock_pdi.keyDown.assert_called_once_with("w")
    mock_pdi.keyUp.assert_called_once_with("w")
    assert unified.mouse.history[0] == {"type": "press", "button": "left"}
    assert unified.mouse.history[1] == {"type": "release", "button": "left"}

    # Action 5: ["s", "aim_right", "mouse_right"] (hybrid aim + click + key)
    mock_pdi.reset_mock()
    unified.mouse.history.clear()
    unified.act(5)
    mock_pdi.keyDown.assert_called_once_with("s")
    mock_pdi.keyUp.assert_called_once_with("s")
    # Aim is executed first, then press/release mouse_right
    assert unified.mouse.history[0] == {"type": "move_rel", "dx": 15, "dy": 0}
    assert unified.mouse.history[1] == {"type": "press", "button": "right"}
    assert unified.mouse.history[2] == {"type": "release", "button": "right"}


def test_unified_input_controller_hold_mode(monkeypatch):
    mock_pdi = MagicMock()
    monkeypatch.setattr(KeyboardController, "_load_backend", lambda self, ctrl: mock_pdi)

    control = ControlConfig(
        action_keys=[
            None,
            "w",
            "mouse_left",
            "w+mouse_left",
            "mouse_right",
        ],
        hold_keys=True,
    )
    unified = UnifiedInputController(control)

    # Hold "w+mouse_left" (action 3)
    unified.act(3)
    assert unified.keyboard._held_keys == {"w"}
    assert unified._held_mouse_buttons == {"left"}
    mock_pdi.keyDown.assert_called_once_with("w")
    assert unified.mouse.history[-1] == {"type": "press", "button": "left"}

    # Switch to "mouse_right" (action 4): w and left released, right pressed
    mock_pdi.reset_mock()
    unified.act(4)
    assert unified.keyboard._held_keys == set()
    assert unified._held_mouse_buttons == {"right"}
    mock_pdi.keyUp.assert_called_once_with("w")
    assert unified.mouse.history[-2] == {"type": "release", "button": "left"}
    assert unified.mouse.history[-1] == {"type": "press", "button": "right"}

    # Release all
    mock_pdi.reset_mock()
    unified.release_all()
    assert unified.keyboard._held_keys == set()
    assert unified._held_mouse_buttons == set()
    assert unified.mouse.history[-1] == {"type": "release_all"}


def test_null_controller_with_mouse_actions():
    control = ControlConfig(
        action_keys=[None, "w", "mouse_left", "aim_left", "w+mouse_left"],
        aim_step=15,
    )
    null_ctrl = NullController(control)
    null_ctrl.act(0)  # None
    null_ctrl.act(2)  # mouse_left
    null_ctrl.act(3)  # aim_left
    null_ctrl.act(4)  # w+mouse_left

    assert null_ctrl.history == [0, 2, 3, 4]
    assert len(null_ctrl.mouse_history) == 3
    assert null_ctrl.mouse_history[0] == {"type": "click", "button": "left"}
    assert null_ctrl.mouse_history[1] == {"type": "move_rel", "dx": -15, "dy": 0}
    assert null_ctrl.mouse_history[2] == {"type": "click", "button": "left"}


def test_mock_controller_with_mouse_actions():
    game = MockArcadeGame()
    action_keys = [None, "up", "mouse_left", "aim_right", "up+mouse_left"]
    mock_ctrl = MockController(game, action_keys)

    # Action 0: None
    mock_ctrl.act(0)
    assert len(mock_ctrl.mouse_events) == 0

    # Action 1: "up"
    mock_ctrl.act(1)
    assert game._pending_key == "up"
    assert len(mock_ctrl.mouse_events) == 0

    # Action 2: "mouse_left"
    mock_ctrl.act(2)
    assert len(mock_ctrl.mouse_events) == 1
    assert mock_ctrl.mouse_events[0] == {"type": "click", "button": "left"}

    # Action 3: "aim_right"
    mock_ctrl.act(3)
    assert len(mock_ctrl.mouse_events) == 2
    assert mock_ctrl.mouse_events[1] == {"type": "move_rel", "dx": 1, "dy": 0}

    # Action 4: "up+mouse_left"
    mock_ctrl.act(4)
    assert game._pending_key == "up"
    assert len(mock_ctrl.mouse_events) == 3
    assert mock_ctrl.mouse_events[2] == {"type": "click", "button": "left"}

    # Restart clears mouse events
    mock_ctrl.restart()
    assert len(mock_ctrl.mouse_events) == 0


def test_build_controller_returns_correct_instance(monkeypatch):
    control = ControlConfig()
    dry_ctrl = build_controller(control, dry_run=True)
    assert isinstance(dry_ctrl, NullController)

    mock_pdi = MagicMock()
    monkeypatch.setattr(KeyboardController, "_load_backend", lambda self, ctrl: mock_pdi)
    unified_ctrl = build_controller(control, dry_run=False)
    assert isinstance(unified_ctrl, UnifiedInputController)


