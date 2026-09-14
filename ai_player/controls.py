"""Action execution: translate discrete agent actions into OS-level key events.

Browser games and most DirectX titles ignore synthetic events produced by the
high-level SendMessage/WM_KEYDOWN path. ``pydirectinput`` uses the Win32
``SendInput`` API with scan codes, which those games do accept -- hence the
hard dependency on it for real play.

The ``ActionController`` protocol keeps the environment decoupled from the
backend so tests and the bundled mock game can plug in their own executor.
"""

from __future__ import annotations

import contextlib
import itertools
import time
from typing import Any, Protocol, runtime_checkable

from .config import ControlConfig

# Canonical mouse action identifiers and mappings
MOUSE_CLICK_ACTIONS: dict[str, str] = {
    "mouse_left": "left",
    "mouse_left_click": "left",
    "click": "left",
    "fire": "left",
    "mouse_right": "right",
    "mouse_right_click": "right",
    "right_click": "right",
    "mouse_middle": "middle",
    "mouse_middle_click": "middle",
    "middle_click": "middle",
}

MOUSE_AIM_ACTIONS: dict[str, tuple[int, int]] = {
    # Directional aims: (dx, dy) unit steps
    "aim_up": (0, -1),
    "aim_down": (0, 1),
    "aim_left": (-1, 0),
    "aim_right": (1, 0),
    "mouse_up": (0, -1),
    "mouse_down": (0, 1),
}

ALL_MOUSE_ACTIONS: set[str] = set(MOUSE_CLICK_ACTIONS.keys()) | set(MOUSE_AIM_ACTIONS.keys())


def is_mouse_action(token: str) -> bool:
    """Return True if the token represents a known mouse action."""
    return token.lower() in ALL_MOUSE_ACTIONS


def format_action_token(token: str) -> str:
    """Format single action token cleanly for HUD (e.g. 'mouse_left' -> 'MOUSE_L')."""
    lower = token.lower()
    if lower in ("mouse_left", "mouse_left_click", "click", "fire"):
        return "MOUSE_L"
    if lower in ("mouse_right", "mouse_right_click", "right_click"):
        return "MOUSE_R"
    if lower in ("mouse_middle", "mouse_middle_click", "middle_click"):
        return "MOUSE_M"
    if lower == "aim_left":
        return "AIM_LEFT"
    if lower == "aim_right":
        return "AIM_RIGHT"
    if lower in ("aim_up", "mouse_up"):
        return "AIM_UP"
    if lower in ("aim_down", "mouse_down"):
        return "AIM_DOWN"
    return token


def parse_action_keys(key_spec: Any) -> tuple[str, ...]:
    """Normalize action key specification into a tuple of key/action names.

    Supports:
    - None -> ()
    - "d" -> ("d",)
    - "d+space" -> ("d", "space")
    - ["d", "space"] -> ("d", "space")
    - "w+mouse_left" -> ("w", "mouse_left")
    - ["w", "mouse_left"] -> ("w", "mouse_left")
    - "aim_left" -> ("aim_left",)
    """
    if key_spec is None:
        return ()
    if isinstance(key_spec, str):
        if "+" in key_spec:
            return tuple(k.strip() for k in key_spec.split("+") if k.strip())
        cleaned = key_spec.strip()
        return (cleaned,) if cleaned else ()
    if isinstance(key_spec, (list, tuple)):
        result: list[str] = []
        for item in key_spec:
            if item is None:
                continue
            s = str(item).strip()
            if "+" in s:
                result.extend(k.strip() for k in s.split("+") if k.strip())
            elif s:
                result.append(s)
        return tuple(result)
    return (str(key_spec).strip(),)


def format_action(key_spec: Any) -> str:
    """Format key spec for user display, logs, and HUD (e.g. 'W + MOUSE_L', 'AIM_LEFT')."""
    keys = parse_action_keys(key_spec)
    if not keys:
        return "IDLE/NONE"
    return " + ".join(format_action_token(k) for k in keys)


def get_effective_action_keys(control: ControlConfig) -> list[Any]:
    """Return the final list of action key specifications.

    If `control.auto_combos` is True, extracts all atomic unique keys from `control.action_keys`
    and generates the full powerset of combinations up to `control.max_combo_size`,
    plus None (idle). This allows the RL agent to discover and learn any key combination.
    """
    if not control.auto_combos:
        return list(control.action_keys)

    # Extract distinct non-empty single keys
    atomic_keys: list[str] = []
    seen: set[str] = set()
    for raw in control.action_keys:
        for k in parse_action_keys(raw):
            if k not in seen:
                seen.add(k)
                atomic_keys.append(k)

    # Build combos: None (IDLE), then size 1..max_combo_size
    combos: list[Any] = [None]
    max_k = max(1, min(len(atomic_keys), control.max_combo_size))
    for r in range(1, max_k + 1):
        for c in itertools.combinations(atomic_keys, r):
            combos.append(list(c))
    return combos


def validate_action_keys(control: ControlConfig) -> list[dict[str, Any]]:
    """Inspect all configured actions, expanding multi-key combos and checking validity."""
    valid_map: set[str] | None = None
    try:
        import pydirectinput

        if hasattr(pydirectinput, "KEYBOARD_MAPPING"):
            valid_map = set(pydirectinput.KEYBOARD_MAPPING.keys())
    except Exception:
        pass

    effective_keys = get_effective_action_keys(control)
    results = []
    for idx, raw in enumerate(effective_keys):
        keys = parse_action_keys(raw)
        formatted = format_action(raw)
        valid = True
        warnings = []
        for k in keys:
            if is_mouse_action(k):
                # Mouse action is always recognized and valid
                continue
            if valid_map is not None and k.lower() not in valid_map:
                valid = False
                warnings.append(f"Key '{k}' not found in pydirectinput mappings")
        results.append(
            {
                "index": idx,
                "raw": raw,
                "keys": keys,
                "formatted": formatted,
                "valid": valid,
                "warnings": warnings,
            }
        )
    return results


@runtime_checkable
class ActionController(Protocol):
    """Executes agent actions and restart sequences."""

    def act(self, action: int) -> None: ...

    def restart(self) -> None: ...

    def release_all(self) -> None: ...


class KeyboardController:
    """``pydirectinput`` backed keyboard driver supporting single and multi-key actions.

    Two modes:

    * **tap** (default) -- press and release within a few milliseconds. Correct
      for discrete inputs like jump/flap/rotate.
    * **hold** -- keep the chosen key(s) down until a different action arrives.
      Correct for continuous movement, where re-tapping would stutter.
    """

    def __init__(self, control: ControlConfig) -> None:
        self._config = control
        self._action_keys = get_effective_action_keys(control)
        self._held_keys: set[str] = set()
        self._pdi = self._load_backend(control)

    @staticmethod
    def _load_backend(control: ControlConfig):
        try:
            import pydirectinput
        except Exception as exc:
            raise RuntimeError(
                "pydirectinput could not be loaded (it only works on Windows). "
                "Run training with --mock to exercise the pipeline elsewhere."
            ) from exc
        # We pace the loop ourselves; the library's implicit pause would add
        # ~100 ms per call and wreck the target FPS.
        pydirectinput.PAUSE = control.pause_between_calls
        pydirectinput.FAILSAFE = control.fail_safe
        return pydirectinput

    def _keys_for(self, action: int) -> tuple[str, ...]:
        keys = self._action_keys
        if not 0 <= action < len(keys):
            raise ValueError(f"Action {action} outside 0..{len(keys) - 1}")
        return parse_action_keys(keys[action])

    def act(self, action: int) -> None:
        """Execute the key(s) bound to ``action`` simultaneously; empty tuple is a no-op."""
        keys = self._keys_for(action)
        if self._config.hold_keys:
            self._apply_hold(keys)
            return
        if not keys:
            return
        backend = self._pdi
        for key in keys:
            backend.keyDown(key)
        try:
            time.sleep(self._config.tap_duration)
        finally:
            for key in reversed(keys):
                backend.keyUp(key)

    def _apply_hold(self, keys: tuple[str, ...]) -> None:
        target = set(keys)
        if target == self._held_keys:
            return
        to_release = self._held_keys - target
        to_press = target - self._held_keys
        for key in to_release:
            self._pdi.keyUp(key)
        for key in to_press:
            self._pdi.keyDown(key)
        self._held_keys = target

    def restart(self) -> None:
        """Tap the configured restart keys in order to begin a new round."""
        self.release_all()
        for raw in self._config.restart_keys:
            keys = parse_action_keys(raw)
            for key in keys:
                self._pdi.keyDown(key)
            time.sleep(self._config.tap_duration)
            for key in reversed(keys):
                self._pdi.keyUp(key)
            # Menus need a beat to process each keystroke.
            time.sleep(0.05)

    def release_all(self) -> None:
        """Drop any held key. Always call this before resetting or exiting."""
        if self._pdi is None:
            return
        for key in list(self._held_keys):
            with contextlib.suppress(Exception):
                self._pdi.keyUp(key)
        self._held_keys.clear()

        # Best effort safety release of all known configured keys
        all_keys: set[str] = set()
        for raw in self._action_keys:
            all_keys.update(parse_action_keys(raw))
        for raw in self._config.restart_keys:
            all_keys.update(parse_action_keys(raw))
        for key in all_keys:
            with contextlib.suppress(Exception):
                self._pdi.keyUp(key)


class MouseController:
    """Cross-platform safe mouse driver supporting relative aiming and clicks.

    On Windows uses ``pydirectinput`` (or fallback) for:
    - move_rel(dx, dy)
    - move_to(x, y)
    - click(button)
    - press(button)
    - release(button)
    - release_all()

    On Linux or headless test environments, falls back gracefully to recording
    actions so unit tests and mock environments run without error.
    """

    def __init__(self, control: ControlConfig) -> None:
        self._config = control
        self._held_buttons: set[str] = set()
        self.history: list[dict[str, Any]] = []  # Recorded actions for testing / inspection
        self._pdi = self._load_backend(control)

    @staticmethod
    def _load_backend(control: ControlConfig):
        try:
            import pydirectinput

            pydirectinput.PAUSE = control.pause_between_calls
            pydirectinput.FAILSAFE = control.fail_safe
            return pydirectinput
        except Exception:
            return None

    def move_rel(self, dx: int, dy: int) -> None:
        """Move mouse relative to current position by (dx, dy) pixels."""
        scaled_dx = round(dx * self._config.mouse_sensitivity)
        scaled_dy = round(dy * self._config.mouse_sensitivity)
        self.history.append({"type": "move_rel", "dx": scaled_dx, "dy": scaled_dy})
        if self._pdi is not None and hasattr(self._pdi, "moveRel"):
            with contextlib.suppress(Exception):
                self._pdi.moveRel(scaled_dx, scaled_dy, relative=True)

    def move_to(self, x: int, y: int) -> None:
        """Move mouse to absolute screen coordinates (x, y)."""
        self.history.append({"type": "move_to", "x": x, "y": y})
        if self._pdi is not None and hasattr(self._pdi, "moveTo"):
            with contextlib.suppress(Exception):
                self._pdi.moveTo(x, y)

    def click(self, button: str = "left") -> None:
        """Click mouse button ('left', 'right', 'middle')."""
        btn = button.lower()
        self.history.append({"type": "click", "button": btn})
        if self._pdi is not None and hasattr(self._pdi, "click"):
            with contextlib.suppress(Exception):
                self._pdi.click(button=btn)
        elif self._pdi is not None and hasattr(self._pdi, "mouseDown") and hasattr(self._pdi, "mouseUp"):
            with contextlib.suppress(Exception):
                self._pdi.mouseDown(button=btn)
                time.sleep(self._config.tap_duration)
                self._pdi.mouseUp(button=btn)

    def press(self, button: str = "left") -> None:
        """Press and hold mouse button ('left', 'right', 'middle')."""
        btn = button.lower()
        self._held_buttons.add(btn)
        self.history.append({"type": "press", "button": btn})
        if self._pdi is not None and hasattr(self._pdi, "mouseDown"):
            with contextlib.suppress(Exception):
                self._pdi.mouseDown(button=btn)

    def release(self, button: str = "left") -> None:
        """Release held mouse button ('left', 'right', 'middle')."""
        btn = button.lower()
        self._held_buttons.discard(btn)
        self.history.append({"type": "release", "button": btn})
        if self._pdi is not None and hasattr(self._pdi, "mouseUp"):
            with contextlib.suppress(Exception):
                self._pdi.mouseUp(button=btn)

    def release_all(self) -> None:
        """Release all currently held mouse buttons."""
        self.history.append({"type": "release_all"})
        for btn in list(self._held_buttons):
            if self._pdi is not None and hasattr(self._pdi, "mouseUp"):
                with contextlib.suppress(Exception):
                    self._pdi.mouseUp(button=btn)
        self._held_buttons.clear()


class UnifiedInputController:
    """Unified hybrid controller executing both keyboard and mouse actions seamlessly.

    Supports:
    - Pure keyboard actions (e.g. ['w'], ['up', 'space'])
    - Pure mouse actions (e.g. ['mouse_left'], ['aim_left'])
    - Compound hybrid actions (e.g. ['w', 'mouse_left'], 'w+mouse_left')
    - Tap and hold modes for keyboard keys and mouse buttons.
    """

    def __init__(
        self,
        control: ControlConfig,
        keyboard: KeyboardController | None = None,
        mouse: MouseController | None = None,
    ) -> None:
        self._config = control
        self._action_keys = get_effective_action_keys(control)
        self.keyboard = keyboard or KeyboardController(control)
        self.mouse = mouse or MouseController(control)
        self._held_mouse_buttons: set[str] = set()

    def _split_action_tokens(self, action: int) -> tuple[list[str], list[str], list[str]]:
        """Split action into (keyboard_keys, mouse_clicks, mouse_aims)."""
        if not 0 <= action < len(self._action_keys):
            raise ValueError(f"Action {action} outside 0..{len(self._action_keys) - 1}")
        tokens = parse_action_keys(self._action_keys[action])
        kb_keys: list[str] = []
        mouse_clicks: list[str] = []
        mouse_aims: list[str] = []
        for token in tokens:
            lower = token.lower()
            if lower in MOUSE_CLICK_ACTIONS:
                mouse_clicks.append(MOUSE_CLICK_ACTIONS[lower])
            elif lower in MOUSE_AIM_ACTIONS:
                mouse_aims.append(lower)
            else:
                kb_keys.append(token)
        return kb_keys, mouse_clicks, mouse_aims

    def act(self, action: int) -> None:
        """Execute keyboard and mouse actions for this step."""
        kb_keys, mouse_clicks, mouse_aims = self._split_action_tokens(action)

        # 1. Execute mouse aim displacements (relative moves)
        for aim in mouse_aims:
            step_unit = MOUSE_AIM_ACTIONS[aim]
            step_px = self._config.aim_step
            self.mouse.move_rel(step_unit[0] * step_px, step_unit[1] * step_px)

        # 2. Execute keyboard action
        if self._config.hold_keys:
            self.keyboard._apply_hold(tuple(kb_keys))
        else:
            if kb_keys:
                backend = self.keyboard._pdi
                for key in kb_keys:
                    backend.keyDown(key)

        # 3. Execute mouse clicks / holds
        if self._config.hold_keys:
            target_buttons = set(mouse_clicks)
            to_release = self._held_mouse_buttons - target_buttons
            to_press = target_buttons - self._held_mouse_buttons
            for btn in to_release:
                self.mouse.release(btn)
            for btn in to_press:
                self.mouse.press(btn)
            self._held_mouse_buttons = target_buttons
        else:
            for btn in mouse_clicks:
                self.mouse.press(btn)

        # In tap mode, pause for tap duration and release tapped keys & buttons
        if not self._config.hold_keys and (kb_keys or mouse_clicks):
            try:
                time.sleep(self._config.tap_duration)
            finally:
                for btn in reversed(mouse_clicks):
                    self.mouse.release(btn)
                if kb_keys:
                    backend = self.keyboard._pdi
                    for key in reversed(kb_keys):
                        backend.keyUp(key)

    def restart(self) -> None:
        """Restart round: release held keys/mouse buttons and tap restart keys."""
        self.release_all()
        self.keyboard.restart()

    def release_all(self) -> None:
        """Release all held keys and mouse buttons."""
        self.keyboard.release_all()
        self.mouse.release_all()
        self._held_mouse_buttons.clear()


class NullController:
    """Records actions without touching the OS. Useful for dry runs and tests."""

    def __init__(self, control: ControlConfig) -> None:
        self._config = control
        self.history: list[int] = []
        self.action_history: list[tuple[str, ...]] = []
        self.mouse_history: list[dict[str, Any]] = []
        self.restarts = 0
        self._action_keys = get_effective_action_keys(control)

    def act(self, action: int) -> None:
        act_int = int(action)
        self.history.append(act_int)
        if 0 <= act_int < len(self._action_keys):
            tokens = parse_action_keys(self._action_keys[act_int])
            self.action_history.append(tokens)
            for token in tokens:
                lower = token.lower()
                if lower in MOUSE_CLICK_ACTIONS:
                    self.mouse_history.append({"type": "click", "button": MOUSE_CLICK_ACTIONS[lower]})
                elif lower in MOUSE_AIM_ACTIONS:
                    step_unit = MOUSE_AIM_ACTIONS[lower]
                    step_px = self._config.aim_step
                    self.mouse_history.append({
                        "type": "move_rel",
                        "dx": step_unit[0] * step_px,
                        "dy": step_unit[1] * step_px,
                    })

    def restart(self) -> None:
        self.restarts += 1

    def release_all(self) -> None:
        return None


def build_controller(control: ControlConfig, *, dry_run: bool = False) -> ActionController:
    """Return the input driver (UnifiedInputController or NullController for dry runs)."""
    if dry_run:
        return NullController(control)
    return UnifiedInputController(control)

