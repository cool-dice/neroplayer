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
import time
from typing import Any, Protocol, runtime_checkable

from .config import ControlConfig


def parse_action_keys(key_spec: Any) -> tuple[str, ...]:
    """Normalize action key specification into a tuple of key names.

    Supports:
    - None -> ()
    - "d" -> ("d",)
    - "d+space" -> ("d", "space")
    - ["d", "space"] -> ("d", "space")
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
    """Format key spec for user display, logs, and HUD."""
    keys = parse_action_keys(key_spec)
    return " + ".join(keys) if keys else "IDLE/NONE"


def validate_action_keys(control: ControlConfig) -> list[dict[str, Any]]:
    """Inspect all configured actions, expanding multi-key combos and checking validity."""
    valid_map: set[str] | None = None
    try:
        import pydirectinput

        if hasattr(pydirectinput, "KEYBOARD_MAPPING"):
            valid_map = set(pydirectinput.KEYBOARD_MAPPING.keys())
    except Exception:
        pass

    results = []
    for idx, raw in enumerate(control.action_keys):
        keys = parse_action_keys(raw)
        formatted = format_action(raw)
        valid = True
        warnings = []
        if valid_map is not None:
            for k in keys:
                if k.lower() not in valid_map:
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
        keys = self._config.action_keys
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
        for raw in self._config.action_keys:
            all_keys.update(parse_action_keys(raw))
        for raw in self._config.restart_keys:
            all_keys.update(parse_action_keys(raw))
        for key in all_keys:
            with contextlib.suppress(Exception):
                self._pdi.keyUp(key)


class NullController:
    """Records actions without touching the OS. Useful for dry runs and tests."""

    def __init__(self, control: ControlConfig) -> None:
        self._config = control
        self.history: list[int] = []
        self.restarts = 0

    def act(self, action: int) -> None:
        self.history.append(int(action))

    def restart(self) -> None:
        self.restarts += 1

    def release_all(self) -> None:
        return None


def build_controller(control: ControlConfig, *, dry_run: bool = False) -> ActionController:
    """Return the real keyboard driver, or a no-op one for dry runs."""
    if dry_run:
        return NullController(control)
    return KeyboardController(control)
