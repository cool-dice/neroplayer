"""Action execution: translate discrete agent actions into OS-level key events.

Browser games and most DirectX titles ignore synthetic events produced by the
high-level SendMessage/WM_KEYDOWN path. ``pydirectinput`` uses the Win32
``SendInput`` API with scan codes, which those games do accept -- hence the
hard dependency on it for real play.

The ``ActionController`` protocol keeps the environment decoupled from the
backend so tests and the bundled mock game can plug in their own executor.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

from .config import ControlConfig


@runtime_checkable
class ActionController(Protocol):
    """Executes agent actions and restart sequences."""

    def act(self, action: int) -> None: ...

    def restart(self) -> None: ...

    def release_all(self) -> None: ...


class KeyboardController:
    """``pydirectinput`` backed keyboard driver.

    Two modes:

    * **tap** (default) -- press and release within a few milliseconds. Correct
      for discrete inputs like jump/flap/rotate.
    * **hold** -- keep the chosen key down until a different action arrives.
      Correct for continuous movement, where re-tapping would stutter.
    """

    def __init__(self, control: ControlConfig) -> None:
        self._config = control
        self._held_key: str | None = None
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

    def act(self, action: int) -> None:
        """Execute the key bound to ``action`` (no-op actions do nothing)."""
        key = self._key_for(action)
        if self._config.hold_keys:
            self._apply_hold(key)
            return
        if key is None:
            return
        self._pdi.keyDown(key)
        time.sleep(self._config.tap_duration)
        self._pdi.keyUp(key)

    def _apply_hold(self, key: str | None) -> None:
        if key == self._held_key:
            return
        if self._held_key is not None:
            self._pdi.keyUp(self._held_key)
        if key is not None:
            self._pdi.keyDown(key)
        self._held_key = key

    def restart(self) -> None:
        """Tap the configured restart keys in order to begin a new round."""
        self.release_all()
        for key in self._config.restart_keys:
            self._pdi.keyDown(key)
            time.sleep(self._config.tap_duration)
            self._pdi.keyUp(key)
            # Menus need a beat to process each keystroke.
            time.sleep(0.05)

    def release_all(self) -> None:
        """Drop any held key. Always call this before resetting or exiting."""
        if self._held_key is not None:
            self._pdi.keyUp(self._held_key)
            self._held_key = None

    def _key_for(self, action: int) -> str | None:
        keys = self._config.action_keys
        if not 0 <= action < len(keys):
            raise ValueError(f"Action {action} outside 0..{len(keys) - 1}")
        return keys[action]


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
