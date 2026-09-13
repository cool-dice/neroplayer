"""Low-level keyboard controls for the game."""

from __future__ import annotations

import time
from typing import Any

from config import GameConfig


class GameController:
    """Translate discrete RL actions into DirectInput keyboard events."""

    def __init__(self, config: GameConfig) -> None:
        self.config = config
        self._input: Any | None = None

    def _backend(self) -> Any:
        if self._input is None:
            try:
                import pydirectinput
            except ImportError as exc:
                raise RuntimeError(
                    "Install 'pydirectinput' on Windows to send game controls."
                ) from exc
            pydirectinput.PAUSE = 0
            pydirectinput.FAILSAFE = True
            self._input = pydirectinput
        return self._input

    @property
    def action_count(self) -> int:
        return len(self.config.action_keys)

    def execute(self, action: int) -> None:
        """Press the configured key for an action; None is a no-op."""
        if action < 0 or action >= self.action_count:
            raise ValueError(f"Invalid action {action}; expected 0..{self.action_count - 1}")
        key = self.config.action_keys[action]
        if key is None:
            return

        backend = self._backend()
        backend.keyDown(key)
        try:
            time.sleep(self.config.action_hold_seconds)
        finally:
            backend.keyUp(key)

    def restart(self) -> None:
        """Press the configured restart key or key combination."""
        backend = self._backend()
        keys = self.config.restart_keys
        if len(keys) == 1:
            backend.press(keys[0])
        else:
            backend.hotkey(*keys)
        time.sleep(self.config.restart_wait_seconds)

    def release_all(self) -> None:
        """Best-effort release of configured keys during shutdown."""
        if self._input is None:
            return
        for key in self.config.action_keys:
            if key is not None:
                self._input.keyUp(key)
