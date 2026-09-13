"""Controls: translates discrete RL actions into OS-level keyboard input.

``pydirectinput`` uses the Win32 ``SendInput`` API with DirectInput scan
codes, which is what most games poll (plain ``pyautogui`` events are ignored
by many DirectX titles). On non-Windows hosts, or when ``dry_run=True``, the
controller logs the keystrokes instead of sending them so the rest of the
pipeline can be exercised without a game.
"""

from __future__ import annotations

import logging
import sys
import time

from config import ControlConfig

logger = logging.getLogger(__name__)


def _load_backend():
    """Import pydirectinput if we are on Windows; otherwise return ``None``."""
    if not sys.platform.startswith("win"):
        return None
    try:
        import pydirectinput

        # The built-in pause between calls would throttle the control loop;
        # we manage timing ourselves in GameController.
        pydirectinput.PAUSE = 0.0
        pydirectinput.FAILSAFE = False
        return pydirectinput
    except Exception as exc:  # pragma: no cover - platform dependent
        logger.warning("pydirectinput unavailable (%s); running controls in dry-run mode", exc)
        return None


class GameController:
    """Executes discrete actions and game-restart sequences.

    Args:
        cfg: Key mapping and timing settings.
        dry_run: Force logging-only mode even on Windows (useful for tests).
    """

    def __init__(self, cfg: ControlConfig, dry_run: bool = False) -> None:
        self.cfg = cfg
        self._backend = None if dry_run else _load_backend()
        self.dry_run = self._backend is None
        self._held: str | None = None
        if self.dry_run:
            logger.info("GameController in dry-run mode: inputs are logged, not sent")

    # -- public API -------------------------------------------------------- #
    @property
    def n_actions(self) -> int:
        return len(self.cfg.actions)

    def action_name(self, action: int) -> str:
        return self.cfg.action_names[action]

    def execute(self, action: int) -> None:
        """Perform ``action`` (index into ``cfg.actions``) as a timed key tap.

        A tap (down, hold, up) rather than a persistent hold keeps the agent's
        effect on the game local to a single step, which makes credit
        assignment easier for the policy.
        """
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action {action} out of range [0, {self.n_actions})")
        key = self.cfg.actions[action]
        if key is None:
            # NoOp: still consume the hold time so every action has equal duration.
            time.sleep(self.cfg.key_hold_seconds)
            return
        self.tap(key, self.cfg.key_hold_seconds)

    def tap(self, key: str, hold: float) -> None:
        """Press ``key`` for ``hold`` seconds."""
        self._key_down(key)
        try:
            time.sleep(hold)
        finally:
            self._key_up(key)

    def restart(self) -> None:
        """Bring the game window into focus and send the restart sequence."""
        self.release_all()
        if self.cfg.focus_click is not None:
            self._click(*self.cfg.focus_click)
            time.sleep(0.1)
        for key in self.cfg.restart_keys:
            self.tap(key, self.cfg.key_hold_seconds)
            time.sleep(0.05)
        # Give the game time to redraw the "new game" state before the env
        # captures the first observation.
        time.sleep(self.cfg.restart_delay_seconds)

    def release_all(self) -> None:
        """Release any key that might be stuck down (e.g. after an exception)."""
        if self._held is not None:
            self._key_up(self._held)

    def close(self) -> None:
        self.release_all()

    # -- backend plumbing ------------------------------------------------- #
    def _key_down(self, key: str) -> None:
        self._held = key
        if self.dry_run:
            logger.debug("[dry-run] keyDown %s", key)
            return
        self._backend.keyDown(key)

    def _key_up(self, key: str) -> None:
        self._held = None
        if self.dry_run:
            logger.debug("[dry-run] keyUp %s", key)
            return
        self._backend.keyUp(key)

    def _click(self, x: int, y: int) -> None:
        if self.dry_run:
            logger.debug("[dry-run] click (%d, %d)", x, y)
            return
        self._backend.moveTo(x, y)
        self._backend.click()
