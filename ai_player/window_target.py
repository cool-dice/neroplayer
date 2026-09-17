"""Locate the live game window so capture can follow the user, not a static box.

The original calibrate flow pinned a desktop rectangle. That is fine for a
single-game proof of concept, but it breaks as soon as you alt-tab to another
title: the agent keeps staring at empty desktop (or the previous game's
chrome) until you re-run calibrate.

This module answers "which window should we capture *right now*?":

* ``follow_foreground`` (default) locks onto the focused window and stays
  sticky when focus moves to a tool we never want to train on (our HUD,
  terminals, the IDE).
* ``window_title`` pins a substring/regex so you can look away without the
  capture jumping.
* A change of window *identity* (OS handle) is a game switch; a move/resize
  of the same handle is not.

OS backends are best-effort. Windows uses ``user32``; Linux tries ``xdotool``
when present. Anything else falls back to the configured static region.
"""

from __future__ import annotations

import contextlib
import logging
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .config import CaptureConfig, Region

logger = logging.getLogger(__name__)

# Substrings matched case-insensitively against the window title. When the
# foreground window looks like one of these we keep capturing the last game
# instead of training on the IDE / our own overlay.
DEFAULT_EXCLUDED_SUBSTRINGS: tuple[str, ...] = (
    "ai-player",
    "calibrate --",
    "agent view",
    "nvidia geforce overlay",
    "windows terminal",
    "powershell",
    "command prompt",
    "task manager",
    "cursor",
)


@dataclass(frozen=True)
class WindowTarget:
    """A visible top-level window and its capture rectangle in screen pixels."""

    title: str
    region: Region
    handle: int | None = None

    @property
    def identity(self) -> tuple[int | None, str]:
        """Stable-enough id: OS handle when we have one, else the title."""
        return (self.handle, self.title if self.handle is None else "")


EnumerateFn = Callable[[], tuple[list[WindowTarget], int | None]]


def title_is_excluded(title: str, extra: Sequence[str] = ()) -> bool:
    """True if ``title`` looks like a tool window we should never train on."""
    lowered = title.strip().lower()
    if not lowered:
        return True
    needles = [n.lower() for n in (*DEFAULT_EXCLUDED_SUBSTRINGS, *extra) if n.strip()]
    return any(n in lowered for n in needles)


def title_matches(title: str, needle: str) -> bool:
    """Case-insensitive substring match; ``^...`` is treated as a regex."""
    needle = needle.strip()
    if not needle:
        return True
    if needle.startswith("^") or needle.endswith("$"):
        try:
            return re.search(needle, title, flags=re.IGNORECASE) is not None
        except re.error:
            return needle.lower() in title.lower()
    return needle.lower() in title.lower()


def pick_target(
    windows: Sequence[WindowTarget],
    *,
    foreground_handle: int | None,
    config: CaptureConfig,
    previous: WindowTarget | None = None,
) -> WindowTarget | None:
    """Choose the window to capture given the live OS snapshot.

    Preference order:
    1. A window whose title matches ``config.window_title`` (foreground first).
    2. The foreground window, if follow-foreground is on and it is not excluded.
    3. The previously locked game, so clicking the HUD / a terminal is a no-op.
    """
    usable = [
        w
        for w in windows
        if w.region.width >= config.min_width and w.region.height >= config.min_height
    ]
    if not usable and previous is not None:
        return previous

    needle = (config.window_title or "").strip()
    if needle:
        matches = [w for w in usable if title_matches(w.title, needle)]
        if not matches:
            return previous
        for w in matches:
            if foreground_handle is not None and w.handle == foreground_handle:
                return w
        return matches[0]

    if not config.follow_foreground:
        return previous

    foreground = None
    if foreground_handle is not None:
        foreground = next((w for w in usable if w.handle == foreground_handle), None)
    if foreground is not None and not title_is_excluded(foreground.title, config.exclude_titles):
        return foreground
    return previous


class WindowLocator:
    """Thin wrapper around the OS backend so tests can inject a fake enumerator."""

    def __init__(self, enumerate_fn: EnumerateFn | None = None) -> None:
        self._enumerate = enumerate_fn or _enumerate_windows

    def locate(self, config: CaptureConfig, previous: WindowTarget | None = None) -> WindowTarget | None:
        try:
            windows, foreground = self._enumerate()
        except Exception as exc:
            logger.debug("Window enumeration failed (%s)", exc)
            return previous
        return pick_target(
            windows,
            foreground_handle=foreground,
            config=config,
            previous=previous,
        )


def _enumerate_windows() -> tuple[list[WindowTarget], int | None]:
    if sys.platform == "win32":
        return _enumerate_win32()
    return _enumerate_xdotool()


def _enumerate_win32() -> tuple[list[WindowTarget], int | None]:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)

    class RECT(ctypes.Structure):
        _fields_ = (
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        )

    GetForegroundWindow = user32.GetForegroundWindow
    GetForegroundWindow.restype = wintypes.HWND
    IsWindowVisible = user32.IsWindowVisible
    GetWindowTextLengthW = user32.GetWindowTextLengthW
    GetWindowTextW = user32.GetWindowTextW
    GetClientRect = user32.GetClientRect
    ClientToScreen = user32.ClientToScreen
    GetWindowRect = user32.GetWindowRect

    def _title(hwnd: int) -> str:
        length = GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        GetWindowTextW(hwnd, buf, length + 1)
        return buf.value

    def _region(hwnd: int) -> Region | None:
        client = RECT()
        if not GetClientRect(hwnd, ctypes.byref(client)):
            return None
        width = int(client.right - client.left)
        height = int(client.bottom - client.top)
        if width <= 1 or height <= 1:
            # Minimised or empty client area: try the outer frame as a fallback.
            outer = RECT()
            if not GetWindowRect(hwnd, ctypes.byref(outer)):
                return None
            width = int(outer.right - outer.left)
            height = int(outer.bottom - outer.top)
            if width <= 1 or height <= 1:
                return None
            return Region(left=int(outer.left), top=int(outer.top), width=width, height=height)
        point = wintypes.POINT(0, 0)
        ClientToScreen(hwnd, ctypes.byref(point))
        return Region(left=int(point.x), top=int(point.y), width=width, height=height)

    windows: list[WindowTarget] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def _callback(hwnd: Any, _lparam: Any) -> bool:
        if not IsWindowVisible(hwnd):
            return True
        title = _title(int(hwnd))
        region = _region(int(hwnd))
        if region is None:
            return True
        windows.append(WindowTarget(title=title, region=region, handle=int(hwnd)))
        return True

    user32.EnumWindows(_callback, 0)
    fg = GetForegroundWindow()
    return windows, (int(fg) if fg else None)


def _enumerate_xdotool() -> tuple[list[WindowTarget], int | None]:
    if shutil.which("xdotool") is None:
        return [], None

    def _run(*args: str) -> str:
        result = subprocess.run(
            ["xdotool", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        return (result.stdout or "").strip()

    def _target_from_id(handle: int) -> WindowTarget | None:
        name = _run("getwindowname", str(handle))
        geom = _run("getwindowgeometry", "--shell", str(handle))
        parsed: dict[str, int] = {}
        for line in geom.splitlines():
            if "=" not in line:
                continue
            key, _, raw = line.partition("=")
            with contextlib.suppress(ValueError, TypeError):
                parsed[key.strip()] = int(raw.strip())
        width = parsed.get("WIDTH", 0)
        height = parsed.get("HEIGHT", 0)
        if width <= 1 or height <= 1:
            return None
        return WindowTarget(
            title=name,
            handle=handle,
            region=Region(left=parsed.get("X", 0), top=parsed.get("Y", 0), width=width, height=height),
        )

    windows: list[WindowTarget] = []
    foreground: int | None = None
    active = _run("getactivewindow")
    if active.isdigit():
        foreground = int(active)
        target = _target_from_id(foreground)
        if target is not None:
            windows.append(target)

    ids = _run("search", "--onlyvisible", "--name", ".")
    for token in ids.split():
        if not token.isdigit():
            continue
        handle = int(token)
        if any(w.handle == handle for w in windows):
            continue
        target = _target_from_id(handle)
        if target is not None:
            windows.append(target)
    return windows, foreground
