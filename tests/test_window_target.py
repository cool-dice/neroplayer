from __future__ import annotations

from ai_player.config import CaptureConfig, Region
from ai_player.window_target import (
    WindowLocator,
    WindowTarget,
    pick_target,
    title_is_excluded,
    title_matches,
)


def _win(
    handle: int, title: str, *, left: int = 0, top: int = 0, width: int = 800, height: int = 600
) -> WindowTarget:
    return WindowTarget(title=title, handle=handle, region=Region(left, top, width, height))


def test_title_exclusion_and_matching():
    assert title_is_excluded("ai-player")
    assert title_is_excluded("foo — Cursor")
    assert title_is_excluded("Windows PowerShell")
    assert not title_is_excluded("Celeste")
    assert title_is_excluded("My Game", extra=["my game"])

    assert title_matches("Celeste", "cele")
    assert title_matches("Cave Story+", r"^cave")
    assert not title_matches("Celeste", "Hollow")


def test_pick_target_follows_foreground_and_stays_sticky_on_tools():
    game = _win(1, "Celeste")
    other = _win(2, "Hollow Knight")
    ide = _win(3, "main.py — neroplayer — Cursor")
    cfg = CaptureConfig(follow_foreground=True, min_width=100, min_height=100)
    windows = [game, other, ide]

    locked = pick_target(windows, foreground_handle=1, config=cfg)
    assert locked is game

    # Alt-tab to another game → switch.
    switched = pick_target(windows, foreground_handle=2, config=cfg, previous=locked)
    assert switched is other

    # Click the IDE → keep the last game.
    sticky = pick_target(windows, foreground_handle=3, config=cfg, previous=switched)
    assert sticky is other


def test_pick_target_pins_window_title_even_when_not_focused():
    game = _win(1, "Celeste")
    other = _win(2, "Hollow Knight")
    cfg = CaptureConfig(follow_foreground=True, window_title="Celeste", min_width=100, min_height=100)
    chosen = pick_target([game, other], foreground_handle=2, config=cfg)
    assert chosen is game


def test_pick_target_ignores_tiny_windows_and_static_mode():
    tiny = _win(1, "Celeste", width=20, height=20)
    cfg = CaptureConfig(follow_foreground=True, min_width=200, min_height=150)
    assert pick_target([tiny], foreground_handle=1, config=cfg) is None

    game = _win(1, "Celeste")
    previous = _win(9, "Old")
    static = CaptureConfig(follow_foreground=False, min_width=100, min_height=100)
    assert pick_target([game], foreground_handle=1, config=static, previous=previous) is previous


def test_locator_uses_injected_enumerator():
    game = _win(7, "Celeste")

    def enumerate_fn():
        return [game], 7

    locator = WindowLocator(enumerate_fn=enumerate_fn)
    found = locator.locate(CaptureConfig(follow_foreground=True, min_width=100, min_height=100))
    assert found == game
