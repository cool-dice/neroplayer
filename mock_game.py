"""In-process dodge arcade used to train and test the pipeline without a window.

The mock speaks the same contracts as the hardware stack:

* :class:`MockArcade` is a ``FrameSource`` (BGR frames the observer can read).
* :class:`MockController` applies discrete actions to that arcade instead of
  sending DirectInput keystrokes.

Collision paints the game-over colour into a known ROI (so ``GameObserver``
fires ``terminated``); passing a hazard lights extra pixels in the score HUD
(so the pixel-change reward fires). ``python train.py --mock --smoke`` is the
fastest way to confirm Gymnasium + PPO before pointing the agent at a real
game.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from config import (
    CONFIG,
    AudioConfig,
    Config,
    ControlConfig,
    Region,
)
from environment import GameEnv
from observer import GameObserver
from perception import AudioCapture, ScreenCapture

MOCK_WIDTH = 320
MOCK_HEIGHT = 240
MOCK_SCORE_REGION = Region(left=10, top=8, width=180, height=40)
MOCK_GAME_OVER_REGION = Region(left=10, top=85, width=300, height=70)
MOCK_GAME_OVER_BGR = (40, 40, 200)


@dataclass
class _Hazard:
    x: int
    y: int
    w: int = 18
    h: int = 48
    speed: int = 8
    counted: bool = False


@dataclass
class MockArcade:
    """Tiny vertical-dodge game rendered with OpenCV."""

    seed: int = 0
    player_x: int = 36
    player_w: int = 18
    player_h: int = 18
    player_speed: int = 14
    spawn_every: int = 22
    rng: np.random.Generator = field(init=False)
    player_y: int = field(init=False)
    hazards: list[_Hazard] = field(init=False)
    score: int = field(init=False)
    over: bool = field(init=False)
    frame_i: int = field(init=False)
    scored_this_step: bool = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.restart()

    # -- FrameSource -------------------------------------------------------
    def grab(self) -> np.ndarray:
        frame = np.full((MOCK_HEIGHT, MOCK_WIDTH, 3), (28, 24, 22), dtype=np.uint8)

        x, y, w, h = MOCK_SCORE_REGION.left, MOCK_SCORE_REGION.top, MOCK_SCORE_REGION.width, MOCK_SCORE_REGION.height
        cv2.rectangle(frame, (x, y), (x + w, y + h), (18, 18, 18), -1)
        bar_w = 6
        for i in range(min(self.score, w // (bar_w + 2))):
            x0 = x + 4 + i * (bar_w + 2)
            cv2.rectangle(frame, (x0, y + 8), (x0 + bar_w, y + h - 8), (240, 240, 240), -1)

        cv2.rectangle(
            frame,
            (self.player_x, self.player_y),
            (self.player_x + self.player_w, self.player_y + self.player_h),
            (220, 200, 40),
            -1,
        )
        for hz in self.hazards:
            cv2.rectangle(frame, (hz.x, hz.y), (hz.x + hz.w, hz.y + hz.h), (40, 90, 230), -1)

        if self.over:
            overlay = frame.copy()
            overlay[:] = (0, 0, 40)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
            gx, gy = MOCK_GAME_OVER_REGION.left, MOCK_GAME_OVER_REGION.top
            gw, gh = MOCK_GAME_OVER_REGION.width, MOCK_GAME_OVER_REGION.height
            # Solid fill last: the colour detector matches mean BGR, and text
            # would pull that mean off the target.
            frame[gy : gy + gh, gx : gx + gw] = MOCK_GAME_OVER_BGR
        return frame

    def close(self) -> None:
        return

    # -- control -----------------------------------------------------------
    def apply_action(self, action: int) -> None:
        if self.over:
            return
        if action == 0:
            self.player_y -= self.player_speed
        elif action == 1:
            self.player_y += self.player_speed
        self.player_y = int(np.clip(self.player_y, 8, MOCK_HEIGHT - self.player_h - 8))
        self._physics()

    def restart(self) -> None:
        self.player_y = MOCK_HEIGHT // 2
        self.hazards = []
        self.score = 0
        self.over = False
        self.frame_i = 0
        self.scored_this_step = False
        self._spawn()

    def _physics(self) -> None:
        self.scored_this_step = False
        self.frame_i += 1
        px1 = self.player_x + self.player_w
        py1 = self.player_y + self.player_h
        remaining: list[_Hazard] = []
        for hz in self.hazards:
            hz.x -= hz.speed
            if self._aabb(self.player_x, self.player_y, px1, py1, hz.x, hz.y, hz.x + hz.w, hz.y + hz.h):
                self.over = True
                remaining.append(hz)
                continue
            if not hz.counted and hz.x + hz.w < self.player_x:
                hz.counted = True
                self.score += 1
                self.scored_this_step = True
            if hz.x + hz.w > 0:
                remaining.append(hz)
        self.hazards = remaining
        if self.frame_i % self.spawn_every == 0:
            self._spawn()

    def _spawn(self) -> None:
        # Aim near the player so a no-op policy dies and the agent must dodge.
        aimed = int(self.player_y - 16)
        jitter = int(self.rng.integers(-12, 13))
        y = int(np.clip(aimed + jitter, 16, MOCK_HEIGHT - 64))
        self.hazards.append(_Hazard(x=MOCK_WIDTH - 20, y=y))

    @staticmethod
    def _aabb(ax0: int, ay0: int, ax1: int, ay1: int, bx0: int, by0: int, bx1: int, by1: int) -> bool:
        return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


class MockController:
    """Same public surface as :class:`GameController`, driving :class:`MockArcade`."""

    def __init__(self, game: MockArcade, cfg: ControlConfig) -> None:
        self.game = game
        self.cfg = cfg
        self.dry_run = True

    @property
    def n_actions(self) -> int:
        return len(self.cfg.actions)

    def action_name(self, action: int) -> str:
        return self.cfg.action_names[action]

    def execute(self, action: int) -> None:
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action {action} out of range [0, {self.n_actions})")
        self.game.apply_action(action)

    def restart(self) -> None:
        self.game.restart()

    def close(self) -> None:
        return


def mock_config() -> Config:
    """Config whose ROIs line up with the mock canvas (no wall-clock sleeps)."""
    return replace(
        CONFIG,
        screen=replace(CONFIG.screen, target_fps=10_000),
        audio=AudioConfig(enabled=False),
        controls=replace(CONFIG.controls, key_hold_seconds=0.0, restart_delay_seconds=0.0, focus_click=None),
        reward=replace(
            CONFIG.reward,
            game_over_template=CONFIG.reward.game_over_template.parent / "missing_mock.png",
            use_ocr=False,
            score_region=MOCK_SCORE_REGION,
            game_over_region=MOCK_GAME_OVER_REGION,
            game_over_color_bgr=MOCK_GAME_OVER_BGR,
            game_over_color_tolerance=15,
            game_over_confirm_frames=1,
            score_change_pixel_fraction=0.02,
        ),
        train=replace(CONFIG.train, max_episode_steps=500),
    )


def build_mock_env(seed: int = 0) -> GameEnv:
    """Fully wired Gymnasium env that never touches the display or keyboard."""
    cfg = mock_config()
    game = MockArcade(seed=seed)
    return GameEnv(
        cfg,
        screen=ScreenCapture(cfg.screen, source=game),
        audio=AudioCapture(cfg.audio, start=False),
        controller=MockController(game, cfg.controls),
        observer=GameObserver(cfg.reward),
    )
