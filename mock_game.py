"""In-process arcade used to exercise the RL pipeline without a real window.

The mock is a tiny dodge game rendered with OpenCV: a player rectangle on
the left, hazards sliding in from the right. Collision paints a GAME OVER
banner that `observer.py` can detect; passing a hazard lights up extra
pixels in the score HUD so the pixel-delta reward fires.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from observer import make_game_over_banner


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
    """Deterministic-enough mini-game that speaks the same interfaces as hardware."""

    width: int = 320
    height: int = 240
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
    _audio: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.restart()

    # -- control interface -------------------------------------------------
    def apply_action(self, action: int) -> None:
        if self.over:
            return
        if action == 0:
            self.player_y -= self.player_speed
        elif action == 1:
            self.player_y += self.player_speed
        self.player_y = int(np.clip(self.player_y, 8, self.height - self.player_h - 8))
        self._physics()

    def restart(self) -> None:
        self.player_y = self.height // 2
        self.hazards = []
        self.score = 0
        self.over = False
        self.frame_i = 0
        self.scored_this_step = False
        self._audio = np.zeros(0, dtype=np.float32)
        self._spawn()

    # -- perception interface ----------------------------------------------
    def grab_bgr(self) -> np.ndarray:
        frame = np.full((self.height, self.width, 3), 24, dtype=np.uint8)
        frame[:, :] = (28, 24, 22)

        # Score HUD: one bright bar per point so the observer can see deltas.
        hud_x, hud_y, hud_w, hud_h = 10, 8, 180, 40
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (18, 18, 18), -1)
        bar_w = 6
        for i in range(min(self.score, hud_w // (bar_w + 2))):
            x0 = hud_x + 4 + i * (bar_w + 2)
            cv2.rectangle(
                frame,
                (x0, hud_y + 8),
                (x0 + bar_w, hud_y + hud_h - 8),
                (240, 240, 240),
                -1,
            )

        color_player = (220, 200, 40)
        cv2.rectangle(
            frame,
            (self.player_x, self.player_y),
            (self.player_x + self.player_w, self.player_y + self.player_h),
            color_player,
            -1,
        )
        for hz in self.hazards:
            cv2.rectangle(frame, (hz.x, hz.y), (hz.x + hz.w, hz.y + hz.h), (40, 90, 230), -1)

        if self.over:
            overlay = frame.copy()
            overlay[:] = (0, 0, 40)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            banner = make_game_over_banner()
            bh, bw = banner.shape[:2]
            x = (self.width - bw) // 2
            y = (self.height - bh) // 2
            x = max(0, x)
            y = max(0, y)
            h = min(bh, self.height - y)
            w = min(bw, self.width - x)
            frame[y : y + h, x : x + w] = banner[:h, :w]
        return frame

    def audio_samples(self, n: int, sample_rate: int) -> np.ndarray:
        """Return a short cue when a point is scored, otherwise silence."""
        t = np.arange(n, dtype=np.float32) / float(sample_rate)
        if self.scored_this_step:
            return (0.35 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)
        if self.over:
            return (0.2 * np.sin(2 * np.pi * 110.0 * t)).astype(np.float32)
        return np.zeros(n, dtype=np.float32)

    # -- internals ---------------------------------------------------------
    def _physics(self) -> None:
        if self.over:
            return
        self.scored_this_step = False
        self.frame_i += 1
        px1, py1 = self.player_x + self.player_w, self.player_y + self.player_h
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
        y = int(self.rng.integers(16, self.height - 64))
        self.hazards.append(_Hazard(x=self.width - 20, y=y))

    @staticmethod
    def _aabb(ax0: int, ay0: int, ax1: int, ay1: int, bx0: int, by0: int, bx1: int, by1: int) -> bool:
        return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0
