"""Controllability Probe and Dynamic Entity Tracker.

Provides autonomous avatar discovery and foreground entity tracking:

1. **ControllabilityProbe**:
   - Injects probe test actions (e.g. left, right, up, down, jump) during early gameplay or on demand.
   - Calculates differential optical flow (Farneback) or dense temporal motion differencing.
   - Identifies the visual cluster / bounding box whose motion correlates directly with input commands.
   - Tracks and updates the player avatar bounding box across frames.

2. **EntityTracker**:
   - Dynamic foreground object segmentation (temporal frame differencing outside player BBox).
   - Classifies detected dynamic entities into:
     * Projectile (fast linear velocity, small bounding box)
     * Threat/Enemy (movement vector oriented toward player, or horizontal platform patrolling)
     * Collectible/Bonus (floating/falling item, slow vertical drift, distinct appearance)
   - Performs causal collision analysis: compares entity intersection against feedback
     (score increase -> collectible reward; life loss / hit flash / penalty -> threat penalty).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

import cv2
import numpy as np

from .config import Region, TrackerConfig


@dataclass
class BBox:
    """Axis-aligned bounding box (x, y, w, h)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def to_region(self) -> Region:
        return Region(left=self.x, top=self.y, width=self.w, height=self.h)

    def intersects(self, other: BBox) -> bool:
        return not (
            self.x2 < other.x
            or self.x > other.x2
            or self.y2 < other.y
            or self.y > other.y2
        )

    def distance_to(self, other: BBox) -> float:
        c1 = self.center
        c2 = other.center
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def iou(self, other: BBox) -> float:
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter_area = iw * ih
        union_area = self.area + other.area - inter_area
        return inter_area / max(union_area, 1)


class EntityKind(str, Enum):
    PLAYER = "player"
    PROJECTILE = "projectile"
    THREAT = "threat"
    COLLECTIBLE = "collectible"
    UNKNOWN = "unknown"


@dataclass
class TrackedEntity:
    """A tracked dynamic entity in the game environment."""

    entity_id: int
    bbox: BBox
    velocity: tuple[float, float] = (0.0, 0.0)  # (vx, vy) px/frame
    kind: EntityKind = EntityKind.UNKNOWN
    confidence: float = 0.5
    age: int = 1  # frames tracked
    disappeared: int = 0
    trajectory: list[tuple[float, float]] = field(default_factory=list)


class ControllabilityProbe:
    """Probes game controls to discover which screen cluster is the player avatar.

    Injects directional probe actions, measures optical flow vectors across candidate
    moving components, and correlates them with expected directional movement.
    """

    # Expected direction signs: (dx, dy)
    DIRECTION_VECTORS: ClassVar[dict[str, tuple[float, float]]] = {
        "left": (-1.0, 0.0),
        "a": (-1.0, 0.0),
        "right": (1.0, 0.0),
        "d": (1.0, 0.0),
        "up": (0.0, -1.0),
        "w": (0.0, -1.0),
        "jump": (0.0, -1.0),
        "space": (0.0, -1.0),
        "down": (0.0, 1.0),
        "s": (0.0, 1.0),
    }

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.player_bbox: BBox | None = None
        self._history: list[tuple[str | None, np.ndarray]] = []
        self._prev_gray: np.ndarray | None = None
        self._is_calibrated: bool = False
        self._confidence: float = 0.0

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated and self.player_bbox is not None

    @property
    def confidence(self) -> float:
        return self._confidence

    def reset(self) -> None:
        self.player_bbox = None
        self._history.clear()
        self._prev_gray = None
        self._is_calibrated = False
        self._confidence = 0.0

    def record_probe_step(self, action_name: str | None, frame: np.ndarray) -> None:
        """Record an action and resulting frame during the probing phase."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        self._history.append((action_name, gray))

    def analyze_probes(self) -> BBox | None:
        """Analyze recorded action-frame sequence to locate the player avatar.

        Returns detected BBox of the player avatar, or None if inconclusive.
        """
        if len(self._history) < 2:
            return None

        # Gather optical flow between consecutive probe steps
        candidate_scores: dict[tuple[int, int, int, int], float] = {}

        for i in range(1, len(self._history)):
            action, cur_gray = self._history[i]
            _, prev_gray = self._history[i - 1]

            if action is None:
                continue

            # Determine expected motion vector
            expected_vec = None
            action_lower = action.lower()
            for key, vec in self.DIRECTION_VECTORS.items():
                if key in action_lower:
                    expected_vec = vec
                    break

            if expected_vec is None:
                continue

            # Compute dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            # Find motion regions
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_mask = (mag > 1.5).astype(np.uint8) * 255

            contours, _ = cv2.findContours(
                motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if not (self.config.min_entity_area <= area <= self.config.max_entity_area):
                    continue

                # Compute mean flow within this contour
                sub_flow = flow[y : y + h, x : x + w]
                mean_dx = float(np.mean(sub_flow[..., 0]))
                mean_dy = float(np.mean(sub_flow[..., 1]))

                flow_mag = math.hypot(mean_dx, mean_dy)
                if flow_mag < 0.5:
                    continue

                # Dot product / cosine similarity with expected vector
                norm_dx = mean_dx / flow_mag
                norm_dy = mean_dy / flow_mag
                similarity = norm_dx * expected_vec[0] + norm_dy * expected_vec[1]

                if similarity > 0.3:
                    # Quantize bbox key to merge close detections
                    key = (x, y, w, h)
                    candidate_scores[key] = candidate_scores.get(key, 0.0) + (similarity * flow_mag)

        if not candidate_scores:
            # Fallback: look for the central/standard candidate moving component
            return self._fallback_avatar_search()

        best_box_coords, best_score = max(candidate_scores.items(), key=lambda item: item[1])
        x, y, w, h = best_box_coords
        self.player_bbox = BBox(max(0, x), max(0, y), max(10, w), max(10, h))
        self._is_calibrated = True
        self._confidence = min(1.0, best_score / 5.0)
        return self.player_bbox

    def _fallback_avatar_search(self) -> BBox | None:
        """Fallback avatar detection: central moving cluster in first frames."""
        if len(self._history) < 2:
            return None
        _, prev_gray = self._history[0]
        _, cur_gray = self._history[-1]
        diff = cv2.absdiff(prev_gray, cur_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Filter contours by area and pick the one closest to horizontal middle
        _, w = cur_gray.shape[:2]
        center_x = w / 2.0
        best_cnt = None
        min_dist = float("inf")

        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if self.config.min_entity_area <= area <= self.config.max_entity_area:
                dist = abs((bx + bw / 2.0) - center_x)
                if dist < min_dist:
                    min_dist = dist
                    best_cnt = (bx, by, bw, bh)

        if best_cnt is not None:
            bx, by, bw, bh = best_cnt
            self.player_bbox = BBox(bx, by, bw, bh)
            self._is_calibrated = True
            self._confidence = 0.5
            return self.player_bbox
        return None

    def update_player_location(self, frame: np.ndarray) -> BBox | None:
        """Track player avatar in current frame using template matching or local flow."""
        if self.player_bbox is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self._prev_gray is None:
            self._prev_gray = gray
            return self.player_bbox

        pb = self.player_bbox
        # Local search window around player
        margin = 40
        h, w = gray.shape[:2]
        x1 = max(0, pb.x - margin)
        y1 = max(0, pb.y - margin)
        x2 = min(w, pb.x2 + margin)
        y2 = min(h, pb.y2 + margin)

        search_area = gray[y1:y2, x1:x2]
        template = self._prev_gray[pb.y : pb.y2, pb.x : pb.x2]

        can_match = (
            template.size > 0
            and search_area.shape[0] >= template.shape[0]
            and search_area.shape[1] >= template.shape[1]
        )
        if can_match:
            res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.4:
                new_x = x1 + max_loc[0]
                new_y = y1 + max_loc[1]
                self.player_bbox = BBox(new_x, new_y, pb.w, pb.h)

        self._prev_gray = gray
        return self.player_bbox


class EntityTracker:
    """Detects, tracks, and classifies dynamic foreground entities.

    Distinguishes projectiles, threats, and collectibles, and performs causal
    collision analysis against player rewards/penalties.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._next_id = 1
        self._tracked_entities: dict[int, TrackedEntity] = {}
        self._prev_frame_gray: np.ndarray | None = None
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=25, detectShadows=False
        )
        self._collision_history: list[dict[str, Any]] = []

    @property
    def entities(self) -> list[TrackedEntity]:
        return list(self._tracked_entities.values())

    @property
    def collision_history(self) -> list[dict[str, Any]]:
        return self._collision_history

    def reset(self) -> None:
        self._next_id = 1
        self._tracked_entities.clear()
        self._prev_frame_gray = None
        self._collision_history.clear()

    def update(
        self,
        frame: np.ndarray,
        player_bbox: BBox | None,
        score_region: Region | None = None,
        game_over_region: Region | None = None,
    ) -> list[TrackedEntity]:
        """Detect and track foreground entities in current frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        h, w = gray.shape[:2]

        # Dynamic foreground detection via temporal frame differencing & thresholding
        detected_boxes: list[BBox] = []
        if self._prev_frame_gray is not None and self._prev_frame_gray.shape == gray.shape:
            # Check for global camera scroll / screen transitions.
            # If the screen scrolls or changes scene, a huge portion of the screen changes.
            # When that happens, treating the whole frame as individual entities blows up CPU,
            # drops FPS, and floods the screen with hundreds of false target boxes!
            temp_diff = cv2.absdiff(gray, self._prev_frame_gray)
            mean_diff = float(np.mean(temp_diff))
            if mean_diff > self.config.scroll_threshold:
                # Camera scrolled significantly: skip foreground entity extraction for this tick
                # to allow the background to settle and maintain target FPS.
                self._prev_frame_gray = gray
                return self.entities

            _, diff_mask = cv2.threshold(temp_diff, 20, 255, cv2.THRESH_BINARY)
            combined_mask = diff_mask
        else:
            combined_mask = np.zeros_like(gray)

        # Mask out player bbox to prevent self-detection
        if player_bbox is not None:
            px1 = max(0, player_bbox.x - 4)
            py1 = max(0, player_bbox.y - 4)
            px2 = min(w, player_bbox.x2 + 4)
            py2 = min(h, player_bbox.y2 + 4)
            combined_mask[py1:py2, px1:px2] = 0

        # Mask out HUD regions
        for roi in (score_region, game_over_region):
            if roi is not None:
                rx1 = max(0, min(roi.left, w))
                ry1 = max(0, min(roi.top, h))
                rx2 = max(rx1, min(roi.left + roi.width, w))
                ry2 = max(ry1, min(roi.top + roi.height, h))
                combined_mask[ry1:ry2, rx1:rx2] = 0

        # Morphological noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Detect candidate bounding boxes
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_boxes: list[BBox] = []

        # If too many contours are detected, it's a parallax background shift or scene flash
        if len(contours) <= self.config.max_active_entities * 2:
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                area = bw * bh
                if self.config.min_entity_area <= area <= self.config.max_entity_area:
                    detected_boxes.append(BBox(x, y, bw, bh))

        # Cap max detections to prevent CPU overload and FPS collapse
        if len(detected_boxes) > self.config.max_active_entities:
            # Sort by area or proximity, keep top max_active_entities
            detected_boxes.sort(key=lambda b: b.area, reverse=True)
            detected_boxes = detected_boxes[: self.config.max_active_entities]

        # Associate detections with existing tracks
        self._associate_and_update(detected_boxes, player_bbox)

        self._prev_frame_gray = gray
        return self.entities

    def _associate_and_update(
        self, detections: list[BBox], player_bbox: BBox | None
    ) -> None:
        """Associate detections with existing tracked entities via nearest neighbor."""
        matched_track_ids: set[int] = set()
        unmatched_detections: list[BBox] = []

        for det in detections:
            best_id: int | None = None
            best_dist = float("inf")

            for t_id, entity in self._tracked_entities.items():
                if t_id in matched_track_ids:
                    continue
                dist = entity.bbox.distance_to(det)
                if dist < 80.0 and dist < best_dist:
                    best_dist = dist
                    best_id = t_id

            if best_id is not None:
                matched_track_ids.add(best_id)
                entity = self._tracked_entities[best_id]
                old_c = entity.bbox.center
                new_c = det.center
                # Update velocity: instant velocity when new, smoothed EMA when mature
                raw_vx = new_c[0] - old_c[0]
                raw_vy = new_c[1] - old_c[1]
                if entity.age <= 1:
                    entity.velocity = (raw_vx, raw_vy)
                else:
                    entity.velocity = (
                        0.5 * entity.velocity[0] + 0.5 * raw_vx,
                        0.5 * entity.velocity[1] + 0.5 * raw_vy,
                    )
                entity.bbox = det
                entity.age += 1
                entity.disappeared = 0
                entity.trajectory.append(new_c)
                if len(entity.trajectory) > self.config.history_len:
                    entity.trajectory.pop(0)
                # Classify entity
                self._classify_entity(entity, player_bbox)
            else:
                unmatched_detections.append(det)

        # Increment disappeared count for unmatched tracks
        to_delete = []
        for t_id, entity in self._tracked_entities.items():
            if t_id not in matched_track_ids:
                entity.disappeared += 1
                if entity.disappeared > 5:
                    to_delete.append(t_id)

        for t_id in to_delete:
            del self._tracked_entities[t_id]

        # Register new tracks
        for det in unmatched_detections:
            new_entity = TrackedEntity(
                entity_id=self._next_id,
                bbox=det,
                velocity=(0.0, 0.0),
                kind=EntityKind.UNKNOWN,
                confidence=0.5,
                age=1,
                trajectory=[det.center],
            )
            self._classify_entity(new_entity, player_bbox)
            self._tracked_entities[self._next_id] = new_entity
            self._next_id += 1

    def _classify_entity(self, entity: TrackedEntity, player_bbox: BBox | None) -> None:
        """Classify dynamic entity into projectile, threat, or collectible."""
        speed = math.hypot(entity.velocity[0], entity.velocity[1])
        area = entity.bbox.area

        # Projectile: fast speed (> 7 px/step) and relatively small
        if speed > 7.0 and area < 800:
            entity.kind = EntityKind.PROJECTILE
            entity.confidence = min(0.95, 0.5 + speed / 20.0)
            return

        if player_bbox is not None:
            ec = entity.bbox.center
            pc = player_bbox.center
            rel_x = pc[0] - ec[0]
            rel_y = pc[1] - ec[1]
            dist = math.hypot(rel_x, rel_y)

            if dist > 0.1 and speed > 0.5:
                # Dot product between entity velocity and vector to player
                dot = (entity.velocity[0] * rel_x + entity.velocity[1] * rel_y) / (speed * dist)
                # Threat: moving towards player (dot > 0.3) or fast horizontal patrol
                if dot > 0.3 or (abs(entity.velocity[0]) > 2.0 and abs(entity.velocity[1]) < 1.0):
                    entity.kind = EntityKind.THREAT
                    entity.confidence = min(0.9, 0.5 + dot * 0.4)
                    return

        # Collectible: slow falling or floating, or upward drift
        if entity.velocity[1] > 0.5 and abs(entity.velocity[0]) < 1.0:
            entity.kind = EntityKind.COLLECTIBLE
            entity.confidence = 0.7
            return

        # Default classification if moving
        if speed > 1.0:
            entity.kind = EntityKind.THREAT
            entity.confidence = 0.5
        else:
            entity.kind = EntityKind.UNKNOWN
            entity.confidence = 0.4

    def check_collisions(
        self,
        player_bbox: BBox | None,
        points_gained: float = 0.0,
        terminated: bool = False,
    ) -> list[dict[str, Any]]:
        """Perform causal collision analysis: correlates intersections with score/death feedback."""
        if player_bbox is None:
            return []

        events: list[dict[str, Any]] = []
        for entity in self.entities:
            dist = entity.bbox.distance_to(player_bbox)
            is_intersecting = (
                entity.bbox.intersects(player_bbox)
                or dist < self.config.collision_distance_threshold
            )

            if is_intersecting:
                outcome = "neutral"
                causal_feedback = 0.0
                if terminated:
                    outcome = "death_collision"
                    causal_feedback = -100.0
                    entity.kind = EntityKind.THREAT
                    entity.confidence = 1.0
                elif points_gained > 0:
                    outcome = "collectible_pickup"
                    causal_feedback = points_gained
                    entity.kind = EntityKind.COLLECTIBLE
                    entity.confidence = 1.0

                event = {
                    "entity_id": entity.entity_id,
                    "kind": entity.kind.value,
                    "distance": dist,
                    "outcome": outcome,
                    "feedback": causal_feedback,
                }
                events.append(event)
                self._collision_history.append(event)
                if len(self._collision_history) > 50:
                    self._collision_history.pop(0)

        return events
