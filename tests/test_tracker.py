from __future__ import annotations

import cv2
import numpy as np

from ai_player.config import Region, TrackerConfig
from ai_player.tracker import BBox, ControllabilityProbe, EntityKind, EntityTracker, TrackedEntity


def test_bbox_properties_and_intersection():
    b1 = BBox(10, 10, 20, 20)
    b2 = BBox(20, 20, 20, 20)
    b3 = BBox(100, 100, 10, 10)

    assert b1.area == 400
    assert b1.center == (20.0, 20.0)
    assert b1.x2 == 30 and b1.y2 == 30
    assert b1.intersects(b2) is True
    assert b1.intersects(b3) is False
    assert b1.distance_to(b2) > 0.0
    assert b1.iou(b2) > 0.0
    assert b1.iou(b3) == 0.0
    assert isinstance(b1.to_region(), Region)


def test_controllability_probe_discovery_with_synthetic_motion():
    cfg = TrackerConfig(probe_actions=3, min_entity_area=50, max_entity_area=2000)
    probe = ControllabilityProbe(cfg)

    # Simulate frame sequence: an avatar block moves to the right when action "right" is applied
    w, h = 320, 240
    frame0 = np.full((h, w, 3), 30, dtype=np.uint8)
    cv2.rectangle(frame0, (50, 100), (80, 130), (255, 255, 255), -1)

    frame1 = np.full((h, w, 3), 30, dtype=np.uint8)
    cv2.rectangle(frame1, (65, 100), (95, 130), (255, 255, 255), -1)

    probe.record_probe_step("idle", frame0)
    probe.record_probe_step("right", frame1)

    bbox = probe.analyze_probes()
    assert bbox is not None
    assert probe.is_calibrated is True
    assert probe.confidence > 0.0
    # Center y should be around 115, x around the moving boundary [45, 95]
    assert 45 <= bbox.x <= 95
    assert 90 <= bbox.y <= 120

    # Test update_player_location
    frame2 = np.full((h, w, 3), 30, dtype=np.uint8)
    cv2.rectangle(frame2, (75, 100), (105, 130), (255, 255, 255), -1)
    updated_box = probe.update_player_location(frame2)
    assert updated_box is not None
    assert updated_box.x >= bbox.x


def test_controllability_probe_fallback():
    probe = ControllabilityProbe()
    probe.record_probe_step("unknown_action", np.zeros((100, 100), dtype=np.uint8))
    f2 = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(f2, (40, 40), (60, 60), 255, -1)
    probe.record_probe_step("unknown_action", f2)

    bbox = probe.analyze_probes()
    assert bbox is not None
    assert probe.is_calibrated is True


def test_entity_tracker_detection_and_classification():
    cfg = TrackerConfig(min_entity_area=20, max_entity_area=1000)
    tracker = EntityTracker(cfg)

    player_box = BBox(50, 100, 20, 20)

    # Frame 1: empty background
    f1 = np.full((240, 320, 3), 20, dtype=np.uint8)
    tracker.update(f1, player_box)

    # Frame 2: small fast projectile moving left rapidly towards player
    f2 = np.full((240, 320, 3), 20, dtype=np.uint8)
    cv2.rectangle(f2, (200, 100), (210, 108), (200, 200, 200), -1)
    tracker.update(f2, player_box)
    assert len(tracker.entities) >= 1
    initial_ent = tracker.entities[0]
    assert initial_ent.entity_id == 1

    # Frame 3: moved from 200 to 180 (vx = -20 px/frame)
    # Note: In Frame 2, the entity appeared at 200 (registered as new entity, velocity=(0,0))
    # Between Frame 2 and Frame 3, diff has TWO blobs: the old position 200 and the new position 180.
    # To test track association and velocity estimation over time:
    f3 = np.full((240, 320, 3), 20, dtype=np.uint8)
    cv2.rectangle(f3, (180, 100), (190, 108), (200, 200, 200), -1)
    # Update with f3: detect motion from f2 -> f3
    _ = tracker.update(f3, player_box)

    # Frame 4: continues to move from 180 to 160 (vx = -20 px/frame)
    f4 = np.full((240, 320, 3), 20, dtype=np.uint8)
    cv2.rectangle(f4, (160, 100), (170, 108), (200, 200, 200), -1)
    entities = tracker.update(f4, player_box)

    assert len(entities) >= 1
    # Find the entity that has moved
    ent = max(entities, key=lambda e: abs(e.velocity[0]))
    assert ent.kind in {EntityKind.PROJECTILE, EntityKind.THREAT}
    assert ent.velocity[0] < 0.0


def test_entity_tracker_causal_collisions():
    tracker = EntityTracker()
    player_box = BBox(50, 50, 30, 30)

    # Fake a tracked entity colliding with player
    colliding_entity = TrackedEntity(
        entity_id=1,
        bbox=BBox(55, 55, 20, 20),
        velocity=(-5.0, 0.0),
        kind=EntityKind.THREAT,
    )
    tracker._tracked_entities[1] = colliding_entity

    # Case 1: Death collision
    events = tracker.check_collisions(player_box, points_gained=0.0, terminated=True)
    assert len(events) == 1
    assert events[0]["outcome"] == "death_collision"
    assert events[0]["feedback"] == -100.0

    # Case 2: Collectible pickup
    colliding_entity2 = TrackedEntity(
        entity_id=2,
        bbox=BBox(55, 55, 20, 20),
        velocity=(0.0, 1.0),
        kind=EntityKind.COLLECTIBLE,
    )
    tracker._tracked_entities[2] = colliding_entity2
    events2 = tracker.check_collisions(player_box, points_gained=5.0, terminated=False)
    assert any(e["outcome"] == "collectible_pickup" for e in events2)
