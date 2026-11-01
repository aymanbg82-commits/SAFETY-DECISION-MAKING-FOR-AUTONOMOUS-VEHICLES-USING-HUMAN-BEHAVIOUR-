"""
Unit tests for 2D geometry, Oriented Bounding Box (OBB),
Separating Axis Theorem (SAT) collision detection, and metric TTC.
"""

import pytest
import math
import numpy as np
from src.simulation.geometry import (
    OrientedBoundingBox,
    check_obb_collision,
    compute_obb_separation_distance,
    calculate_physical_ttc
)


def test_obb_corners_axis_aligned():
    # Box centered at (0, 0), length=4, width=2, heading=0
    box = OrientedBoundingBox(0.0, 0.0, 4.0, 2.0, 0.0)
    corners = box.get_corners()
    # Front-Right: (2, -1), Front-Left: (2, 1), Rear-Left: (-2, 1), Rear-Right: (-2, -1)
    expected = np.array([
        [2.0, -1.0],
        [2.0, 1.0],
        [-2.0, 1.0],
        [-2.0, -1.0]
    ])
    np.testing.assert_allclose(corners, expected, atol=1e-5)


def test_obb_collision_detection_overlapping():
    box1 = OrientedBoundingBox(0.0, 0.0, 4.0, 2.0, 0.0)
    box2 = OrientedBoundingBox(2.0, 0.5, 4.0, 2.0, 0.0)  # Overlaps in both X and Y
    assert check_obb_collision(box1, box2) is True


def test_obb_collision_detection_disjoint():
    box1 = OrientedBoundingBox(0.0, 0.0, 4.0, 2.0, 0.0)
    box2 = OrientedBoundingBox(10.0, 0.0, 4.0, 2.0, 0.0)  # 6m clearance in X
    assert check_obb_collision(box1, box2) is False


def test_obb_collision_detection_rotated():
    # Two boxes that would overlap if axis-aligned, but rotate away from each other
    box1 = OrientedBoundingBox(0.0, 0.0, 4.0, 1.0, math.pi / 4)
    box2 = OrientedBoundingBox(3.0, 3.0, 4.0, 1.0, math.pi / 4)
    assert check_obb_collision(box1, box2) is False

    # Crossing perpendicular boxes
    box_a = OrientedBoundingBox(0.0, 0.0, 6.0, 2.0, 0.0)
    box_b = OrientedBoundingBox(0.0, 0.0, 6.0, 2.0, math.pi / 2)
    assert check_obb_collision(box_a, box_b) is True


def test_obb_separation_distance():
    box1 = OrientedBoundingBox(0.0, 0.0, 4.0, 2.0, 0.0)
    box2 = OrientedBoundingBox(10.0, 0.0, 4.0, 2.0, 0.0)
    dist = compute_obb_separation_distance(box1, box2)
    assert dist > 0.0
    # Overlapping boxes should return 0.0
    overlap_box = OrientedBoundingBox(1.0, 0.0, 4.0, 2.0, 0.0)
    assert compute_obb_separation_distance(box1, overlap_box) == 0.0


def test_calculate_physical_ttc():
    # Ego at (0, 0) moving forward at 10 m/s
    # Actor at (50, 0) stationary (0 m/s)
    # Distance to contact ~ 48m, closing speed 10 m/s -> TTC ~ 4.8s
    ttc = calculate_physical_ttc(
        ego_pos=(0.0, 0.0),
        ego_vel=(10.0, 0.0),
        actor_pos=(50.0, 0.0),
        actor_vel=(0.0, 0.0)
    )
    assert ttc is not None
    assert 4.0 < ttc < 5.0

    # Diverging paths (actor moving away faster than ego)
    ttc_diverging = calculate_physical_ttc(
        ego_pos=(0.0, 0.0),
        ego_vel=(5.0, 0.0),
        actor_pos=(50.0, 0.0),
        actor_vel=(20.0, 0.0)
    )
    assert ttc_diverging is None
