"""
2D Geometric primitives, Oriented Bounding Boxes (OBB),
Separating Axis Theorem (SAT) collision detection, and metric separation distance.
"""

import math
from typing import Tuple, List, Optional
import numpy as np


class OrientedBoundingBox:
    """
    Oriented 2D Bounding Box represented by center (x, y), length (L), width (W),
    and heading orientation angle theta (radians counter-clockwise from +X axis).
    """

    def __init__(self, x: float, y: float, length: float, width: float, heading: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.length = float(length)
        self.width = float(width)
        self.heading = float(heading)

    def get_corners(self) -> np.ndarray:
        """
        Compute the 4 metric corner coordinates (x, y) of the oriented rectangle in SI metres.
        Returns shape (4, 2) in order: Front-Right, Front-Left, Rear-Left, Rear-Right.
        """
        c = math.cos(self.heading)
        s = math.sin(self.heading)
        hl = self.length / 2.0
        hw = self.width / 2.0

        # Local offsets relative to center
        # Front: +x_local, Rear: -x_local, Left: +y_local, Right: -y_local
        corners_local = [
            (hl, -hw),   # Front-Right
            (hl, hw),    # Front-Left
            (-hl, hw),   # Rear-Left
            (-hl, -hw),  # Rear-Right
        ]

        corners = []
        for lx, ly in corners_local:
            gx = self.x + (lx * c - ly * s)
            gy = self.y + (lx * s + ly * c)
            corners.append([gx, gy])

        return np.array(corners, dtype=np.float64)

    def get_axes(self) -> List[np.ndarray]:
        """
        Get the 2 unique normal projection axes for Separating Axis Theorem (SAT).
        """
        c = math.cos(self.heading)
        s = math.sin(self.heading)
        # Unit axis along longitudinal heading and perpendicular lateral axis
        axis_longitudinal = np.array([c, s], dtype=np.float64)
        axis_lateral = np.array([-s, c], dtype=np.float64)
        return [axis_longitudinal, axis_lateral]


def check_obb_collision(box1: OrientedBoundingBox, box2: OrientedBoundingBox) -> bool:
    """
    Exact 2D collision detection between two oriented bounding boxes using
    the Separating Axis Theorem (SAT).
    Two convex polygons do NOT collide if and only if there exists a projection axis
    along which their 1D projected intervals are disjoint.
    """
    axes = box1.get_axes() + box2.get_axes()
    corners1 = box1.get_corners()
    corners2 = box2.get_corners()

    for axis in axes:
        # Project all corners of both boxes onto the current axis
        proj1 = np.dot(corners1, axis)
        proj2 = np.dot(corners2, axis)

        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)

        # Check for separation interval
        if max1 < min2 or max2 < min1:
            return False  # Separating axis found -> No collision

    return True  # Overlap on all 4 axes -> Collision


def compute_obb_separation_distance(box1: OrientedBoundingBox, box2: OrientedBoundingBox) -> float:
    """
    Compute conservative separation distance (in metres) between two oriented bounding boxes.
    If boxes collide, distance is 0.0.
    Otherwise, computes minimum Euclidean distance between corner sets and edge midpoints.
    """
    if check_obb_collision(box1, box2):
        return 0.0

    corners1 = box1.get_corners()
    corners2 = box2.get_corners()

    # Minimum corner-to-corner distance
    diff = corners1[:, np.newaxis, :] - corners2[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    min_dist = float(np.min(dist_matrix))

    # Center-to-center distance minus maximum bounding radius
    radius1 = math.hypot(box1.length / 2.0, box1.width / 2.0)
    radius2 = math.hypot(box2.length / 2.0, box2.width / 2.0)
    center_dist = math.hypot(box1.x - box2.x, box1.y - box2.y)
    radial_clearance = max(0.0, center_dist - (radius1 + radius2))

    return max(radial_clearance, min_dist * 0.75)


def calculate_physical_ttc(
    ego_pos: Tuple[float, float],
    ego_vel: Tuple[float, float],
    actor_pos: Tuple[float, float],
    actor_vel: Tuple[float, float],
    ego_dim: Tuple[float, float] = (4.5, 2.0),
    actor_dim: Tuple[float, float] = (2.2, 0.9),
    max_ttc_horizon: float = 10.0
) -> Optional[float]:
    """
    Compute physical Time-To-Collision (TTC) under constant velocity assumptions.
    Solves for the earliest time t > 0 where distance between bounding centers
    collapses below the combined dimensional clearance.

    Returns:
        float: Time in seconds to collision if approaching, or None if trajectories diverge.
    """
    rx = actor_pos[0] - ego_pos[0]
    ry = actor_pos[1] - ego_pos[1]
    rvx = actor_vel[0] - ego_vel[0]
    rvy = actor_vel[1] - ego_vel[1]

    # Effective contact radius
    r_contact = 0.5 * (math.hypot(*ego_dim) + math.hypot(*actor_dim)) * 0.7

    # Center-to-center distance: d(t)^2 = (rx + rvx*t)^2 + (ry + rvy*t)^2
    # Expanding: (rvx^2 + rvy^2)*t^2 + 2*(rx*rvx + ry*rvy)*t + (rx^2 + ry^2 - r_contact^2) = 0
    a = rvx * rvx + rvy * rvy
    b = 2.0 * (rx * rvx + ry * rvy)
    c = (rx * rx + ry * ry) - (r_contact * r_contact)

    if c <= 0:
        return 0.0  # Already touching or within collision boundary

    if a < 1e-6:
        return None  # Relative velocity is zero (parallel identical motion)

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        return None  # Trajectories never enter the collision radius

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # We want the smallest positive time
    valid_times = [t for t in (t1, t2) if 0.0 <= t <= max_ttc_horizon]
    if not valid_times:
        return None

    return min(valid_times)
