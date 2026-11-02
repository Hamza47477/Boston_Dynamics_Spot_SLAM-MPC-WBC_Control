"""Pure-math helpers for body attitude estimation (no ROS dependency)."""
import math

import numpy as np


def fit_ground_plane(feet):
    """Least-squares plane z = a*x + b*y + c through the foot points (body frame).

    Returns (height, roll, pitch) of the body relative to that plane. Assumes all four
    feet are on flat ground.
    """
    pts = np.array(feet)
    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    a, b, c = np.linalg.lstsq(A, pts[:, 2], rcond=None)[0]
    n = np.array([-a, -b, 1.0])
    n /= np.linalg.norm(n)                 # world "up" expressed in the body frame
    height = -c * n[2]
    pitch = math.asin(max(-1.0, min(1.0, -n[0])))
    roll = math.atan2(n[1], n[2])
    return height, roll, pitch


def quat_to_roll_pitch(x, y, z, w):
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch
