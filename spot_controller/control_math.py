"""Pure-math helpers for the stance controller (no ROS dependency)."""
import numpy as np

from .constants import webots_to_ik


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def rot_error(R_des, R):
    """Orientation error vector in the body frame (0 when R == R_des)."""
    E = 0.5 * (R_des.T @ R - R.T @ R_des)
    return np.array([E[2, 1], E[0, 2], E[1, 0]])


def leg_jacobian(kin, leg, angles, h=1e-6):
    """3x3 d(foot position)/d(joint angles) of one leg, body-aligned axes (Webots convention)."""
    def foot(a):
        return np.array(kin.legs[leg].compute_fk(*webots_to_ik(*a)))

    J = np.zeros((3, 3))
    base = np.array(angles, dtype=float)
    for k in range(3):
        plus, minus = base.copy(), base.copy()
        plus[k] += h
        minus[k] -= h
        J[:, k] = (foot(plus) - foot(minus)) / (2.0 * h)
    return J
