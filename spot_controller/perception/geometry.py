"""Small SE(3) and camera-frame helpers shared by the perception stack.

Frames
  world / map   z up. `map` is the SLAM frame (origin at the start pose, like `odom`).
  base_link     robot body, x forward, y left, z up (FLU).
  cam_flu       the left camera in Webots convention, same axes as base_link.
  cam_cv        OpenCV convention: x right, y down, z forward (used for projection).
"""
import math

import numpy as np

from ..stereo_sensors import BASELINE, LEFT_IN_BODY      # noqa: F401  (re-exported)

# x_flu = z_cv, y_flu = -x_cv, z_flu = -y_cv
R_FLU_FROM_CV = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T):
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_R(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    """Return (x, y, z, w)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        return ((R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s, 0.25 / s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)


def T_body_camcv():
    """Transform from the OpenCV camera frame to the body frame (fixed mount)."""
    return make_T(R_FLU_FROM_CV, np.array(LEFT_IN_BODY))


class Intrinsics:
    """Pinhole model of the rectified left camera (no distortion)."""

    def __init__(self, fx, cx, cy, width, height, baseline=BASELINE):
        self.fx, self.cx, self.cy = fx, cx, cy
        self.width, self.height = width, height
        self.baseline = baseline
        self.K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]])

    @classmethod
    def from_info(cls, msg):
        return cls(msg.k[0], msg.k[2], msg.k[5], msg.width, msg.height)

    def backproject(self, u, v, z):
        """Pixels (u, v) with depth z -> points (N, 3) in the OpenCV camera frame."""
        return np.stack([(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fx, z], axis=-1)

    def project(self, pts):
        """Points (N, 3) in the camera frame -> pixels (N, 2) and depth."""
        z = pts[:, 2]
        return np.stack([self.fx * pts[:, 0] / z + self.cx, self.fx * pts[:, 1] / z + self.cy],
                        axis=-1), z


class PoseBuffer:
    """Time-indexed 4x4 poses with interpolation (linear translation, normalised-lerp rotation)."""

    def __init__(self, maxlen=2000):
        self.t, self.T = [], []
        self.maxlen = maxlen

    def add(self, t, T):
        self.t.append(t)
        self.T.append(T)
        if len(self.t) > self.maxlen:
            del self.t[:len(self.t) - self.maxlen]
            del self.T[:len(self.T) - self.maxlen]

    def get(self, t):
        """Pose at time t, or None if t is outside the buffered interval (small slack allowed)."""
        import bisect
        if not self.t or t < self.t[0] - 0.05 or t > self.t[-1] + 0.05:
            return None
        i = bisect.bisect_left(self.t, t)
        if i == 0:
            return self.T[0]
        if i >= len(self.t):
            return self.T[-1]
        t0, t1 = self.t[i - 1], self.t[i]
        a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        q0, q1 = np.array(R_to_quat(self.T[i - 1][:3, :3])), np.array(R_to_quat(self.T[i][:3, :3]))
        if np.dot(q0, q1) < 0:
            q1 = -q1
        q = (1 - a) * q0 + a * q1
        return make_T(quat_to_R(*q), (1 - a) * self.T[i - 1][:3, 3] + a * self.T[i][:3, 3])
