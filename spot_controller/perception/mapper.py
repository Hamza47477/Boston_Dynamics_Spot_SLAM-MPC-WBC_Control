"""Mapping on top of SLAM poses: depth frames -> voxel map, rebuilt when loop closure moves keyframes.

Every integrated frame is stored as (reference keyframe, pose relative to it, points in the camera
frame). SLAM keeps optimising the keyframe poses; whenever they move noticeably the whole map is rebuilt
from the stored frames at the corrected poses, so walls seen twice end up in the same place instead
of being smeared by drift.
"""
import time

import numpy as np

from .geometry import inv_T, T_body_camcv
from .voxel_map import VoxelMap


class Mapper:

    def __init__(self, intr, resolution=0.10, stride=6, max_range=8.0, robot_height=0.48,
                 min_move=0.08, min_turn=0.05, max_gap=1.0, rebuild_move=0.03):
        self.intr = intr
        self.vm = VoxelMap(resolution=resolution)
        self.stride, self.max_range = stride, max_range
        self.robot_height = robot_height
        self.min_move, self.min_turn, self.max_gap, self.rebuild_move = min_move, min_turn, max_gap, rebuild_move
        self.T_base_cam = T_body_camcv()
        self.frames = []                       # dict(kf, T_kf_cam, pts)
        self.kf_poses = []                     # camera poses of the keyframes (map frame)
        self.last = None                       # (pose, stamp) of the last integrated frame
        self.floor_samples = []
        self.floor_z = 0.0
        self.rebuilds = 0
        self.ms_integrate = 0.0
        self._u, self._v = np.meshgrid(np.arange(stride // 2, intr.width, stride),
                                       np.arange(stride // 2, intr.height, stride))

    def points_cam(self, depth, boxes=None):
        """Subsample the depth image into camera-frame points (N, 3), skipping masked pixels."""
        z = depth[self._v, self._u]
        ok = (z > 0.3) & (z < self.max_range)
        if boxes:
            for u0, v0, u1, v1 in boxes:
                ok &= ~((self._u >= u0) & (self._u <= u1) & (self._v >= v0) & (self._v <= v1))
        return self.intr.backproject(self._u[ok].astype(np.float64), self._v[ok].astype(np.float64),
                                     z[ok].astype(np.float64))

    def update_floor(self, T_w_cam):
        T_base = T_w_cam @ inv_T(self.T_base_cam)
        self.floor_samples.append(T_base[2, 3] - self.robot_height)
        self.floor_samples = self.floor_samples[-300:]
        self.floor_z = float(np.median(self.floor_samples))

    def _moved(self, T, stamp):
        if self.last is None:
            return True
        dT = inv_T(self.last[0]) @ T
        turn = np.arccos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1))
        return (np.linalg.norm(dT[:3, 3]) > self.min_move or turn > self.min_turn
                or stamp - self.last[1] > self.max_gap)

    def add_frame(self, stamp, kf_id, T_kf_cam, T_w_cam, depth, boxes=None):
        """Integrate a frame if the camera moved enough. Returns True if it was integrated."""
        self.update_floor(T_w_cam)
        if not self._moved(T_w_cam, stamp):
            return False
        t0 = time.perf_counter()
        pts = self.points_cam(depth, boxes)
        self._integrate(T_w_cam, pts)
        self.frames.append(dict(kf=kf_id, T_kf_cam=T_kf_cam, pts=pts.astype(np.float16)))
        self.last = (T_w_cam, stamp)
        self.ms_integrate = 0.8 * self.ms_integrate + 0.2 * (time.perf_counter() - t0) * 1e3
        return True

    def _integrate(self, T_w_cam, pts_cam):
        pw = pts_cam @ T_w_cam[:3, :3].T + T_w_cam[:3, 3]
        self.vm.integrate(T_w_cam[:3, 3], pw, self.max_range)

    def update_keyframes(self, poses):
        """New keyframe poses (list of 4x4). Rebuild the map if any existing keyframe moved."""
        moved = False
        for i, T in enumerate(poses):
            if i < len(self.kf_poses) and np.linalg.norm(T[:3, 3] - self.kf_poses[i][:3, 3]) > self.rebuild_move:
                moved = True
                break
        self.kf_poses = [T.copy() for T in poses]
        if moved:
            self.rebuild()
        return moved

    def rebuild(self):
        self.vm.clear()
        for fr in self.frames:
            if fr['kf'] < len(self.kf_poses):
                T = self.kf_poses[fr['kf']] @ fr['T_kf_cam']
                self._integrate(T, fr['pts'].astype(np.float64))
        self.rebuilds += 1

    def grid_2d(self):
        return self.vm.occupancy_2d(self.floor_z)


def points_to_scan(pts_base, robot_height, angle_min=-0.72, angle_max=0.72, n_beams=116,
                   band=(0.15, 1.4), floor_tol=0.12, max_range=4.0):
    """Pseudo laser scan from camera points in the base frame (x forward, y left, z up).

    A beam reports the nearest point inside the obstacle height band. A beam that sees the floor but no
    obstacle reports +inf (free: Nav2 clears space along it, so a person who walks away leaves nothing
    behind). A beam that saw neither is NaN (no information).
    """
    ranges = np.full(n_beams, np.nan, np.float32)
    if len(pts_base) == 0:
        return ranges
    h = pts_base[:, 2] + robot_height                      # height above the floor
    rng = np.hypot(pts_base[:, 0], pts_base[:, 1])
    bearing = np.arctan2(pts_base[:, 1], pts_base[:, 0])
    beam = np.floor((bearing - angle_min) / ((angle_max - angle_min) / n_beams)).astype(int)
    ok = (beam >= 0) & (beam < n_beams) & (rng < max_range + 1.0)
    obstacle = ok & (h > band[0]) & (h < band[1])
    floor = ok & (np.abs(h) < floor_tol)
    seen_floor = np.zeros(n_beams, bool)
    seen_floor[beam[floor]] = True
    ranges[seen_floor] = np.inf
    nearest = np.full(n_beams, np.inf)
    np.minimum.at(nearest, beam[obstacle], rng[obstacle])
    hit = np.isfinite(nearest)
    ranges[hit] = nearest[hit]
    return ranges
