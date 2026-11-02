"""Stereo visual SLAM (RGB-D style) on the CPU: ORB tracking, keyframes, loop closure, pose graph.

Per frame
  1. ORB features on the left image (dynamic objects masked out); their 3D position comes from the
     stereo depth image.
  2. Prediction: the previous pose moved by the leg-odometry/EKF motion (robust to motion blur).
  3. Local map: landmarks of the last few keyframes, stored in keyframe coordinates, are projected into
     the predicted view and matched to the new features (descriptor + pixel gate).
  4. PnP + RANSAC (+ LM refinement) gives the camera pose; inliers are counted.
  5. Keyframe when the view changed enough (few tracked landmarks, distance, rotation or time).
At a keyframe
  6. New landmarks from unmatched features with depth; an odometry edge is added to the pose graph.
  7. Loop closure: older keyframes that are close in space or look similar are checked by descriptor
     matching and a RANSAC rigid alignment of the matched 3D points. A verified loop adds an edge and
     the 4-DoF pose graph (see pose_graph.py) is optimised: every keyframe, and with it every
     landmark, moves, and the drift that accumulated over the loop is removed.
If tracking fails, the pose follows the odometry and relocalisation against all keyframes is tried.
"""
import math
import time

import cv2
import numpy as np

from .geometry import inv_T, make_T, R_FLU_FROM_CV, rot_z, T_body_camcv, wrap, yaw_of
from .pose_graph import PoseGraph
from .rigid import ransac_rigid

DEFAULTS = dict(
    n_features=1000, local_kfs=8, max_new_points=450,
    min_depth=0.5, max_depth=10.0,
    gate_px=22.0, gate_px_wide=45.0, match_dist=70, ratio=0.85,
    pnp_reproj=3.0, pnp_iters=80, min_inliers=15, max_jump_m=0.6, max_jump_rad=0.5,
    kf_min_frames=3, kf_dist=0.35, kf_rot=0.3, kf_time=2.5, kf_track_ratio=0.6,
    loop_min_path=8.0, loop_cooldown=8, loop_radius=2.0, loop_radius_per_m=0.03, loop_yaw=1.0,
    loop_max_correction=0.5, loop_correction_per_m=0.03, loop_candidates=5,
    loop_ratio=0.8, loop_min_matches=25, loop_inlier_m=0.2, loop_min_inliers=30,
    sigma_odom_t=0.03, sigma_odom_psi=0.01, sigma_loop_t=0.06, sigma_loop_psi=0.02,
    lost_reloc_after=3, reloc_max_dist=1.5, reloc_min_inliers=35, lost_reinit_after=4,
    f2f_gate_px=80.0, weak_sigma_scale=4.0, gravity_align=True,
)

T_FLU_FROM_CV = make_T(R_FLU_FROM_CV.T, np.zeros(3))      # T_w_flu = T_w_cv @ this


def cv_to_flu(T_w_cv):
    return T_w_cv @ T_FLU_FROM_CV


def flu_to_cv(T_w_flu):
    return T_w_flu @ inv_T(T_FLU_FROM_CV)


class KeyFrame:
    def __init__(self, kf_id, stamp, T_w_cam, kps, desc, pts_cam, thumb):
        self.id, self.stamp, self.T_w_cam = kf_id, stamp, T_w_cam
        self.path = 0.0                     # distance travelled up to this keyframe (m)
        self.kps, self.desc, self.pts_cam, self.thumb = kps, desc, pts_cam, thumb


class PointStore:
    """Landmarks in the coordinates of the keyframe that created them."""

    def __init__(self):
        self.kf = np.zeros(0, np.int32)
        self.p = np.zeros((0, 3), np.float32)
        self.desc = np.zeros((0, 32), np.uint8)
        self.nobs = np.zeros(0, np.int32)

    def add(self, kf_id, p, desc):
        n = len(p)
        self.kf = np.concatenate([self.kf, np.full(n, kf_id, np.int32)])
        self.p = np.vstack([self.p, p.astype(np.float32)])
        self.desc = np.vstack([self.desc, desc])
        self.nobs = np.concatenate([self.nobs, np.ones(n, np.int32)])
        return np.arange(len(self.kf) - n, len(self.kf))

    def __len__(self):
        return len(self.kf)


class SlamResult:
    def __init__(self):
        self.state = 'init'
        self.T_w_cam = np.eye(4)
        self.n_matches = self.n_inliers = self.n_visible = 0
        self.keyframe = False
        self.kf_id = 0
        self.T_kf_cam = np.eye(4)
        self.loop = None
        self.graph_changed = False
        self.ms = {}
        self.f2f = False                                    # tracked against the previous frame only
        self.inlier_xy = np.zeros((0, 2), np.float32)       # tracked features (for the debug view)
        self.outlier_xy = np.zeros((0, 2), np.float32)


class StereoSlam:

    def __init__(self, intr, **params):
        self.intr = intr
        self.cfg = dict(DEFAULTS, **params)
        self.orb = cv2.ORB_create(nfeatures=self.cfg['n_features'], scaleFactor=1.2, nlevels=8,
                                  fastThreshold=10)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.T_base_cam = T_body_camcv()
        self.kfs = []
        self.points = PointStore()
        self.graph = PoseGraph()
        self.state = 'init'
        self.T_w_cam = np.eye(4)
        self.T_o_cam_prev = None
        self.T_rel_kf = np.eye(4)               # current camera relative to the last keyframe
        self.dr_rel = np.eye(4)                 # same, from dead reckoning only (used when lost)
        self.last_kf_stamp = 0.0
        self.frames_since_kf = 0
        self.lost_frames = 0
        self.velocity = np.eye(4)               # constant-velocity fallback (cam frame increment)
        self.loop_count = 0
        self.rng = np.random.default_rng(0)
        self._force_kf = False
        self._gray_for_thumb = None
        self.ref_inliers = 0                    # landmarks tracked when the last keyframe was made
        self.prev = None                        # previous frame (for the frame-to-frame fallback)
        self._weak_kf = False
        self.last_loop_kf = -10 ** 6

    # ------------------------------------------------------------------ features
    def _detect(self, gray, exclude):
        mask = None
        if exclude:
            mask = np.full(gray.shape, 255, np.uint8)
            for u0, v0, u1, v1 in exclude:
                mask[max(0, int(v0)):int(v1) + 1, max(0, int(u0)):int(u1) + 1] = 0
        kps, desc = self.orb.detectAndCompute(gray, mask)
        if desc is None or len(kps) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8), np.zeros(0)
        xy = np.array([k.pt for k in kps], np.float32)
        resp = np.array([k.response for k in kps], np.float32)
        return xy, desc, resp

    def _lift(self, xy, depth):
        """3D camera-frame position of each keypoint from the depth image (NaN without depth)."""
        h, w = depth.shape
        u = np.clip(np.round(xy[:, 0]).astype(int), 1, w - 2)
        v = np.clip(np.round(xy[:, 1]).astype(int), 1, h - 2)
        centre = depth[v, u]
        patch = np.stack([depth[v + dy, u + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=1)
        has = patch > 0
        zmax = np.where(has, patch, -np.inf).max(1)
        zmin = np.where(has, patch, np.inf).min(1)
        ok = (centre > self.cfg['min_depth']) & (centre < self.cfg['max_depth']) \
            & ((zmax - zmin) < 0.06 * centre + 0.05) & (has.sum(1) >= 6)
        pts = np.full((len(xy), 3), np.nan, np.float32)
        pts[ok] = self.intr.backproject(xy[ok, 0], xy[ok, 1], centre[ok]).astype(np.float32)
        return pts

    @staticmethod
    def _thumb(gray):
        t = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        t -= t.mean()
        return t / (np.linalg.norm(t) + 1e-6)

    # ------------------------------------------------------------------ local map
    def _local_points(self):
        ids = {k.id for k in self.kfs[-self.cfg['local_kfs']:]}
        sel = np.where(np.isin(self.points.kf, list(ids)))[0]
        if len(sel) == 0:
            return sel, np.zeros((0, 3))
        Pw = np.zeros((len(sel), 3))
        kfs = self.points.kf[sel]
        for kid in ids:
            m = kfs == kid
            if m.any():
                T = self.kfs[kid].T_w_cam
                Pw[m] = self.points.p[sel[m]].astype(np.float64) @ T[:3, :3].T + T[:3, 3]
        return sel, Pw

    def _match_local(self, xy, desc, T_pred, gate):
        sel, Pw = self._local_points()
        if len(sel) < 8 or len(desc) < 8:
            return None
        T_cw = inv_T(T_pred)
        Pc = Pw @ T_cw[:3, :3].T + T_cw[:3, 3]
        uv, z = self.intr.project(Pc)
        vis = (z > 0.3) & (z < self.cfg['max_depth'] + 2) & (uv[:, 0] > 0) & (uv[:, 0] < self.intr.width) \
            & (uv[:, 1] > 0) & (uv[:, 1] < self.intr.height)
        vi = np.where(vis)[0]
        if len(vi) < 8:
            return None
        knn = self.bf.knnMatch(np.ascontiguousarray(self.points.desc[sel[vi]]), desc, k=2)
        best = {}
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance > self.cfg['match_dist'] or m.distance > self.cfg['ratio'] * n.distance:
                continue
            q = vi[m.queryIdx]
            if np.linalg.norm(uv[q] - xy[m.trainIdx]) > gate:
                continue
            old = best.get(m.trainIdx)
            if old is None or m.distance < old[1]:
                best[m.trainIdx] = (q, m.distance)
        if not best:
            return None
        kp_idx = np.array(list(best.keys()))
        map_idx = np.array([best[k][0] for k in kp_idx])
        return sel[map_idx], kp_idx, Pw[map_idx], len(vi)

    def _pnp(self, obj, img, T_guess):
        T_cw = inv_T(T_guess)
        rvec, _ = cv2.Rodrigues(T_cw[:3, :3])
        tvec = T_cw[:3, 3].reshape(3, 1).copy()
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj.astype(np.float64), img.astype(np.float64), self.intr.K, None, rvec, tvec, True,
            iterationsCount=self.cfg['pnp_iters'], reprojectionError=self.cfg['pnp_reproj'],
            confidence=0.995, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inl is None or len(inl) < self.cfg['min_inliers']:
            return None
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl].astype(np.float64), img[inl].astype(np.float64),
                                          self.intr.K, None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        return inv_T(make_T(R, tvec.ravel())), inl

    # ------------------------------------------------------------------ keyframes
    def _make_keyframe(self, stamp, xy, desc, resp, pts, matched_kp, matched_map, T_w_cam, odom_rel,
                       weak=False):
        kf_id = len(self.kfs)
        kf = KeyFrame(kf_id, stamp, T_w_cam.copy(), xy, desc, pts, self._thumb(self._gray_for_thumb))
        if kf_id > 0:
            prev = self.kfs[-1]
            kf.path = prev.path + float(np.linalg.norm(T_w_cam[:3, 3] - prev.T_w_cam[:3, 3]))
        # landmarks from unmatched keypoints with depth, spread over the image (one per 32 px cell)
        have = np.isfinite(pts[:, 0])
        have[matched_kp] = False
        cand = np.where(have)[0]
        cand = cand[np.argsort(-resp[cand])]
        taken, chosen = set(), []
        for i in cand:
            cell = (int(xy[i, 0]) // 24, int(xy[i, 1]) // 24)
            if cell in taken:
                continue
            taken.add(cell)
            chosen.append(i)
            if len(chosen) >= self.cfg['max_new_points']:
                break
        chosen = np.array(chosen, int)
        if len(chosen):
            self.points.add(kf_id, pts[chosen], desc[chosen])
        if len(matched_map):
            self.points.nobs[matched_map] += 1
        self.kfs.append(kf)
        node = self.graph.add_node(cv_to_flu(T_w_cam))
        if kf_id > 0:
            Ti = cv_to_flu(self.kfs[kf_id - 1].T_w_cam)
            rel = inv_T(Ti) @ cv_to_flu(self.kfs[kf_id - 1].T_w_cam @ odom_rel)
            t = rel[:3, 3]
            psi = yaw_of(rel[:3, :3])
            n = np.linalg.norm(t)
            k = self.cfg['weak_sigma_scale'] if weak else 1.0       # dead-reckoned link: trust less
            self.graph.add_edge(kf_id - 1, node, t, psi,
                                k * self.cfg['sigma_odom_t'] * (1 + n), k * self.cfg['sigma_odom_psi'] * (1 + n))
        self.last_kf_stamp = stamp
        self.frames_since_kf = 0
        self.T_rel_kf = np.eye(4)
        self.dr_rel = np.eye(4)
        return kf

    # ------------------------------------------------------------------ loop closure
    def _verify(self, kf, cand):
        """Match kf against an older keyframe; return (T_cand_cur, n_inliers) or None."""
        have_a = np.isfinite(kf.pts_cam[:, 0])
        have_b = np.isfinite(cand.pts_cam[:, 0])
        ia, ib = np.where(have_a)[0], np.where(have_b)[0]
        if len(ia) < 20 or len(ib) < 20:
            return None
        knn = self.bf.knnMatch(np.ascontiguousarray(kf.desc[ia]), np.ascontiguousarray(cand.desc[ib]), k=2)
        qa, qb = [], []
        for pair in knn:
            if len(pair) == 2 and pair[0].distance < self.cfg['match_dist'] \
                    and pair[0].distance < self.cfg['loop_ratio'] * pair[1].distance:
                qa.append(ia[pair[0].queryIdx])
                qb.append(ib[pair[0].trainIdx])
        if len(qa) < self.cfg['loop_min_matches']:
            return None
        P = kf.pts_cam[qa].astype(np.float64)
        Q = cand.pts_cam[qb].astype(np.float64)
        res = ransac_rigid(P, Q, iters=200, thresh=self.cfg['loop_inlier_m'],
                           min_inliers=self.cfg['loop_min_inliers'], rng=self.rng)
        if res is None:
            return None
        R, t, inl = res
        if inl.sum() < 0.25 * len(qa):
            return None
        T = make_T(R, t)                                   # cand <- cur (3D-3D, coarse)
        # refine with the 2D-3D reprojection error (more precise than noisy depth points)
        qa_i, qb_i = np.array(qa)[inl], np.array(qb)[inl]
        T_cur_cand = inv_T(T)
        rvec, _ = cv2.Rodrigues(T_cur_cand[:3, :3])
        tvec = T_cur_cand[:3, 3].reshape(3, 1).copy()
        obj = cand.pts_cam[qb_i].astype(np.float64)
        img = kf.kps[qa_i].astype(np.float64)
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj, img, self.intr.K, None, rvec, tvec)
            Rr, _ = cv2.Rodrigues(rvec)
            T_new = inv_T(make_T(Rr, tvec.ravel()))
            proj, _ = self.intr.project(obj @ Rr.T + tvec.ravel())
            if np.median(np.linalg.norm(proj - img, axis=1)) < 3.0:
                T = T_new
        except cv2.error:
            pass
        return T, int(inl.sum())

    def _loop_candidates(self, kf):
        cfg = self.cfg
        if kf.id - getattr(self, 'last_loop_kf', -10 ** 6) < cfg['loop_cooldown']:
            return []
        old = [k for k in self.kfs[:-1] if kf.path - k.path >= cfg['loop_min_path']]
        if not old:
            return []
        scores = []
        p = kf.T_w_cam[:3, 3]
        yaw = yaw_of(cv_to_flu(kf.T_w_cam)[:3, :3])
        for k in old:
            # the odometry says we are near k; allow for the drift expected over the path in between
            allowed = cfg['loop_radius'] + cfg['loop_radius_per_m'] * (kf.path - k.path)
            d = np.linalg.norm(k.T_w_cam[:3, 3] - p)
            dyaw = abs(wrap(yaw - yaw_of(cv_to_flu(k.T_w_cam)[:3, :3])))
            if d > allowed or dyaw > cfg['loop_yaw']:
                continue
            scores.append((float(kf.thumb @ k.thumb), k))
        scores.sort(key=lambda s: -s[0])
        return [k for s, k in scores[:cfg['loop_candidates']]]

    def _try_loop(self, kf):
        for cand in self._loop_candidates(kf):
            res = self._verify(kf, cand)
            if res is None:
                continue
            T_cand_cur, n_inl = res
            T_now = inv_T(cand.T_w_cam) @ kf.T_w_cam               # what the trajectory currently says
            correction = np.linalg.norm(T_now[:3, 3] - T_cand_cur[:3, 3])
            if correction > self.cfg['loop_max_correction'] + self.cfg['loop_correction_per_m'] * (kf.path - cand.path):
                continue                                           # looks like perceptual aliasing
            T_rel_flu = inv_T(cv_to_flu(cand.T_w_cam)) @ cv_to_flu(cand.T_w_cam @ T_cand_cur)
            t = T_rel_flu[:3, 3]
            self.graph.add_edge(cand.id, kf.id, t, yaw_of(T_rel_flu[:3, :3]),
                                self.cfg['sigma_loop_t'], self.cfg['sigma_loop_psi'], loop=True)
            self.loop_count += 1
            self.last_loop_kf = kf.id
            return cand.id, n_inl
        return None

    def _apply_graph(self):
        """Copy optimised poses back to the keyframes and move the current pose with the last one."""
        for i, kf in enumerate(self.kfs):
            kf.T_w_cam = flu_to_cv(self.graph.pose(i))
        self.T_w_cam = self.kfs[-1].T_w_cam @ self.T_rel_kf

    # ------------------------------------------------------------------ relocalisation
    def _relocalize(self, xy, desc, pts, T_pred):
        best = None
        p = T_pred[:3, 3]
        order = sorted(self.kfs, key=lambda k: np.linalg.norm(k.T_w_cam[:3, 3] - p))[:6]
        for kf in order:
            ok = np.isfinite(kf.pts_cam[:, 0])
            idx = np.where(ok)[0]
            if len(idx) < 20:
                continue
            knn = self.bf.knnMatch(np.ascontiguousarray(desc), np.ascontiguousarray(kf.desc[idx]), k=2)
            qa, qb = [], []
            for pair in knn:
                if len(pair) == 2 and pair[0].distance < self.cfg['match_dist'] \
                        and pair[0].distance < self.cfg['loop_ratio'] * pair[1].distance:
                    qa.append(pair[0].queryIdx)
                    qb.append(idx[pair[0].trainIdx])
            if len(qa) < 20:
                continue
            Pw = kf.pts_cam[qb].astype(np.float64) @ kf.T_w_cam[:3, :3].T + kf.T_w_cam[:3, 3]
            res = self._pnp(Pw, xy[qa], kf.T_w_cam)
            if res and len(res[1]) >= self.cfg['reloc_min_inliers'] \
                    and np.linalg.norm(res[0][:3, 3] - T_pred[:3, 3]) < self.cfg['reloc_max_dist'] \
                    and (best is None or len(res[1]) > best[1]):
                best = (res[0], len(res[1]), kf.id)
        return best

    def _track_prev(self, xy, desc, T_pred):
        """Fallback: pose from matches with the previous frame's 3D points (no map needed)."""
        prev = self.prev
        idx = np.where(np.isfinite(prev['pts'][:, 0]))[0]
        if len(idx) < 25 or len(desc) < 25:
            return None
        knn = self.bf.knnMatch(np.ascontiguousarray(prev['desc'][idx]), desc, k=2)
        qa, qb = [], []
        for pair in knn:
            if len(pair) == 2 and pair[0].distance < self.cfg['match_dist'] \
                    and pair[0].distance < 0.8 * pair[1].distance:
                a, b = idx[pair[0].queryIdx], pair[0].trainIdx
                if np.linalg.norm(prev['xy'][a] - xy[b]) < self.cfg['f2f_gate_px']:
                    qa.append(a)
                    qb.append(b)
        if len(qa) < self.cfg['min_inliers']:
            return None
        T = prev['T']
        Pw = prev['pts'][qa].astype(np.float64) @ T[:3, :3].T + T[:3, 3]
        return self._pnp(Pw, xy[qb], T_pred)

    def _gravity_align(self, T_cam, T_o_base):
        """Keep x, y and heading from vision; take roll, pitch and height from the EKF.

        The IMU makes gravity (roll, pitch) and, on flat ground, the height observable with ~0.1 deg
        and ~1 cm accuracy, while a camera alone lets them drift. Standard visual-inertial practice.
        """
        T_b = T_cam @ inv_T(self.T_base_cam)
        yaw = yaw_of(T_b[:3, :3])
        yaw_o = yaw_of(T_o_base[:3, :3])
        R_rp = rot_z(-yaw_o) @ T_o_base[:3, :3]
        t = T_b[:3, 3].copy()
        t[2] = T_o_base[2, 3]
        return make_T(rot_z(yaw) @ R_rp, t) @ self.T_base_cam

    # ------------------------------------------------------------------ main entry
    def process(self, stamp, gray, depth, T_o_base=None, exclude=None):
        """Track one frame. gray: uint8 HxW, depth: float32 HxW (metres, 0 invalid)."""
        out = SlamResult()
        t_all = time.perf_counter()
        self._gray_for_thumb = gray
        t0 = time.perf_counter()
        xy, desc, resp = self._detect(gray, exclude)
        pts = self._lift(xy, depth)
        out.ms['features'] = (time.perf_counter() - t0) * 1e3
        T_o_cam = None if T_o_base is None else T_o_base @ self.T_base_cam

        if self.state == 'init':
            if np.isfinite(pts[:, 0]).sum() < 40:
                out.state = 'init'
                return out
            T0 = T_o_cam if T_o_cam is not None else np.eye(4)
            self.T_w_cam = T0.copy()
            self._make_keyframe(stamp, xy, desc, resp, pts, np.zeros(0, int), np.zeros(0, int), T0,
                                np.eye(4))
            self.state = 'tracking'
            self.T_o_cam_prev = T_o_cam
            out.state, out.keyframe, out.kf_id, out.T_w_cam = 'tracking', True, 0, self.T_w_cam.copy()
            return out

        # prediction from odometry (preferred) or constant velocity
        if T_o_cam is not None and self.T_o_cam_prev is not None:
            delta = inv_T(self.T_o_cam_prev) @ T_o_cam
        else:
            delta = self.velocity
        T_pred = self.T_w_cam @ delta

        t0 = time.perf_counter()
        match = self._match_local(xy, desc, T_pred, self.cfg['gate_px'])
        if match is None or len(match[1]) < self.cfg['min_inliers']:
            match = self._match_local(xy, desc, T_pred, self.cfg['gate_px_wide'])
        out.ms['match'] = (time.perf_counter() - t0) * 1e3

        solved = None
        if match is not None and len(match[1]) >= self.cfg['min_inliers']:
            map_idx, kp_idx, Pw, n_vis = match
            out.n_matches, out.n_visible = len(kp_idx), n_vis
            t0 = time.perf_counter()
            solved = self._pnp(Pw, xy[kp_idx], T_pred)
            out.ms['pnp'] = (time.perf_counter() - t0) * 1e3
            if solved is not None:
                T_new, inl = solved
                jump_t = np.linalg.norm(T_new[:3, 3] - T_pred[:3, 3])
                jump_r = math.acos(max(-1.0, min(1.0, (np.trace(
                    T_new[:3, :3].T @ T_pred[:3, :3]) - 1.0) / 2.0)))
                if jump_t > self.cfg['max_jump_m'] or jump_r > self.cfg['max_jump_rad']:
                    solved = None
        used_f2f = False
        if solved is None and self.prev is not None:
            fallback = self._track_prev(xy, desc, T_pred)
            if fallback is not None:
                jump_t = np.linalg.norm(fallback[0][:3, 3] - T_pred[:3, 3])
                if jump_t < self.cfg['max_jump_m']:
                    solved, used_f2f = fallback, True
        if solved is not None and T_o_base is not None and self.cfg['gravity_align']:
            solved = (self._gravity_align(solved[0], T_o_base), solved[1])
        if solved is not None:
            T_new, inl = solved
            out.n_inliers = len(inl)
            self.velocity = inv_T(self.T_w_cam) @ T_new
            self.T_w_cam = T_new
            self.T_rel_kf = inv_T(self.kfs[-1].T_w_cam) @ T_new
            self.dr_rel = self.T_rel_kf.copy()
            self.state, self.lost_frames = 'tracking', 0
            if used_f2f:
                matched_map = matched_kp = np.zeros(0, int)
                out.f2f = True
                self._force_kf = True            # the local map no longer covers this view: new keyframe
            else:
                matched_map, matched_kp = map_idx[inl], kp_idx[inl]
                out.inlier_xy = xy[matched_kp]
                out.outlier_xy = xy[np.setdiff1d(kp_idx, matched_kp)]
        else:
            self.lost_frames += 1
            self.state = 'lost'
            self.T_w_cam = T_pred
            self.T_rel_kf = inv_T(self.kfs[-1].T_w_cam) @ T_pred
            self.dr_rel = self.T_rel_kf.copy()
            matched_map = matched_kp = np.zeros(0, int)
            if self.lost_frames >= self.cfg['lost_reloc_after'] and self.lost_frames % 2 == 1:
                rel = self._relocalize(xy, desc, pts, T_pred)
                if rel is not None:
                    self.T_w_cam, n_inl, _ = rel
                    if T_o_base is not None and self.cfg['gravity_align']:
                        self.T_w_cam = self._gravity_align(self.T_w_cam, T_o_base)
                    self.T_rel_kf = inv_T(self.kfs[-1].T_w_cam) @ self.T_w_cam
                    self.state, self.lost_frames = 'tracking', 0
                    out.n_inliers = n_inl
                    matched_map = matched_kp = np.zeros(0, int)
                    self._force_kf = True
            elif self.lost_frames >= self.cfg['lost_reinit_after'] and np.isfinite(pts[:, 0]).sum() >= 40:
                # nothing works: start a new keyframe at the odometry pose, linked by a weak edge
                self._force_kf = self._weak_kf = True
                self.state, self.lost_frames = 'tracking', 0
                out.n_inliers = 0
        self.T_o_cam_prev = T_o_cam if T_o_cam is not None else self.T_o_cam_prev
        self.frames_since_kf += 1
        out.state = self.state

        # keyframe decision
        last = self.kfs[-1]
        dist = np.linalg.norm(self.T_rel_kf[:3, 3])
        rot = math.acos(max(-1.0, min(1.0, (np.trace(self.T_rel_kf[:3, :3]) - 1) / 2)))
        ratio = out.n_inliers / max(self.ref_inliers, 1)         # tracked landmarks vs last keyframe
        need = (self.state == 'tracking' and self.frames_since_kf >= self.cfg['kf_min_frames']
                and out.n_inliers >= self.cfg['min_inliers'] + 4
                and (ratio < self.cfg['kf_track_ratio'] or dist > self.cfg['kf_dist']
                     or rot > self.cfg['kf_rot'] or stamp - self.last_kf_stamp > self.cfg['kf_time']))
        weak = self._weak_kf
        if self._force_kf:
            need, self._force_kf, self._weak_kf = True, False, False
        if need:
            t0 = time.perf_counter()
            kf = self._make_keyframe(stamp, xy, desc, resp, pts, matched_kp, matched_map, self.T_w_cam,
                                     self.dr_rel if weak else self.T_rel_kf, weak=weak)
            out.keyframe, out.kf_id = True, kf.id
            self.ref_inliers = max(out.n_inliers, self.cfg['min_inliers'])
            loop = self._try_loop(kf)
            if loop:
                out.loop = loop
                c0, c1 = self.graph.optimize()
                self._apply_graph()
                out.graph_changed = True
            out.ms['keyframe'] = (time.perf_counter() - t0) * 1e3
        self.prev = dict(xy=xy, desc=desc, pts=pts, T=self.T_w_cam.copy())
        out.kf_id = len(self.kfs) - 1
        out.T_w_cam = self.T_w_cam.copy()
        out.T_kf_cam = inv_T(self.kfs[-1].T_w_cam) @ self.T_w_cam
        out.ms['total'] = (time.perf_counter() - t_all) * 1e3
        return out

    def landmarks_world(self, max_points=20000):
        """All landmarks in map coordinates (N, 3), thinned to max_points."""
        n = len(self.points)
        if n == 0:
            return np.zeros((0, 3))
        idx = np.arange(n) if n <= max_points else np.linspace(0, n - 1, max_points).astype(int)
        out = np.zeros((len(idx), 3))
        for kid in np.unique(self.points.kf[idx]):
            m = self.points.kf[idx] == kid
            T = self.kfs[kid].T_w_cam
            out[m] = self.points.p[idx[m]].astype(np.float64) @ T[:3, :3].T + T[:3, 3]
        return out

    @property
    def T_w_base(self):
        return self.T_w_cam @ inv_T(self.T_base_cam)
