import math

import numpy as np

from spot_controller.perception.geometry import make_T, rot_z, PoseBuffer, quat_to_R, R_to_quat
from spot_controller.perception.pose_graph import ate_rmse, PoseGraph
from spot_controller.perception.rigid import kabsch, ransac_rigid


def test_kabsch_recovers_transform():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(50, 3))
    R = rot_z(0.7)
    t = np.array([1.0, -2.0, 0.5])
    Rk, tk = kabsch(P, P @ R.T + t)
    assert np.allclose(Rk, R, atol=1e-9) and np.allclose(tk, t, atol=1e-9)


def test_ransac_rejects_outliers():
    rng = np.random.default_rng(2)
    P = rng.uniform(-3, 3, size=(120, 3))
    R, t = rot_z(-0.4), np.array([0.3, 0.1, -0.2])
    Q = P @ R.T + t + rng.normal(0, 0.01, P.shape)
    Q[:40] = rng.uniform(-3, 3, size=(40, 3))             # 33 % gross outliers
    Rr, tr, inl = ransac_rigid(P, Q, thresh=0.1)
    assert np.allclose(Rr, R, atol=0.02) and np.allclose(tr, t, atol=0.03)
    assert inl[40:].mean() > 0.95 and inl[:40].mean() < 0.2


def test_quat_roundtrip():
    R = rot_z(1.1) @ np.array([[1, 0, 0], [0, math.cos(.3), -math.sin(.3)], [0, math.sin(.3), math.cos(.3)]])
    assert np.allclose(quat_to_R(*R_to_quat(R)), R, atol=1e-9)


def test_pose_buffer_interpolates():
    buf = PoseBuffer()
    buf.add(0.0, make_T(np.eye(3), [0, 0, 0]))
    buf.add(1.0, make_T(rot_z(0.4), [2, 0, 0]))
    T = buf.get(0.5)
    assert abs(T[0, 3] - 1.0) < 1e-9 and abs(math.atan2(T[1, 0], T[0, 0]) - 0.2) < 0.01


def _square_loop(n_per_side=10, drift_yaw=0.01, drift_scale=1.03):
    """Ground truth square path and a drifting odometry version of it (+ loop edge)."""
    pts, yaws = [], []
    x = y = yaw = 0.0
    for side in range(4):
        for _ in range(n_per_side):
            pts.append((x, y))
            yaws.append(yaw)
            x += math.cos(yaw)
            y += math.sin(yaw)
        yaw += math.pi / 2
    pts.append((0.0, 0.0))
    yaws.append(2 * math.pi)
    return np.array(pts), np.array(yaws)


def test_pose_graph_closes_drifting_loop():
    gt_xy, gt_yaw = _square_loop()
    n = len(gt_xy)
    g = PoseGraph()
    est_p, est_yaw = np.zeros(2), 0.0
    nodes = []
    for k in range(n):
        if k == 0:
            T = np.eye(4)
        else:                                            # integrate drifting relative motion
            d = gt_xy[k] - gt_xy[k - 1]
            dist = np.linalg.norm(d) * 1.03
            dyaw = gt_yaw[k] - gt_yaw[k - 1] + 0.012
            heading = est_yaw
            est_p = est_p + dist * np.array([math.cos(heading), math.sin(heading)])
            est_yaw += dyaw
            T = make_T(rot_z(est_yaw), [est_p[0], est_p[1], 0.0])
        nodes.append(g.add_node(T))
    for k in range(1, n):
        t, psi = g.measure(k - 1, k)
        g.add_edge(k - 1, k, t, psi, 0.05, 0.02)
    before = np.array([g.p[k][:2] for k in range(n)])
    err_before = np.linalg.norm(before[-1] - gt_xy[-1])
    g.add_edge(0, n - 1, [0.0, 0.0, 0.0], 2 * math.pi, 0.05, 0.02, loop=True)   # same place as node 0
    c0, c1 = g.optimize()
    after = np.array([g.p[k][:2] for k in range(n)])
    assert c1 < 0.05 * c0
    assert np.linalg.norm(after[-1] - gt_xy[-1]) < 0.3 * err_before
    assert ate_rmse(np.c_[after, np.zeros(n)], np.c_[gt_xy, np.zeros(n)])[0] < 0.5
