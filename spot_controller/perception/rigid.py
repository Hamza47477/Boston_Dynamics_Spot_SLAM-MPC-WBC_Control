"""Rigid alignment of two 3D point sets: Kabsch (least squares) and RANSAC around it."""
import numpy as np


def kabsch(P, Q, w=None):
    """Return R, t minimising sum ||R p + t - q||^2 (P, Q: (N, 3))."""
    if w is None:
        w = np.ones(len(P))
    w = w / w.sum()
    cp, cq = (w[:, None] * P).sum(0), (w[:, None] * Q).sum(0)
    H = ((P - cp) * w[:, None]).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cq - R @ cp


def ransac_rigid(P, Q, iters=150, thresh=0.15, min_inliers=12, rng=None):
    """Robust rigid transform with Q ~ R P + t. Returns (R, t, inlier mask) or None."""
    n = len(P)
    if n < max(4, min_inliers):
        return None
    rng = rng or np.random.default_rng(0)
    best, best_count = None, 0
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        a, b = P[idx], Q[idx]
        if np.linalg.norm(np.cross(a[1] - a[0], a[2] - a[0])) < 1e-3:       # degenerate sample
            continue
        R, t = kabsch(a, b)
        err = np.linalg.norm(P @ R.T + t - Q, axis=1)
        inl = err < thresh
        count = int(inl.sum())
        if count > best_count:
            best, best_count = inl, count
            if count > 0.8 * n:
                break
    if best is None or best_count < min_inliers:
        return None
    for _ in range(3):                                  # refine on all inliers
        R, t = kabsch(P[best], Q[best])
        best = np.linalg.norm(P @ R.T + t - Q, axis=1) < thresh
        if best.sum() < min_inliers:
            return None
    R, t = kabsch(P[best], Q[best])
    return R, t, best
