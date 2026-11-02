"""4-DoF pose graph (x, y, z, yaw): roll and pitch are observable from gravity and stay fixed.

Each node is a body pose R_i = Rz(psi_i) @ Rr_i, p_i with Rr_i fixed. An edge (i, j) is a
measurement of the relative pose seen from node i:

    e_t   = R_i^T (p_j - p_i) - t_ij            (3)
    e_psi = wrap(psi_j - psi_i - psi_ij)        (1)

Visual odometry gives a chain of edges between consecutive keyframes (they drift); a loop
closure adds an edge between two keyframes that saw the same place. Gauss-Newton with a Huber
weight on loop edges moves all keyframes to the best compromise, node 0 stays fixed.
"""
import math

import numpy as np

from .geometry import rot_z, wrap


def _dRz_T(psi):
    """d/dpsi of Rz(psi)^T."""
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[-s, c, 0.0], [-c, -s, 0.0], [0.0, 0.0, 0.0]])


class PoseGraph:

    def __init__(self):
        self.p = []          # (3,) positions
        self.psi = []        # yaw
        self.Rr = []         # fixed roll/pitch part
        self.edges = []      # dict(i, j, t, psi, w_t, w_psi, loop)

    def __len__(self):
        return len(self.p)

    def add_node(self, T):
        """Add a node from a 4x4 pose (z up); returns its index."""
        R = T[:3, :3]
        psi = math.atan2(R[1, 0], R[0, 0])
        self.p.append(T[:3, 3].copy())
        self.psi.append(psi)
        self.Rr.append(rot_z(-psi) @ R)
        return len(self.p) - 1

    def pose(self, i):
        T = np.eye(4)
        T[:3, :3] = rot_z(self.psi[i]) @ self.Rr[i]
        T[:3, 3] = self.p[i]
        return T

    def measure(self, i, j):
        """Relative measurement of the current estimate: (t_ij, psi_ij)."""
        Ri = rot_z(self.psi[i]) @ self.Rr[i]
        return Ri.T @ (self.p[j] - self.p[i]), wrap(self.psi[j] - self.psi[i])

    def add_edge(self, i, j, t_ij, psi_ij, sigma_t=0.05, sigma_psi=0.02, loop=False):
        self.edges.append(dict(i=i, j=j, t=np.asarray(t_ij, float), psi=float(psi_ij),
                               w_t=1.0 / sigma_t ** 2, w_psi=1.0 / sigma_psi ** 2, loop=loop))

    def residual(self, e):
        Ri = rot_z(self.psi[e['i']]) @ self.Rr[e['i']]
        et = Ri.T @ (self.p[e['j']] - self.p[e['i']]) - e['t']
        ep = wrap(self.psi[e['j']] - self.psi[e['i']] - e['psi'])
        return et, ep

    def cost(self):
        total = 0.0
        for e in self.edges:
            et, ep = self.residual(e)
            total += e['w_t'] * float(et @ et) + e['w_psi'] * ep * ep
        return total

    def optimize(self, iterations=20, huber=3.0, tol=1e-6):
        """Levenberg-Marquardt over all nodes except node 0. Returns (initial cost, final cost)."""
        n = len(self.p)
        if n < 2 or not self.edges:
            return 0.0, 0.0
        cost0 = last = self.cost()
        lam = 1e-3
        for _ in range(iterations):
            H = np.zeros((4 * n, 4 * n))
            b = np.zeros(4 * n)
            for e in self.edges:
                i, j = e['i'], e['j']
                Ri = rot_z(self.psi[i]) @ self.Rr[i]
                d = self.p[j] - self.p[i]
                et = Ri.T @ d - e['t']
                ep = wrap(self.psi[j] - self.psi[i] - e['psi'])
                chi = math.sqrt(e['w_t'] * float(et @ et) + e['w_psi'] * ep * ep)
                k = 1.0 if (not e['loop'] or chi <= huber) else huber / chi      # Huber on loops
                J = np.zeros((4, 8))                                          # [p_i psi_i p_j psi_j]
                J[:3, 0:3] = -Ri.T
                J[:3, 3] = self.Rr[i].T @ _dRz_T(self.psi[i]) @ d
                J[:3, 4:7] = Ri.T
                J[3, 3], J[3, 7] = -1.0, 1.0
                W = np.diag([e['w_t'] * k] * 3 + [e['w_psi'] * k])
                r = np.concatenate([et, [ep]])
                idx = np.r_[4 * i:4 * i + 4, 4 * j:4 * j + 4]
                H[np.ix_(idx, idx)] += J.T @ W @ J
                b[idx] += J.T @ W @ r
            H[:4, :4] += np.eye(4) * 1e9                                      # gauge: fix node 0
            improved = False
            for _try in range(6):
                dx = np.linalg.solve(H + lam * np.eye(4 * n), -b)
                saved = ([q.copy() for q in self.p], list(self.psi))
                for k_ in range(n):
                    self.p[k_] = self.p[k_] + dx[4 * k_:4 * k_ + 3]
                    self.psi[k_] = wrap(self.psi[k_] + dx[4 * k_ + 3])
                new = self.cost()
                if new < last:
                    lam = max(lam / 3.0, 1e-9)
                    improved = True
                    break
                self.p, self.psi = saved
                lam *= 5.0
            if not improved or last - new < tol * max(1.0, last):
                last = new if improved else last
                break
            last = new
        return cost0, last


def umeyama(src, dst):
    """Rigid alignment R, t with dst ~ R src + t (no scale), for trajectory comparison."""
    from .rigid import kabsch
    return kabsch(np.asarray(src, float), np.asarray(dst, float))


def ate_rmse(est, gt):
    """Absolute trajectory error (RMSE, m) after the best rigid alignment; also max error."""
    est, gt = np.asarray(est, float), np.asarray(gt, float)
    R, t = umeyama(est, gt)
    err = np.linalg.norm(est @ R.T + t - gt, axis=1)
    return float(np.sqrt(np.mean(err ** 2))), float(err.max())
