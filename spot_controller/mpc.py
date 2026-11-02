"""Convex MPC for the ground reaction forces (single rigid body, Di Carlo et al. 2018).

State  x = [roll, pitch, yaw, p(3), omega(3, world), v(3), g]  (13), g = -9.81 is a constant state
so gravity enters the linear model.  Input: one force f_i (3) per foot, world frame.

    d(Theta)/dt = Rz(yaw)^T omega            (small roll/pitch)
    dp/dt       = v
    d(omega)/dt = I_w^-1  sum_i  r_i x f_i   (r_i: foot relative to the CoM, world axes)
    dv/dt       = (1/m) sum_i f_i + g

Over a horizon of N steps the contact schedule says which feet can push (swing feet are forced to
zero), friction pyramids bound the horizontal force, and the QP minimises state tracking error plus
a small force penalty. Only the forces of the first step are used (receding horizon).
"""
import math

import numpy as np

from .qp_solver import solve_qp
from .quaternion import skew


def rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class ConvexMPC:

    def __init__(self, mass, inertia_body, horizon=10, dt=0.03, mu=0.6, fz_min=0.0, fz_max=220.0,
                 q_weights=None, force_weight=1e-4, max_iter=300, eps=2e-3, force_scale=100.0):
        self.m = mass
        inertia = np.asarray(inertia_body, dtype=float)
        self.I_body = np.diag(inertia) if inertia.ndim == 1 else inertia
        self.I_inv_body = np.linalg.inv(self.I_body)
        self.N = horizon
        self.dt = dt
        self.mu = mu
        self.fz_min = fz_min
        self.fz_max = fz_max
        self.Q = np.diag(q_weights if q_weights is not None else
                         [60, 60, 20,  1, 1, 150,  0.5, 0.5, 0.8,  8, 8, 20, 0.0])
        self.alpha = force_weight
        self.force_scale = force_scale
        self.max_iter = max_iter
        self.eps = eps
        self._x_prev = None
        self._y_prev = None
        self.last_solve = None

    # ------------------------------------------------------------------ model
    def _discrete(self, yaw, r_feet):
        """Second-order discretisation of the continuous model for one step (closed form).

        A has only three non-zero blocks (Rz', I and gravity), so Ad = I + A dt + A^2 dt^2/2 and
        Bd = (I dt + A dt^2/2) B are written out directly.
        """
        dt, m = self.dt, self.m
        Rz = rot_z(yaw)
        I_inv = Rz @ self.I_inv_body @ Rz.T
        Ad = np.eye(13)
        Ad[0:3, 6:9] = Rz.T * dt
        Ad[3:6, 9:12] = np.eye(3) * dt
        Ad[11, 12] = dt
        Ad[5, 12] = 0.5 * dt * dt
        Bd = np.zeros((13, 12))
        half = 0.5 * dt * dt
        eye_m = np.eye(3) / m
        for i, r in enumerate(r_feet):
            torque_map = I_inv @ skew(r)
            c = 3 * i
            Bd[0:3, c:c + 3] = half * (Rz.T @ torque_map)
            Bd[3:6, c:c + 3] = half * eye_m
            Bd[6:9, c:c + 3] = dt * torque_map
            Bd[9:12, c:c + 3] = dt * eye_m
        return Ad, Bd

    # ------------------------------------------------------------------ solve
    def solve(self, x0, x_ref, r_feet, contact, yaw_seq):
        """Plan forces.

        x0: (13,) current state. x_ref: (N, 13) reference at steps 1..N.
        r_feet: (N, 4, 3) foot position relative to the CoM (world axes) at each step.
        contact: (N, 4) bool, foot can push at that step. yaw_seq: (N,) yaw used by the model.
        Returns forces (4, 3) for the first step, or None if the QP failed.
        """
        N = self.N
        nu = 12 * N
        Aqp = np.zeros((13 * N, 13))
        Bqp = np.zeros((13 * N, nu))
        Ak = np.eye(13)
        for k in range(N):
            Ad, Bd = self._discrete(yaw_seq[k], r_feet[k])
            Ak = Ad @ Ak
            Aqp[13 * k:13 * k + 13] = Ak
            for j in range(k):
                Bqp[13 * k:13 * k + 13, 12 * j:12 * j + 12] = Ad @ Bqp[13 * (k - 1):13 * k, 12 * j:12 * j + 12]
            Bqp[13 * k:13 * k + 13, 12 * k:12 * k + 12] = Bd
        q_diag = np.tile(np.diag(self.Q), N)
        xref = np.asarray(x_ref, dtype=float).reshape(-1)

        # Only feet that can push are decision variables (swing feet are exactly zero).
        active = [(k, i) for k in range(N) for i in range(4) if contact[k][i]]
        if not active:
            return np.zeros((4, 3))
        cols = np.array([12 * k + 3 * i + c for (k, i) in active for c in range(3)])
        Ba = Bqp[:, cols]
        s = self.force_scale                                   # u = f / s keeps the QP well scaled
        P = s * s * ((Ba.T * q_diag) @ Ba + self.alpha * np.eye(len(cols)))
        q = s * ((Ba.T * q_diag) @ (Aqp @ x0 - xref))

        mu = self.mu
        n_act = len(active)
        lo_b = np.tile([-mu * self.fz_max / s, -mu * self.fz_max / s, self.fz_min / s], n_act)
        hi_b = np.tile([mu * self.fz_max / s, mu * self.fz_max / s, self.fz_max / s], n_act)
        pyr = np.zeros((4 * n_act, 3 * n_act))
        for a_i in range(n_act):
            for r_i, (sx, ax) in enumerate(((1, 0), (-1, 0), (1, 1), (-1, 1))):
                pyr[4 * a_i + r_i, 3 * a_i + ax] = sx
                pyr[4 * a_i + r_i, 3 * a_i + 2] = -mu
        Acon = np.vstack([np.eye(3 * n_act), pyr])
        l = np.concatenate([lo_b, np.full(4 * n_act, -np.inf)])
        u = np.concatenate([hi_b, np.zeros(4 * n_act)])

        x_init = None
        if self._x_prev is not None and self._x_prev.shape[0] == 12 * N:
            shifted = np.concatenate([self._x_prev[12:], self._x_prev[-12:]])      # one step ahead
            x_init = shifted[cols] / s
        res = solve_qp(P, q, Acon, l, u, x0=x_init, max_iter=self.max_iter, eps=self.eps)
        self.last_solve = res
        if not np.all(np.isfinite(res.x)):
            return None
        full = np.zeros(12 * N)
        full[cols] = res.x * s
        self._x_prev = full
        return full[:12].reshape(4, 3)
