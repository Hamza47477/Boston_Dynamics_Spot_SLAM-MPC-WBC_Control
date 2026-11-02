"""Whole-body controller: one QP turns planned contact forces and motion tasks into joint torques.

Decision variables  x = [ du/dt (18: base linear, base angular, 12 joint accelerations),
                          F (3 per stance foot: ground reaction force, world frame) ]

Equality constraints (hard)
    floating-base dynamics      [M du/dt + h - Jc' F]_base = 0
    stance feet do not move     Jc du/dt + Jc_dot u = 0
Inequality constraints
    friction pyramid and fz bounds on every F
    motor torque limits         |[M du/dt + h - Jc' F]_joints| <= tau_max
Weighted tasks (soft, least squares)
    follow the MPC contact forces           F ~ F_mpc                       (high weight)
    swing foot acceleration                 Js du/dt = a_swing - Js_dot u   (high weight)
    base attitude / height PD accelerations                                 (low weight)
    small regularisation on accelerations

Output torques  tau = [M du/dt + h - Jc' F]_joints  (feed-forward torque, inverse dynamics).
"""
import numpy as np

from .dynamics import SpotDynamics
from .qp_solver import solve_qp

FORCE_SCALE = 50.0           # decision variable is F / FORCE_SCALE


class WholeBodyController:

    def __init__(self, dynamics=None, mu=0.6, tau_max=50.0, fz_max=300.0,
                 w_force=1.0, w_swing=30.0, w_orient=0.5, w_height=0.5, w_reg=1e-4,
                 w_joint_reg=1e-3, max_iter=400, eps=1e-4):
        self.dyn = dynamics or SpotDynamics()
        self.mu = mu
        self.tau_max = tau_max
        self.fz_max = fz_max
        self.w = dict(force=w_force, swing=w_swing, orient=w_orient, height=w_height,
                      reg=w_reg, joint_reg=w_joint_reg)
        self.max_iter = max_iter
        self.eps = eps
        self.last = None

    def compute(self, R, q, u, stance, F_des, swing, base_acc_des):
        """Solve the QP.

        R: body rotation; q: {leg: joint angles}; u: generalised velocity (18,).
        stance: list of stance foot indices; F_des: (len(stance), 3) MPC forces (world).
        swing: list of (foot index, a_des (3,) world foot acceleration to achieve).
        base_acc_des: (alpha_des (3,) angular accel, z_acc_des) desired base accelerations.
        Returns dict(tau (12,), F (n,3), udd (18,), converged) or None on failure.
        """
        n_c = len(stance)
        M, h = self.dyn.terms(R, q, u)
        Jc, bias_c = self.dyn.contact_jacobians(R, q, u, stance)
        nv = 18 + 3 * n_c
        s = FORCE_SCALE

        # ---- tasks as least squares  sum w ||A x - b||^2
        P = np.zeros((nv, nv))
        g = np.zeros(nv)

        def add(A, b, w):
            nonlocal P, g
            P += w * A.T @ A
            g -= w * A.T @ b

        A = np.zeros((3 * n_c, nv))
        A[:, 18:] = np.eye(3 * n_c) * s
        add(A, np.asarray(F_des).reshape(-1), self.w['force'] / 400.0)       # 20 N scale

        for foot, a_des in swing:
            Js, bias_s = self.dyn.contact_jacobians(R, q, u, [foot])
            A = np.zeros((3, nv))
            A[:, :18] = Js
            add(A, np.asarray(a_des) - bias_s, self.w['swing'])

        alpha_des, z_acc_des = base_acc_des
        A = np.zeros((3, nv))
        A[:, 3:6] = np.eye(3)
        add(A, np.asarray(alpha_des), self.w['orient'])
        A = np.zeros((1, nv))
        A[0, 2] = 1.0
        add(A, np.array([z_acc_des]), self.w['height'])
        reg = np.zeros(nv)
        reg[:6] = self.w['reg']
        reg[6:18] = self.w['joint_reg']
        P += np.diag(reg)
        P += 1e-9 * np.eye(nv)

        # ---- constraints
        rows, lo, hi = [], [], []
        Aeq = np.zeros((6, nv))                                    # floating-base dynamics
        Aeq[:, :18] = M[:6]
        Aeq[:, 18:] = -(Jc[:, :6].T if n_c else np.zeros((6, 0))) * s
        rows.append(Aeq)
        lo.append(-h[:6])
        hi.append(-h[:6])
        if n_c:
            Ac = np.zeros((3 * n_c, nv))                           # stance feet: zero acceleration
            Ac[:, :18] = Jc
            rows.append(Ac)
            lo.append(-bias_c)
            hi.append(-bias_c)
            for k in range(n_c):                                   # friction pyramid, 0 <= fz
                fx = np.zeros((4, nv))
                base = 18 + 3 * k
                for r, (sx, ax) in enumerate(((1, 0), (-1, 0), (1, 1), (-1, 1))):
                    fx[r, base + ax] = sx
                    fx[r, base + 2] = -self.mu
                rows.append(fx)
                lo.append(np.full(4, -np.inf))
                hi.append(np.zeros(4))
                fz = np.zeros((1, nv))
                fz[0, base + 2] = 1.0
                rows.append(fz)
                lo.append(np.array([0.0]))
                hi.append(np.array([self.fz_max / s]))
        At = np.zeros((12, nv))                                    # motor torque limits
        At[:, :18] = M[6:]
        if n_c:
            At[:, 18:] = -Jc[:, 6:].T * s
        rows.append(At)
        lo.append(-self.tau_max - h[6:])
        hi.append(self.tau_max - h[6:])

        Acon = np.vstack(rows)
        l = np.concatenate(lo)
        uu = np.concatenate(hi)
        res = solve_qp(P, g, Acon, l, uu, max_iter=self.max_iter, eps=self.eps)
        self.last = res
        if not np.all(np.isfinite(res.x)):
            return None
        udd = res.x[:18]
        F = res.x[18:].reshape(n_c, 3) * s
        tau = (M @ udd + h)[6:] - (Jc[:, 6:].T @ F.reshape(-1) if n_c else 0.0)
        acceptable = res.converged or (res.primal_residual < 0.05 and res.dual_residual < 0.05)
        return dict(tau=np.asarray(tau), F=F, udd=udd, converged=acceptable,
                    residual=(res.primal_residual, res.dual_residual))
