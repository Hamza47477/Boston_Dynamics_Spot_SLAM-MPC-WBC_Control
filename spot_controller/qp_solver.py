"""Small dense QP solver (OSQP-style ADMM) in pure NumPy.

    minimize    0.5 x'Px + q'x
    subject to  l <= A x <= u          (use l == u for equality rows, +-inf for one-sided rows)

Same splitting scheme as OSQP (Stellato et al., 2020): one factorisation of
P + sigma I + A' diag(rho) A per solve, then cheap iterations with over-relaxation. Warm starts are
accepted. Meant for the problems here (tens to ~150 variables); no sparse algebra.
"""
import numpy as np


class QPResult:
    def __init__(self, x, z, y, iterations, primal_residual, dual_residual, converged):
        self.x, self.z, self.y = x, z, y
        self.iterations = iterations
        self.primal_residual = primal_residual
        self.dual_residual = dual_residual
        self.converged = converged


def solve_qp(P, q, A, l, u, x0=None, y0=None, rho=0.1, sigma=1e-6, alpha=1.6,
             max_iter=500, eps=1e-4, equality_rho_scale=1e3, check_every=10, adapt=True):
    P = np.asarray(P, dtype=float)
    q = np.asarray(q, dtype=float)
    A = np.asarray(A, dtype=float)
    l = np.asarray(l, dtype=float)
    u = np.asarray(u, dtype=float)
    n, m = P.shape[0], A.shape[0]

    is_eq = np.abs(u - l) < 1e-9

    def factor(rho_value):
        rho_v = np.full(m, rho_value)
        rho_v[is_eq] *= equality_rho_scale                        # equality rows converge faster
        K = P + sigma * np.eye(n) + A.T @ (rho_v[:, None] * A)
        return rho_v, np.linalg.inv(K)

    rho_vec, K_inv = factor(rho)

    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    y = np.zeros(m) if y0 is None else np.array(y0, dtype=float)
    z = np.clip(A @ x, l, u)

    r_prim = r_dual = np.inf
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        rhs = sigma * x - q + A.T @ (rho_vec * z - y)
        x_tilde = K_inv @ rhs
        z_tilde = A @ x_tilde
        x = alpha * x_tilde + (1.0 - alpha) * x
        z_relaxed = alpha * z_tilde + (1.0 - alpha) * z
        z_new = np.clip(z_relaxed + y / rho_vec, l, u)
        y = y + rho_vec * (z_relaxed - z_new)
        z = z_new
        if it % check_every == 0 or it == max_iter:
            Ax = A @ x
            r_prim = np.max(np.abs(Ax - z)) if m else 0.0
            r_dual = np.max(np.abs(P @ x + q + A.T @ y))
            scale_p = max(np.max(np.abs(Ax)) if m else 0.0, np.max(np.abs(z)) if m else 0.0, 1.0)
            scale_d = max(np.max(np.abs(P @ x)), np.max(np.abs(A.T @ y)) if m else 0.0,
                          np.max(np.abs(q)), 1.0)
            if r_prim <= eps * scale_p and r_dual <= eps * scale_d:
                converged = True
                break
            if adapt and it % (2 * check_every) == 0:                # OSQP-style step-size adaptation
                ratio = np.sqrt((r_prim / scale_p) / max(r_dual / scale_d, 1e-12))
                if ratio > 3.0 or ratio < 1.0 / 3.0:
                    rho = float(np.clip(rho * ratio, 1e-4, 1e4))
                    rho_vec, K_inv = factor(rho)
    return QPResult(x, z, y, it, r_prim, r_dual, converged)
