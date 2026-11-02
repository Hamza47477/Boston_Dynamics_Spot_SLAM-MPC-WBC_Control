"""Distribute a desired body wrench over the stance feet (ground reaction forces).

Solves   min  0.5 |W (A f - w)|^2 + 0.5 alpha |f - f_ref|^2
         s.t. f_z in [f_min, f_max],  |f_x|, |f_y| <= mu f_z      (per foot, friction pyramid)
with projected gradient descent. A f = w is the Newton-Euler balance: the sum of foot forces is
the net force, the sum of (foot position relative to the CoM) x force is the net moment.
"""
import numpy as np

from .quaternion import skew


def grasp_matrix(feet_rel_com):
    """6 x 3n matrix mapping stacked foot forces (world frame) to [net force; net moment]."""
    n = len(feet_rel_com)
    A = np.zeros((6, 3 * n))
    for k, rho in enumerate(feet_rel_com):
        A[0:3, 3 * k:3 * k + 3] = np.eye(3)
        A[3:6, 3 * k:3 * k + 3] = skew(rho)
    return A


def project_friction(f, mu, f_min, f_max):
    """Clamp each foot force into the friction pyramid (feasible, convex set).

    f_max may be a scalar or one value per foot (used to ramp load on and off a foot).
    """
    f = f.reshape(-1, 3).copy()
    f[:, 2] = np.clip(f[:, 2], f_min, np.maximum(f_max, f_min))
    limit = mu * f[:, 2]
    f[:, 0] = np.clip(f[:, 0], -limit, limit)
    f[:, 1] = np.clip(f[:, 1], -limit, limit)
    return f.reshape(-1)


def distribute_forces(feet_rel_com, wrench, mu=0.6, f_min=1.0, f_max=300.0,
                      force_scale=10.0, moment_scale=2.0, alpha=1e-3, iterations=300):
    """Return an (n, 3) array of foot forces (world frame, ground reaction on the feet).

    feet_rel_com: foot positions relative to the centre of mass, world-aligned axes.
    wrench: [Fx, Fy, Fz, Mx, My, Mz] required on the body about the centre of mass.
    """
    n = len(feet_rel_com)
    A = grasp_matrix(feet_rel_com)
    w = np.asarray(wrench, dtype=float)
    scale = np.array([1 / force_scale] * 3 + [1 / moment_scale] * 3)
    Aw = A * scale[:, None]
    ww = w * scale
    f_ref = np.tile(w[0:3] / n, n)

    # Closed-form unconstrained solution as a warm start
    H = Aw.T @ Aw + alpha * np.eye(3 * n)
    f = np.linalg.solve(H, Aw.T @ ww + alpha * f_ref)
    f = project_friction(f, mu, f_min, f_max)

    step = 1.0 / np.linalg.eigvalsh(H)[-1]
    for _ in range(iterations):
        grad = H @ f - (Aw.T @ ww + alpha * f_ref)
        f_new = project_friction(f - step * grad, mu, f_min, f_max)
        if np.max(np.abs(f_new - f)) < 1e-6:
            f = f_new
            break
        f = f_new
    return f.reshape(n, 3)
