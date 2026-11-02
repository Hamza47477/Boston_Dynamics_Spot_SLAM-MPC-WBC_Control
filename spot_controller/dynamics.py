"""Floating-base rigid-body model of Spot for the whole-body controller.

Generalised velocity  u = [v (3, world), omega (3, world), qdot (12, Webots joint convention)].
Equation of motion    M(q) du/dt + h(q, u) = S' tau + Jc' F       S selects the 12 joints.

Mass model (from the Webots PROTO files): the body is a rigid body (14.41 kg, CoM near the body
origin); each leg adds three point masses: the shoulder block at the abduction axis (2.13 kg), the
thigh (0.94 kg, at 41 % of the thigh) and the calf (0.14 kg, at 45 % of the calf). Link rotational
inertia is neglected (the legs are light), the body keeps its full inertia tensor.
"""
import numpy as np

from .constants import (CALF_LENGTH, HIP_MOUNT, HIP_OFFSET_Y, ik_to_webots, LEG_ORDER, LEG_SIDE,
                        THIGH_LENGTH, webots_to_ik)
from .quaternion import q_exp, skew

GRAVITY = 9.81
BODY_MASS = 14.41
BODY_COM = np.array([0.0224, 0.0, 0.0203])
BODY_INERTIA = np.diag([0.11, 0.90, 0.91])          # body-only, about its CoM (box estimate)
SHOULDER_MASS, THIGH_MASS, CALF_MASS = 2.132, 0.935, 0.137
THIGH_FRACTION, CALF_FRACTION = 0.41, 0.45


def leg_point(leg, q_leg, kind):
    """Position (body frame) of a leg's mass point or foot for Webots joint angles q_leg."""
    t1, t2, t3 = webots_to_ik(*q_leg)
    dy = LEG_SIDE[leg] * HIP_OFFSET_Y
    l1, l2 = THIGH_LENGTH, CALF_LENGTH
    if kind == 'shoulder':
        return np.array(HIP_MOUNT[leg])
    if kind == 'thigh':
        a = THIGH_FRACTION * l1
        x, zp = a * np.sin(t2), -a * np.cos(t2)
    elif kind == 'calf':
        b = CALF_FRACTION * l2
        x = l1 * np.sin(t2) + b * np.sin(t2 + t3)
        zp = -(l1 * np.cos(t2) + b * np.cos(t2 + t3))
    else:                                              # foot
        x = l1 * np.sin(t2) + l2 * np.sin(t2 + t3)
        zp = -(l1 * np.cos(t2) + l2 * np.cos(t2 + t3))
    y = dy * np.cos(t1) - zp * np.sin(t1)
    z = dy * np.sin(t1) + zp * np.cos(t1)
    return np.array(HIP_MOUNT[leg]) + np.array([x, y, z])


def leg_point_jacobian_numeric(leg, q_leg, kind, h=1e-6):
    """Central-difference reference for leg_point_jacobian (used in tests)."""
    J = np.zeros((3, 3))
    base = np.array(q_leg, dtype=float)
    for k in range(3):
        plus, minus = base.copy(), base.copy()
        plus[k] += h
        minus[k] -= h
        J[:, k] = (leg_point(leg, plus, kind) - leg_point(leg, minus, kind)) / (2 * h)
    return J


def leg_point_jacobian(leg, q_leg, kind):
    """d(point in body frame)/d(Webots joint angles of this leg), 3x3, closed form."""
    if kind == 'shoulder':
        return np.zeros((3, 3))
    t1, t2, t3 = webots_to_ik(*q_leg)
    dy = LEG_SIDE[leg] * HIP_OFFSET_Y
    l1 = THIGH_LENGTH
    if kind == 'thigh':
        a = THIGH_FRACTION * l1
        zp = -a * np.cos(t2)
        dx2, dx3 = a * np.cos(t2), 0.0
        dz2, dz3 = a * np.sin(t2), 0.0
    else:
        b = CALF_FRACTION * CALF_LENGTH if kind == 'calf' else CALF_LENGTH
        c23, s23 = np.cos(t2 + t3), np.sin(t2 + t3)
        zp = -(l1 * np.cos(t2) + b * c23)
        dx2, dx3 = l1 * np.cos(t2) + b * c23, b * c23
        dz2, dz3 = l1 * np.sin(t2) + b * s23, b * s23
    c1, s1 = np.cos(t1), np.sin(t1)
    J = np.zeros((3, 3))
    J[1, 0] = -(-dy * s1 - zp * c1)                 # d/dt1 with dt1/dq0 = -1
    J[2, 0] = -(dy * c1 - zp * s1)
    J[0, 1], J[1, 1], J[2, 1] = dx2, -s1 * dz2, c1 * dz2
    J[0, 2], J[1, 2], J[2, 2] = dx3, -s1 * dz3, c1 * dz3
    return J


class SpotDynamics:

    MASS_POINTS = (('shoulder', SHOULDER_MASS), ('thigh', THIGH_MASS), ('calf', CALF_MASS))

    def __init__(self):
        self.total_mass = BODY_MASS + 4 * (SHOULDER_MASS + THIGH_MASS + CALF_MASS)

    def com_body(self, q):
        """Centre of mass of the whole robot in the body frame for joint angles q."""
        total = BODY_MASS * BODY_COM
        for leg in LEG_ORDER:
            for kind, m in self.MASS_POINTS:
                total = total + m * leg_point(leg, q[leg], kind)
        return total / self.total_mass

    # ---- Jacobians of world points w.r.t. u
    @staticmethod
    def _point_jacobian(R, b, leg_index, J_leg):
        """3 x 18 Jacobian of a world point p + R b (b: body-frame position, J_leg: 3x3 d b/d q)."""
        J = np.zeros((3, 18))
        J[:, 0:3] = np.eye(3)
        J[:, 3:6] = -skew(R @ b)
        J[:, 6 + 3 * leg_index:9 + 3 * leg_index] = R @ J_leg
        return J

    def mass_points(self, R, q):
        """List of (mass, J (3x18), body position b) for the base and the 12 leg mass points."""
        pts = [(BODY_MASS, self._point_jacobian(R, BODY_COM, 0, np.zeros((3, 3))), BODY_COM)]
        for i, leg in enumerate(LEG_ORDER):
            for kind, m in self.MASS_POINTS:
                b = leg_point(leg, q[leg], kind)
                J_leg = leg_point_jacobian(leg, q[leg], kind) if kind != 'shoulder' else np.zeros((3, 3))
                pts.append((m, self._point_jacobian(R, b, i, J_leg), b))
        return pts

    def foot_jacobian(self, R, q, leg_index):
        leg = LEG_ORDER[leg_index]
        b = leg_point(leg, q[leg], 'foot')
        return self._point_jacobian(R, b, leg_index, leg_point_jacobian(leg, q[leg], 'foot')), b

    # ---- dynamics terms
    def terms(self, R, q, u, dt=1e-4):
        """Return M (18x18) and h (18): M du/dt + h = S' tau + Jc' F.

        q: {leg: (hip_x, hip_y, knee)}; u: generalised velocity (18,). h = Coriolis + gravity.
        """
        pts = self.mass_points(R, q)
        M = np.zeros((18, 18))
        h = np.zeros(18)
        omega = u[3:6]
        I_world = R @ BODY_INERTIA @ R.T
        M[3:6, 3:6] += I_world
        h[3:6] += np.cross(omega, I_world @ omega)
        # advance the state by dt along u to get d(J)/dt * u numerically
        R2 = q_exp(omega * dt)
        from .quaternion import q_to_R
        R_next = q_to_R(R2) @ R
        q_next = {leg: tuple(np.array(q[leg]) + dt * u[6 + 3 * i:9 + 3 * i])
                  for i, leg in enumerate(LEG_ORDER)}
        pts_next = self.mass_points(R_next, q_next)
        for (m, J, _), (_, J2, _) in zip(pts, pts_next):
            M += m * J.T @ J
            h += m * J.T @ (((J2 - J) / dt) @ u)                  # Coriolis / centripetal
            h += m * J.T @ np.array([0.0, 0.0, GRAVITY])            # gravity (opposes -g)
        return M, h

    def contact_jacobians(self, R, q, u, feet, dt=1e-4):
        """Stacked foot Jacobians Jc (3n x 18) and bias Jc_dot u for the given foot indices."""
        from .quaternion import q_to_R
        omega = u[3:6]
        R_next = q_to_R(q_exp(omega * dt)) @ R
        q_next = {leg: tuple(np.array(q[leg]) + dt * u[6 + 3 * i:9 + 3 * i])
                  for i, leg in enumerate(LEG_ORDER)}
        Jc, bias = [], []
        for i in feet:
            J, _ = self.foot_jacobian(R, q, i)
            J2, _ = self.foot_jacobian(R_next, q_next, i)
            Jc.append(J)
            bias.append(((J2 - J) / dt) @ u)
        if not feet:
            return np.zeros((0, 18)), np.zeros(0)
        return np.vstack(Jc), np.concatenate(bias)
