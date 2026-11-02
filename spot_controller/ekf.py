"""Error-state extended Kalman filter: IMU prediction, leg-kinematics corrections.

Nominal state : position p, velocity v (world), attitude q (body -> world), accel bias ba,
                gyro bias bg.
Error state   : dx = [dp, dv, dtheta, dba, dbg]  (15), with  q_true = q (x) exp(dtheta)
                (dtheta is a small rotation in the body frame).
"""
import numpy as np

from .leg_odometry import body_velocity_from_foot
from .quaternion import q_exp, q_mult, q_normalize, q_to_R, skew

P_IDX = slice(0, 3)
V_IDX = slice(3, 6)
TH_IDX = slice(6, 9)
BA_IDX = slice(9, 12)
BG_IDX = slice(12, 15)


class ErrorStateEKF:

    def __init__(self, gravity=9.81, accel_noise=0.05, gyro_noise=0.003,
                 accel_bias_walk=0.002, gyro_bias_walk=0.0005,
                 leg_velocity_sigma=0.05, foot_position_sigma=0.01,
                 gravity_sigma=0.3, gravity_tolerance=0.8, slip_gate=16.0):
        self.g = np.array([0.0, 0.0, -gravity])
        self.gravity = gravity
        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise
        self.accel_bias_walk = accel_bias_walk
        self.gyro_bias_walk = gyro_bias_walk
        self.leg_velocity_sigma = leg_velocity_sigma
        self.foot_position_sigma = foot_position_sigma
        self.gravity_sigma = gravity_sigma
        self.gravity_tolerance = gravity_tolerance
        self.slip_gate = slip_gate        # chi-square threshold (3 dof); 16 ~ 99.9 %

        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        self.P = np.eye(15)
        self.anchors = {}          # foot index -> world position where it touched down
        self.slips = 0             # number of detected foot slips
        self.ground_z = None       # flat floor: height of a planted foot centre (None = unknown)

    def initialize(self, p, q, v=None):
        self.p = np.array(p, dtype=float)
        self.q = q_normalize(np.array(q, dtype=float))
        self.v = np.zeros(3) if v is None else np.array(v, dtype=float)
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        self.anchors = {}
        self.P = np.diag(np.concatenate([
            np.full(3, 0.01 ** 2),      # position
            np.full(3, 0.2 ** 2),       # velocity
            np.full(3, 0.05 ** 2),      # attitude
            np.full(3, 0.1 ** 2),       # accel bias
            np.full(3, 0.02 ** 2)]))   # gyro bias

    # ----------------------------------------------------------------- prediction
    def predict(self, gyro, accel, dt):
        """Propagate with one IMU sample (gyro rad/s, accel m/s^2, both in the body frame)."""
        R = q_to_R(self.q)
        a = np.asarray(accel) - self.ba
        w = np.asarray(gyro) - self.bg
        a_world = R @ a + self.g

        F = np.zeros((15, 15))
        F[P_IDX, V_IDX] = np.eye(3)
        F[V_IDX, TH_IDX] = -R @ skew(a)
        F[V_IDX, BA_IDX] = -R
        F[TH_IDX, TH_IDX] = -skew(w)
        F[TH_IDX, BG_IDX] = -np.eye(3)
        Fd = np.eye(15) + F * dt + 0.5 * (F @ F) * dt * dt

        self.p = self.p + self.v * dt + 0.5 * a_world * dt * dt
        self.v = self.v + a_world * dt
        self.q = q_normalize(q_mult(self.q, q_exp(w * dt)))

        q_diag = np.concatenate([
            np.full(3, (0.5 * self.accel_noise * dt * dt) ** 2),
            np.full(3, (self.accel_noise * dt) ** 2),
            np.full(3, (self.gyro_noise * dt) ** 2),
            np.full(3, self.accel_bias_walk ** 2 * dt),
            np.full(3, self.gyro_bias_walk ** 2 * dt)])
        self.P = Fd @ self.P @ Fd.T + np.diag(q_diag)
        self.P = 0.5 * (self.P + self.P.T)

    # ----------------------------------------------------------------- generic update
    def _update(self, innovation, H, sigma, gate=None):
        """Kalman update. If `gate` is given and the normalised innovation squared exceeds it,
        the measurement is rejected (returns False): an outlier such as a slipping foot."""
        Rm = np.eye(len(innovation)) * sigma ** 2
        S = H @ self.P @ H.T + Rm
        S_inv = np.linalg.inv(S)
        if gate is not None and innovation @ S_inv @ innovation > gate:
            return False
        K = self.P @ H.T @ S_inv
        dx = K @ innovation
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T      # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        self.p = self.p + dx[P_IDX]
        self.v = self.v + dx[V_IDX]
        self.q = q_normalize(q_mult(self.q, q_exp(dx[TH_IDX])))
        self.ba = self.ba + dx[BA_IDX]
        self.bg = self.bg + dx[BG_IDX]
        return True

    # ----------------------------------------------------------------- corrections
    def update_gravity(self, accel):
        """Accelerometer as a gravity-direction measurement (only when nearly un-accelerated)."""
        accel = np.asarray(accel)
        if abs(np.linalg.norm(accel) - self.gravity) > self.gravity_tolerance:
            return False
        R = q_to_R(self.q)
        up_body = R.T @ np.array([0.0, 0.0, self.gravity])
        H = np.zeros((3, 15))
        H[:, BA_IDX] = np.eye(3)
        H[:, TH_IDX] = skew(up_body)
        self._update(accel - (self.ba + up_body), H, self.gravity_sigma)
        return True

    def update_velocity(self, v_body_measured):
        """Measurement of the body-frame velocity R^T v (from a stance foot). False if rejected."""
        R = q_to_R(self.q)
        predicted = R.T @ self.v
        H = np.zeros((3, 15))
        H[:, V_IDX] = R.T
        H[:, TH_IDX] = skew(predicted)
        return self._update(np.asarray(v_body_measured) - predicted, H, self.leg_velocity_sigma,
                            self.slip_gate)

    def update_foot_anchor(self, foot_index, r_body):
        """Stance foot is fixed in the world: p + R r must equal its anchor position.

        A residual far outside its expected spread means the foot slipped: the foot is then
        re-anchored where it is now instead of corrupting position and attitude. Returns False
        on slip."""
        R = q_to_R(self.q)
        predicted = self.p + R @ r_body
        H = np.zeros((3, 15))
        H[:, P_IDX] = np.eye(3)
        H[:, TH_IDX] = -R @ skew(r_body)
        if self._update(self.anchors[foot_index] - predicted, H, self.foot_position_sigma,
                        self.slip_gate):
            return True
        self.anchors[foot_index] = predicted.copy()
        if self.ground_z is not None:
            self.anchors[foot_index][2] = self.ground_z
        return False

    def update_stance(self, stance, feet_r, feet_rdot, gyro):
        """Apply the leg-kinematics corrections for all feet that are in stance.

        stance: list of bool; feet_r / feet_rdot: body-frame foot position / velocity.
        """
        omega = np.asarray(gyro) - self.bg
        used = 0
        for i, in_stance in enumerate(stance):
            if not in_stance:
                self.anchors.pop(i, None)
                continue
            if i not in self.anchors:                      # touchdown: remember where
                self.anchors[i] = self.p + q_to_R(self.q) @ feet_r[i]
                if self.ground_z is not None:
                    self.anchors[i][2] = self.ground_z     # flat floor: no vertical drift
            elif not self.update_foot_anchor(i, feet_r[i]):
                self.slips += 1                            # slipping: no corrections from it
                continue
            if self.update_velocity(body_velocity_from_foot(feet_r[i], feet_rdot[i], omega)):
                used += 1
        return used
