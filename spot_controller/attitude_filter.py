"""Mahony complementary filter: roll/pitch from gyro + accelerometer only."""
import math

import numpy as np

from .quaternion import q_exp, q_from_rpy, q_mult, q_normalize, q_to_R, q_to_rpy


class MahonyFilter:
    """Gyro integration (fast, drifts) corrected toward the accelerometer's gravity direction.

    Correction term: e = a_measured_up x a_predicted_up, fed back as an angular-rate
    correction (proportional gain kp) and a gyro-bias correction (integral gain ki).
    Yaw is not observable from gravity, so it only follows the gyro.
    """

    def __init__(self, kp=2.0, ki=0.05, gravity=9.81, accel_tolerance=0.8):
        self.kp = kp
        self.ki = ki
        self.gravity = gravity
        self.accel_tolerance = accel_tolerance
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.bias = np.zeros(3)

    def initialize(self, accel, yaw=0.0):
        roll = math.atan2(accel[1], accel[2])
        pitch = math.atan2(-accel[0], math.hypot(accel[1], accel[2]))
        self.q = q_from_rpy(roll, pitch, yaw)
        self.bias = np.zeros(3)

    def update(self, gyro, accel, dt):
        R = q_to_R(self.q)
        up_pred = R[2, :]                          # R^T * e_z : world "up" in the body frame
        error = np.zeros(3)
        norm = np.linalg.norm(accel)
        if abs(norm - self.gravity) < self.accel_tolerance:
            error = np.cross(accel / norm, up_pred)
            self.bias -= self.ki * error * dt
        omega = np.asarray(gyro) - self.bias + self.kp * error
        self.q = q_normalize(q_mult(self.q, q_exp(omega * dt)))

    def rpy(self):
        return q_to_rpy(self.q)
