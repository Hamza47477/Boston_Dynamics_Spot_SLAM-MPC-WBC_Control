"""Offline tests of the attitude filter and EKF on a synthetic trajectory (no ROS, no Webots).

Truth: the body moves and tilts while four feet stay fixed on the ground. The IMU readings are
generated from the truth, with constant bias and Gaussian noise added.
"""
import math

import numpy as np
import pytest

from spot_controller.attitude_filter import MahonyFilter
from spot_controller.ekf import ErrorStateEKF
from spot_controller.quaternion import (q_from_rpy, q_to_R, q_to_rpy, skew, wrap_angle)
from spot_controller.stance import detect_stance

G = 9.81
DT = 0.03
FEET_WORLD = np.array([[0.36, 0.17, 0.0], [0.36, -0.17, 0.0],
                       [-0.31, 0.17, 0.0], [-0.31, -0.17, 0.0]])
GYRO_BIAS = np.array([0.010, -0.008, 0.005])
ACCEL_BIAS = np.array([0.05, -0.04, 0.06])
YAW0 = 1.83


def truth_pos(t):
    return np.array([0.10 * math.sin(0.5 * t), 0.08 * math.sin(0.7 * t),
                     0.50 + 0.03 * math.sin(1.2 * t)])


def truth_rpy(t):
    return (0.08 * math.sin(1.5 * t), 0.06 * math.sin(1.2 * t),
            YAW0 + 0.2 * math.sin(0.4 * t))


def truth_R(t):
    return q_to_R(q_from_rpy(*truth_rpy(t)))


def foot_body(t, i):
    return truth_R(t).T @ (FEET_WORLD[i] - truth_pos(t))


def truth_accel_world(t, h=1e-3):
    return (truth_pos(t + h) - 2 * truth_pos(t) + truth_pos(t - h)) / h ** 2


def truth_omega_body(t, h=1e-5):
    dR = (truth_R(t + h) - truth_R(t - h)) / (2 * h)
    S = truth_R(t).T @ dR
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def run(duration=30.0, seed=3, use_legs=True):
    rng = np.random.default_rng(seed)
    ekf = ErrorStateEKF()
    cf = MahonyFilter()

    def imu(t):
        a = truth_R(t).T @ (truth_accel_world(t) + np.array([0, 0, G]))
        w = truth_omega_body(t)
        return (w + GYRO_BIAS + rng.normal(0, 0.003, 3),
                a + ACCEL_BIAS + rng.normal(0, 0.05, 3))

    gyro, accel = imu(0.0)
    ekf.initialize(truth_pos(0.0), q_from_rpy(*truth_rpy(0.0)))   # v = 0 (truth is not 0)
    cf.initialize(accel, yaw=truth_rpy(0.0)[2])

    log = []
    n = int(duration / DT)
    for k in range(1, n + 1):
        t = k * DT
        gyro, accel = imu(t)
        cf.update(gyro, accel, DT)
        ekf.predict(gyro, accel, DT)
        ekf.update_gravity(accel)
        if use_legs:
            feet_r, feet_rdot = [], []
            for i in range(4):
                r = foot_body(t, i)
                rdot = (foot_body(t + 1e-5, i) - foot_body(t - 1e-5, i)) / 2e-5
                feet_r.append(r + rng.normal(0, 0.002, 3))
                feet_rdot.append(rdot + rng.normal(0, 0.01, 3))
            stance = detect_stance(feet_r, q_to_R(ekf.q))
            ekf.update_stance(stance, feet_r, feet_rdot, gyro)
        true_v = (truth_pos(t + 1e-5) - truth_pos(t - 1e-5)) / 2e-5
        e_rpy = np.array(q_to_rpy(ekf.q)) - np.array(truth_rpy(t))
        e_rpy[2] = wrap_angle(e_rpy[2])
        c_rpy = np.array(cf.rpy()) - np.array(truth_rpy(t))
        log.append((t, np.linalg.norm(ekf.p - truth_pos(t)), np.linalg.norm(ekf.v - true_v),
                    np.degrees(np.abs(e_rpy)), np.degrees(np.abs(c_rpy[:2]))))
    return ekf, cf, log


def test_detect_stance_marks_raised_foot_as_swing():
    feet = [np.array([0.3, 0.17, -0.5]), np.array([0.3, -0.17, -0.5]),
            np.array([-0.3, 0.17, -0.5]), np.array([-0.3, -0.17, -0.42])]
    assert detect_stance(feet, np.eye(3)) == [True, True, True, False]


def test_complementary_filter_roll_pitch_error_below_1_5_deg():
    _, _, log = run(use_legs=False)
    cf_err = np.array([entry[4] for entry in log[200:]])       # after 6 s settling
    assert cf_err.max() < 1.5


def test_ekf_tracks_position_velocity_and_attitude():
    ekf, _, log = run()
    late = log[300:]                                           # after 9 s
    assert max(e[1] for e in late) < 0.03                      # position error [m]
    assert max(e[2] for e in late) < 0.05                      # velocity error [m/s]
    assert max(e[3][0] for e in late) < 1.5                    # roll error [deg]
    assert max(e[3][1] for e in late) < 1.5                    # pitch error [deg]
    assert max(e[3][2] for e in late) < 2.0                    # yaw error [deg]


def test_ekf_estimates_gyro_bias():
    ekf, _, _ = run(duration=60.0)
    assert np.linalg.norm(ekf.bg - GYRO_BIAS) < 0.004


def test_leg_measurements_help_versus_imu_only():
    _, _, with_legs = run(duration=20.0)
    _, _, imu_only = run(duration=20.0, use_legs=False)
    assert with_legs[-1][1] < 0.1 * imu_only[-1][1]


def test_velocity_measurement_jacobian_matches_finite_difference():
    ekf = ErrorStateEKF()
    ekf.initialize([0, 0, 0.5], q_from_rpy(0.1, -0.2, 0.7), v=[0.3, -0.1, 0.05])
    R = q_to_R(ekf.q)
    dtheta = np.array([1e-6, -2e-6, 1.5e-6])
    R_pert = q_to_R(ekf.q) @ q_to_R(np.concatenate(([1.0], 0.5 * dtheta)))
    numeric = R_pert.T @ ekf.v - R.T @ ekf.v
    analytic = skew(R.T @ ekf.v) @ dtheta
    assert numeric == pytest.approx(analytic, abs=1e-10)


def test_foot_slip_does_not_corrupt_attitude():
    rng = np.random.default_rng(5)
    ekf = ErrorStateEKF()
    ekf.initialize(truth_pos(0.0), q_from_rpy(*truth_rpy(0.0)))
    slip = np.array([0.05, -0.03, 0.0])          # foot 0 slides 5 cm at t = 12 s
    worst = 0.0
    for k in range(1, int(30.0 / DT) + 1):
        t = k * DT
        accel = (truth_R(t).T @ (truth_accel_world(t) + np.array([0, 0, G])) + ACCEL_BIAS
                 + rng.normal(0, 0.05, 3))
        gyro = truth_omega_body(t) + GYRO_BIAS + rng.normal(0, 0.003, 3)
        ekf.predict(gyro, accel, DT)
        ekf.update_gravity(accel)
        feet_r, feet_rdot = [], []
        for i in range(4):
            world_foot = FEET_WORLD[i] + (slip if (i == 0 and t > 12.0) else 0.0)
            r = truth_R(t).T @ (world_foot - truth_pos(t))
            rdot = (foot_body(t + 1e-5, i) - foot_body(t - 1e-5, i)) / 2e-5
            feet_r.append(r + rng.normal(0, 0.002, 3))
            feet_rdot.append(rdot + rng.normal(0, 0.01, 3))
        ekf.update_stance([True] * 4, feet_r, feet_rdot, gyro)
        if t > 14.0:
            err = np.degrees(np.abs(np.array(q_to_rpy(ekf.q)) - np.array(truth_rpy(t))))
            worst = max(worst, err[0], err[1])
    assert ekf.slips >= 1
    assert worst < 2.0


def test_flat_ground_anchor_removes_height_drift():
    rng = np.random.default_rng(7)
    ekf = ErrorStateEKF()
    ekf.initialize(truth_pos(0.0) + np.array([0.0, 0.0, 0.06]), q_from_rpy(*truth_rpy(0.0)))
    ekf.ground_z = 0.0                                   # feet centres on the floor plane z = 0
    for k in range(1, int(20.0 / DT) + 1):
        t = k * DT
        accel = (truth_R(t).T @ (truth_accel_world(t) + np.array([0, 0, G])) + ACCEL_BIAS
                 + rng.normal(0, 0.05, 3))
        gyro = truth_omega_body(t) + GYRO_BIAS + rng.normal(0, 0.003, 3)
        ekf.predict(gyro, accel, DT)
        ekf.update_gravity(accel)
        feet_r = [foot_body(t, i) + rng.normal(0, 0.002, 3) for i in range(4)]
        feet_rdot = [(foot_body(t + 1e-5, i) - foot_body(t - 1e-5, i)) / 2e-5
                     + rng.normal(0, 0.01, 3) for i in range(4)]
        ekf.update_stance([True] * 4, feet_r, feet_rdot, gyro)
    assert abs(ekf.p[2] - truth_pos(20.0)[2]) < 0.015
