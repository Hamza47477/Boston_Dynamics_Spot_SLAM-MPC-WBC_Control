import numpy as np
import pytest

from spot_controller.force_distribution import distribute_forces, grasp_matrix

M = 27.2
FEET = [np.array([0.34, 0.17, -0.5]), np.array([0.34, -0.17, -0.5]),
        np.array([-0.33, 0.17, -0.5]), np.array([-0.33, -0.17, -0.5])]


def test_weight_support_balances_force_and_moment():
    w = np.array([0, 0, M * 9.81, 0, 0, 0])
    f = distribute_forces(FEET, w)
    assert (grasp_matrix(FEET) @ f.reshape(-1)) == pytest.approx(w, abs=0.5)
    assert np.all(f[:, 2] > 0)


def test_com_offset_shifts_load_to_nearer_feet():
    feet = [r - np.array([0.05, 0, 0]) for r in FEET]       # CoM 5 cm ahead of the origin
    f = distribute_forces(feet, np.array([0, 0, M * 9.81, 0, 0, 0]))
    assert f[0, 2] + f[1, 2] > f[2, 2] + f[3, 2]


def test_friction_cone_is_respected_for_a_big_lateral_demand():
    w = np.array([400.0, 0, M * 9.81, 0, 0, 0])
    f = distribute_forces(FEET, w, mu=0.6)
    assert np.all(np.abs(f[:, 0]) <= 0.6 * f[:, 2] + 1e-6)
    assert np.all(f[:, 2] >= 1.0 - 1e-9)


def test_pitch_moment_is_produced():
    w = np.array([0, 0, M * 9.81, 0, 30.0, 0])
    f = distribute_forces(FEET, w)
    assert (grasp_matrix(FEET) @ f.reshape(-1))[4] == pytest.approx(30.0, abs=1.0)


def test_rotation_matrix_quaternion_round_trip():
    from spot_controller.quaternion import q_from_rpy, q_to_R, R_to_q
    for rpy in [(0.1, -0.2, 0.7), (3.0, 0.2, -2.5), (0.0, 1.4, 0.3)]:
        R = q_to_R(q_from_rpy(*rpy))
        assert q_to_R(R_to_q(R)) == pytest.approx(R, abs=1e-12)


def test_rot_error_is_zero_at_target_and_signed():
    from spot_controller.quaternion import q_from_rpy, q_to_R
    from spot_controller.control_math import rot_error
    R_des = q_to_R(q_from_rpy(0, 0, 0.5))
    assert rot_error(R_des, R_des) == pytest.approx(np.zeros(3), abs=1e-12)
    tilted = q_to_R(q_from_rpy(0.1, 0, 0.5))
    assert rot_error(R_des, tilted)[0] > 0


def test_jacobian_matches_foot_velocity_and_static_torque_sign():
    from spot_controller.control_math import leg_jacobian
    from spot_controller.spot_4leg_ik import Spot4LegsIK
    kin = Spot4LegsIK()
    q = {leg: (0.05, -0.38, 0.32) for leg in ('FL', 'FR', 'RL', 'RR')}
    J = leg_jacobian(kin, 'FL', q['FL'])
    dq = np.array([0.0, 0.0, 1e-4])
    q2 = dict(q)
    q2['FL'] = tuple(np.array(q['FL']) + dq)
    moved = np.array(kin.compute_body_fk(q2)['FL']) - np.array(kin.compute_body_fk(q)['FL'])
    assert moved == pytest.approx(J @ dq, abs=1e-8)
    # An upward ground reaction on the foot must be held by a nonzero joint torque
    assert np.linalg.norm(-J.T @ np.array([0, 0, 68.0])) > 1.0
