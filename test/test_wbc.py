import numpy as np
import pytest

from spot_controller.constants import HIP_MOUNT, LEG_ORDER, LEG_SIDE
from spot_controller.control_math import leg_jacobian
from spot_controller.dynamics import SpotDynamics
from spot_controller.gait import swing_kinematics, swing_position
from spot_controller.spot_4leg_ik import Spot4LegsIK
from spot_controller.wbc import WholeBodyController


def standing(height=0.5):
    kin = Spot4LegsIK()
    feet = {l: [HIP_MOUNT[l][0], LEG_SIDE[l] * 0.17, 0.0] for l in LEG_ORDER}
    ang = kin.compute_body_ik_webots([0, 0, height], (0, 0, 0), feet)
    return kin, {l: tuple(ang[3 * i:3 * i + 3]) for i, l in enumerate(LEG_ORDER)}


def test_mass_matrix_is_symmetric_positive_definite_with_total_mass():
    _, q = standing()
    dyn = SpotDynamics()
    M, h = dyn.terms(np.eye(3), q, np.zeros(18))
    assert np.abs(M - M.T).max() < 1e-9
    assert np.linalg.eigvalsh(M).min() > 0.0
    assert np.diag(M[:3, :3]) == pytest.approx([dyn.total_mass] * 3)
    assert h[2] == pytest.approx(dyn.total_mass * 9.81)             # weight acts on the base


def test_static_torques_match_jacobian_transpose_mapping():
    kin, q = standing()
    dyn = SpotDynamics()
    R, u = np.eye(3), np.zeros(18)
    M, h = dyn.terms(R, q, u)
    Jc, _ = dyn.contact_jacobians(R, q, u, [0, 1, 2, 3])
    F = np.linalg.lstsq(Jc[:, :6].T, h[:6], rcond=None)[0]          # forces that carry the weight
    tau_full = h[6:] - Jc[:, 6:].T @ F
    tau_jt = []
    for i, leg in enumerate(LEG_ORDER):
        tau_jt.extend(-leg_jacobian(kin, leg, q[leg]).T @ F.reshape(4, 3)[i])
    assert np.abs(tau_full - np.array(tau_jt)).max() < 3.0           # differs by leg-weight terms only


def test_wbc_reproduces_mpc_forces_when_standing():
    _, q = standing()
    wbc = WholeBodyController()
    F_des = np.zeros((4, 3))
    F_des[:, 2] = [62.6, 62.6, 71.0, 71.0]
    out = wbc.compute(np.eye(3), q, np.zeros(18), [0, 1, 2, 3], F_des, [], (np.zeros(3), 0.0))
    assert out['converged']
    assert np.abs(out['F'] - F_des).max() < 1.0
    assert np.abs(out['udd'][:6]).max() < 0.05
    assert np.abs(out['tau']).max() <= 50.0 + 1e-3


def test_wbc_achieves_swing_foot_acceleration_without_moving_stance_feet():
    _, q = standing()
    dyn = SpotDynamics()
    wbc = WholeBodyController(dyn)
    R, u = np.eye(3), np.zeros(18)
    F = np.zeros((2, 3))
    F[:, 2] = 133.5
    want = np.array([2.0, 0.0, 6.0])
    out = wbc.compute(R, q, u, [0, 3], F, [(1, want), (2, want)], (np.zeros(3), 0.0))
    assert out['converged']
    for foot in (1, 2):
        Js, bias = dyn.contact_jacobians(R, q, u, [foot])
        assert Js @ out['udd'] + bias == pytest.approx(want, abs=0.05)
    Jc, bias_c = dyn.contact_jacobians(R, q, u, [0, 3])
    assert Jc @ out['udd'] + bias_c == pytest.approx(np.zeros(6), abs=1e-3)   # planted feet fixed
    assert np.abs(out['tau']).max() <= 50.0 + 1e-3


def test_swing_kinematics_matches_numerical_derivatives():
    lo, goal = np.array([0.0, 0.0, 0.033]), np.array([0.12, 0.02, 0.033])
    T, s, d = 0.21, 0.4, 1e-4
    pos, vel, acc = swing_kinematics(lo, goal, s, 0.06, 0.033, T)
    p_plus = swing_position(lo, goal, s + d, 0.06, 0.033)
    p_minus = swing_position(lo, goal, s - d, 0.06, 0.033)
    assert pos == pytest.approx(swing_position(lo, goal, s, 0.06, 0.033))
    assert vel == pytest.approx((p_plus - p_minus) / (2 * d) / T, rel=0.02, abs=1e-3)
    assert acc == pytest.approx((p_plus - 2 * pos + p_minus) / d ** 2 / T ** 2, rel=0.03, abs=0.05)
