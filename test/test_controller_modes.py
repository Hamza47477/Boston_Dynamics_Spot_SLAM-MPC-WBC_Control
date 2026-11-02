import numpy as np
import pytest

from spot_controller.constants import JOINT_LIMITS, JOINT_NAMES, LEG_ORDER, LEG_SIDE, HIP_MOUNT
from spot_controller.locomotion_core import LocomotionCore
from spot_controller.spot_4leg_ik import Spot4LegsIK


def standing_state(height=0.5):
    kin = Spot4LegsIK()
    feet = {l: [HIP_MOUNT[l][0], LEG_SIDE[l] * 0.17, 0.0] for l in LEG_ORDER}
    ang = kin.compute_body_ik_webots([0, 0, height], (0, 0, 0), feet)
    q = {l: tuple(ang[3 * i:3 * i + 3]) for i, l in enumerate(LEG_ORDER)}
    return q, {l: (0.0, 0.0, 0.0) for l in LEG_ORDER}


@pytest.mark.parametrize('controller', ['vmc', 'mpc', 'mpc_wbc'])
def test_every_controller_mode_runs_a_walking_sequence_cleanly(controller):
    q, qd = standing_state()
    core = LocomotionCore(controller=controller, height=0.5, com_mode='model')
    core.set_command(0.3, 0.0, 0.2)
    weight = 27.2 * 9.81
    for k in range(700):
        out = core.compute(k * 0.008, np.array([0, 0, 0.5]), np.eye(3), np.array([0.3, 0, 0]),
                           np.zeros(3), q, qd)
        assert np.all(np.isfinite(out.tau_ff)) and np.all(np.isfinite(out.q_des))
        assert np.max(np.abs(out.tau_ff)) < 80.0
    for name, value in zip(JOINT_NAMES, out.q_des):
        assert JOINT_LIMITS[name][0] <= value <= JOINT_LIMITS[name][1]
    assert core.mpc_fallbacks == 0 and core.wbc_fallbacks == 0
    assert out.status[9] == 1.0                                     # gait running
    assert 0.3 * weight < out.status[8] < 1.6 * weight              # supports roughly the weight


def test_mpc_and_force_controller_agree_on_standing_load():
    q, qd = standing_state()
    totals = {}
    for controller in ('vmc', 'mpc'):
        core = LocomotionCore(controller=controller, height=0.5, com_mode='model')
        for k in range(200):
            out = core.compute(k * 0.008, np.array([0, 0, 0.5]), np.eye(3), np.zeros(3),
                               np.zeros(3), q, qd)
        totals[controller] = out.status[8]
    assert totals['mpc'] == pytest.approx(totals['vmc'], rel=0.05)
