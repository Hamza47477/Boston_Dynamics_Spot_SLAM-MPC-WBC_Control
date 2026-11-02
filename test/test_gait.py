import math

import numpy as np
import pytest

from spot_controller.constants import JOINT_LIMITS, JOINT_NAMES, LEG_ORDER, LEG_SIDE
from spot_controller.gait import quintic, raibert_target, swing_position, TrotGait
from spot_controller.locomotion_core import LocomotionCore
from spot_controller.spot_4leg_ik import Spot4LegsIK


def run_gait(gait, seconds, want, dt=0.008):
    pattern = []
    for _ in range(int(seconds / dt)):
        gait.update(dt, want)
        pattern.append(tuple(gait.stance_flags()))
    return pattern


def test_trot_pairs_move_together_and_one_pair_always_down():
    pattern = run_gait(TrotGait(), 3.0, True)
    for fl, fr, rl, rr in pattern:
        assert fl == rr and fr == rl                # diagonal pairs
        assert fl or fr                             # never both pairs in the air
    assert any(all(p) for p in pattern)             # four-foot phases exist
    assert any(not p[0] for p in pattern) and any(not p[1] for p in pattern)


def test_stopping_lands_swing_feet_then_stands_on_four():
    gait = TrotGait()
    run_gait(gait, 1.3, True)
    run_gait(gait, 1.0, False)
    assert gait.stance_flags() == [True] * 4 and not gait.active


def test_restart_after_stop_begins_in_stance():
    gait = TrotGait()
    run_gait(gait, 1.0, True)
    run_gait(gait, 1.0, False)
    gait.update(0.008, True)
    assert gait.stance_flags() == [True] * 4


def test_swing_arch_endpoints_and_apex():
    lo, goal = np.array([0.0, 0.0, 0.03]), np.array([0.1, 0.02, 0.03])
    assert swing_position(lo, goal, 0.0, 0.07, 0.03) == pytest.approx(lo)
    assert swing_position(lo, goal, 1.0, 0.07, 0.03) == pytest.approx(goal)
    assert swing_position(lo, goal, 0.5, 0.07, 0.03)[2] == pytest.approx(0.10)
    assert quintic(0.5) == pytest.approx(0.5)


def test_raibert_steps_ahead_when_moving_and_slows_when_too_fast():
    nominal = [0.4, 0.17]
    at_rest = raibert_target([0, 0], 0.0, nominal, [0, 0], [0, 0], 0.0, 0.42)
    assert at_rest == pytest.approx(nominal)
    walking = raibert_target([0, 0], 0.0, nominal, [0.3, 0], [0.3, 0], 0.0, 0.42)
    assert walking[0] == pytest.approx(0.4 + 0.3 * 0.21)
    too_fast = raibert_target([0, 0], 0.0, nominal, [0.5, 0], [0.2, 0], 0.0, 0.42)
    assert too_fast[0] > walking[0]                 # extra step ahead, to brake
    pushed_left = raibert_target([0, 0], 0.0, nominal, [0, 0.4], [0, 0], 0.0, 0.42)
    assert pushed_left[1] > nominal[1] + 0.05       # steps toward the push direction


def standing_state(height=0.5):
    kin = Spot4LegsIK()
    feet = {leg: [kin.hip_positions[leg][0], LEG_SIDE[leg] * 0.17, 0.0] for leg in LEG_ORDER}
    ang = kin.compute_body_ik_webots([0, 0, height], (0, 0, 0), feet)
    q = {leg: tuple(ang[3 * i:3 * i + 3]) for i, leg in enumerate(LEG_ORDER)}
    return q, {leg: (0.0, 0.0, 0.0) for leg in LEG_ORDER}


def test_locomotion_core_stands_supports_weight_and_respects_joint_limits():
    q, qd = standing_state()
    core = LocomotionCore(height=0.5)
    out = None
    for k in range(600):
        out = core.compute(k * 0.008, np.array([0, 0, 0.5]), np.eye(3), np.zeros(3),
                           np.zeros(3), q, qd)
    assert out.status[8] == pytest.approx(27.2 * 9.81, rel=0.03)      # total Fz
    assert out.status[7] == 4
    for name, value in zip(JOINT_NAMES, out.q_des):
        assert JOINT_LIMITS[name][0] <= value <= JOINT_LIMITS[name][1]


def test_locomotion_core_walks_and_swing_legs_get_zero_feedforward_torque():
    q, qd = standing_state()
    core = LocomotionCore(height=0.5)
    core.set_command(0.3, 0.0, 0.0)
    seen_swing = False
    for k in range(1500):
        out = core.compute(k * 0.008, np.array([0, 0, 0.5]), np.eye(3), np.zeros(3),
                           np.zeros(3), q, qd)
        flags = core.gait.stance_flags()
        for i, in_stance in enumerate(flags):
            if not in_stance:
                seen_swing = True
                assert out.tau_ff[3 * i:3 * i + 3] == [0.0, 0.0, 0.0]
                assert all(math.isfinite(x) for x in out.q_des[3 * i:3 * i + 3])
    assert seen_swing and out.status[9] == 1.0


def test_push_while_standing_triggers_recovery_steps_then_stops():
    q, qd = standing_state()
    core = LocomotionCore(height=0.5)
    t = 0.0
    for _ in range(600):
        core.compute(t, np.array([0, 0, 0.5]), np.eye(3), np.zeros(3), np.zeros(3), q, qd)
        t += 0.008
    for _ in range(100):                            # shoved: 0.5 m/s sideways
        core.compute(t, np.array([0, 0, 0.5]), np.eye(3), np.array([0, 0.5, 0]), np.zeros(3),
                     q, qd)
        t += 0.008
    assert core.gait.active
    for _ in range(int(3.0 / 0.008)):               # calm again
        core.compute(t, np.array([0, 0, 0.5]), np.eye(3), np.zeros(3), np.zeros(3), q, qd)
        t += 0.008
    assert not core.gait.active
