import math

import pytest

from spot_controller.constants import (HIP_MOUNT, HIP_OFFSET_Y, JOINT_LIMITS, JOINT_NAMES,
                                       LEG_ORDER, LEG_SIDE)
from spot_controller.ik_solver import SpotIK
from spot_controller.spot_4leg_ik import Spot4LegsIK
from spot_controller.body_state import fit_ground_plane

STANCE = {leg: [HIP_MOUNT[leg][0], LEG_SIDE[leg] * (abs(HIP_MOUNT['FL'][1]) + HIP_OFFSET_Y), 0.0]
          for leg in LEG_ORDER}


@pytest.mark.parametrize('target', [(0.0, 0.1172, -0.5), (0.1, 0.15, -0.45), (-0.08, 0.1, -0.55)])
def test_leg_ik_fk_round_trip(target):
    leg = SpotIK()
    assert leg.compute_fk(*leg.compute_ik(*target)) == pytest.approx(target, abs=1e-9)


def test_unreachable_foot_raises():
    with pytest.raises(ValueError):
        SpotIK().compute_ik(0.0, 0.1172, -1.5)


@pytest.mark.parametrize('height,rpy', [(0.5, (0, 0, 0)), (0.47, (0.08, 0.06, 0.05))])
def test_body_ik_within_joint_limits(height, rpy):
    angles = Spot4LegsIK().compute_body_ik_webots([0, 0, height], rpy, STANCE)
    for name, value in zip(JOINT_NAMES, angles):
        low, high = JOINT_LIMITS[name]
        assert low <= value <= high, name


def test_fk_recovers_height_and_level_attitude():
    kin = Spot4LegsIK()
    angles = kin.compute_body_ik_webots([0, 0, 0.5], (0, 0, 0), STANCE)
    per_leg = {leg: tuple(angles[3 * i:3 * i + 3]) for i, leg in enumerate(LEG_ORDER)}
    feet = kin.compute_body_fk(per_leg)
    height, roll, pitch = fit_ground_plane([feet[leg] for leg in LEG_ORDER])
    assert height == pytest.approx(0.5, abs=1e-6)
    assert math.degrees(roll) == pytest.approx(0.0, abs=1e-6)
    assert math.degrees(pitch) == pytest.approx(0.0, abs=1e-6)
