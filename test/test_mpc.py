import math

import numpy as np
import pytest

from rigid_body_sim import HEIGHT, Sim


def test_mpc_stands_and_supports_weight_inside_friction_cone():
    sim = Sim()
    assert sim.run(4.0)
    assert sim.pc[2] == pytest.approx(HEIGHT, abs=0.02)
    assert sim.f[:, 2].sum() == pytest.approx(27.2 * 9.81, rel=0.05)
    assert sim.max_friction_violation < 1.0
    assert sim.failures == 0


def test_mpc_tracks_forward_velocity_while_trotting():
    sim = Sim()
    sim.cmd = np.array([0.3, 0.0, 0.0])
    assert sim.run(10.0)
    assert np.linalg.norm(sim.v[:2]) == pytest.approx(0.3, abs=0.06)
    assert max(abs(sim.rpy()[0]), abs(sim.rpy()[1])) < math.radians(8)
    assert sim.max_friction_violation < 1.0


def test_mpc_walking_survives_a_sideways_push():
    sim = Sim()
    sim.cmd = np.array([0.3, 0.0, 0.0])
    assert sim.run(5.0)
    sim.push = (np.array([0.0, 120.0, 0.0]), sim.t + 0.2)
    assert sim.run(6.0)
    assert np.linalg.norm(sim.v[:2]) == pytest.approx(0.3, abs=0.1)


def test_mpc_solve_time_is_reasonable():
    sim = Sim()
    sim.cmd = np.array([0.3, 0.0, 0.0])
    sim.run(3.0)
    assert np.median(sim.solve_times) < 0.05
