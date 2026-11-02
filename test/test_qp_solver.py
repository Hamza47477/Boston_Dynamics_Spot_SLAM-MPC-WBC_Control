import time

import numpy as np
import pytest

from spot_controller.qp_solver import solve_qp


def test_box_constrained_projection():
    c = np.array([2.0, -3.0, 0.5])
    res = solve_qp(np.eye(3), -c, np.eye(3), -np.ones(3), np.ones(3))
    assert res.converged
    assert res.x == pytest.approx(np.clip(c, -1, 1), abs=1e-3)


def test_equality_constrained_least_norm():
    # min 0.5|x|^2  s.t. x1 + x2 + x3 = 3  ->  x = (1, 1, 1)
    res = solve_qp(np.eye(3), np.zeros(3), np.ones((1, 3)), [3.0], [3.0])
    assert res.converged
    assert res.x == pytest.approx(np.ones(3), abs=1e-3)


def test_kkt_conditions_on_random_problems():
    rng = np.random.default_rng(0)
    for _ in range(5):
        n, m = 12, 20
        M = rng.normal(size=(n, n))
        P = M @ M.T + np.eye(n)
        q = rng.normal(size=n)
        A = rng.normal(size=(m, n))
        x_feas = rng.normal(size=n)
        l = A @ x_feas - rng.uniform(0.1, 1.0, m)
        u = A @ x_feas + rng.uniform(0.1, 1.0, m)
        res = solve_qp(P, q, A, l, u, eps=1e-6, max_iter=4000)
        assert res.converged
        Ax = A @ res.x
        assert np.all(Ax >= l - 1e-4) and np.all(Ax <= u + 1e-4)            # primal feasible
        assert np.max(np.abs(P @ res.x + q + A.T @ res.y)) < 1e-3          # stationarity


def test_friction_pyramid_problem_is_fast_enough():
    n = 120                                                                 # MPC-sized
    rng = np.random.default_rng(1)
    M = rng.normal(size=(n, n))
    P = M @ M.T / n + 0.1 * np.eye(n)
    q = rng.normal(size=n)
    A = np.vstack([rng.normal(size=(160, n)) * 0.1, np.eye(n)])
    l = np.concatenate([-np.ones(160), -np.ones(n)])
    u = np.concatenate([np.ones(160), np.ones(n)])
    start = time.perf_counter()
    res = solve_qp(P, q, A, l, u)
    elapsed = time.perf_counter() - start
    assert res.converged
    assert elapsed < 0.25
