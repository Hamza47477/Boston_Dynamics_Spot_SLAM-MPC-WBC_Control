"""Reduced simulation used to test the MPC: one rigid body, planted feet, trot schedule, pushes."""
import math

import numpy as np

from spot_controller.constants import LEG_ORDER
from spot_controller.gait import raibert_target, TrotGait
from spot_controller.mpc import ConvexMPC, rot_z
from spot_controller.mpc_planner import plan_inputs
from spot_controller.quaternion import q_exp, q_to_R, q_to_rpy, R_to_q, skew

MASS = 27.2
I_BODY = np.diag([0.55, 2.4, 2.5])
NOMINAL = [np.array(a) for a in ([0.406, 0.17], [0.406, -0.17], [-0.266, 0.17], [-0.266, -0.17])]
FOOT_Z = 0.033
HEIGHT = 0.50


class Sim:
    def __init__(self, mpc=None, control_dt=0.03, seed=0):
        self.mpc = mpc or ConvexMPC(MASS, I_BODY)
        self.control_dt = control_dt
        self.pc = np.array([0.0, 0.0, HEIGHT])
        self.v = np.zeros(3)
        self.R = np.eye(3)
        self.w = np.zeros(3)
        self.gait = TrotGait(0.6, 0.65, 0.06)
        self.feet = [np.array([n[0], n[1], FOOT_Z]) + np.array([0.0, 0.0, 0.0]) for n in NOMINAL]
        self.cmd = np.zeros(3)
        self.push = None                           # (force vector, end time)
        self.t = 0.0
        self.f = np.zeros((4, 3))
        self.hold = (0.0, 0.0, 0.0)
        self.max_friction_violation = 0.0
        self.solve_times = []
        self.failures = 0
        self.was_stance = [True] * 4

    def rpy(self):
        return q_to_rpy(R_to_q(self.R))

    def step_control(self):
        yaw = self.rpy()[2]
        moving = np.linalg.norm(self.cmd[:2]) > 0.02 or abs(self.cmd[2]) > 0.05
        speed = np.linalg.norm(self.v[:2])
        tilt = max(abs(self.rpy()[0]), abs(self.rpy()[1]))
        if not moving and (speed > 0.25 or tilt > math.radians(9)):
            self.recover_until = self.t + 1.2
        want = moving or self.t < getattr(self, 'recover_until', -1)
        was_active = self.gait.active
        self.gait.update(self.control_dt, want)
        if was_active and not self.gait.active:
            self.hold = (self.pc[0], self.pc[1], yaw)
        walking = self.gait.active
        v_des = rot_z(yaw) @ np.array([self.cmd[0], self.cmd[1], 0.0]) if moving else np.zeros(3)
        stance = self.gait.stance_flags()
        for i, leg in enumerate(LEG_ORDER):                      # touchdown: plant the foot
            if stance[i] and not self.was_stance[i]:
                td = raibert_target(self.pc[:2], yaw, NOMINAL[i], self.v[:2], v_des[:2],
                                    self.cmd[2] if moving else 0.0, 0.65 * 0.6, 0.2)
                self.feet[i] = np.array([td[0], td[1], FOOT_Z])
        self.was_stance = list(stance)
        inp = plan_inputs(self.mpc.N, self.mpc.dt, self.pc, self.R, self.v, np.linalg.inv(self.R) @ self.w,
                          self.feet, dict(self.gait.phase), self.gait.active, 0.65, 0.6,
                          v_des[:2], self.cmd[2] if moving else 0.0, self.hold[:2], self.hold[2],
                          walking, HEIGHT, FOOT_Z, NOMINAL, np.zeros(3))
        import time
        t0 = time.perf_counter()
        f = self.mpc.solve(inp['x0'], inp['x_ref'], inp['r_feet'], inp['contact'], inp['yaw_seq'])
        self.solve_times.append(time.perf_counter() - t0)
        if f is None:
            self.failures += 1
            f = self.f
        f = np.array(f)
        for i in range(4):
            if not stance[i]:
                f[i] = 0.0
        self.f = f

    def step_physics(self, dt):
        F = self.f.sum(axis=0) + MASS * np.array([0, 0, -9.81])
        if self.push is not None and self.t < self.push[1]:
            F = F + self.push[0]
        torque = np.zeros(3)
        for i in range(4):
            torque += np.cross(self.feet[i] - self.pc, self.f[i])
            if self.f[i, 2] > 1.0:
                self.max_friction_violation = max(
                    self.max_friction_violation, np.max(np.abs(self.f[i, :2])) - 0.6 * self.f[i, 2])
        I_w = self.R @ I_BODY @ self.R.T
        w_dot = np.linalg.solve(I_w, torque - np.cross(self.w, I_w @ self.w))
        self.v = self.v + F / MASS * dt
        self.pc = self.pc + self.v * dt
        self.w = self.w + w_dot * dt
        self.R = q_to_R(np.array([1.0, 0, 0, 0])) @ self.R if False else \
            (q_to_R(q_exp(self.w * dt)) @ self.R)
        self.t += dt

    def run(self, seconds, physics_dt=0.004):
        steps_per_control = int(round(self.control_dt / physics_dt))
        for _ in range(int(seconds / self.control_dt)):
            self.step_control()
            for _ in range(steps_per_control):
                self.step_physics(physics_dt)
            if self.pc[2] < 0.25 or max(abs(self.rpy()[0]), abs(self.rpy()[1])) > math.radians(45):
                return False
        return True
