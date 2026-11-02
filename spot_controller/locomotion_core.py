"""ROS-free locomotion controller: standing, trotting, push recovery.

Runs inside the plugin every simulation step (no communication delay) on the estimated state.

Stance legs  - virtual model control: a body wrench (velocity / height / attitude spring-dampers
               plus weight support) is split into ground reaction forces over the feet that are
               in stance (force_distribution.py); joint torques are tau = -J^T f.
Swing legs   - the foot follows a smooth arch to a Raibert touchdown point, joint-PD tracked.
Standing     - holds position; if it is pushed hard it takes recovery steps, then stops.
"""
import math
import time
from collections import namedtuple

import numpy as np

from .constants import (clamp_joint, COM_OFFSET, HIP_MOUNT, ik_to_webots, JOINT_KD, JOINT_KP,
                        JOINT_NAMES, LEG_ORDER, ROBOT_INERTIA, ROBOT_MASS)
from .control_math import leg_jacobian, rot_error, smoothstep
from .force_distribution import distribute_forces
from .gait import raibert_target, swing_kinematics, swing_position, TrotGait
from .leg_odometry import foot_states
from .dynamics import SpotDynamics
from .mpc import ConvexMPC
from .mpc_planner import plan_inputs
from .quaternion import q_from_rpy, q_to_R, q_to_rpy, R_to_q
from .spot_4leg_ik import Spot4LegsIK
from .wbc import WholeBodyController

GRAVITY = 9.81
RAMP_SECONDS = 3.0
STATUS_FIELDS = ['height_m', 'roll_deg', 'pitch_deg', 'yaw_deg', 'vx_body', 'vy_body',
                 'speed_cmd', 'n_stance', 'fz_total_n', 'gait_active', 'x_err_m', 'y_err_m']

Output = namedtuple('Output', 'q_des qd_des tau_ff kp kd stance_est status')

DEFAULTS = dict(
    height=0.50,
    # standing hold (position spring-damper, about 0.7 x critical damping for 27 kg)
    kp_pos=(900.0, 900.0, 1500.0), kd_pos=(220.0, 220.0, 300.0),
    # attitude (damping above about 14..18 destabilised the robot in simulation)
    kp_rot=(250.0, 300.0, 150.0), kd_rot=(14.0, 18.0, 10.0),
    mu=0.6, mu_wbc=0.75, max_xy_error=0.06, friction_margin=0.8,
    # walking
    kd_vel=(160.0, 160.0), ki_vel=(120.0, 120.0), max_vel_integral=50.0,
    period=0.6, duty=0.65, swing_height=0.06, k_raibert=0.2,
    swing_kp=160.0, swing_kd=4.0,
    max_step_offset=0.18, max_inward=0.03, max_fore_aft=0.18, vel_filter=0.15,
    load_ramp_up=0.10, load_ramp_down=0.05, max_yaw_lead=0.2,
    # push recovery while standing: start stepping above these, stop when calm
    recover_speed=0.25, recover_tilt_deg=9.0, recover_time=1.2,
    max_speed=(0.5, 0.25), max_yaw_rate=0.8,
    # controller: 'vmc' (spring-damper wrench + force distribution) or 'mpc' (convex MPC forces)
    com_mode='model', mpc_vel_ki=0.12, mpc_vel_int_max=0.04, controller='vmc', mpc_horizon=10, mpc_dt=0.03, mpc_every=1, mpc_q=None, mpc_alpha=1e-4,
    # whole-body control (controller='mpc_wbc'): task-space gains (1/s^2, 1/s)
    wbc_min_load=15.0, swing_task_kp=400.0, swing_task_kd=35.0, base_att_kp=60.0, base_att_kd=12.0,
    base_z_kp=60.0, base_z_kd=15.0)


def yaw_matrix(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class LocomotionCore:

    def __init__(self, kin=None, **params):
        self.kin = kin or Spot4LegsIK()
        self.params = dict(DEFAULTS, **params)
        prm = self.params
        self.gait = TrotGait(prm['period'], prm['duty'], prm['swing_height'])
        self.com = np.array(COM_OFFSET)
        zero = {leg: (0.0, 0.0, 0.0) for leg in LEG_ORDER}
        self.nominal = self.kin.compute_body_fk(zero)       # foot positions at the Webots zero pose
        self.cmd = np.zeros(3)                              # vx, vy (m/s), yaw rate (rad/s)
        self.hold = None                                    # [x0, y0, yaw0, z0, t0]
        self.foot_z = 0.0
        self.t_prev = None
        self.yaw_des = 0.0
        self.recover_until = -1.0
        self.vel_integral = np.zeros(2)
        self.v_filt = None
        self.use_mpc = str(prm['controller']).startswith('mpc')
        self.mpc = (ConvexMPC(ROBOT_MASS, ROBOT_INERTIA, horizon=prm['mpc_horizon'],
                              dt=prm['mpc_dt'], mu=prm['mu'], q_weights=prm['mpc_q'],
                              force_weight=prm['mpc_alpha']) if self.use_mpc else None)
        self.mpc_f = None
        self.mpc_vint = np.zeros(2)
        self.mpc_flags = None
        self.mpc_counter = 0
        self.mpc_info = (0, True, 0.0)                      # iterations, converged, solve seconds
        self.mpc_fallbacks = 0
        self.use_wbc = str(prm['controller']) == 'mpc_wbc'
        self.dyn = SpotDynamics() if (self.use_wbc or prm['com_mode'] == 'model') else None
        self.wbc = WholeBodyController(self.dyn, mu=prm['mu_wbc']) if self.use_wbc else None
        self.wbc_info = (0, True, 0.0)
        self.wbc_fallbacks = 0
        self.v_des_world = np.zeros(3)
        self.vel_integral = np.zeros(2)
        self.was_walking = False
        self.liftoff = {}
        self.prev_swing_q = {}
        self.debug_feet = None
        self.debug = {}                                     # per-leg info for the flight recorder

    # ------------------------------------------------------------------ helpers
    def set_command(self, vx, vy, wz):
        mx, my = self.params['max_speed']
        self.cmd = np.array([np.clip(vx, -mx, mx), np.clip(vy, -my, my),
                             np.clip(wz, -self.params['max_yaw_rate'],
                                     self.params['max_yaw_rate'])])

    def _leg_ik(self, leg, p_body, R_body, foot_world):
        """Webots joint angles that put a leg's foot at a world position; None if unreachable."""
        hip = np.array(HIP_MOUNT[leg])
        rel = R_body.T @ (foot_world - p_body) - hip
        try:
            return np.array(ik_to_webots(*self.kin.legs[leg].compute_ik(*rel)))
        except ValueError:
            return None

    def _limit_step(self, p_xy, yaw, nominal_xy, target, leg):
        """Keep a touchdown point near its nominal spot, and never inside the other legs' lane."""
        prm = self.params
        c, s = math.cos(yaw), math.sin(yaw)
        d = np.asarray(target) - np.asarray(p_xy)
        body = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])        # world -> body yaw frame
        dev = body - nominal_xy
        dev[0] = np.clip(dev[0], -prm['max_fore_aft'], prm['max_fore_aft'])
        side = 1.0 if leg in ('FL', 'RL') else -1.0                          # +1 left leg
        dev[1] = max(dev[1] * side, -prm['max_inward']) * side               # limit inward step
        body = nominal_xy + dev
        return np.asarray(p_xy) + np.array([c * body[0] - s * body[1], s * body[0] + c * body[1]])

    def _wbc_torques(self, R, v, w_body, q, qd, idx, f_world, feet_w, h_des, p, R_des,
                     w_des_body):
        """Joint torques from the whole-body QP; None if it fails (caller keeps J^T torques)."""
        prm = self.params
        u = np.concatenate([v, R @ w_body] + [np.array(qd[leg]) for leg in LEG_ORDER])
        swing = []
        for i, (pos_d, vel_d, acc_d) in self.swing_targets.items():
            Js, _ = self.dyn.foot_jacobian(R, q, i)
            vel_m = Js @ u
            a_cmd = (acc_d + prm['swing_task_kp'] * (pos_d - feet_w[i])
                     + prm['swing_task_kd'] * (vel_d - vel_m))
            swing.append((i, a_cmd))
        alpha_des = R @ (-prm['base_att_kp'] * rot_error(R_des, R)
                         - prm['base_att_kd'] * (w_body - w_des_body))
        z_acc = prm['base_z_kp'] * (h_des - p[2]) - prm['base_z_kd'] * v[2]
        # Feet with almost no planned load (about to lift off or just landed) are left free: a
        # planted-foot constraint with a ~0 N cone cannot hold the few newtons the others demand.
        loaded = [k for k in range(len(idx)) if f_world[k][2] > prm['wbc_min_load']]
        contact_feet = [idx[k] for k in loaded]
        F_loaded = np.asarray(f_world)[loaded] if loaded else np.zeros((0, 3))
        start = time.perf_counter()
        try:
            out = self.wbc.compute(R, q, u, contact_feet, F_loaded, swing, (alpha_des, z_acc))
        except (np.linalg.LinAlgError, ValueError):
            out = None
        res = self.wbc.last
        self.wbc_info = (res.iterations if res else 0, bool(res.converged) if res else False,
                         time.perf_counter() - start)
        if out is None or not out['converged'] or not np.all(np.isfinite(out['tau'])):
            self.wbc_fallbacks += 1
            return None
        return out['tau']

    def _mpc_forces(self, p, R, v, w_body, feet_w, stance, v_des, yaw_rate_cmd, walking, h_des,
                    hold):
        """Ground reaction forces (4, 3) from the convex MPC; re-solved every `mpc_every` steps and on contact changes."""
        prm = self.params
        flags = tuple(stance)
        if self.mpc_f is None or self.mpc_counter % prm['mpc_every'] == 0 or flags != self.mpc_flags:
            nominal = [np.array(self.nominal[leg][:2]) for leg in LEG_ORDER]
            inp = plan_inputs(prm['mpc_horizon'], prm['mpc_dt'], p, R, v, w_body, feet_w,
                              dict(self.gait.phase), self.gait.active, self.gait.duty,
                              self.gait.period, v_des[:2], yaw_rate_cmd, (hold[0], hold[1]),
                              hold[2], walking, h_des, self.foot_z, nominal, self.com,
                              self.v_filt, prm['k_raibert'], stance_time=self.gait.stance_time,
                              max_step=prm['max_step_offset'])
            start = time.perf_counter()
            f = self.mpc.solve(inp['x0'], inp['x_ref'], inp['r_feet'], inp['contact'],
                               inp['yaw_seq'])
            res = self.mpc.last_solve
            self.mpc_info = (res.iterations if res else 0, bool(res.converged) if res else False,
                             time.perf_counter() - start)
            if f is None:
                return None
            self.mpc_f = f
            self.mpc_flags = flags
        self.mpc_counter += 1
        f = self.mpc_f.copy()
        for i, in_stance in enumerate(stance):
            if not in_stance:
                f[i] = 0.0
        return f

    # ------------------------------------------------------------------ main step
    def compute(self, t, p, R, v, w_body, q, qd):
        """Return an Output. q, qd: {leg: (hip_x, hip_y, knee)}; p, R, v, w_body: estimate."""
        prm = self.params
        dt = 0.008 if self.t_prev is None else float(np.clip(t - self.t_prev, 1e-3, 0.05))
        self.t_prev = t
        feet_r, _ = foot_states(self.kin, q, qd)
        if prm['com_mode'] == 'model':                        # CoM follows the leg configuration
            self.com = self.dyn.com_body(q)
        feet_w = [p + R @ r for r in feet_r]
        self.debug_feet = [[round(float(a), 3) for a in r] for r in feet_r]
        roll, pitch, yaw = q_to_rpy(R_to_q(R))
        alpha = dt / (prm['vel_filter'] + dt)
        self.v_filt = v[:2].copy() if self.v_filt is None else self.v_filt + alpha * (v[:2] - self.v_filt)

        if self.hold is None:                                # first call
            self.foot_z = float(np.mean([f[2] for f in feet_w]))   # = foot radius on flat ground
            self.hold = [p[0], p[1], yaw, p[2], t]
            self.yaw_des = yaw
        x0, y0, yaw0, z0, t0 = self.hold

        # ---- mode: stand / walk / recover
        settled = (t - t0) > RAMP_SECONDS + 0.5
        cmd = self.cmd
        moving = settled and (np.linalg.norm(cmd[:2]) > 0.02 or abs(cmd[2]) > 0.05)
        speed = float(np.linalg.norm(v[:2]))
        tilt = max(abs(roll), abs(pitch))
        if settled and not moving and (speed > prm['recover_speed']
                                       or tilt > math.radians(prm['recover_tilt_deg'])):
            self.recover_until = t + prm['recover_time']      # shoved hard: take steps
        want_step = settled and (moving or t < self.recover_until)
        self.gait.update(dt, want_step)
        walking = self.gait.active
        if self.was_walking and not walking:                  # stopped: hold where we are now
            self.hold[0], self.hold[1], self.hold[2] = p[0], p[1], yaw
            x0, y0, yaw0 = p[0], p[1], yaw
            self.yaw_des = yaw
        if walking and not self.was_walking:
            self.yaw_des = yaw
        self.was_walking = walking

        # ---- body targets and wrench
        h_des = z0 + (prm['height'] - z0) * smoothstep((t - t0) / RAMP_SECONDS)
        yaw_rate_cmd = cmd[2] if moving else 0.0
        if walking:
            self.yaw_des += yaw_rate_cmd * dt
            lead = (self.yaw_des - yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw_des = yaw + float(np.clip(lead, -prm['max_yaw_lead'], prm['max_yaw_lead']))
            v_des = yaw_matrix(yaw) @ np.array([cmd[0], cmd[1], 0.0]) if moving \
                else np.zeros(3)
            p_des = np.array([p[0], p[1], h_des])
            vel_err = v_des[:2] - v[:2]
            self.vel_integral = np.clip(self.vel_integral + np.array(prm['ki_vel']) * vel_err * dt,
                                        -prm['max_vel_integral'], prm['max_vel_integral'])
            force_xy = np.array(prm['kd_vel']) * vel_err + self.vel_integral
        else:
            self.vel_integral[:] = 0.0
            self.yaw_des = yaw0
            v_des = np.zeros(3)
            p_des = np.array([x0, y0, h_des])
            err = np.clip(p_des[:2] - p[:2], -prm['max_xy_error'], prm['max_xy_error'])
            force_xy = np.array(prm['kp_pos'][:2]) * err - np.array(prm['kd_pos'][:2]) * v[:2]
        force_z = (ROBOT_MASS * GRAVITY + prm['kp_pos'][2] * (h_des - p[2])
                   - prm['kd_pos'][2] * v[2])
        force = np.array([force_xy[0], force_xy[1], max(force_z, 1.0)])
        limit = prm['friction_margin'] * prm['mu'] * force[2]   # stay inside the friction cone
        horizontal = np.linalg.norm(force[:2])
        if horizontal > limit:
            force[:2] *= limit / horizontal

        self.v_des_world = v_des
        R_des = q_to_R(q_from_rpy(0.0, 0.0, self.yaw_des))
        w_des_body = R.T @ np.array([0.0, 0.0, yaw_rate_cmd])
        moment_body = (-np.array(prm['kp_rot']) * rot_error(R_des, R)
                       - np.array(prm['kd_rot']) * (w_body - w_des_body))
        wrench = np.concatenate([force, R @ moment_body])

        # ---- stance legs: forces -> torques, joint targets from the desired body pose
        stance = self.gait.stance_flags()
        idx = [i for i, s in enumerate(stance) if s]
        tau = np.zeros(12)
        fz_total = 0.0
        if idx:
            rho = [R @ (feet_r[i] - self.com) for i in idx]
            scale = []
            for i in idx:
                leg = LEG_ORDER[i]
                up = min(1.0, 0.1 + self.gait.stance_time[leg] / prm['load_ramp_up'])
                to_lift = (self.gait.duty - self.gait.phase[leg]) * self.gait.period
                down = 1.0 if not self.gait.active else min(1.0, 0.1 + to_lift / prm['load_ramp_down'])
                scale.append(min(up, down))
            f_world = None
            if self.use_mpc:
                if walking:
                    self.mpc_vint = np.clip(
                        self.mpc_vint + prm['mpc_vel_ki'] * (v_des[:2] - self.v_filt) * dt,
                        -prm['mpc_vel_int_max'], prm['mpc_vel_int_max'])
                else:
                    self.mpc_vint[:] = 0.0
                v_ref = v_des + np.array([self.mpc_vint[0], self.mpc_vint[1], 0.0])
                f_all = self._mpc_forces(p, R, v, w_body, feet_w, stance, v_ref, yaw_rate_cmd,
                                         walking, h_des, (x0, y0, yaw0))
                if f_all is not None:
                    f_world = np.array([f_all[i] for i in idx])
                else:
                    self.mpc_fallbacks += 1
            if f_world is None:                                  # spring-damper wrench + split
                f_world = distribute_forces(rho, wrench, mu=prm['mu'],
                                            f_max=300.0 * np.array(scale))
            fz_total = float(f_world[:, 2].sum())
            for k, i in enumerate(idx):
                J = leg_jacobian(self.kin, LEG_ORDER[i], q[LEG_ORDER[i]])
                tau[3 * i:3 * i + 3] = -J.T @ (R.T @ f_world[k])

        self.swing_targets = {}
        q_des = np.zeros(12)
        qd_des = np.zeros(12)
        kp = np.zeros(12)
        kd = np.zeros(12)
        for i, leg in enumerate(LEG_ORDER):
            suffix = ('hip_x', 'hip_y', 'knee')
            gains_kp = [JOINT_KP[s] for s in suffix]
            gains_kd = [JOINT_KD[s] for s in suffix]
            target = None
            if stance[i]:
                self.debug.pop(leg, None)
                self.liftoff.pop(leg, None)
                self.prev_swing_q.pop(leg, None)
                foot = feet_w[i].copy()
                foot[2] = self.foot_z
                target = self._leg_ik(leg, p_des, R_des, foot)
            else:
                s = self.gait.swing_progress(leg)
                if leg not in self.liftoff:
                    self.liftoff[leg] = feet_w[i].copy()
                nominal_xy = np.array(self.nominal[leg][:2])
                td = raibert_target(p[:2], yaw, nominal_xy, self.v_filt, v_des[:2], yaw_rate_cmd,
                                    self.gait.stance_duration(), prm['k_raibert'],
                                    prm['max_step_offset'])
                td = self._limit_step(p[:2], yaw, nominal_xy, td, leg)
                goal = np.array([td[0], td[1], self.foot_z])
                foot = swing_position(self.liftoff[leg], goal, s, prm['swing_height'],
                                      self.foot_z)
                if self.use_wbc:
                    self.swing_targets[i] = swing_kinematics(
                        self.liftoff[leg], goal, s, prm['swing_height'], self.foot_z,
                        self.gait.swing_duration())
                self.debug[leg] = dict(s=round(s, 3), goal=[round(float(a), 3) for a in goal],
                                       foot=[round(float(a), 3) for a in foot],
                                       liftoff=[round(float(a), 3) for a in self.liftoff[leg]])
                target = self._leg_ik(leg, p, R, foot)
                gains_kp = [prm['swing_kp']] * 3
                gains_kd = [prm['swing_kd']] * 3
                if target is not None:
                    prev = self.prev_swing_q.get(leg)
                    if prev is not None:
                        qd_des[3 * i:3 * i + 3] = (target - prev) / dt
                    self.prev_swing_q[leg] = target.copy()
            if target is None:                               # unreachable: hold the current pose
                target = np.array(q[leg])
            q_des[3 * i:3 * i + 3] = target
            kp[3 * i:3 * i + 3] = gains_kp
            kd[3 * i:3 * i + 3] = gains_kd

        if self.use_wbc:
            tau_w = self._wbc_torques(R, v, w_body, q, qd, idx, f_world, feet_w, h_des, p,
                                      R_des, w_des_body)
            if tau_w is not None:
                tau = tau_w
        q_des = [clamp_joint(JOINT_NAMES[j], q_des[j]) for j in range(12)]

        v_body = R.T @ v
        stance_est = [st and self.gait.stance_time[leg] > 0.05
                      for st, leg in zip(stance, LEG_ORDER)]
        status = [p[2], math.degrees(roll), math.degrees(pitch), math.degrees(yaw),
                  float(v_body[0]), float(v_body[1]), float(np.linalg.norm(cmd[:2])),
                  float(len(idx)), fz_total, float(walking),
                  float(p[0] - x0), float(p[1] - y0)]
        return Output([float(a) for a in q_des], [float(a) for a in qd_des],
                      [float(a) for a in tau], [float(a) for a in kp], [float(a) for a in kd],
                      stance_est, status)
