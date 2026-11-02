"""Builds the MPC inputs (reference, foot positions, contacts) from the robot state and the gait."""
import math

import numpy as np

from .constants import LEG_ORDER
from .gait import raibert_target
from .mpc import rot_z
from .quaternion import q_to_rpy, R_to_q

GRAVITY = 9.81


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def plan_inputs(horizon, dt, p, R, v, w_body, feet_w, phases, gait_active, duty, period,
                v_des_xy, yaw_rate, hold_xy, hold_yaw, walking, h_des, foot_z, nominal_xy,
                com_body, v_filt=None, k_raibert=0.2, max_hold_error=0.06, stance_time=None,
                touch_delay=0.03, max_step=0.10):
    """Return dict(x0, x_ref, r_feet (N,4,3), contact (N,4), yaw_seq (N,)).

    phases: {leg: gait phase}; feet_w: 4 world foot positions (measured, valid for feet in stance).
    walking: velocity tracking (no position hold); otherwise hold hold_xy / hold_yaw.
    """
    roll, pitch, yaw = q_to_rpy(R_to_q(R))
    x0 = np.concatenate([[roll, pitch, yaw], p, R @ w_body, v, [-GRAVITY]])
    t_stance = duty * period
    v_ref_xy = np.asarray(v_des_xy, dtype=float) if walking else np.zeros(2)
    wz = yaw_rate if walking else 0.0
    v_filt = np.asarray(v[:2] if v_filt is None else v_filt, dtype=float)

    if walking:
        yaw_target0 = yaw
    else:
        yaw_target0 = yaw + wrap(hold_yaw - yaw)
        err = np.clip(np.asarray(hold_xy) - p[:2], -max_hold_error, max_hold_error)
    x_ref = np.zeros((horizon, 13))
    yaw_seq = np.zeros(horizon)
    p_ref = np.zeros((horizon, 3))
    for k in range(horizon):
        tk = (k + 1) * dt
        yaw_k = yaw_target0 + wz * tk
        yaw_seq[k] = yaw_target0 + wz * (k + 0.5) * dt
        if walking:
            xy = p[:2] + v_ref_xy * tk
        else:
            xy = p[:2] + err                      # constant: pull towards the held position
        p_ref[k] = [xy[0], xy[1], h_des]
        x_ref[k] = np.concatenate([[0.0, 0.0, yaw_k], p_ref[k], [0.0, 0.0, wz],
                                   [v_ref_xy[0], v_ref_xy[1], 0.0], [-GRAVITY]])

    # contact schedule and foot positions over the horizon
    contact = np.ones((horizon, 4), dtype=bool)
    r_feet = np.zeros((horizon, 4, 3))
    for i, leg in enumerate(LEG_ORDER):
        pos = np.array(feet_w[i], dtype=float) if phases[leg] < duty else None
        for k in range(horizon):
            if gait_active:
                phase = (phases[leg] + (k + 0.5) * dt / period) % 1.0
                stance = phase < duty
            else:
                stance = True
            if k == 0 and stance and gait_active and stance_time is not None \
                    and stance_time[leg] < touch_delay:
                contact[k, i] = False                  # just landed: let the other feet carry it
            else:
                contact[k, i] = stance
            com_k = p_ref[k] + rot_z(yaw_seq[k]) @ com_body
            if stance:
                if pos is None:                       # landing during the horizon
                    n = np.asarray(nominal_xy[i])
                    td = raibert_target(p_ref[k, :2], yaw_seq[k], n, v_filt, v_ref_xy, wz,
                                        t_stance, k_raibert, max_step)
                    pos = np.array([td[0], td[1], foot_z])
                r_feet[k, i] = pos - com_k
            else:
                pos = None
                r_feet[k, i] = np.array([0.0, 0.0, -0.4])      # unused (force forced to zero)
    return dict(x0=x0, x_ref=x_ref, r_feet=r_feet, contact=contact, yaw_seq=yaw_seq)
