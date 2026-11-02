"""Trot gait: contact schedule, swing-foot trajectory and Raibert foot placement (ROS-free)."""
import math

import numpy as np

from .constants import LEG_ORDER

# Diagonal pairs move together: FL+RR, then FR+RL
TROT_OFFSETS = {'FL': 0.0, 'RR': 0.0, 'FR': 0.5, 'RL': 0.5}


class TrotGait:
    """Phase per leg in [0, 1): stance for phase < duty, swing for phase >= duty.

    With duty > 0.5 both diagonal pairs are on the ground for part of each cycle.
    When stepping stops, legs in stance freeze, legs in swing finish and land, then freeze.
    """

    def __init__(self, period=0.6, duty=0.7, swing_height=0.07):
        self.period = period
        self.duty = duty
        self.swing_height = swing_height
        self.phase = dict(TROT_OFFSETS)
        self.active = False
        self.stance_time = {leg: 1.0 for leg in LEG_ORDER}    # seconds since touchdown

    def update(self, dt, want_step):
        """Advance the schedule by dt. want_step: keep stepping (True) or come to a stop."""
        was_active = self.active
        swinging = [self.in_swing(leg) for leg in LEG_ORDER]
        if want_step and not was_active and not any(swinging):
            self.phase = dict(TROT_OFFSETS)                    # restart cleanly from standing
        for leg in LEG_ORDER:
            ph = self.phase[leg]
            if want_step or ph >= self.duty:
                ph += dt / self.period
                if ph >= 1.0:
                    ph -= 1.0
                    if not want_step:
                        ph = 0.0                               # landed: freeze at touchdown
                        self.stance_time[leg] = 0.0
                    else:
                        self.stance_time[leg] = 0.0
                self.phase[leg] = ph
            if self.phase[leg] < self.duty:
                self.stance_time[leg] += dt
            else:
                self.stance_time[leg] = 0.0
        self.active = want_step or any(self.in_swing(leg) for leg in LEG_ORDER)

    def in_swing(self, leg):
        return self.phase[leg] >= self.duty

    def swing_progress(self, leg):
        """0 at lift-off, 1 at touchdown."""
        return min(1.0, max(0.0, (self.phase[leg] - self.duty) / (1.0 - self.duty)))

    def stance_flags(self):
        return [not self.in_swing(leg) for leg in LEG_ORDER]

    def swing_duration(self):
        return (1.0 - self.duty) * self.period

    def stance_duration(self):
        return self.duty * self.period


def quintic(s):
    """Smooth 0 -> 1 with zero velocity and acceleration at both ends."""
    s = min(1.0, max(0.0, s))
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def swing_position(liftoff, target, s, height, foot_z):
    """World position of the swing foot at progress s (horizontal quintic, vertical arch)."""
    b = quintic(s)
    pos = liftoff + (target - liftoff) * b
    pos = np.array(pos, dtype=float)
    pos[2] = foot_z + height * math.sin(math.pi * s) ** 2
    return pos


def raibert_target(p_xy, yaw, nominal_xy, v_xy, v_des_xy, yaw_rate, stance_time,
                   k_v=0.25, max_offset=0.10):
    """Touchdown point (world xy) of a foot.

    nominal_xy: foot position under the body in the body frame at rest (x forward, y left).
    Foot lands where the body will have moved to by mid-stance (v * T/2), plus a correction
    k_v * (v - v_des) that makes the robot slow down (or speed up) to the commanded velocity.
    A yaw rate adds a tangential offset so turning steps follow the arc.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    nominal_world = np.array([c * nominal_xy[0] - s * nominal_xy[1],
                              s * nominal_xy[0] + c * nominal_xy[1]])
    tangential = yaw_rate * 0.5 * stance_time * np.array([-nominal_world[1], nominal_world[0]])
    offset = (np.asarray(v_xy) * 0.5 * stance_time + k_v * (np.asarray(v_xy) - v_des_xy)
              + tangential)
    norm = np.linalg.norm(offset)
    if norm > max_offset:
        offset *= max_offset / norm
    return np.asarray(p_xy) + nominal_world + offset


def swing_kinematics(liftoff, target, s, height, foot_z, swing_time, eps=0.02):
    """Desired foot position, velocity and acceleration (world) on the swing arch.

    Derivatives w.r.t. time come from finite differences of the arch in the progress s
    (ds/dt = 1 / swing_time) with the landing target held fixed.
    """
    s0, s1 = max(0.0, s - eps), min(1.0, s + eps)
    pos = swing_position(liftoff, target, s, height, foot_z)
    p0 = swing_position(liftoff, target, s0, height, foot_z)
    p1 = swing_position(liftoff, target, s1, height, foot_z)
    dsdt = 1.0 / swing_time
    vel = (p1 - p0) / max(s1 - s0, 1e-9) * dsdt
    ds_up, ds_dn = s1 - s, s - s0
    if ds_up > 1e-9 and ds_dn > 1e-9:
        acc = ((p1 - pos) / ds_up - (pos - p0) / ds_dn) / (0.5 * (ds_up + ds_dn)) * dsdt * dsdt
    else:
        acc = np.zeros(3)
    return pos, vel, acc
