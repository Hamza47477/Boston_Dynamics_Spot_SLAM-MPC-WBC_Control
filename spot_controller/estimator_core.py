"""ROS-free state estimator: complementary filter + stance detection + leg odometry + EKF.

Used by the ROS node (ekf_node.py) and directly by the Webots plugin (torque mode), where it runs
in the same simulation step as the sensors, with no communication delay.
"""
import math

import numpy as np

from .attitude_filter import MahonyFilter
from .ekf import ErrorStateEKF
from .leg_odometry import foot_states
from .quaternion import q_from_rpy, q_to_R, q_to_rpy
from .spot_4leg_ik import Spot4LegsIK
from .stance import detect_stance

STILL_SAMPLES = 15           # IMU samples of quiet before the filter starts
STANCE_HEIGHT = 0.03         # m; a foot higher than this above the lowest foot is in swing


class EstimatorCore:

    def __init__(self, kin=None):
        self.kin = kin or Spot4LegsIK()
        self.ekf = ErrorStateEKF()
        self.cf = MahonyFilter()
        self.initialized = False
        self.still_count = 0
        self.t_last = None
        self.gyro = np.zeros(3)
        self.n_stance = 0

    def step_imu(self, t, gyro, accel, gt_pose):
        """Process one IMU sample. gt_pose = (position, quaternion wxyz) is used only to
        initialise yaw and position (a real robot has no compass/GPS). Returns True when the
        filter is running and has been updated."""
        gyro, accel = np.asarray(gyro, float), np.asarray(accel, float)
        self.gyro = gyro
        if not self.initialized:
            # Wait until the robot is quiet (accelerometer ~ gravity only), otherwise the first
            # attitude is wrong and the foot anchors lock that error in.
            still = abs(np.linalg.norm(accel) - 9.81) < 0.3 and np.linalg.norm(gyro) < 0.3
            self.still_count = self.still_count + 1 if still else 0
            if gt_pose is None or self.still_count < STILL_SAMPLES:
                return False
            p0, q_gt = gt_pose
            yaw0 = q_to_rpy(q_gt)[2]
            roll0 = math.atan2(accel[1], accel[2])
            pitch0 = math.atan2(-accel[0], math.hypot(accel[1], accel[2]))
            self.ekf.initialize(p0, q_from_rpy(roll0, pitch0, yaw0))
            self.cf.initialize(accel, yaw=yaw0)
            self.t_last = t
            self.initialized = True
            return False

        dt = t - self.t_last
        if dt <= 0.0 or dt > 0.25:
            self.t_last = t
            return False
        self.cf.update(gyro, accel, dt)
        self.ekf.predict(gyro, accel, dt)
        self.ekf.update_gravity(accel)
        self.t_last = t
        return True

    def step_legs(self, q, qd, stance=None):
        """Leg corrections. q, qd: {leg: (hip_x, hip_y, knee)} (Webots convention).

        stance: contact flags from the gait schedule; if None they are detected from the foot
        heights (attitude taken from the complementary filter, independent of the EKF).
        """
        feet_r, feet_rdot = foot_states(self.kin, q, qd)
        if self.ekf.ground_z is None:                     # first call, robot standing: flat floor
            R0 = q_to_R(self.ekf.q)
            self.ekf.ground_z = float(np.mean([(self.ekf.p + R0 @ r)[2] for r in feet_r]))
        if stance is None:
            stance = detect_stance(feet_r, q_to_R(self.cf.q), STANCE_HEIGHT)
        self.n_stance = self.ekf.update_stance(stance, feet_r, feet_rdot, self.gyro)
        return feet_r

    def state(self):
        """(position, R body->world, world velocity, body angular velocity)."""
        R = q_to_R(self.ekf.q)
        return self.ekf.p.copy(), R, self.ekf.v.copy(), self.gyro - self.ekf.bg
