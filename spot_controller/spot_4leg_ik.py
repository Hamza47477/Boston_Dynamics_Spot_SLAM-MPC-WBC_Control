"""Whole-body kinematics: body pose -> 12 joint angles, joint angles -> foot positions."""
import math

from .constants import (HIP_MOUNT, HIP_OFFSET_Y, ik_to_webots, LEG_ORDER, LEG_SIDE,
                        webots_to_ik)
from .ik_solver import SpotIK


class Spot4LegsIK:
    """Kinematics for all four legs in the body frame (x forward, y left, z up)."""

    def __init__(self):
        self.legs = {leg: SpotIK(hip_offset_y=LEG_SIDE[leg] * HIP_OFFSET_Y)
                     for leg in LEG_ORDER}
        self.hip_positions = {leg: list(pos) for leg, pos in HIP_MOUNT.items()}

    @staticmethod
    def _euler_to_rot_matrix(roll, pitch, yaw):
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]

    @staticmethod
    def _matrix_vec_mult(R, v):
        return [R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2] for i in range(3)]

    def compute_body_ik(self, body_pos, body_rpy, foot_targets):
        """Return 12 angles (IK convention, order FL, FR, RL, RR x hip_x, hip_y, knee).

        body_pos / body_rpy: body pose in the world; foot_targets: world foot positions.
        """
        R = self._euler_to_rot_matrix(*body_rpy)
        R_inv = [[R[j][i] for j in range(3)] for i in range(3)]

        all_joints = []
        for leg in LEG_ORDER:
            foot_w = foot_targets[leg]
            to_foot_w = [foot_w[i] - body_pos[i] for i in range(3)]
            to_foot_b = self._matrix_vec_mult(R_inv, to_foot_w)
            hip = self.hip_positions[leg]
            rel = [to_foot_b[i] - hip[i] for i in range(3)]
            all_joints.extend(self.legs[leg].compute_ik(*rel))
        return all_joints

    def compute_body_ik_webots(self, body_pos, body_rpy, foot_targets):
        """Same as compute_body_ik but returns raw Webots motor angles."""
        ik = self.compute_body_ik(body_pos, body_rpy, foot_targets)
        out = []
        for i in range(len(LEG_ORDER)):
            out.extend(ik_to_webots(*ik[3 * i:3 * i + 3]))
        return out

    def compute_body_fk(self, webots_angles):
        """Foot positions in the body frame from Webots-convention joint angles.

        webots_angles: dict leg -> (hip_x, hip_y, knee). Returns dict leg -> [x, y, z].
        """
        feet = {}
        for leg in LEG_ORDER:
            ik_angles = webots_to_ik(*webots_angles[leg])
            local = self.legs[leg].compute_fk(*ik_angles)
            hip = self.hip_positions[leg]
            feet[leg] = [hip[i] + local[i] for i in range(3)]
        return feet
