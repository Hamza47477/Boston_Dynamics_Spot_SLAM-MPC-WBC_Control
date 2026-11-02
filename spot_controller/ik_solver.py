"""Analytic inverse and forward kinematics for one 3-DOF Spot leg (IK convention)."""
import math

from .constants import CALF_LENGTH, HIP_OFFSET_Y, THIGH_LENGTH


class SpotIK:
    """One leg. Foot positions are relative to the abduction axis, x forward, z up."""

    def __init__(self, hip_offset_y=HIP_OFFSET_Y, thigh_length=THIGH_LENGTH,
                 calf_length=CALF_LENGTH):
        self.dy = hip_offset_y
        self.l1 = thigh_length
        self.l2 = calf_length

    def compute_ik(self, x, y, z):
        """Return [theta1, theta2, theta3]; raise ValueError if the foot is unreachable."""
        r = math.sqrt(y ** 2 + z ** 2)
        if r < abs(self.dy):
            raise ValueError('foot too close to the hip axis (r={:.3f})'.format(r))

        theta1 = math.atan2(y, -z) - math.asin(self.dy / r)

        z_prime = -math.sqrt(r ** 2 - self.dy ** 2)
        d_sq = x ** 2 + z_prime ** 2
        cos_theta3 = (d_sq - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
        if not -1.0 <= cos_theta3 <= 1.0:
            raise ValueError('foot unreachable (distance {:.3f} m, reach {:.3f} m)'
                             .format(math.sqrt(d_sq), self.l1 + self.l2))
        theta3 = math.acos(cos_theta3)

        theta2 = math.atan2(x, -z_prime) - math.atan2(
            self.l2 * math.sin(theta3), self.l1 + self.l2 * math.cos(theta3))
        return [theta1, theta2, theta3]

    def compute_fk(self, theta1, theta2, theta3):
        """Return (x, y, z) of the foot relative to the abduction axis."""
        x = self.l1 * math.sin(theta2) + self.l2 * math.sin(theta2 + theta3)
        z_prime = -(self.l1 * math.cos(theta2) + self.l2 * math.cos(theta2 + theta3))
        y = self.dy * math.cos(theta1) - z_prime * math.sin(theta1)
        z = self.dy * math.sin(theta1) + z_prime * math.cos(theta1)
        return x, y, z
