"""Leg-kinematics body-state estimate (height, roll, pitch), compared against ground truth."""
import math

import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .body_state import fit_ground_plane, quat_to_roll_pitch
from .constants import LEG_ORDER, LEG_ROS_PREFIX
from .spot_4leg_ik import Spot4LegsIK


class StateEstimator(Node):
    """Subscribes /joint_states and /ground_truth/pose, publishes /fk_state (x=roll, y=pitch, z=height)."""

    def __init__(self):
        super().__init__('state_estimator')
        self.kin = Spot4LegsIK()
        self.imu_rp = None
        self.create_subscription(JointState, '/joint_states', self.joint_states_cb, 10)
        self.create_subscription(PoseStamped, '/ground_truth/pose', self.gt_cb, 10)
        self.pub = self.create_publisher(Vector3Stamped, '/fk_state', 10)
        self.get_logger().info('State estimator started (FK height/roll/pitch vs ground truth).')

    def gt_cb(self, msg):
        q = msg.pose.orientation
        self.imu_rp = quat_to_roll_pitch(q.x, q.y, q.z, q.w)

    def joint_states_cb(self, msg):
        angles = dict(zip(msg.name, msg.position))
        try:
            per_leg = {leg: tuple(angles['{}_{}'.format(LEG_ROS_PREFIX[leg], s)]
                                  for s in ('hip_x', 'hip_y', 'knee'))
                       for leg in LEG_ORDER}
        except KeyError:
            return
        feet = self.kin.compute_body_fk(per_leg)
        height, roll, pitch = fit_ground_plane([feet[leg] for leg in LEG_ORDER])

        out = Vector3Stamped()
        out.header.stamp = msg.header.stamp
        out.vector.x, out.vector.y, out.vector.z = roll, pitch, height
        self.pub.publish(out)

        text = 'FK  height {:.3f} m | roll {:+.2f} deg | pitch {:+.2f} deg'.format(
            height, math.degrees(roll), math.degrees(pitch))
        if self.imu_rp is not None:
            text += ' || truth roll {:+.2f} deg | pitch {:+.2f} deg'.format(
                math.degrees(self.imu_rp[0]), math.degrees(self.imu_rp[1]))
        self.get_logger().info(text, throttle_duration_sec=0.5)


def main(args=None):
    rclpy.init(args=args)
    node = StateEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
