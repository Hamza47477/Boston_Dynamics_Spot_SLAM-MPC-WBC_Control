"""Demo node: smooth body-pose motion converted to joint commands through the IK."""
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .constants import HIP_MOUNT, HIP_OFFSET_Y, JOINT_NAMES, LEG_ORDER, LEG_SIDE
from .spot_4leg_ik import Spot4LegsIK

HIP_Y = abs(HIP_MOUNT['FL'][1])
START_HEIGHT = 0.58   # body height of the Webots zero pose (legs at 0 rad)
STAND_HEIGHT = 0.50
RAMP_SECONDS = 3.0


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


class SpotPublisherNode(Node):
    """Publishes /joint_commands at 50 Hz. Parameter `motion`: 'stand' or 'sway'."""

    def __init__(self):
        super().__init__('spot_publisher_node')
        self.declare_parameter('motion', 'sway')
        self.publisher_ = self.create_publisher(JointState, '/joint_commands', 10)
        self.ik_solver = Spot4LegsIK()
        self.t0 = self.get_clock().now().nanoseconds / 1e9
        self.timer = self.create_timer(0.02, self.publish_joint_commands)

        # Feet directly below the hips, on the ground (world z = 0)
        self.foot_stance = {
            leg: [HIP_MOUNT[leg][0], LEG_SIDE[leg] * (HIP_Y + HIP_OFFSET_Y), 0.0]
            for leg in LEG_ORDER}

    def publish_joint_commands(self):
        t = self.get_clock().now().nanoseconds / 1e9 - self.t0
        ramp = smoothstep(t / RAMP_SECONDS)
        amp = ramp if self.get_parameter('motion').value == 'sway' else 0.0

        height = START_HEIGHT + (STAND_HEIGHT - START_HEIGHT) * ramp
        body_pos = [0.02 * amp * math.cos(1.0 * t),
                    0.02 * amp * math.sin(1.5 * t),
                    height + 0.03 * amp * math.sin(1.2 * t)]
        body_rpy = [0.08 * amp * math.sin(1.5 * t),
                    0.06 * amp * math.sin(1.2 * t),
                    0.05 * amp * math.cos(1.0 * t)]

        try:
            angles = self.ik_solver.compute_body_ik_webots(
                body_pos, body_rpy, self.foot_stance)
        except ValueError as exc:
            self.get_logger().error('IK error: {}'.format(exc), throttle_duration_sec=2.0)
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = angles
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SpotPublisherNode()
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
