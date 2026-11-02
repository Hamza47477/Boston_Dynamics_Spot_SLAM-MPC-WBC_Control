"""TF tree and /odom for Nav2.

    map --(SLAM correction, stereo_slam)--> odom --(this node, leg-odometry EKF)--> base_link
    base_link --> stereo_left (FLU) --> stereo_left_optical (OpenCV axes)

/odom_estimated (frame "world" = the EKF's start-pose frame) is re-published as /odom (odom -> base_link).
"""
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from .geometry import LEFT_IN_BODY, R_FLU_FROM_CV, R_to_quat
import numpy as np


def make_tf(stamp, parent, child, translation, quat):
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id, msg.child_frame_id = parent, child
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = \
        (float(v) for v in translation)
    msg.transform.rotation.x, msg.transform.rotation.y = float(quat[0]), float(quat[1])
    msg.transform.rotation.z, msg.transform.rotation.w = float(quat[2]), float(quat[3])
    return msg


class TfBridge(Node):

    def __init__(self):
        super().__init__('tf_bridge')
        self.tf = TransformBroadcaster(self)
        static = StaticTransformBroadcaster(self)
        zero = self.get_clock().now().to_msg()
        static.sendTransform([
            make_tf(zero, 'base_link', 'stereo_left', LEFT_IN_BODY, (0.0, 0.0, 0.0, 1.0)),
            make_tf(zero, 'stereo_left', 'stereo_left_optical', (0.0, 0.0, 0.0),
                    R_to_quat(R_FLU_FROM_CV))])
        self.pub = self.create_publisher(Odometry, '/odom', 20)
        self.create_subscription(Odometry, '/odom_estimated', self.cb, 50)

    def cb(self, msg):
        p = msg.pose.pose
        self.tf.sendTransform(make_tf(
            msg.header.stamp, 'odom', 'base_link', (p.position.x, p.position.y, p.position.z),
            (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)))
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id, out.child_frame_id = 'odom', 'base_link'
        out.pose, out.twist = msg.pose, msg.twist
        self.pub.publish(out)


def main():
    rclpy.init()
    node = TfBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
