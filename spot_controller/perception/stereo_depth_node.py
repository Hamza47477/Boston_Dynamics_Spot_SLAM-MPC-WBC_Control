"""ROS node: /stereo/left|right/image_raw -> /stereo/depth/image_raw (32FC1 metres) + scores."""
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from .geometry import Intrinsics
from .ros_utils import image_to_numpy, numpy_to_image, PairSync, SENSOR_QOS
from .stats import StageStats
from .stereo_depth import depth_error, StereoMatcher


class StereoDepthNode(Node):

    def __init__(self):
        super().__init__('stereo_depth')
        self.declare_parameter('num_disp', 96)
        self.declare_parameter('block', 7)
        cv2.setNumThreads(2)
        self.matcher = None
        self.sync = PairSync()
        self.recent = {}                 # stamp -> depth, to score when the ground truth arrives
        self.errors = []
        self.stats = StageStats(self, 'stereo_depth')
        self.pub_depth = self.create_publisher(Image, '/stereo/depth/image_raw', SENSOR_QOS)
        self.pub_disp = self.create_publisher(Image, '/stereo/disparity/image_raw', SENSOR_QOS)
        self.create_subscription(CameraInfo, '/stereo/left/camera_info', self.info_cb, 5)
        self.create_subscription(Image, '/stereo/left/image_raw', self.left_cb, SENSOR_QOS)
        self.create_subscription(Image, '/stereo/right/image_raw', self.right_cb, SENSOR_QOS)
        self.create_subscription(Image, '/stereo/depth_gt/image_raw', self.gt_cb, SENSOR_QOS)

    def info_cb(self, msg):
        if self.matcher is None:
            self.matcher = StereoMatcher(Intrinsics.from_info(msg),
                                         self.get_parameter('num_disp').value,
                                         self.get_parameter('block').value)
            self.get_logger().info('stereo depth ready (fx {:.1f}, baseline {:.2f} m)'.format(
                msg.k[0], self.matcher.intr.baseline))

    def gt_cb(self, msg):
        depth = self.recent.get((msg.header.stamp.sec, msg.header.stamp.nanosec))
        if depth is not None:
            score = depth_error(depth, image_to_numpy(msg))
            if score:
                self.errors = (self.errors + [score])[-20:]

    def left_cb(self, msg):
        pair = self.sync.add_left(msg)
        if pair:
            self.process(*pair)

    def right_cb(self, msg):
        pair = self.sync.add_right(msg)
        if pair:
            self.process(*pair)

    def process(self, left_msg, right_msg):
        if self.matcher is None:
            return
        with self.stats.measure():
            left = cv2.cvtColor(image_to_numpy(left_msg), cv2.COLOR_BGR2GRAY)
            right = image_to_numpy(right_msg)
            disp, depth = self.matcher.compute(left, right)
        self.pub_depth.publish(numpy_to_image(depth, left_msg.header.stamp, 'stereo_left', '32FC1'))
        self.pub_disp.publish(numpy_to_image(disp, left_msg.header.stamp, 'stereo_left', '32FC1'))
        key = (left_msg.header.stamp.sec, left_msg.header.stamp.nanosec)
        self.recent[key] = depth
        while len(self.recent) > 4:
            self.recent.pop(min(self.recent))
        extra = {}
        if self.errors:
            n = len(self.errors)
            extra = dict(valid=round(sum(e[0] for e in self.errors) / n, 3),
                         mae_m=round(sum(e[1] for e in self.errors) / n, 3),
                         rel_err=round(sum(e[2] for e in self.errors) / n, 3),
                         bad_pct=round(sum(e[3] for e in self.errors) / n, 1))
        self.stats.publish(**extra)


def main():
    rclpy.init()
    node = StereoDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
