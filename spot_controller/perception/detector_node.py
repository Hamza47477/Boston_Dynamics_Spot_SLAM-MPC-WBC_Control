"""ROS node: left image -> 2D detections of people and cars.

Parameter `backend`: 'auto' (YOLOv4-tiny if its files are in ~/.spot_models, otherwise the HOG people
detector), 'yolo', 'hog' or 'oracle' (simulation ground truth projected into the image: a cheat for testing).
Detection runs on every `every`-th frame (it is the slowest stage) and publishes /detections
(vision_msgs/Detection2DArray, stamp = image stamp) and /detections/debug_image.
"""
import json

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2D, Detection2DArray, Detection3DArray, ObjectHypothesisWithPose

from .detector import CLASS_COLORS, make_detector
from .geometry import Intrinsics, PoseBuffer, T_body_camcv
from .ros_utils import image_to_numpy, msg_to_T, numpy_to_image, PairSync, SENSOR_QOS, stamp_to_sec
from .stats import StageStats


class DetectorNode(Node):

    def __init__(self):
        super().__init__('detector')
        self.declare_parameter('backend', 'auto')
        self.declare_parameter('every', 2)
        cv2.setNumThreads(3)
        self.backend_name = self.get_parameter('backend').value
        self.detector = None
        self.intr = None
        self.sync = PairSync()
        self.count = 0
        self.gt_pose = PoseBuffer()
        self.gt_objects = []
        self.stats = StageStats(self, 'detector')
        self.pub = self.create_publisher(Detection2DArray, '/detections', 5)
        self.pub_debug = self.create_publisher(Image, '/detections/debug_image', SENSOR_QOS)
        self.create_subscription(CameraInfo, '/stereo/left/camera_info', self.info_cb, 5)
        self.create_subscription(Image, '/stereo/left/image_raw', self.left_cb, SENSOR_QOS)
        self.create_subscription(Image, '/stereo/depth/image_raw', self.depth_cb, SENSOR_QOS)
        if self.backend_name == 'oracle':
            self.create_subscription(PoseStamped, '/ground_truth/pose', lambda m: self.gt_pose.add(
                stamp_to_sec(m.header.stamp), msg_to_T(m.pose)), 50)
            self.create_subscription(Detection3DArray, '/ground_truth/objects', self.objects_cb, 5)

    def info_cb(self, msg):
        if self.detector is None:
            self.intr = Intrinsics.from_info(msg)
            self.detector = make_detector(self.backend_name, self.intr, T_body_camcv())
            self.get_logger().info('detector backend: {}{}'.format(
                self.detector.name, '  (SIMULATION GROUND TRUTH - not a real detector)'
                if self.detector.name == 'oracle' else ''))

    def objects_cb(self, msg):
        self.gt_objects = [dict(cls=d.results[0].hypothesis.class_id,
                                pos=[d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z],
                                yaw=2 * float(np.arctan2(d.bbox.center.orientation.z, d.bbox.center.orientation.w)),
                                size=[d.bbox.size.x, d.bbox.size.y, d.bbox.size.z]) for d in msg.detections]

    def left_cb(self, msg):
        pair = self.sync.add_left(msg)
        if pair:
            self.process(*pair)

    def depth_cb(self, msg):
        pair = self.sync.add_right(msg)
        if pair:
            self.process(*pair)

    def process(self, left_msg, depth_msg):
        if self.detector is None:
            return
        self.count += 1
        if self.count % self.get_parameter('every').value:
            return
        img = image_to_numpy(left_msg)
        with self.stats.measure():
            if self.detector.name == 'oracle':
                T = self.gt_pose.get(stamp_to_sec(left_msg.header.stamp))
                dets = [] if T is None else self.detector.detect(T, self.gt_objects, image_to_numpy(depth_msg))
            else:
                dets = self.detector.detect(img)
        out = Detection2DArray()
        out.header = left_msg.header
        for d in dets:
            u0, v0, u1, v1 = d.box
            det = Detection2D()
            det.header = left_msg.header
            det.bbox.center.position.x, det.bbox.center.position.y = (u0 + u1) / 2, (v0 + v1) / 2
            det.bbox.size_x, det.bbox.size_y = float(u1 - u0), float(v1 - v0)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = d.cls, float(d.score)
            det.results.append(hyp)
            out.detections.append(det)
        self.pub.publish(out)
        if self.pub_debug.get_subscription_count() > 0:
            vis = img.copy()
            for d in dets:
                u0, v0, u1, v1 = (int(v) for v in d.box)
                col = CLASS_COLORS.get(d.cls, (200, 200, 200))
                cv2.rectangle(vis, (u0, v0), (u1, v1), col, 2)
                cv2.putText(vis, '{} {:.2f}'.format(d.cls, d.score), (u0, max(12, v0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            cv2.putText(vis, 'backend: ' + self.detector.name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            self.pub_debug.publish(numpy_to_image(vis, left_msg.header.stamp, 'stereo_left', 'bgr8'))
        self.stats.publish(backend=self.detector.name, objects=len(dets))


def main():
    rclpy.init()
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
