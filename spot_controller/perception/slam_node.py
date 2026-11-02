"""ROS node around StereoSlam.

In : /stereo/left/image_raw, /stereo/depth/image_raw, /stereo/left/camera_info,
     /odom_estimated (leg-odometry EKF, used as motion prior), /perception/dynamic_boxes (optional)
Out: TF map -> odom (the SLAM correction of the odometry), /slam/odom, /slam/keyframes, /slam/path,
     /slam/landmarks, /slam/debug_image, /pipeline/stats.

/slam/odom is the base_link pose in `map`. Its child_frame_id is "kf<N>" (the reference keyframe) and its
twist carries the pose of the camera relative to that keyframe (linear = translation, angular = rotation
vector). The mapper uses this to re-attach every frame to its keyframe after a loop closure.
"""
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_ros import TransformBroadcaster
from vision_msgs.msg import Detection2DArray

from .geometry import inv_T, Intrinsics, PoseBuffer, R_to_quat, T_body_camcv
from .pose_graph import ate_rmse
from .ros_utils import (image_to_numpy, make_cloud, msg_to_T, numpy_to_image, PairSync, pose_to_msg,
                        SENSOR_QOS, stamp_to_sec)
from .stats import StageStats
from .stereo_slam import StereoSlam


class SlamNode(Node):

    def __init__(self):
        super().__init__('stereo_slam')
        self.declare_parameter('use_odom_prior', True)
        cv2.setNumThreads(2)
        self.slam = None
        self.sync = PairSync()
        self.odom, self.gt = PoseBuffer(), PoseBuffer()
        self.T_base_cam = T_body_camcv()
        self.boxes = []
        self.kf_stamps = []
        self.last_ate = None
        self.stats = StageStats(self, 'slam')
        self.tf = TransformBroadcaster(self)
        self.pub_odom = self.create_publisher(Odometry, '/slam/odom', 10)
        self.pub_kfs = self.create_publisher(PoseArray, '/slam/keyframes', 5)
        self.pub_path = self.create_publisher(Path, '/slam/path', 5)
        self.pub_points = self.create_publisher(PointCloud2, '/slam/landmarks', 2)
        self.pub_debug = self.create_publisher(Image, '/slam/debug_image', SENSOR_QOS)
        self.create_subscription(CameraInfo, '/stereo/left/camera_info', self.info_cb, 5)
        self.create_subscription(Odometry, '/odom_estimated', self.odom_cb, 100)
        self.create_subscription(PoseStamped, '/ground_truth/pose', self.gt_cb, 50)
        self.create_subscription(Detection2DArray, '/perception/dynamic_boxes', self.boxes_cb, 5)
        self.create_subscription(Image, '/stereo/left/image_raw', self.left_cb, SENSOR_QOS)
        self.create_subscription(Image, '/stereo/depth/image_raw', self.depth_cb, SENSOR_QOS)
        self.last_cloud_pub = 0.0

    def info_cb(self, msg):
        if self.slam is None:
            self.slam = StereoSlam(Intrinsics.from_info(msg))
            self.get_logger().info('stereo SLAM ready')

    def odom_cb(self, msg):
        self.odom.add(stamp_to_sec(msg.header.stamp), msg_to_T(msg.pose.pose))

    def gt_cb(self, msg):
        self.gt.add(stamp_to_sec(msg.header.stamp), msg_to_T(msg.pose))

    def boxes_cb(self, msg):
        self.boxes = [(d.bbox.center.position.x - d.bbox.size_x / 2, d.bbox.center.position.y - d.bbox.size_y / 2,
                       d.bbox.center.position.x + d.bbox.size_x / 2, d.bbox.center.position.y + d.bbox.size_y / 2)
                      for d in msg.detections]

    def left_cb(self, msg):
        pair = self.sync.add_left(msg)
        if pair:
            self.process(*pair)

    def depth_cb(self, msg):
        pair = self.sync.add_right(msg)
        if pair:
            self.process(*pair)

    # ------------------------------------------------------------------
    def process(self, left_msg, depth_msg):
        if self.slam is None:
            return
        t = stamp_to_sec(left_msg.header.stamp)
        T_o_base = self.odom.get(t) if self.get_parameter('use_odom_prior').value else None
        gray = cv2.cvtColor(image_to_numpy(left_msg), cv2.COLOR_BGR2GRAY)
        depth = image_to_numpy(depth_msg)
        with self.stats.measure():
            res = self.slam.process(t, gray, depth, T_o_base, self.boxes or None)
        if res.state == 'init':
            return
        stamp = left_msg.header.stamp
        T_w_base = res.T_w_cam @ inv_T(self.T_base_cam)
        self.publish_tf(stamp, T_w_base, T_o_base)
        self.publish_odom(stamp, res, T_w_base)
        if res.keyframe:
            self.kf_stamps.append(t)
        if res.keyframe or res.graph_changed:
            self.publish_keyframes(stamp)
            self.update_ate()
        if self.pub_debug.get_subscription_count() > 0:
            self.pub_debug.publish(numpy_to_image(self.debug_image(left_msg, res), stamp, 'stereo_left', 'bgr8'))
        if t - self.last_cloud_pub > 3.0 and self.pub_points.get_subscription_count() > 0:
            self.last_cloud_pub = t
            self.pub_points.publish(make_cloud(self.slam.landmarks_world(15000), 'map', stamp))
        self.stats.publish(state=res.state, kfs=len(self.slam.kfs), loops=self.slam.loop_count,
                           inliers=res.n_inliers, ate_m=self.last_ate,
                           ms_feat=round(res.ms.get('features', 0), 1),
                           ms_match=round(res.ms.get('match', 0), 1),
                           ms_pnp=round(res.ms.get('pnp', 0), 1))

    def publish_tf(self, stamp, T_w_base, T_o_base):
        T_map_odom = np.eye(4) if T_o_base is None else T_w_base @ inv_T(T_o_base)
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id, msg.child_frame_id = 'map', 'odom'
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = \
            (float(v) for v in T_map_odom[:3, 3])
        q = R_to_quat(T_map_odom[:3, :3])
        msg.transform.rotation.x, msg.transform.rotation.y = float(q[0]), float(q[1])
        msg.transform.rotation.z, msg.transform.rotation.w = float(q[2]), float(q[3])
        self.tf.sendTransform(msg)

    def publish_odom(self, stamp, res, T_w_base):
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id, msg.child_frame_id = 'map', 'kf{}'.format(res.kf_id)
        pose_to_msg(T_w_base, msg.pose.pose)
        rvec, _ = cv2.Rodrigues(res.T_kf_cam[:3, :3])
        msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z = \
            (float(v) for v in res.T_kf_cam[:3, 3])
        msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z = \
            (float(v) for v in rvec.ravel())
        msg.pose.covariance[0] = float(res.n_inliers)            # tracking quality, for the viewer
        msg.pose.covariance[1] = {'tracking': 0.0, 'lost': 1.0}.get(res.state, 2.0)
        self.pub_odom.publish(msg)

    def publish_keyframes(self, stamp):
        arr = PoseArray()
        arr.header.stamp = stamp
        arr.header.frame_id = 'map'
        path = Path()
        path.header = arr.header
        for kf in self.slam.kfs:
            arr.poses.append(pose_to_msg(kf.T_w_cam, Pose()))
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.header.stamp = stamp
            pose_to_msg(kf.T_w_cam @ inv_T(self.T_base_cam), ps.pose)
            path.poses.append(ps)
        self.pub_kfs.publish(arr)
        self.pub_path.publish(path)

    def update_ate(self):
        est, gt = [], []
        for kf, t in zip(self.slam.kfs, self.kf_stamps):
            g = self.gt.get(t)
            if g is not None:
                est.append(kf.T_w_cam[:3, 3])
                gt.append((g @ self.T_base_cam)[:3, 3])
        if len(est) >= 5:
            self.last_ate = round(ate_rmse(np.array(est), np.array(gt))[0], 3)

    def debug_image(self, left_msg, res):
        img = image_to_numpy(left_msg).copy()
        for u, v in res.outlier_xy:
            cv2.circle(img, (int(u), int(v)), 2, (0, 0, 255), -1)
        for u, v in res.inlier_xy:
            cv2.circle(img, (int(u), int(v)), 3, (0, 255, 0), 1)
        colour = {'tracking': (0, 255, 0), 'lost': (0, 0, 255)}.get(res.state, (0, 255, 255))
        cv2.putText(img, '{} | tracked {} | kf {} | loops {}'.format(
            res.state, res.n_inliers, len(self.slam.kfs), self.slam.loop_count), (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
        return img


def main():
    rclpy.init()
    node = SlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
