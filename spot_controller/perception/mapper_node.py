"""ROS node: depth + SLAM poses -> 3D voxel map, 2D occupancy grid, obstacle cloud for navigation.

In : /stereo/depth/image_raw, /slam/odom, /slam/keyframes, /perception/dynamic_boxes (optional)
Out: /map (OccupancyGrid, 1 Hz), /map_3d (PointCloud2 of occupied voxels, coloured by height, 0.5 Hz),
     /mapper/obstacles (PointCloud2 in base_link: what the camera sees right now above the floor) and
     /mapper/scan (the same as a pseudo laser scan, marks AND clears: used by the Nav2 costmaps),
     /pipeline/stats.
"""
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2
from vision_msgs.msg import Detection2DArray

from .geometry import inv_T, Intrinsics, T_body_camcv
from .mapper import Mapper, points_to_scan
from .ros_utils import image_to_numpy, make_cloud, msg_to_T, PairSync, SENSOR_QOS, stamp_to_sec
from .stats import StageStats


def height_colors(z, lo=-0.1, hi=2.0):
    """Blue (low) to red (high) colour ramp for a point cloud."""
    t = np.clip((z - lo) / (hi - lo), 0, 1)
    rgb = cv2.applyColorMap((t * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    return rgb[:, ::-1].copy()


class MapperNode(Node):

    def __init__(self):
        super().__init__('mapper')
        self.mapper = None
        self.sync = PairSync()
        self.boxes = []
        self.T_base_cam = T_body_camcv()
        self.stats = StageStats(self, 'mapper')
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.pub_map = self.create_publisher(OccupancyGrid, '/map', latched)
        self.pub_3d = self.create_publisher(PointCloud2, '/map_3d', 2)
        self.pub_obs = self.create_publisher(PointCloud2, '/mapper/obstacles', SENSOR_QOS)
        self.pub_scan = self.create_publisher(LaserScan, '/mapper/scan', SENSOR_QOS)
        self.create_subscription(CameraInfo, '/stereo/left/camera_info', self.info_cb, 5)
        self.create_subscription(Image, '/stereo/depth/image_raw', self.depth_cb, SENSOR_QOS)
        self.create_subscription(Odometry, '/slam/odom', self.odom_cb, 20)
        self.create_subscription(PoseArray, '/slam/keyframes', self.kfs_cb, 5)
        self.create_subscription(Detection2DArray, '/perception/dynamic_boxes', self.boxes_cb, 5)
        self.create_timer(1.0, self.publish_maps)
        self.last_stamp = None

    def info_cb(self, msg):
        if self.mapper is None:
            self.mapper = Mapper(Intrinsics.from_info(msg))
            self.get_logger().info('mapper ready')

    def boxes_cb(self, msg):
        self.boxes = [(d.bbox.center.position.x - d.bbox.size_x / 2, d.bbox.center.position.y - d.bbox.size_y / 2,
                       d.bbox.center.position.x + d.bbox.size_x / 2, d.bbox.center.position.y + d.bbox.size_y / 2)
                      for d in msg.detections]

    def depth_cb(self, msg):
        pair = self.sync.add_left(msg)
        if pair:
            self.process(*pair)

    def odom_cb(self, msg):
        pair = self.sync.add_right(msg)
        if pair:
            self.process(*pair)

    def kfs_cb(self, msg):
        if self.mapper is None:
            return
        poses = [msg_to_T(p) for p in msg.poses]
        if self.mapper.update_keyframes(poses):
            self.get_logger().info('loop closure: map rebuilt from {} frames ({} rebuilds)'.format(
                len(self.mapper.frames), self.mapper.rebuilds))

    def process(self, depth_msg, odom_msg):
        if self.mapper is None or not odom_msg.child_frame_id.startswith('kf'):
            return
        t = stamp_to_sec(depth_msg.header.stamp)
        depth = image_to_numpy(depth_msg)
        T_w_base = msg_to_T(odom_msg.pose.pose)
        T_w_cam = T_w_base @ self.T_base_cam
        kf_id = int(odom_msg.child_frame_id[2:])
        rvec = np.array([odom_msg.twist.twist.angular.x, odom_msg.twist.twist.angular.y,
                         odom_msg.twist.twist.angular.z])
        T_kf_cam = np.eye(4)
        T_kf_cam[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_kf_cam[:3, 3] = [odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y,
                           odom_msg.twist.twist.linear.z]
        if odom_msg.pose.covariance[1] > 0.5:          # SLAM lost: do not map from a guessed pose
            return
        with self.stats.measure():
            self.mapper.add_frame(t, kf_id, T_kf_cam, T_w_cam, depth, self.boxes or None)
        self.last_stamp = depth_msg.header.stamp
        self.publish_obstacles(depth_msg, depth)
        self.stats.publish(frames=len(self.mapper.frames), rebuilds=self.mapper.rebuilds,
                           ms_integrate=round(self.mapper.ms_integrate, 1),
                           floor_z=round(self.mapper.floor_z, 2))

    def publish_obstacles(self, depth_msg, depth):
        """Current points between 15 cm and 1.4 m above the floor, within 4 m, in base_link."""
        pts = self.mapper.points_cam(depth, None)        # people must stay visible here
        if len(pts) == 0:
            return
        pb = pts @ self.T_base_cam[:3, :3].T + self.T_base_cam[:3, 3]
        # height above floor: base is 0.48 m above the floor in the robot's own frame (roll/pitch ignored)
        h = pb[:, 2] + self.mapper.robot_height
        keep = (h > 0.15) & (h < 1.4) & (np.hypot(pb[:, 0], pb[:, 1]) < 4.0)
        self.pub_obs.publish(make_cloud(pb[keep], 'base_link', depth_msg.header.stamp))
        scan = LaserScan()
        scan.header.stamp, scan.header.frame_id = depth_msg.header.stamp, 'base_link'
        scan.angle_min, scan.angle_max = -0.72, 0.72
        scan.angle_increment = (scan.angle_max - scan.angle_min) / 116
        scan.range_min, scan.range_max = 0.3, 4.0
        scan.ranges = points_to_scan(pb, self.mapper.robot_height).tolist()
        self.pub_scan.publish(scan)

    def publish_maps(self):
        if self.mapper is None or self.last_stamp is None or len(self.mapper.frames) == 0:
            return
        vm = self.mapper.vm
        grid = self.mapper.grid_2d()
        msg = OccupancyGrid()
        msg.header.stamp = self.last_stamp
        msg.header.frame_id = 'map'
        msg.info.resolution = vm.res
        msg.info.width, msg.info.height = vm.nx, vm.ny
        msg.info.origin.position.x, msg.info.origin.position.y = vm.x0, vm.y0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.ravel().tolist()
        self.pub_map.publish(msg)
        if self.pub_3d.get_subscription_count() > 0:
            pts = vm.occupied_points(1.0, z_min=self.mapper.floor_z + 0.08)
            if len(pts) > 60000:
                pts = pts[np.linspace(0, len(pts) - 1, 60000).astype(int)]
            self.pub_3d.publish(make_cloud(pts, 'map', self.last_stamp,
                                           height_colors(pts[:, 2] - self.mapper.floor_z)))


def main():
    rclpy.init()
    node = MapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
