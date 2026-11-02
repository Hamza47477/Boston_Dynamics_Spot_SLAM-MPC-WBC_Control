"""Frontier exploration: drive to the edge between known free space and unknown space, repeat.

A frontier is a free cell that touches unknown cells. The explorer groups frontier cells into clusters, picks
the nearest cluster that is big enough and reachable (far enough from walls), and sends it to Nav2 as a goal.
When Nav2 reaches it (or fails, in which case the spot is blacklisted) the map has grown, and it picks the
next one. It stops when no frontier is left. Needs Nav2 running (nav:=true).
"""
import math

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener

from .geometry import wrap


def find_frontiers(grid, res, min_cells=6, clearance=0.5):
    """grid: int8 (H, W) with -1 unknown, 0 free, 100 occupied. Returns list of (row, col, size) clusters."""
    free = grid == 0
    unknown = grid == -1
    near_unknown = np.zeros_like(free)
    near_unknown[1:, :] |= unknown[:-1, :]
    near_unknown[:-1, :] |= unknown[1:, :]
    near_unknown[:, 1:] |= unknown[:, :-1]
    near_unknown[:, :-1] |= unknown[:, 1:]
    frontier = free & near_unknown
    # only frontiers with some clearance from obstacles are reachable
    dist = cv2.distanceTransform((grid != 100).astype(np.uint8), cv2.DIST_L2, 3) * res
    frontier &= dist > clearance
    mask = cv2.dilate(frontier.astype(np.uint8), np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for k in range(1, n):
        cells = np.argwhere((labels == k) & frontier)
        if len(cells) >= min_cells:
            out.append((cells, len(cells)))
    return out


class Explorer(Node):

    def __init__(self):
        super().__init__('explorer')
        self.grid = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.goal_handle = None
        self.goal_xy = None
        self.goal_time = 0.0
        self.blacklist = []
        self.empty_cycles = 0
        self.looked_around = False
        self.spin_until = 0.0
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, latched)
        self.create_timer(2.0, self.tick)
        self.create_timer(0.1, self.spin)
        self.get_logger().info('explorer waiting for /map and Nav2')

    def map_cb(self, msg):
        self.grid = (np.array(msg.data, np.int8).reshape(msg.info.height, msg.info.width),
                     msg.info.origin.position.x, msg.info.origin.position.y, msg.info.resolution)

    def robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        except Exception:
            return None
        return t.transform.translation.x, t.transform.translation.y

    def spin(self):
        if self.get_clock().now().nanoseconds * 1e-9 < self.spin_until:
            cmd = Twist()
            cmd.angular.z = 0.35
            self.pub_cmd.publish(cmd)

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now < self.spin_until:
            return
        if self.goal_handle is not None:
            if now - self.goal_time > 120.0:                      # stuck: give up on this goal
                self.get_logger().warn('goal timed out; blacklisting it')
                self.blacklist.append(self.goal_xy)
                self.goal_handle.cancel_goal_async()
                self.goal_handle = None
            return
        if self.grid is None or not self.client.server_is_ready():
            return
        pos = self.robot_xy()
        if pos is None:
            return
        grid, x0, y0, res = self.grid
        clusters = find_frontiers(grid, res)
        best, best_score = None, 1e9
        for cells, size in clusters:
            xy = np.stack([x0 + (cells[:, 1] + 0.5) * res, y0 + (cells[:, 0] + 0.5) * res], axis=1)
            d = np.hypot(xy[:, 0] - pos[0], xy[:, 1] - pos[1])
            d_far = np.where(d >= 1.0, d, np.inf)          # the edge of the view right next to us is no frontier
            i = int(np.argmin(d_far))
            if not np.isfinite(d_far[i]) or any(math.hypot(xy[i][0] - b[0], xy[i][1] - b[1]) < 0.8 for b in self.blacklist):
                continue
            score = d[i] - 0.03 * size
            if score < best_score:
                best, best_score = (xy[i][0], xy[i][1]), score
        if best is None:
            self.empty_cycles += 1
            if not self.looked_around:                      # no map yet: turn on the spot to reveal the surroundings
                self.looked_around = True
                self.spin_until = now + 20.0
                self.get_logger().info('no frontier yet: turning on the spot to look around')
                return
            if self.empty_cycles == 3:
                self.get_logger().info('no reachable frontier left: exploration finished')
            return
        self.empty_cycles = 0
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        yaw = math.atan2(best[1] - pos[1], best[0] - pos[0])
        goal.pose.pose.position.x, goal.pose.pose.position.y = best
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.goal_xy, self.goal_time = best, now
        self.get_logger().info('exploring towards frontier ({:.1f}, {:.1f}), {:.1f} m away, {} clusters'.format(
            best[0], best[1], math.hypot(best[0] - pos[0], best[1] - pos[1]), len(clusters)))
        future = self.client.send_goal_async(goal)
        future.add_done_callback(self.accepted)
        self.goal_handle = 'pending'

    def accepted(self, future):
        handle = future.result()
        if not handle.accepted:
            self.blacklist.append(self.goal_xy)
            self.goal_handle = None
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(self.finished)

    def finished(self, future):
        status = future.result().status
        if status != 4:                                       # 4 = SUCCEEDED
            self.blacklist.append(self.goal_xy)
            self.get_logger().info('goal failed (status {}); blacklisted'.format(status))
        self.goal_handle = None


def main():
    rclpy.init()
    node = Explorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
