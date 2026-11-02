"""Drive Spot around a list of waypoints (Webots world coordinates, from /ground_truth/pose).

Only for recording repeatable data: it uses ground truth, so it is NOT part of the perception stack.
    python3 tools/waypoint_drive.py --laps 2            # default loop around partition A
"""
import argparse
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node

# around partition A (x = -2, y from -8 to 1): up the left side, over the open end, down the right
# side, back through the 2 m gap at y = -9
LOOP = [(-3.5, -5.0), (-3.5, 3.0), (0.3, 3.0), (0.3, -9.0), (-3.5, -9.0)]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Driver(Node):

    def __init__(self, waypoints, laps, speed, wz_max):
        super().__init__('waypoint_drive')
        self.wps, self.laps, self.speed, self.wz_max = waypoints, laps, speed, wz_max
        self.idx, self.lap, self.pose = 0, 0, None
        self.done = False
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(PoseStamped, '/ground_truth/pose', self.pose_cb, 10)
        self.create_timer(0.05, self.tick)

    def pose_cb(self, msg):
        q = msg.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (msg.pose.position.x, msg.pose.position.y, yaw)

    def tick(self):
        if self.pose is None or self.done:
            return
        x, y, yaw = self.pose
        tx, ty = self.wps[self.idx]
        dist = math.hypot(tx - x, ty - y)
        if dist < 0.4:
            self.idx += 1
            if self.idx >= len(self.wps):
                self.idx, self.lap = 0, self.lap + 1
                self.get_logger().info('lap {} done'.format(self.lap))
                if self.lap >= self.laps:
                    self.done = True
                    self.pub.publish(Twist())
                    return
            return
        err = wrap(math.atan2(ty - y, tx - x) - yaw)
        cmd = Twist()
        cmd.angular.z = max(-self.wz_max, min(self.wz_max, 1.2 * err))
        cmd.linear.x = self.speed * max(0.0, math.cos(err)) if abs(err) < 0.6 else 0.05
        self.pub.publish(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--laps', type=int, default=1)
    ap.add_argument('--speed', type=float, default=0.3)
    ap.add_argument('--wz', type=float, default=0.5)
    args = ap.parse_args()
    rclpy.init()
    node = Driver(LOOP, args.laps, args.speed, args.wz)
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
