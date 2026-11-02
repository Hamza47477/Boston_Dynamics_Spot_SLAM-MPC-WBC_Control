"""Push test: (optionally walk), push the robot, report how it reacts (ground truth)."""
import argparse
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist, Wrench, TwistStamped
from rclpy.node import Node

from .quaternion import q_to_rpy


class PushTest(Node):

    def __init__(self, force, direction, settle, cmd):
        super().__init__('push_test')
        self.force, self.direction, self.settle = force, direction, settle
        self.cmd = cmd
        self.pub = self.create_publisher(Wrench, '/disturbance', 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speed_before = None
        self.create_subscription(TwistStamped, '/ground_truth/twist', self.twist_cb, 10)
        self.vel = (0.0, 0.0)
        self.create_timer(0.1, self.send_cmd)
        self.create_subscription(PoseStamped, '/ground_truth/pose', self.pose_cb, 10)
        self.t0 = None
        self.t_push = None
        self.base = None
        self.max_tilt = 0.0
        self.max_disp = 0.0
        self.max_dz = 0.0
        self.last = None
        self.done = False

    def twist_cb(self, msg):
        self.vel = (msg.twist.linear.x, msg.twist.linear.y)

    def send_cmd(self):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = self.cmd
        self.pub_cmd.publish(msg)

    def pose_cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p, o = msg.pose.position, msg.pose.orientation
        roll, pitch, _ = q_to_rpy([o.w, o.x, o.y, o.z])
        if self.t0 is None:
            self.t0 = t
        if self.t_push is None and t - self.t0 > self.settle:
            self.base = (p.x, p.y, p.z)
            self.speed_before = math.hypot(*self.vel)
            self.t_push = t
            f = Wrench()
            f.force.x = self.force * self.direction[0]
            f.force.y = self.force * self.direction[1]
            self.pub.publish(f)
            self.get_logger().info('pushing with {:.0f} N, height before: {:.3f} m'.format(
                self.force, p.z))
        if self.t_push is not None:
            self.max_tilt = max(self.max_tilt, math.degrees(max(abs(roll), abs(pitch))))
            self.max_disp = max(self.max_disp, math.hypot(p.x - self.base[0], p.y - self.base[1]))
            self.max_dz = max(self.max_dz, abs(p.z - self.base[2]))
            self.last = (math.hypot(p.x - self.base[0], p.y - self.base[1]), p.z - self.base[2],
                         math.degrees(roll), math.degrees(pitch))
            if t - self.t_push > 6.0:
                self.done = True


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', type=float, default=100.0, help='push force in N')
    parser.add_argument('--direction', type=float, nargs=2, default=[0.0, 1.0],
                        help='x y direction in the world frame')
    parser.add_argument('--vx', type=float, default=0.0, help='forward speed command, m/s')
    parser.add_argument('--vy', type=float, default=0.0, help='sideways speed command, m/s')
    parser.add_argument('--wz', type=float, default=0.0, help='yaw-rate command, rad/s')
    parser.add_argument('--settle', type=float, default=8.0, help='simulation seconds to wait')
    ns, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    norm = math.hypot(*ns.direction) or 1.0
    node = PushTest(ns.force, [d / norm for d in ns.direction], ns.settle, (ns.vx, ns.vy, ns.wz))
    deadline = time.time() + 600
    while rclpy.ok() and not node.done and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.done:
        print('\nPush {:.0f} N for 0.2 s:'.format(ns.force))
        print('  speed before push  : {:.2f} m/s (command {:.2f})'.format(
            node.speed_before, math.hypot(ns.vx, ns.vy)))
        print('  result             : {}'.format('FELL' if node.max_tilt > 45.0 else 'stayed up'))
        print('  peak tilt          : {:.2f} deg'.format(node.max_tilt))
        print('  peak displacement  : {:.3f} m'.format(node.max_disp))
        print('  peak height change : {:.3f} m'.format(node.max_dz))
        print('  after 6 s          : shift {:.3f} m, dz {:+.3f} m, roll {:+.2f} deg, '
              'pitch {:+.2f} deg'.format(*node.last))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
