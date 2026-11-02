"""Keyboard teleoperation: publishes /cmd_vel. Run in a terminal; keys are read raw (no Enter)."""
import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist

HELP = """Spot teleop
  w / s : forward / backward      a / d : step left / right
  q / e : turn left / right       space : stop       x : quit
  + / - : change speed
"""
KEYS = {'w': (1, 0, 0), 's': (-1, 0, 0), 'a': (0, 1, 0), 'd': (0, -1, 0),
        'q': (0, 0, 1), 'e': (0, 0, -1), ' ': (0, 0, 0)}


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('spot_teleop')
    pub = node.create_publisher(Twist, '/cmd_vel', 10)
    settings = termios.tcgetattr(sys.stdin)
    scale = 0.5
    cmd = (0, 0, 0)
    print(HELP)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                if key == 'x':
                    break
                if key in KEYS:
                    cmd = KEYS[key]
                elif key == '+':
                    scale = min(1.0, scale + 0.1)
                elif key == '-':
                    scale = max(0.2, scale - 0.1)
                print('\rcommand {} speed scale {:.1f}   '.format(cmd, scale), end='', flush=True)
            msg = Twist()
            msg.linear.x = cmd[0] * 0.5 * scale
            msg.linear.y = cmd[1] * 0.25 * scale
            msg.angular.z = cmd[2] * 0.8 * scale
            pub.publish(msg)                      # repeated, the plugin stops after 1 s of silence
    finally:
        pub.publish(Twist())
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
