"""Send Spot to a point on the map:   ros2 run spot_controller goto X Y [YAW]    (map frame, metres, radians)."""
import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    x, y = float(sys.argv[1]), float(sys.argv[2])
    yaw = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    rclpy.init()
    node = Node('goto')
    client = ActionClient(node, NavigateToPose, 'navigate_to_pose')
    if not client.wait_for_server(timeout_sec=10.0):
        print('Nav2 is not running (start the launch file with nav:=true)')
        return
    goal = NavigateToPose.Goal()
    goal.pose = PoseStamped()
    goal.pose.header.frame_id = 'map'
    goal.pose.pose.position.x, goal.pose.pose.position.y = x, y
    goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    last = [0.0]

    def feedback(msg):
        d = msg.feedback.distance_remaining
        if abs(d - last[0]) > 0.5:
            last[0] = d
            print('  {:.1f} m to go'.format(d))

    handle = client.send_goal_async(goal, feedback_callback=feedback)
    rclpy.spin_until_future_complete(node, handle)
    handle = handle.result()
    if not handle.accepted:
        print('goal rejected')
        return
    print('goal accepted: going to ({:.1f}, {:.1f})'.format(x, y))
    result = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result)
    print('finished with status', {4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}.get(result.result().status, result.result().status))
    rclpy.shutdown()


if __name__ == '__main__':
    main()
