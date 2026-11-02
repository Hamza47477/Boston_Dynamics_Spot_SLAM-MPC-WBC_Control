"""ROS node around EstimatorCore: scored against GPS ground truth (position-control mode)."""
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

from .estimator_core import EstimatorCore
from .leg_odometry import joints_to_leg_dict
from .quaternion import q_to_R, q_to_rpy, wrap_angle

ERROR_FIELDS = ['cf_roll_err_deg', 'cf_pitch_err_deg', 'ekf_roll_err_deg', 'ekf_pitch_err_deg',
                'ekf_yaw_err_deg', 'vel_err_mps', 'pos_err_m', 'n_stance_feet']


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def error_vector(core, gt_pose, gt_vel):
    """Estimator errors against ground truth, in the order of ERROR_FIELDS."""
    p_gt, q_gt = gt_pose
    gt = np.array(q_to_rpy(q_gt))
    d_ekf = np.degrees([wrap_angle(a) for a in np.array(q_to_rpy(core.ekf.q)) - gt])
    d_cf = np.degrees([wrap_angle(a) for a in np.array(core.cf.rpy()) - gt])
    return [float(d_cf[0]), float(d_cf[1]), float(d_ekf[0]), float(d_ekf[1]), float(d_ekf[2]),
            float(np.linalg.norm(core.ekf.v - gt_vel)), float(np.linalg.norm(core.ekf.p - p_gt)),
            float(core.n_stance)]


def make_odometry(stamp, core):
    p, R, v, w_body = core.state()
    odom = Odometry()
    odom.header.stamp = stamp
    odom.header.frame_id = 'world'
    odom.child_frame_id = 'base_link'
    odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = p
    w, x, y, z = core.ekf.q
    odom.pose.pose.orientation.x, odom.pose.pose.orientation.y = x, y
    odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = z, w
    v_body = R.T @ v
    odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = v_body
    odom.twist.twist.angular.x, odom.twist.twist.angular.y = w_body[0], w_body[1]
    odom.twist.twist.angular.z = w_body[2]
    return odom


class EkfEstimator(Node):
    """Subscribes /imu, /joint_states, /ground_truth/*; publishes /odom_estimated, /estimator/error.

    /estimator/error is a Float64MultiArray with the fields listed in ERROR_FIELDS.
    Ground truth initialises yaw and position and scores the estimate; it is never fed to the
    filters afterwards.
    """

    def __init__(self):
        super().__init__('ekf_estimator')
        self.core = EstimatorCore()
        self.pending_joints = None
        self.t_last = None
        self.gt_pose = None
        self.gt_vel = None
        self.announced = False

        self.create_subscription(Imu, '/imu', self.imu_cb, 50)
        self.create_subscription(JointState, '/joint_states', self.joints_cb, 50)
        self.create_subscription(PoseStamped, '/ground_truth/pose', self.gt_pose_cb, 10)
        self.create_subscription(TwistStamped, '/ground_truth/twist', self.gt_twist_cb, 10)
        self.pub_odom = self.create_publisher(Odometry, '/odom_estimated', 10)
        self.pub_err = self.create_publisher(Float64MultiArray, '/estimator/error', 10)
        self.get_logger().info('EKF estimator waiting for IMU and ground-truth pose...')

    def gt_pose_cb(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        self.gt_pose = (np.array([p.x, p.y, p.z]), np.array([o.w, o.x, o.y, o.z]))

    def gt_twist_cb(self, msg):
        v = msg.twist.linear
        self.gt_vel = np.array([v.x, v.y, v.z])

    def imu_cb(self, msg):
        t = stamp_to_sec(msg.header.stamp)
        gyro = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        accel = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        updated = self.core.step_imu(t, gyro, accel, self.gt_pose)
        if self.core.initialized and not self.announced:
            self.announced = True
            self.get_logger().info('EKF initialised (roll/pitch from accelerometer, '
                                   'yaw/position from ground truth).')
        if not updated:
            return
        self.t_last = t
        if self.pending_joints is not None and abs(self.pending_joints[0] - t) < 1e-6:
            self.core.step_legs(*self.pending_joints[1:])
            self.pending_joints = None
        self.publish(msg.header.stamp)

    def joints_cb(self, msg):
        if len(msg.velocity) != len(msg.position):
            return
        q = joints_to_leg_dict(msg.name, msg.position)
        qd = joints_to_leg_dict(msg.name, msg.velocity)
        if q is None or qd is None:
            return
        t = stamp_to_sec(msg.header.stamp)
        self.pending_joints = (t, q, qd)
        if self.core.initialized and self.t_last is not None and abs(self.t_last - t) < 1e-6:
            self.core.step_legs(q, qd)                       # IMU for this stamp came first
            self.pending_joints = None

    def publish(self, stamp):
        self.pub_odom.publish(make_odometry(stamp, self.core))
        if self.gt_pose is None or self.gt_vel is None:
            return
        errors = error_vector(self.core, self.gt_pose, self.gt_vel)
        self.pub_err.publish(Float64MultiArray(data=errors))
        self.get_logger().info(
            'CF roll/pitch err {:+.2f}/{:+.2f} deg | EKF roll/pitch/yaw err {:+.2f}/{:+.2f}/'
            '{:+.2f} deg | vel err {:.3f} m/s | pos err {:.3f} m | stance {:.0f} | gyro bias '
            '({:+.3f} {:+.3f} {:+.3f})'.format(*errors, *self.core.ekf.bg),
            throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = EkfEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
