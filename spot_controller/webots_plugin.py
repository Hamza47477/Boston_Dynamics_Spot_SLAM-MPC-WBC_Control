"""Webots <-> ROS 2 bridge plugin, loaded by webots_ros2_driver from the URDF in the launch file."""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')        # tiny matrices: threading only slows NumPy
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import collections
import json
import random
import time

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped, Wrench
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
import numpy as np
import rclpy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from .constants import (ACCEL_BIAS, ACCEL_SIGMA, clamp_joint, GYRO_BIAS, GYRO_SIGMA,
                        IMU_NOISE_ENABLED, JOINT_LIMITS, JOINT_NAMES, LEG_ORDER, MOTOR_DEVICE,
                        SENSOR_DEVICE, TORQUE_LIMIT)
from .ekf_node import error_vector, make_odometry
from .estimator_core import EstimatorCore
from .leg_odometry import joints_to_leg_dict
from .locomotion_core import LocomotionCore
from .scene import Scene
from .stereo_sensors import StereoSensors

PUSH_DURATION = 0.2


class SpotMotorsPlugin:
    """Publishes /joint_states (with velocity), /imu and ground truth; drives the motors.

    Control modes (URDF plugin parameter `control_mode`):
      position (default): /joint_commands.position goes to the Webots position servo.
      torque: the state estimator and the locomotion controller (stand / trot / push recovery)
              run INSIDE this plugin, in the same
              simulation step as the sensors (no communication delay). Until the estimator has
              started, the stiff Webots servo holds the zero pose. Then every step
                tau = Kp (q_des - q) + Kd (qd_des - qdot) + tau_ff
              is applied. It also publishes /odom_estimated, /estimator/error and
              /locomotion/status. Velocity commands: /cmd_vel (Twist: x, y in m/s, z = yaw rate).
    A force sent to /disturbance (geometry_msgs/Wrench, world frame, N) is applied to the robot's
    centre of mass for PUSH_DURATION seconds (needs `supervisor TRUE` in the world).

    All stamps are Webots simulation time, so nodes are independent of real-time speed.
    /imu carries NO orientation (like a real IMU). Ground truth for evaluation:
    /ground_truth/pose (GPS position + inertial-unit attitude), /ground_truth/twist (GPS velocity).
    """

    def init(self, webots_node, properties):
        self.robot = webots_node.robot
        self.timestep = int(self.robot.getBasicTimeStep())

        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node('spot_webots_plugin')
        log = self.node.get_logger()

        self.motors = {}
        self.sensors = {}
        for name in JOINT_NAMES:
            motor = self.robot.getDevice(MOTOR_DEVICE[name])
            sensor = self.robot.getDevice(SENSOR_DEVICE[name])
            if motor is None or sensor is None:
                log.error('Webots device missing for joint {} ({} / {})'.format(
                    name, MOTOR_DEVICE[name], SENSOR_DEVICE[name]))
                continue
            sensor.enable(self.timestep)
            self.motors[name] = motor
            self.sensors[name] = sensor
        log.info('{} / {} joints connected'.format(len(self.motors), len(JOINT_NAMES)))

        self.accel = self._enable('accelerometer')
        self.gyro = self._enable('gyro')
        self.inertial = self._enable('inertial unit')
        self.gps = self._enable('gps')

        self.rng = random.Random(1)
        self.prev_pos = {}
        self.vel_filtered = {name: 0.0 for name in JOINT_NAMES}
        self.prev_time = None

        self.torque_mode = str(properties.get('control_mode', 'position')) == 'torque'
        self.q_des = {}
        self.qd_des = {name: 0.0 for name in JOINT_NAMES}
        self.kp = {name: 0.0 for name in JOINT_NAMES}
        self.kd = {name: 0.0 for name in JOINT_NAMES}
        self.tau_ff = {name: 0.0 for name in JOINT_NAMES}
        self.torque_active = False
        if self.torque_mode:
            for motor in self.motors.values():          # stiff servo until torque control starts
                motor.setPosition(0.0)
            self.est = EstimatorCore()
            overrides = json.loads(os.environ.get('SPOT_STANCE_PARAMS', '{}'))
            overrides.setdefault('height', float(properties.get('height', 0.5)))
            overrides.setdefault('controller', str(properties.get('controller', 'vmc')))
            self.ctrl = LocomotionCore(**overrides)   # e.g. SPOT_STANCE_PARAMS='{"duty":0.65}'
            self.cmd_time = -1e9
            self.last_stance = None
            self.last_status = None
            self.flight = collections.deque(maxlen=600)    # flight recorder: last ~5 s
            self.flight_dumped = False
            self.node.create_subscription(Twist, '/cmd_vel', self.cmd_vel_cb, 10)
            self.pub_odom = self.node.create_publisher(Odometry, '/odom_estimated', 10)
            self.pub_err = self.node.create_publisher(Float64MultiArray, '/estimator/error', 10)
            self.pub_stand = self.node.create_publisher(Float64MultiArray, '/locomotion/status', 10)
            self.pub_debug = self.node.create_publisher(Float64MultiArray, '/locomotion/debug', 10)
            self.sat_count = 0
            self.sat_names = ''
            self.limit_margin = 9.0
            self.compute_time = 0.0
            self.compute_steps = 0
            self.last_report = time.time()
        log.info('control mode: ' + ('torque' if self.torque_mode else 'position'))

        self.push_force = None
        self.push_until = 0.0
        self.self_node = None
        try:
            self.self_node = self.robot.getSelf()
        except Exception:                                   # not a supervisor robot
            log.warn('disturbance pushes unavailable (robot is not a supervisor)')
        self.node.create_subscription(Wrench, '/disturbance', self.push_cb, 10)

        self.pub_clock = self.node.create_publisher(Clock, '/clock', 10)   # simulation time
        self.stereo = None
        if str(properties.get('stereo', 'false')) == 'true':
            self.stereo = StereoSensors(self.robot, self.node, self.timestep)
            if not self.stereo.active:
                log.error('stereo cameras requested but the world has no stereo_left/right')
                self.stereo = None
        self.scene = None                                   # moving cars + ground-truth objects
        self.step_count = 0
        if self.self_node is not None:
            scene = Scene(self.robot, log)
            if scene.active:
                self.scene = scene
                self.pub_gt_objects = self.node.create_publisher(
                    Detection3DArray, '/ground_truth/objects', 10)

        self.pub_states = self.node.create_publisher(JointState, '/joint_states', 10)
        self.pub_imu = self.node.create_publisher(Imu, '/imu', 10)
        self.pub_gt_pose = self.node.create_publisher(PoseStamped, '/ground_truth/pose', 10)
        self.pub_gt_twist = self.node.create_publisher(TwistStamped, '/ground_truth/twist', 10)
        self.node.create_subscription(JointState, '/joint_commands', self.cmd_cb, 10)

    def _enable(self, device_name):
        device = self.robot.getDevice(device_name)
        if device is None:
            self.node.get_logger().error('Webots device missing: ' + device_name)
        else:
            device.enable(self.timestep)
        return device

    def cmd_cb(self, msg):
        """Position-mode joint commands (Webots convention)."""
        if self.torque_mode:
            return
        for name, pos in zip(msg.name, msg.position):
            if name in self.motors:
                self.motors[name].setPosition(clamp_joint(name, pos))

    def cmd_vel_cb(self, msg):
        self.ctrl.set_command(msg.linear.x, msg.linear.y, msg.angular.z)
        self.cmd_time = self.robot.getTime()

    def push_cb(self, msg):
        self.push_force = [msg.force.x, msg.force.y, msg.force.z]
        self.push_until = self.robot.getTime() + PUSH_DURATION
        self.node.get_logger().info('push {} N for {} s'.format(self.push_force, PUSH_DURATION))

    def _apply_torques(self):
        """Inner loop: joint PD + feedforward torque, clamped to the motor limit."""
        sat, margin, names = 0, 9.0, ''
        taus = {}
        for name, motor in self.motors.items():
            q = self.sensors[name].getValue()
            tau = (self.kp[name] * (self.q_des[name] - q)
                   + self.kd[name] * (self.qd_des[name] - self.vel_filtered[name])
                   + self.tau_ff[name])
            if abs(tau) >= TORQUE_LIMIT:
                sat += 1
                names += name + ' '
            low, high = JOINT_LIMITS[name]
            margin = min(margin, q - low, high - q)
            taus[name] = tau
            motor.setTorque(max(-TORQUE_LIMIT, min(TORQUE_LIMIT, tau)))
        self.sat_count, self.sat_names, self.limit_margin = sat, names, margin
        return taus

    def _noisy(self, values, bias, sigma):
        if not IMU_NOISE_ENABLED:
            return list(values)
        return [v + b + self.rng.gauss(0.0, sigma) for v, b in zip(values, bias)]

    def _torque_step(self, sim_time, stamp, gyro, accel, positions, gt_pose, gt_vel):
        """Estimator + stance controller for this simulation step."""
        t0 = time.perf_counter()
        updated = self.est.step_imu(sim_time, gyro, accel, gt_pose)
        if updated:
            q = joints_to_leg_dict(JOINT_NAMES, [positions[n] for n in JOINT_NAMES])
            qd = joints_to_leg_dict(JOINT_NAMES, [self.vel_filtered[n] for n in JOINT_NAMES])
            self.est.step_legs(q, qd, self.last_stance)
            p, R, v, w_body = self.est.state()
            if sim_time - self.cmd_time > 1.0:             # command timeout: stop
                self.ctrl.set_command(0.0, 0.0, 0.0)
            try:
                out = self.ctrl.compute(sim_time, p, R, v, w_body, q, qd)
            except Exception as exc:                     # keep the last command, report it
                self.node.get_logger().error('controller error: {!r}'.format(exc),
                                             throttle_duration_sec=1.0)
                out = None
            if out is None:
                self.compute_time += time.perf_counter() - t0
                self.compute_steps += 1
                return
            self.last_stance = out.stance_est
            self.last_status = status = out.status
            for i, name in enumerate(JOINT_NAMES):
                self.q_des[name] = out.q_des[i]
                self.qd_des[name] = out.qd_des[i]
                self.kp[name] = out.kp[i]
                self.kd[name] = out.kd[i]
                self.tau_ff[name] = out.tau_ff[i]
            self.torque_active = True
            self.pub_stand.publish(Float64MultiArray(data=status))
            self.pub_debug.publish(Float64MultiArray(
                data=[self.ctrl.gait.phase[leg] for leg in LEG_ORDER]
                + [float(self.sat_count), float(self.limit_margin)]))
            self.pub_odom.publish(make_odometry(stamp, self.est))
            if gt_pose is not None and gt_vel is not None:
                self.pub_err.publish(Float64MultiArray(
                    data=error_vector(self.est, gt_pose, gt_vel)))
        self.compute_time += time.perf_counter() - t0
        self.compute_steps += 1
        if time.time() - self.last_report > 5.0 and self.torque_active:
            status = self.last_status
            self.last_report = time.time()
            self.node.get_logger().info(
                'torque loop: {:.1f} ms/step | height {:.3f} m | roll/pitch {:+.1f}/{:+.1f} deg | '
                'v body ({:+.2f}, {:+.2f}) m/s | cmd {:.2f} | stance feet {:.0f} | stepping {:.0f} | '
                'mpc {:.1f} ms ({} fb) | wbc {:.1f} ms ({} fb)'
                .format(1e3 * self.compute_time / max(1, self.compute_steps), status[0],
                        status[1], status[2], status[4], status[5], status[6], status[7],
                        status[9], 1e3 * self.ctrl.mpc_info[2], self.ctrl.mpc_fallbacks,
                        1e3 * self.ctrl.wbc_info[2], self.ctrl.wbc_fallbacks))
            self.compute_time, self.compute_steps = 0.0, 0

    def _record(self, t, positions, taus, gt_pose):
        """Flight recorder: on a big tilt, write the last seconds of controller data to a file."""
        p, R, v, _ = self.est.state()
        st = self.last_status
        self.flight.append((t, st[1], st[2], float(p[2]), v, dict(self.ctrl.gait.phase), int(st[7]),
                            self.ctrl.debug.copy(), self.ctrl.debug_feet, p.copy(), st[3],
                            self.ctrl.cmd.copy(), self.ctrl.yaw_des, self.ctrl.v_des_world.copy(),
                            self.ctrl.v_filt.copy(), dict(positions), dict(self.q_des), dict(taus),
                            None if gt_pose is None else gt_pose[1].copy()))
        if not self.flight_dumped and max(abs(st[1]), abs(st[2])) > 12.0:
            self.flight_dumped = True
            rows = []
            for (tt, r_, p_, h_, v_, ph, n_, dbg, feet, pos, yaw, cmd, yd, vd, vf, q_, qd_, tau_,
                 gtq) in self.flight:
                rows.append(dict(
                    t=round(tt, 3), tilt=[round(r_, 2), round(p_, 2)], h=round(h_, 3),
                    v=[round(float(a), 3) for a in v_], phase={l: round(x, 3) for l, x in ph.items()},
                    n=n_, debug=dbg, feet_r=feet, p=[round(float(a), 3) for a in pos],
                    yaw=round(yaw, 1), cmd=[round(float(a), 2) for a in cmd],
                    yaw_des=round(float(np.degrees(yd)), 1),
                    v_des=[round(float(a), 2) for a in vd[:2]],
                    v_filt=[round(float(a), 2) for a in vf],
                    q={k: round(x, 3) for k, x in q_.items()},
                    q_des={k: round(x, 3) for k, x in qd_.items()},
                    tau={k: round(x, 1) for k, x in tau_.items()},
                    gt_tilt=None if gtq is None else [round(float(a), 2) for a in self._gt_rp(gtq)]))
            with open('/tmp/flight_record.json', 'w') as handle:
                json.dump(rows, handle)
            self.node.get_logger().warn('tilt > 12 deg: flight record written to /tmp/flight_record.json')

    @staticmethod
    def _gt_rp(q_wxyz):
        w, x, y, z = q_wxyz
        roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = np.degrees(np.arcsin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
        return roll, pitch

    def _scene_step(self, sim_time, stamp):
        if self.step_count % 2 == 0:
            self.scene.step(sim_time, 2 * self.timestep / 1000.0)
        if self.step_count % 12 == 0:
            msg = Detection3DArray()
            msg.header.stamp = stamp
            msg.header.frame_id = 'world'
            for obj in self.scene.ground_truth():
                det = Detection3D()
                det.id = obj['name']
                det.bbox.center.position.x, det.bbox.center.position.y = obj['position'][:2]
                det.bbox.center.position.z = float(obj['position'][2])
                det.bbox.center.orientation.z = float(np.sin(obj['yaw'] / 2))
                det.bbox.center.orientation.w = float(np.cos(obj['yaw'] / 2))
                det.bbox.size.x, det.bbox.size.y, det.bbox.size.z = [float(v) for v in obj['size']]
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = obj['cls']
                hyp.hypothesis.score = 1.0
                det.results.append(hyp)
                msg.detections.append(det)
            self.pub_gt_objects.publish(msg)

    def step(self):
        rclpy.spin_once(self.node, timeout_sec=0)
        sim_time = self.robot.getTime()
        self.step_count += 1
        sec = int(sim_time)
        stamp = Time(sec=sec, nanosec=int((sim_time - sec) * 1e9))
        dt = None if self.prev_time is None else sim_time - self.prev_time
        self.pub_clock.publish(Clock(clock=stamp))

        # IMU first, then joint states, so an estimator sees them in the same order.
        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = 'base_link'
        imu.orientation_covariance[0] = -1.0           # orientation not provided
        a = g = (0.0, 0.0, 0.0)
        if self.accel:
            a = self._noisy(self.accel.getValues(), ACCEL_BIAS, ACCEL_SIGMA)
            imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = a
            imu.linear_acceleration_covariance[0] = ACCEL_SIGMA ** 2
        else:
            imu.linear_acceleration_covariance[0] = -1.0
        if self.gyro:
            g = self._noisy(self.gyro.getValues(), GYRO_BIAS, GYRO_SIGMA)
            imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = g
            imu.angular_velocity_covariance[0] = GYRO_SIGMA ** 2
        else:
            imu.angular_velocity_covariance[0] = -1.0
        self.pub_imu.publish(imu)

        states = JointState()
        states.header.stamp = stamp
        positions = {}
        for name in JOINT_NAMES:
            if name in self.sensors:
                pos = self.sensors[name].getValue()
                positions[name] = pos
                states.name.append(name)
                states.position.append(pos)
                prev = self.prev_pos.get(name)
                raw = 0.0 if prev is None or not dt else (pos - prev) / dt
                self.vel_filtered[name] = 0.6 * self.vel_filtered[name] + 0.4 * raw
                states.velocity.append(self.vel_filtered[name])
                self.prev_pos[name] = pos
        if states.name:
            self.pub_states.publish(states)
        self.prev_time = sim_time

        gt_pose = gt_vel = None
        if self.gps and self.inertial:
            position = self.gps.getValues()
            qx, qy, qz, qw = self.inertial.getQuaternion()
            speed = self.gps.getSpeedVector()
            gt_pose = (np.array(position), np.array([qw, qx, qy, qz]))
            gt_vel = np.array(speed)

            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = 'world'
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
            pose.pose.orientation.x, pose.pose.orientation.y = qx, qy
            pose.pose.orientation.z, pose.pose.orientation.w = qz, qw
            self.pub_gt_pose.publish(pose)

            twist = TwistStamped()
            twist.header.stamp = stamp
            twist.header.frame_id = 'world'
            twist.twist.linear.x, twist.twist.linear.y, twist.twist.linear.z = speed
            self.pub_gt_twist.publish(twist)

        if self.torque_mode and len(positions) == len(JOINT_NAMES):
            self._torque_step(sim_time, stamp, g, a, positions, gt_pose, gt_vel)
            if self.torque_active:
                taus = self._apply_torques()
                self._record(sim_time, positions, taus, gt_pose)

        if self.scene is not None:
            self._scene_step(sim_time, stamp)
        if self.stereo is not None:
            self.stereo.publish(self.step_count, stamp)

        if self.push_force is not None and self.self_node is not None:
            if sim_time < self.push_until:
                self.self_node.addForce(self.push_force, False)
            else:
                self.push_force = None
