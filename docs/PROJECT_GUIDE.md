# Spot Controller: Project Guide

Living document. Update it whenever you add a node, topic, command or algorithm.
Last updated: 2026-10-26 (after fixing issues in section 5).

Stack: ROS 2 Jazzy, Webots R2025a, Python 3.12, `webots_ros2_driver`.
Package: `src/spot_controller` (ament_python). Workspace: `~/ros2_ws`.

---

## 1. Commands cheat sheet

### Build
```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select spot_controller --symlink-install
source install/setup.bash
```
`--symlink-install` means edits to Python files take effect without rebuilding.
You still need a rebuild after changing `setup.py`, launch files or worlds (unless symlinked).

### Run the simulation
```bash
ros2 launch spot_controller spot_launch.py                    # Webots + plugin + estimator + demo motion
ros2 launch spot_controller spot_launch.py motion:=stand      # hold a neutral stance (no sway)
ros2 launch spot_controller spot_launch.py demo:=false        # no demo publisher (send your own commands)
```
The launch file starts everything. To run nodes by hand instead (separate terminals, after
`source install/setup.bash`):
```bash
ros2 run spot_controller spot_publisher --ros-args -p motion:=sway
ros2 run spot_controller state_estimator
ros2 run spot_controller ekf_estimator
```
On a laptop with weak graphics, `export LIBGL_ALWAYS_SOFTWARE=1` before launching (as in your other script).

### Torque-mode controllers (MPC, whole-body control)
```bash
ros2 launch spot_controller spot_launch.py control:=torque controller:=vmc      # force controller (default)
ros2 launch spot_controller spot_launch.py control:=torque controller:=mpc      # convex MPC
ros2 launch spot_controller spot_launch.py control:=torque controller:=mpc_wbc  # MPC + whole-body QP
ros2 run spot_controller teleop                                                 # drive it
ros2 run spot_controller push_test --vx 0.3 --wz 0.3 --force 100 --direction 0 1 --settle 4
```
Extra launch arguments: `gui:=false`, `mode:=fast`, `port:=1240`, `world:=spot_flat.wbt`. Details in
[LOCOMOTION_CONTROL.md](LOCOMOTION_CONTROL.md).

### Unit tests (no ROS or Webots needed)
```bash
cd ~/ros2_ws/src/spot_controller && python3 -m pytest test/test_kinematics.py test/test_estimation.py -q
```

### Inspect
```bash
ros2 topic list
ros2 topic echo /joint_states --once
ros2 topic echo /imu --once
ros2 topic hz /imu                  # should match the Webots timestep rate
ros2 topic echo /estimator/error     # EKF / complementary filter errors vs ground truth (fields in docs/STATE_ESTIMATION.md)
ros2 topic echo /odom_estimated --once
ros2 topic echo /fk_state           # x=roll, y=pitch (rad), z=body height (m) from leg kinematics
ros2 topic echo /joint_commands --once
ros2 node list
ros2 run rqt_graph rqt_graph        # node/topic graph
ros2 run plotjuggler plotjuggler    # plot IMU / joints live (if installed)
```

### Debug
```bash
colcon build --packages-select spot_controller --event-handlers console_direct+
ros2 launch spot_controller spot_launch.py --debug
ls ~/ros2_ws/log/latest_build          # build logs
python3 -m pytest src/spot_controller/test -q   # lint tests (flake8, pep257)
```

---

## 2. Architecture

```
                 Webots (my_spot.wbt, Spot robot)
                         |  devices: motors, position sensors,
                         |  Accelerometer, Gyro, InertialUnit, GPS
                         v
       webots_ros2_driver  (WebotsController, robot_name='Spot')
                         |  loads plugin from the URDF string in spot_launch.py
                         v
              webots_plugin.py  (Python plugin, runs inside the driver)
        publishes:  /joint_states  (sensor_msgs/JointState)
                    /imu           (sensor_msgs/Imu)
        subscribes: /joint_commands (sensor_msgs/JointState)
                         ^                         |
                         |                         v
  spot_publisher_node.py                   state_estimator.py
  (IK demo, 50 Hz)                         (FK -> height, pitch, roll)
  uses spot_4leg_ik.py + ik_solver.py
```

### Files
| File | Role |
|---|---|
| `launch/spot_launch.py` | Starts Webots with the world and the `WebotsController`; injects the plugin via an inline URDF. |
| `worlds/my_spot.wbt` | Arena, stairs, obstacles, and the Spot robot. |
| `spot_controller/webots_plugin.py` | Bridge: reads sensors, publishes ROS topics, applies motor commands. |
| `spot_controller/ik_solver.py` | `SpotIK`: analytic IK for one 3-DOF leg. |
| `spot_controller/constants.py` | Single source of truth: joint names, Webots device names, limits, leg geometry, angle conversions. |
| `spot_controller/body_state.py` | ROS-free math: ground-plane fit (height/roll/pitch), quaternion to roll/pitch. |
| `spot_controller/ekf_node.py` | ROS node: complementary filter, stance detection, leg odometry, EKF, scoring against ground truth. |
| `spot_controller/ekf.py` | Error-state EKF (15 states). |
| `spot_controller/attitude_filter.py` | Mahony complementary filter. |
| `spot_controller/stance.py`, `leg_odometry.py`, `quaternion.py` | Contact detection, foot velocity to body velocity, rotation helpers. |
| `spot_controller/spot_4leg_ik.py` | `Spot4LegsIK`: body pose (position and roll/pitch/yaw) to 12 joint angles. |
| `spot_controller/spot_publisher_node.py` | Demo motion generator, publishes commands. |
| `spot_controller/locomotion_core.py`, `gait.py`, `force_distribution.py` | Torque-mode controller: stand / trot / push recovery (see [LOCOMOTION_CONTROL.md](LOCOMOTION_CONTROL.md)). |
| `spot_controller/mpc.py`, `mpc_planner.py`, `qp_solver.py` | Convex MPC for ground reaction forces and the pure-NumPy QP solver it uses. |
| `spot_controller/dynamics.py`, `wbc.py` | Floating-base dynamics model and the whole-body QP. |
| `spot_controller/state_estimator.py` | FK-based body state estimate (logs only for now). |
| `test/test_estimation.py` | pytest: complementary filter, EKF on a synthetic trajectory, Jacobian check. |
| `test/test_kinematics.py` | pytest: IK to FK round trip, reach, joint limits, attitude recovery. |

### Topics
| Topic | Type | Publisher | Subscriber |
|---|---|---|---|
| `/joint_states` | JointState | plugin | state_estimator |
| `/imu` | Imu (no orientation, simulated bias/noise) | plugin | ekf_estimator |
| `/ground_truth/pose`, `/ground_truth/twist` | PoseStamped, TwistStamped | plugin | ekf_estimator, state_estimator (scoring only) |
| `/odom_estimated` | Odometry | ekf_estimator | (controllers, later) |
| `/estimator/error` | Float64MultiArray | ekf_estimator | plots / logs |
| `/joint_commands` | JointState | spot_publisher | plugin |
| `/fk_state` | Vector3Stamped | state_estimator | (plot / EKF) |

---

## 3. Algorithms used

### 3.1 Leg inverse kinematics (`ik_solver.py`)
3-DOF leg: hip abduction (roll) theta1, hip pitch theta2, knee theta3.
1. theta1 = atan2(y, -z) - asin(dy / r), with r = sqrt(y^2 + z^2). The lateral hip offset dy is removed.
2. Planar 2-link problem in the leg plane: z' = -sqrt(r^2 - dy^2), d^2 = x^2 + z'^2.
3. Law of cosines: cos(theta3) = (d^2 - l1^2 - l2^2) / (2 l1 l2).
4. theta2 = atan2(x, -z') - atan2(l2 sin(theta3), l1 + l2 cos(theta3)).

### 3.2 Body IK (`spot_4leg_ik.py`)
For a commanded body pose (position p, rotation R from roll/pitch/yaw):
foot position in body frame = R^T (foot_world - p); subtract the hip mount position; solve leg IK per leg.
Order of output: FL, FR, RL, RR, each (hip_x, hip_y, knee).

### 3.3 Forward kinematics (`state_estimator.py`)
Foot position relative to the hip from the three joint angles. From the four feet:
- height = -mean(z_foot)
- pitch = atan2(z_rear_avg - z_front_avg, body_length)
- roll = atan2(z_right_avg - z_left_avg, body_width)

Assumption: all four feet are on flat ground. This is what the IMU will replace.

### 3.4 Sensor fusion (done): see [STATE_ESTIMATION.md](STATE_ESTIMATION.md)
Complementary filter (roll/pitch), stance detection, leg odometry, and an error-state EKF (position, velocity, attitude, IMU biases). Theory, equations, results and how to read the topics are in that document.

### 3.5 MPC and whole-body control (done): see [LOCOMOTION_CONTROL.md](LOCOMOTION_CONTROL.md)

### 3.6 Stereo SLAM, 3D mapping, object tracking, navigation (done): see [STEREO_SLAM.md](STEREO_SLAM.md)
Stereo depth, ORB-based SLAM with loop closure, a 3D voxel map, car/person detection and tracking with dynamic
masking, Nav2 and frontier exploration, all on the CPU. Launch: `ros2 launch spot_controller spot_slam_launch.py`.
The MPC and the whole-body QP below are implemented; the section that follows is the original plan.

### 3.5b Original MPC plan
Single-rigid-body model with ground reaction forces at the feet (friction cone constraints), solved as a QP (OSQP / qpOASES / CasADi) at 30-100 Hz. Output forces f = J^T tau mapped to joint torques, or feet targets mapped through the IK above.

---

## 4. Webots facts (checked against the R2025a `Spot.proto`)

Device names in the Spot PROTO are **not** `front_left_hip_x` etc. They are:

| Joint | Motor name | Sensor name | Limits (rad) |
|---|---|---|---|
| shoulder abduction | `<leg> shoulder abduction motor` | `<leg> shoulder abduction sensor` | -0.6 .. 0.5 |
| shoulder rotation | `<leg> shoulder rotation motor` | `<leg> shoulder rotation sensor` | -1.7 .. 1.7 |
| elbow | `<leg> elbow motor` | `<leg> elbow sensor` | -0.45 .. 1.6 |

`<leg>` is one of: `front left`, `front right`, `rear left`, `rear right`. Max torque is 50 N m per joint.

Body sensors (default names): `accelerometer`, `gyro`, `inertial unit`, `gps`.
The proto's default controller is `spot_moving_demo`. For ROS 2 control it must be `<extern>` (done in `my_spot.wbt`).

Zero pose: thigh points down and 35.6 deg backward, shin down and 39.9 deg forward, so all-zero joints stand at about 0.58 m. Left abduction negative = foot moves outward (+y); right abduction positive = outward. Thigh length is about 0.358 m and calf about 0.392 m (measured from the PROTO).

Frames: Webots world is ENU in R2025a (x forward, y left, z up). An accelerometer at rest reads about +9.81 on z.

---

## 5. Issues found and how they were resolved (2026-10-26)

| # | Problem | Fix |
|---|---|---|
| 1 | Launch loaded `SpotMotorsPlugin` but the class was `WebotsPlugin` | Class renamed to `SpotMotorsPlugin` |
| 2 | Plugin used joint names that are not Webots devices, so no joint was found | Names mapped in `constants.py` (`front left shoulder abduction motor`, ...) |
| 3 | Publisher sent `/joint_command`, plugin listened on `/joint_commands` | Publisher now uses `/joint_commands` |
| 4 | Spot in `my_spot.wbt` had no `controller "<extern>"` | Added `name "Spot"` and `controller "<extern>"` |
| 5 | IMU message had no orientation and no covariance | Orientation from the `inertial unit`; covariances set; `rclpy.init` guarded |
| 6 | IK angles were not Webots angles (zero pose has a bent knee, abduction sign is opposite) | `ik_to_webots()` / `webots_to_ik()` in `constants.py` |
| 7 | Estimator and IK used different leg lengths and different formulas | Both use `constants.py` and the same FK |
| 8 | Silent clamp in IK for unreachable targets, demo height beyond reach | IK raises `ValueError`; demo uses 0.50 m +- 0.03 m, all joints inside limits (tested) |
| 9 | Unsafe: motor commands were not limited | Plugin clamps every command to the joint limits |
| 11 | IMU orientation was ground truth, no way to test an estimator | `/imu` has no orientation; ground truth on `/ground_truth/*`; stamps are simulation time |
| 10 | Launch XML string started with whitespace, deps missing in `package.xml` | Fixed |

### Still unverified (needs a real run in Webots)
- Leg lengths and the zero-pose offsets were read from the PROTO files (about +-1 cm). If the robot
  looks wrong, check `THIGH_ZERO_ANGLE`, `KNEE_ZERO_ANGLE` and the abduction sign in `constants.py`.
- The IMU frame: `inertial unit` roll/pitch should be about 0 while standing. `az` should be about +9.81.
- The FK roll/pitch sign relative to the IMU. The estimator log prints both: they should agree
  in sign and size when the demo is swaying. If one is mirrored, flip the sign in `body_state.py`.
- The stray file `~/ros2_ws/spot_publisher_node.py` (workspace root) is an old copy. It is not used.

## 6. Roadmap

- [x] Fix plugin, topics, world, joint names (section 5)
- [x] Shared `constants.py` (leg lengths, hip positions, joint map, limits)
- [x] Unit tests: IK then FK round trip (`pytest`), reach and limits
- [x] Confirm in Webots: 12 joints in `/joint_states`, FK vs inertial-unit roll/pitch agree
- [x] Complementary filter for roll/pitch (0.1-0.6 deg error in Webots)
- [x] Stance detection, leg odometry, error-state EKF (about 1 cm / 0.02 m/s while standing)
- [ ] Re-tune estimator while walking (stance detector, anchor noise); publish TF
- [ ] Stand and balance controller (PD in joint space), then body-pose control
- [ ] Gait generator (trot): swing trajectories, contact schedule
- [ ] MPC (single rigid body) plus force-to-torque mapping
- [ ] Closed-loop walking on flat ground, then stairs

---

## 7. Learning resources
- MIT Cheetah 3 convex MPC: Di Carlo et al., "Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control" (2018)
- State estimation: Bloesch et al., "State Estimation for Legged Robots: Consistent Fusion of Leg Kinematics and IMU" (2012)
- Quaternions and EKF basics: Joan Sola, "Quaternion kinematics for the error-state Kalman filter"
- webots_ros2 docs: https://cyberbotics.github.io/webots_ros2/
