# State Estimation: Theory and Implementation

What the robot needs to know: where the body is (position p), how fast it moves (velocity v),
and how it is oriented (attitude q), in the world frame. Nothing measures these directly.
We have an IMU (gyro + accelerometer) and joint encoders. This document explains how they are fused.

Conventions: world frame ENU (x forward at start, z up). Body frame x forward, y left, z up.
`R` rotates body to world. Hamilton quaternions (w, x, y, z). `[a]x` is the skew-symmetric matrix of a.
g = 9.81 m/s^2, gravity vector `g_w = (0, 0, -g)`.

---

## 1. Sensors and their weaknesses

| Sensor | Gives | Problem |
|---|---|---|
| Gyro | angular velocity w (body) | constant bias b_g plus noise. Integrating it gives attitude that **drifts**. |
| Accelerometer | specific force a = R^T (a_world - g_w) | bias b_a plus noise. At rest it reads +g on z. Double integration for position **drifts fast**. It also senses body acceleration, so it is a gravity reference only when the robot is not accelerating. |
| Joint encoders | joint angles q, rates qdot | exact in simulation. Through the kinematics they give foot positions in the body frame. |
| Feet on the ground | a foot in contact does not move in the world | needs contact detection. |

No sensor alone is enough. The IMU is fast but drifts; the legs have no drift but only work when feet are planted. Fusion uses each where it is strong.

---

## 2. Step 1: Complementary filter (roll and pitch)
File: `attitude_filter.py` (Mahony filter).

Idea: the gyro is good for fast changes, the accelerometer's gravity direction is good for the long term. Split by frequency: trust the gyro at high frequency and the accelerometer at low frequency.

Algorithm each IMU sample:
1. Predicted "up" direction in the body frame: `u_pred = R^T e_z`.
2. Measured "up": `u_meas = a / |a|`. Only used if `| |a| - g | < 0.8` (the robot is not accelerating hard).
3. Error `e = u_meas x u_pred` (a rotation axis that tilts the prediction toward the measurement).
4. Corrected rate: `w_c = w_gyro - b + kp * e`. Bias: `b <- b - ki * e * dt`.
5. Integrate the quaternion: `q <- q (x) exp(w_c * dt)`.

Gravity says nothing about rotation around the vertical axis, so **yaw is not observable** and only follows the gyro.
Result in Webots: roll/pitch error about 0.1-0.5 degrees against the inertial unit.

---

## 3. Step 2: Stance (contact) detection
File: `stance.py`.

Foot positions in the body frame `r_i` come from forward kinematics (`spot_4leg_ik.py`).
Rotating them by the attitude gives their heights in a gravity-aligned frame: `z_i = (R r_i)_z`.
On flat ground, a foot that is more than 3 cm higher than the lowest foot is in swing; the others are in stance.
The attitude used here comes from the complementary filter, not from the EKF, so the contact decision does not depend on the corrections it triggers.
Limitation: it assumes flat ground. A stair or a gait scheduler should replace or confirm it later (the commanded gait phase is a better contact signal).

---

## 4. Step 3: Leg odometry
File: `leg_odometry.py`.

A stance foot is stationary in the world: `f_i = p + R r_i = const`. Differentiate:

```
v + R ( rdot_i + w x r_i ) = 0     =>     v_body = R^T v = -( rdot_i + w x r_i )
```
- `r_i`: foot position in the body frame (forward kinematics).
- `rdot_i = J(q) qdot`: foot velocity in the body frame, from the joint rates. Computed by a central difference of the forward kinematics along `qdot`.
- `w`: body angular velocity, the gyro minus the estimated bias.

So every stance foot gives a measurement of the body velocity. Four feet give four measurements, which averages the noise.

---

## 5. Step 4: Error-state Extended Kalman Filter
Files: `ekf.py` (filter), `ekf_node.py` (ROS node).

### 5.1 Why an EKF, and why the error-state form
A Kalman filter is the optimal linear estimator when you have a model (how the state evolves) and measurements (noisy functions of the state), each with known noise. The problem is **nonlinear** (rotations, the gravity coupling), so we linearise around the current estimate: that is the *extended* Kalman filter.

Attitude is a rotation, which lives on a curved manifold (unit quaternion, 4 numbers for 3 degrees of freedom). The standard solution is the **error-state** (indirect) form:
- Keep a *nominal state* (p, v, q, b_a, b_g), propagated with the full nonlinear equations.
- Estimate a small *error state* `dx = [dp, dv, dtheta, db_a, db_g]` (15 values) with the Kalman filter. Its dynamics are well approximated by a linear model because the error is small.
- After each correction, inject the error into the nominal state and reset it to zero. The attitude error is applied as `q <- q (x) exp(dtheta)`.

It avoids quaternion constraint problems and keeps the covariance 15x15 and well-behaved.

### 5.2 Prediction (every IMU sample)
With `a = a_meas - b_a` and `w = w_meas - b_g`:
```
a_world = R a + g_w
p <- p + v dt + 1/2 a_world dt^2
v <- v + a_world dt
q <- q (x) exp(w dt)
```
Error dynamics (continuous), `dx' = F dx + noise`:
```
dp'     = dv
dv'     = -R [a]x dtheta - R db_a
dtheta' = -[w]x dtheta - db_g
db_a' = 0,  db_g' = 0   (random walk)
```
Discretised: `F_d = I + F dt + 1/2 F^2 dt^2`, `P <- F_d P F_d^T + Q`.
`Q` holds the IMU noise (accel noise integrates into velocity, gyro noise into attitude) and the bias random-walk.

### 5.3 Corrections
General update: `S = H P H^T + R_m`, `K = P H^T S^-1`, `dx = K y`, Joseph-form covariance update, then inject `dx`.
`y` is the innovation (measured minus predicted).

| Update | Measurement | Predicted | Jacobian H | What it fixes |
|---|---|---|---|---|
| Gravity (when not accelerating) | accel | `b_a + R^T (0,0,g)` | `[I (b_a), [R^T g]x (theta)]` | roll, pitch, accel bias |
| Leg velocity (each stance foot) | `-(rdot + w x r)` | `R^T v` | `[R^T (v), [R^T v]x (theta)]` | velocity, attitude, gyro bias |
| Foot anchor (each stance foot) | anchor position `f_i` | `p + R r_i` | `[I (p), -R [r_i]x (theta)]` | position, attitude, **yaw** |

Foot anchor: when a foot touches down, remember `f_i = p + R r_i`. While it stays in stance it cannot move, so any change in `p + R r_i` is estimation error. This stops position drift while standing and also observes yaw (feet fixed, so the body cannot have rotated about the vertical). The anchor is dropped when the foot lifts.

Noise values (`ErrorStateEKF` constructor): accel 0.05 m/s^2, gyro 0.003 rad/s, leg velocity 0.05 m/s, foot anchor 1 cm, gravity measurement 0.3 m/s^2. Tune these on real data.

### 5.4 What is and is not observable
- Roll, pitch, velocity, gyro bias: observable. Estimated well.
- Yaw and absolute position: only *relative* to the start. Without a compass or GPS they drift when the robot moves (the anchors only fix them while feet are planted). The filter is initialised with yaw and position from ground truth, as a real robot would from a known start pose.
- Accelerometer bias: only weakly observable (tilt and bias look alike). It stays inaccurate; this is normal for this filter structure.

---

## 6. Simulation setup
- Webots IMU is perfect, so the plugin adds a constant bias and Gaussian noise (`constants.py`: `IMU_NOISE_ENABLED`, `GYRO_BIAS`, `ACCEL_BIAS`, sigmas). True biases are known, so we can check that the EKF finds them.
- `/imu` carries no orientation (like real hardware).
- Ground truth (evaluation only): `/ground_truth/pose` (GPS position + inertial-unit attitude), `/ground_truth/twist` (GPS velocity). It sets the initial yaw/position and scores the filter. It is never fed into the filters afterwards.
- All stamps are Webots simulation time, so results do not depend on how fast Webots runs.

---

## 7. Results

Webots, 4 feet planted, demo sway (position of the body varies +-2 cm, roll/pitch +-4 deg):

| Quantity | Result |
|---|---|
| Complementary filter roll/pitch error | 0.1-0.6 deg |
| EKF roll/pitch error | 0.1-0.8 deg |
| EKF yaw error | about 0.2 deg |
| Velocity error vs GPS | 0.01-0.03 m/s |
| Position error vs GPS | about 1 cm |
| Gyro bias estimate | (0.010, -0.008, 0.006) vs true (0.010, -0.008, 0.005) rad/s |
| Stance feet detected | 4 / 4 |

Offline tests (`test/test_estimation.py`, synthetic trajectory with fixed feet, IMU bias and noise): 30 s run, position error < 3 cm, velocity error < 0.05 m/s, attitude < 1.5 deg; IMU alone drifts to 18 cm position error in 20 s while legs + IMU stay at 0.6 cm.

**Not tested yet:** walking. With feet lifting, anchors are reset at each touchdown, so position will drift by a small amount per step. The stance detector and noise values need tuning once a gait exists.

## 8. How to run and read it
```bash
ros2 launch spot_controller spot_launch.py        # starts the EKF node too
ros2 topic echo /estimator/error                  # see fields below
ros2 topic echo /odom_estimated --once            # estimated pose + body velocity
```
`/estimator/error` (`Float64MultiArray`) fields, in order:
`cf_roll_err_deg, cf_pitch_err_deg, ekf_roll_err_deg, ekf_pitch_err_deg, ekf_yaw_err_deg, vel_err_mps, pos_err_m, n_stance_feet`.
Plot them with `ros2 run plotjuggler plotjuggler` or watch the one-line log of `ekf_estimator` once per second.

## 9. References
- J. Sola, "Quaternion kinematics for the error-state Kalman filter" (2017): the formulas for section 5.
- R. Mahony, T. Hamel, J.-M. Pflimlin, "Nonlinear complementary filters on the special orthogonal group" (2008): section 2.
- M. Bloesch et al., "State estimation for legged robots: consistent fusion of leg kinematics and IMU" (2012): foot-anchor idea in section 5.3.
- G. Bledt et al., "MIT Cheetah 3: design and control of a robust, dynamic quadruped robot" (2018): leg-odometry and Kalman estimator on a real quadruped.
