# Locomotion control: force controller, MPC, whole-body control

Everything runs inside the Webots plugin, in the same 8 ms simulation step as the sensors (no
communication delay). Three interchangeable controllers share the estimator (EKF), the trot gait,
the swing-leg logic and the joint loop; they differ in **how the stance-foot forces are chosen**.

| `controller:=` | Stance forces | Joint torques |
|---|---|---|
| `vmc` (default) | Virtual model control: spring-damper body wrench, split over the feet by a friction-limited least-squares problem | `tau = -J^T R^T f` |
| `mpc` | Convex MPC over 0.3 s using the contact schedule (ground reaction forces, friction pyramids) | `tau = -J^T R^T f` |
| `mpc_wbc` | Same MPC forces | Whole-body QP on a floating-base dynamics model: tracks the MPC forces and the swing-foot motion, respects contact, friction and torque limits, outputs inverse-dynamics torques |

## Run it

Inside `distrobox enter ros2-jazzy` after `source /opt/ros/jazzy/setup.bash` and `source ~/ros2_ws/install/setup.bash`:

```bash
colcon build --packages-select spot_controller --symlink-install
ros2 launch spot_controller spot_launch.py control:=torque controller:=mpc
ros2 launch spot_controller spot_launch.py control:=torque controller:=mpc_wbc
ros2 launch spot_controller spot_launch.py control:=torque controller:=vmc
```

Other launch arguments: `gui:=false` (no window), `mode:=fast` (not real-time), `port:=1240` (Webots port, so a second
simulation can run next to the first; also use another `ROS_DOMAIN_ID`), `world:=spot_flat.wbt` (light test world),
`height:=0.5`. Drive it with `ros2 run spot_controller teleop` or publish `/cmd_vel`; push it with
`ros2 run spot_controller push_test --vx 0.3 --wz 0.3 --force 100 --direction 0 1 --settle 4`.

Parameters can be overridden without editing code, for example
`SPOT_STANCE_PARAMS='{"com_mode":"model","mpc_alpha":1e-3}' ros2 launch ...` (any key of `DEFAULTS` in
`locomotion_core.py`).

Tests (no ROS or Webots needed): `python3 -m pytest test/test_qp_solver.py test/test_mpc.py test/test_wbc.py test/test_controller_modes.py`.

## Files

| File | Purpose |
|---|---|
| `qp_solver.py` | Dense OSQP-style ADMM QP solver in pure NumPy (adaptive step size, warm start). No external solver is installed here. |
| `mpc.py` | Convex MPC: single rigid body, 13 states, condensed QP, friction pyramids, swing feet fixed to zero force. |
| `mpc_planner.py` | Builds the MPC inputs: reference (velocity/yaw/height or position hold), contact schedule from the gait phases, foot positions (planted feet, predicted touchdowns). |
| `dynamics.py` | Floating-base model: mass matrix `M(q)`, Coriolis, gravity, foot Jacobians (closed form). |
| `wbc.py` | Whole-body QP. |
| `locomotion_core.py` | Selects the controller, runs the gait, swing legs, force-to-torque mapping, WBC call, fallbacks. |
| `gait.py` | Trot schedule, swing arch (position, velocity, acceleration), Raibert foot placement. |

## MPC

State `x = [roll, pitch, yaw, p, omega, v, g]` (13), input one force per foot (world frame):

```
d(Theta)/dt = Rz(yaw)^T omega      dp/dt = v
d(omega)/dt = I_w^-1 sum_i r_i x f_i      dv/dt = (1/m) sum_i f_i + g
```
`r_i` is the foot position relative to the centre of mass, taken along the reference trajectory; the contact
schedule says which feet can push at each of the N = 10 steps of dt = 30 ms. Cost: weighted state error plus a
small force penalty; constraints: `|f_x|, |f_y| <= mu f_z` (mu = 0.6), `0 <= f_z <= 220 N`, swing feet exactly zero.
The state error weights are `[60 60 20, 1 1 150, 0.5 0.5 0.8, 8 8 20, 0]` (attitude, position, angular rate, velocity).
The QP is condensed (states eliminated), swing-foot variables are removed, forces are scaled by 100 N, and it is solved
with the ADMM solver in about 10-20 iterations, re-solved every simulation step (a 32 ms hold made it unstable: see below).
An integral term on the velocity reference removes the steady speed shortfall; the centre of mass comes from the mass
model (`com_mode: "model"`), not a guess.

## Whole-body controller

Generalised velocity `u = [v, omega, qdot]` (18), `M(q) du/dt + h(q,u) = S^T tau + Jc^T F`. QP variables: `du/dt` and
the stance forces `F`. Hard: floating-base dynamics, planted feet have zero acceleration, friction cones (mu = 0.75, a bit
wider than the MPC's so the MPC forces are always feasible), torque limit 50 N m. Soft tasks: track the MPC forces
(high weight), swing-foot acceleration `a_des + Kp (x_des - x) + Kd (v_des - v)` (high), base attitude and height
accelerations (low), small regularisation. Torque `tau = [M du/dt + h - Jc^T F]_joints` goes to the joint loop as
feed-forward on top of a joint PD. Feet with almost no planned load (< 15 N: about to lift off or just landed) are
left free, otherwise the near-zero friction cone makes the QP infeasible. If a solve ever fails, that step uses the plain
`-J^T f` torques (fallback counters are logged every 5 s).

Mass model from the Webots PROTO files: body 14.41 kg with inertia (0.11, 0.90, 0.91) kg m^2; per leg a 2.13 kg
shoulder block, a 0.94 kg thigh and a 0.14 kg calf as point masses (link rotational inertia neglected). Total 27.2 kg,
centre of mass in standing pose (0.006, 0, -0.004) m, composite inertia about it (0.36, 2.5, 2.6) kg m^2.

## Problems found and fixed

| Symptom | Cause | Fix |
|---|---|---|
| MPC robot fell within 0.2 s of torque control starting (growing 14 Hz roll oscillation) | MPC solved at 31 Hz and held for 4 steps: 32 ms of delay in a loop as stiff as the force controller | Re-solve every simulation step |
| MPC solve took 260 ms | NumPy multi-threading on tiny matrices (CPU oversubscribed); QP not converging in 300 iterations (poor scaling, swing variables included) | Single-threaded BLAS, scaled forces, removed swing variables, adaptive ADMM step: 10 ms, 20 iterations |
| MPC walked 8-10 % slower than commanded | No integral action | Integral on the velocity reference |
| Large pushes were not caught | Recovery steps limited to 0.10-0.12 m, but the capture point of a 0.75 m/s shove is about 0.17 m ahead | Step limits 0.18 m (inward limit unchanged) |
| Whole-body QP fell back on 3 % of steps | Near-zero cone of feet at lift-off / touchdown | Leave unloaded feet free; mu_wbc = 0.75; accept near-converged solutions |
| Webots killed by memory pressure when a second simulation ran | 15 GB RAM shared with an interactive simulation | Light `spot_flat.wbt`, separate port and ROS domain, headless |

## Results (Webots, flat test world, 8 ms step; one run per cell)

Pushes are 0.2 s forces at the body centre (100 N is about 0.7 m/s of velocity change), directions rotating through the world axes.

| Test | `vmc` | `mpc` | `mpc_wbc` |
|---|---|---|---|
| Walk on a circle at 0.30 m/s: mean speed / max tilt | 0.32 m/s / 1.5 deg | 0.32 m/s / 1.6 deg | 0.33 m/s / 2.3 deg (full world, 0 fallbacks) |
| Standing, peak tilt after 100 / 120 / 140 / 160 / 180 / 200 N | 2.4 / 5.9 / 8.9 / 8.2 / 8.8 / 16.1 | 1.5 / 4.5 / 4.1 / 6.9 / 4.2 / 15.2 | 3.1 / 8.0 / 4.3 / 8.6 / 6.4 / 16.7 |
| Walking, peak tilt after 100 / 140 / 160 / 180 / 200 / 220 / 240 N | 5.0 / 4.8 / 8.6 / 10.4 / 6.9 / 26.7 / 4.7 | 4.7 / 4.1 / 7.4 / 10.0 / 5.5 / 25.8 / 2.5 | 5.4 / 6.7 / 5.0 / 15.4 / 10.5 / 21.2 / 5.8 |
| Standing: largest push survived | 200 N (fell at 240) | 240 N (fell at 280) | 200 N (fell at 240) |
| Walking: largest push survived | 220 N; 240 N depends on direction | 220 N; 240 N depends on direction | 280 N (fell at 320) |
| Loop cost per step (loaded laptop) | 12-15 ms | 25-30 ms | 35-65 ms |

Read this with care: one run per condition, and the push direction changes the outcome as much as the controller does.
The MPC is gentler on standing pushes; the whole-body variant showed no consistent gain. Summary report:
[MPC_WBC_Report.pdf](MPC_WBC_Report.pdf).
