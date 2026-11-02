# Stereo SLAM, 3D mapping, object tracking and navigation

Spot gets a stereo camera and builds a 3D map of the place it walks through, finds and tracks cars and
people, keeps the moving ones out of the map, and navigates in the map it built. Everything runs on the CPU.

```
 Webots (spot_slam.wbt)                       separate ROS 2 nodes, each one a stage you can watch
 +--------------------+
 | stereo_left/right  |--> stereo_depth ---------+--> stereo_slam --> TF map->odom, /slam/odom, keyframes
 | (512x288, 12 cm)   |    SGBM block matching   |    ORB + PnP + keyframes + loop closure + pose graph
 | plugin publishes   |                          |           |
 | with sim-time stamps                          +--> mapper <-+  3D voxel map, 2D /map, /mapper/scan
 +--------------------+                          |
        |                                        +--> detector --> tracker --> /tracks, dynamic masks
        |                                              (HOG or YOLO)   |             |
 leg-odometry EKF (existing) --> tf_bridge (odom->base_link, /odom)    |             +--> SLAM + mapper
        |                                                              |                  ignore those pixels
        +--> motion prior for SLAM, gravity (roll, pitch, height)      v
                                                       Nav2 (planner, controller) --> /cmd_vel --> gait
```

## Run it

Inside `distrobox enter ros2-jazzy`, after `source /opt/ros/jazzy/setup.bash` and
`source ~/ros2_ws/install/setup.bash`:

```bash
colcon build --packages-select spot_controller --symlink-install
ros2 launch spot_controller spot_slam_launch.py                    # sim + perception + dashboard window
ros2 launch spot_controller spot_slam_launch.py nav:=true          # + Nav2
ros2 launch spot_controller spot_slam_launch.py nav:=true explore:=true   # explores the whole place by itself
ros2 launch spot_controller spot_slam_launch.py rviz:=true         # + RViz (3D map, tracks, plan)
```

Then, in a second terminal: `ros2 run spot_controller teleop` (drive by hand), or with `nav:=true`
`ros2 run spot_controller goto X Y` (coordinates in the map frame: the origin is where Spot started, x forward).

Launch options: `controller:=vmc|mpc|mpc_wbc`, `detector:=auto|hog|yolo|oracle`, `viewer:=false`,
`save_view:=/tmp/view.png`, `mode:=fast`, `port:=1240`. The Webots window must stay open: with `gui:=false`
Webots does not render, so there would be no camera images.

The simulation runs at roughly 0.4x real time on a laptop (rendering and the 8 ms controller), so every
"fps" in the dashboard is per wall-clock second and about 2.5x lower than per simulated second.

### The dashboard

One window with the SLAM view (green = features used for the pose, red = rejected), the detections, the
stereo depth, a top-down map (white free, black occupied, green robot, red path, cyan Nav2 plan, coloured
dots for tracked objects with yellow velocity arrows) and a table with the rate, the milliseconds per frame and
the key numbers of every stage (SLAM error against ground truth, keyframes, loop closures, tracked objects...).

## The world and the sensors

`worlds/spot_slam.wbt`: a 20 x 20 m arena, two partition walls, crates, barrels, cones, two parked cars, one car
that drives around, two pedestrians that walk back and forth, one that stands. Spot carries two cameras 12 cm
apart at the front (rectified: parallel axes, 512 x 288, 80 degrees) and a hidden range finder at the left camera
that is used only to score the stereo depth.

* The walls use generated textures (`tools/make_textures.py`: posters with text, brick, graffiti, tiles, plus fine
  grain). Plain coloured walls give a visual tracker nothing to lock onto. Each wall is different, so places look
  different.
* The plugin publishes the images itself (`stereo_sensors.py`) with simulation-time stamps and publishes `/clock`.
  The Webots ROS driver would stamp them with wall-clock time and publish every camera of the robot, which
  slowed the simulation to a crawl.
* `scene.py` moves the car and publishes the ground-truth boxes of every car and person on
  `/ground_truth/objects`. Ground truth is used for scoring only (and by `detector:=oracle`, see below).

## Stages

### 1. Stereo depth (`perception/stereo_depth.py`)
For a rectified pair a point at depth `z` appears shifted by `d = f B / z` pixels between the images (`f` focal length in
pixels, `B` = 0.12 m baseline). OpenCV's semi-global matching finds `d` for every pixel (cost aggregated along
several image directions, left-right consistency check), and `z = f B / d`. Depth noise grows with `z^2`, so
anything beyond about 6 m is rough. Measured against the hidden range finder: see Results.

### 2. Visual odometry and SLAM (`perception/stereo_slam.py`)
RGB-D style: ORB features are detected in the left image and get their 3D position from the depth image.

1. **Prediction**: the previous pose moved by the leg-odometry EKF's motion since the last frame (robust to blur).
2. **Local map**: landmarks of the last 8 keyframes, stored in the coordinates of the keyframe that created them,
   are projected into the predicted view and matched to the new features (Hamming distance, ratio test, a
   pixel gate around the projection).
3. **Pose**: PnP inside RANSAC on the 2D-3D matches, then Levenberg-Marquardt on the inliers.
4. **Gravity**: roll, pitch and height are taken from the EKF (the IMU measures gravity to about 0.1 degrees and the
   robot walks on flat ground); only x, y and heading come from vision. Without this the camera's tilt drifted by
   several degrees and its height by metres, and the map filled up with fake obstacles.
5. **Fallbacks**: if the local map no longer covers the view (after a turn), track against the previous frame; if that
   fails too, follow the odometry and start a new keyframe linked by a weak edge.
6. **Keyframes** when the tracked landmarks drop below 60 % of the reference keyframe, or after 0.35 m, 0.3 rad or
   2.5 s. New landmarks come from the features that were not matched.
7. **Pose graph** (`pose_graph.py`): keyframes are nodes `(x, y, z, yaw)` (roll and pitch stay fixed), consecutive
   keyframes are joined by odometry edges `e = R_i^T (p_j - p_i) - t_ij`, `e_yaw = yaw_j - yaw_i - yaw_ij`. Gauss-Newton
   (Levenberg-Marquardt) with analytic Jacobians; node 0 is fixed.
8. **Loop closure**: an older keyframe is a candidate only if it is close to where the odometry says we are (allowed
   distance 2 m + 3 % of the path travelled since), and at least 8 m of path ago. Its descriptors are matched, a RANSAC
   rigid alignment of the matched 3D points must keep 30 or more inliers, and the pose is refined with PnP. The
   implied correction must be plausible (0.5 m + 3 % of the path). An accepted loop adds an edge (Huber-robust) and the
   graph is optimised: every keyframe, and with it every landmark, moves.

### 3. Mapping (`perception/mapper.py`, `voxel_map.py`)
A 3D grid of 10 cm voxels holds a log-odds value per voxel. Each depth point marks its voxel occupied (+0.85) and
every voxel along the ray from the camera free (-0.4); values are clamped to [-2, 2.4], so something that moves
away is cleared within about 4 frames. Every integrated frame is stored with its reference keyframe: after a loop
closure moves the keyframes, the whole map is rebuilt at the corrected poses. The 2D `/map` flattens the band
15 cm to 1.4 m above the floor. `/mapper/scan` turns the current depth image into a pseudo laser scan that both
marks obstacles and clears free floor, for the Nav2 costmaps.

### 4. Detection and tracking (`detector.py`, `tracker.py`)
* **Backends**: `hog` (OpenCV's people detector, needs nothing), `yolo` (YOLOv4-tiny through OpenCV's DNN module:
  people, cars, trucks; needs the model files in `~/.spot_models/`), `oracle` (simulation ground truth projected into
  the image: a cheat to test the stages after it, never chosen automatically). `auto` = YOLO if installed, else HOG.
* **Tracker**: the median depth inside the detection's centre gives a 3D position in the map; a constant-velocity
  Kalman filter per object, nearest-neighbour association within 1.6 m. A confirmed track faster than 0.35 m/s is
  `moving`. Static tracks (parked cars) are remembered for 2 minutes: that is the semantic map.
* **Dynamic masking**: people and moving things are published as image boxes on `/perception/dynamic_boxes`; SLAM
  ignores features inside them and the mapper ignores those depth pixels. Parked cars stay in the static map.

### 5. Navigation (`config/nav2_spot.yaml`, `explorer_node.py`)
Nav2's planner, controller (regulated pure pursuit, 0.3 m/s), behaviours and velocity smoother run on the live `/map`
and TF (`map -> odom` from SLAM, `odom -> base_link` from the EKF). The global costmap is the map plus the live scan;
the local costmap is a 6 m rolling window fed by the scan, so people are avoided as they move. `explorer_node` picks
frontiers (free cells next to unknown space), sends them to Nav2 and repeats until none is left.

## Tools
* `tools/record_dataset.py` records stereo frames, depth, odometry and ground truth; `tools/replay_slam.py` replays them
  through SLAM (and `--map`) without the simulator, scores the trajectory against ground truth and draws it.
  `--mask oracle` uses ground-truth boxes to measure what dynamic masking is worth, `--param key=value` overrides a SLAM
  parameter. `tools/waypoint_drive.py` walks a repeatable loop (it uses ground truth, only for recording).
* Tests (no ROS or Webots needed): `python3 -m pytest test/test_slam_math.py test/test_voxel_map.py test/test_tracker.py`.

## Results (simulation, all stages running together on the laptop)

| Test | Result |
|---|---|
| Stereo depth against the hidden range finder (live) | 98.8 % of pixels valid, mean error 6.7-8.4 cm, median relative error 1.1-1.3 %, 1.5-2.2 % clearly wrong pixels |
| Visual odometry on a straight 13.5 m walk (recorded) | 3.4 cm error, the leg-odometry EKF alone: 25 cm |
| One 31 m lap around the partition, whole stack live | SLAM error 0.4-0.7 cm at the end (noise-free simulated images), 117 keyframes |
| 14 minutes of autonomous exploration (Nav2 + frontier explorer, nobody driving) | walked 38.5 m, mapped about 317 of 400 m², 10 goals reached and none failed, never tilted more than 3.5 degrees, 225 keyframes, 7 loop closures, 6 map rebuilds, SLAM error 3 cm |
| Tests | 62 functional tests pass (`pytest -k 'not flake8 and not pep257 and not copyright'`) |

Rates in the dashboard (wall-clock, everything running): stereo depth 4 fps (54-85 ms), SLAM 3.5-4 fps (44-60 ms),
mapper 4 fps, detector 1.2-2 fps (HOG, every 3rd frame, 170 ms), tracker 1-2 fps. That is about 0.2-0.4x real time because Webots,
the 8 ms controller and the nodes share a 15 W laptop CPU (2 fast cores, 8 slow ones) with about 14 GB of memory.

## Problems found and how they were fixed

| Symptom | Cause | Fix |
|---|---|---|
| Camera stamps did not match the IMU, simulation at under half speed | the Webots ROS driver stamps with wall-clock time and publishes all 5 built-in 1080p cameras | the plugin publishes the stereo pair itself with simulation-time stamps and `/clock`; the other cameras are switched off |
| Tracker had nothing to lock onto, lost on close walls | plain coloured walls, then huge smooth polygons | generated textures with text, brick, tiles and fine grain (`tools/make_textures.py`) |
| Camera tilt error of 4.6 degrees on average (49 at the 95th percentile), height drifting from 0.13 to 2.08 m; the map filled with fake obstacles | vision alone cannot hold gravity | roll, pitch and height come from the EKF, only x, y and heading from vision |
| Error 2.7 m on the first lap | a pedestrian 1.5 m from the camera and a moving car dominated the image | dynamic masking: with ground-truth boxes the error fell to 0.26 m (see Limits: the real detectors are weaker) |
| Tracking lost for hundreds of frames after a turn | no keyframe is created while lost, so no map in the new direction | frame-to-frame fallback, then a new keyframe at the odometry pose with a weak edge |
| Error jumped from 7 cm to 0.6-1.9 m at the first loop closure | the wall textures repeat; on a flat wall all matches fit one rigid shift, so a false loop passes geometric verification | candidates must be where the odometry says we are (+3 % of the path), and the correction must be plausible |
| 19 % of the depth image invalid | OpenCV SGBM leaves the first `num_disp` columns empty | both images are padded on the left |
| Explorer idle forever, or hopping 1 m at a time | it skipped every frontier cluster whose nearest cell was under 0.8 m away (always true at the apex of the view) and had nothing to do before the first map existed | frontier cell at least 1 m away, and a 20 s turn on the spot first |
| A static car was masked because another car moved | the mask rule looked at classes, not at which track a detection belongs to | a detection is masked only if its own track is dynamic; missed dynamic tracks are projected into the image |

## Limits (read before trusting the numbers)

* **Simulation only.** The cameras are noise-free, there is no motion blur or lighting change, and the textures were designed to be
  trackable. Expect a worse trajectory on a real robot. The errors above are from one start and one world.
* **Detection is the weakest stage.** The built-in HOG detector finds people only, and on the recorded frames it found
  47 % of the people at least 50 px tall (88 % precision), and it misses people cut off by the image edge. It finds no cars.
  The YOLO path (cars and people) is written but has never been run, because the model files were not downloaded
  (`bash tools/get_yolo_model.sh` fetches them: about 23 MB from github.com/AlexeyAB/darknet). The benefit of masking was
  measured with ground-truth boxes (`--mask oracle`), not with a real detector.
* **Tracked object positions are rough**: a car's centre is estimated 1 m behind its visible surface; the tracker's position error against
  ground truth was 0.1-0.7 m. Moving-object velocity was not scored: the driving car was not in view during the live tests.
* **The map was only checked by eye**, not scored against the world's geometry. Nav2's avoidance of moving people was not tested.
* **Exploration is greedy**: it picks the nearest frontier and sometimes hops back and forth in one spot (the map grew little between
  minutes 4 and 9 of the run) before a long goal pays off.
* **Loop closure** was exercised by 7 accepted loops in the exploration run and by a recorded two-lap run; the gate numbers
  (2 m + 3 % of the path, 0.5 m + 3 %) were set from this one world.
* Stereo depth is rough beyond about 6 m (error grows with range squared), so landmarks and map entries far away are noisy.
* The packaged flake8/pep257/copyright tests fail for the whole package (line length etc.); they were failing before this work.
