# Roadmap: robot having fun autonomously

Goal: the Go2 explores on its own, finds cones, sniffs them and pees on them, driven by an LLM over MCP.

Each phase ends with a test on the real robot. Phases are in order; each one needs the one before.

## Phase 1: Real-robot navigation (now)

Working: the L1 lidar (mount, yaw, range offset, leg filter), SLAM in a room, and Nav2 goals from the app.

- [ ] **Save a map.** Build a map of a room with SLAM, then save it from the robot's Docker:
      `ros2 run nav2_map_server map_saver_cli -f maps/<name>`. Not in the launcher or the app yet.
- [ ] **Big loop.** Walk one large loop (a corridor, or around a building) and check:
  - when the robot gets back to the start, the map closes without a doubled wall;
  - how far the pose drifted before SLAM corrected it.
  If leg odometry drifts badly, compare it with Unitree's lidar odometry.
- [ ] **Keepout zones** on the saved map (`scripts/make_keepout_mask.py`).
- [ ] **Localize on the saved map:**
  1. AMCL with a known start pose.
  2. Unknown start pose plus the `localize` skill. Never confirmed, not even in Gazebo, so try Gazebo first.
- [ ] **Goals and obstacles:** several goals on the saved map from the app, then someone stepping into the path.
- [ ] **Jetson load:** run `top` with the policy, SLAM or AMCL, and Nav2 all running.
- [ ] **Lidar checks:**
  - the 12 cm range offset, against walls to the side and behind at tape-measured distances;
  - narrowing the body box back from ±0.35 m toward ±0.25 m, now that the leg filter exists.

**Done when:** the robot reaches 10 goals in a row across the saved map with no help.

## Phase 2: Camera

- [ ] **Camera bridge.** `VideoClient` JPEG frames to `/camera/image_raw/compressed` plus `/camera/camera_info`. The app should only show a low-rate preview.
- [ ] **Calibration:** checkerboard with `camera_calibration`, saved to a yaml.
- [ ] **Camera mount check:** draw the lidar points on the image; they should land on the objects.

**Done when:** a lidar point on a cone lands on the cone in the image.

## Phase 3: Cone detection

- [ ] **Colour filter first:** orange pixels → blobs → lidar distance. About 1–2 ms per image on the CPU, with no training or GPU. Move on to YOLO only if it gets fooled by other orange things or bad lighting.
- [ ] **GPU in the robot's Docker**, needed for YOLO. See [GPU on the robot](#gpu-on-the-robot-ros-with-cuda-torch) below.
- [ ] **Dataset:**
  - Gazebo images, labelled automatically from the known cone poses;
  - 100–200 real photos, labelled by hand.
- [ ] **Model:** a small YOLO, exported to TensorRT on the Jetson. Measure FPS.
- [ ] **`cone_detector` node:**
  - the box gives a direction from the camera, and the lidar gives the distance;
  - the cone becomes a point on the map, and repeated sightings are merged;
  - the list is published on `/cones`.
- [ ] **Show the cones** in rviz and in the app's map tab.

**Done when:** every cone in a room appears on the map within about 20 cm.

## Phase 4: Skills

- [ ] **`go_to_cone`:** a Nav2 goal about 0.5 m from the cone, facing it.
- [ ] **`pee`:** lift a back leg and hold it. Balance it in sim first.
- [ ] **`wiggle`.**
- [ ] **`explore`:** frontier exploration, so the robot finds cones in an unknown place.
- [ ] **`follow_me`:** track the person's legs in the lidar beside the robot, published on `/cmd_vel/skill`.

All skills go through `/skill`.

**Done when:** go to cone → sniff → pee works on the real robot.

## Phase 5: LLM brain over MCP

- [ ] **MCP server** with these tools:
  - `get_state`, `list_cones`, `go_to(x, y)`, `skill(name)`, `explore`, `stop`;
  - `describe_surroundings()`: the map condensed to a few lines of text, see [3D mapping](#3d-mapping-lidar-inertial-odometry).
- [ ] **LLM on the laptop.** Calls cross Wi-Fi over TCP or HTTP, not DDS.
- [ ] **Safety:**
  - the LLM only reaches skills and `/cmd_vel/llm`, the lowest priority;
  - the app's e-stop and the joystick always win.
- [ ] **A "having fun" prompt:** explore, pick a cone, go sniff it, sometimes pee, then repeat.

**Done when:** 10 minutes of finding, sniffing and peeing on cones with no human input.

## 3D mapping: lidar-inertial odometry

A parallel track, best started after Phase 1, or sooner if the big loop shows the leg odometry drifting.
A LIO method combines the L1 with an IMU into accurate odometry and a 3D point-cloud map, and five things
build on it.

**Options:**
- **Point-LIO:** Unitree's own `point_lio_unilidar` (https://github.com/unitreerobotics/point_lio_unilidar),
  tuned for the L1: lidar-IMU offset, noise and point-time format. It's ROS 1 (Noetic) and reads the
  standalone L1 driver (`unilidar_sdk`), not the Go2's internal cloud over DDS. Use a ROS 2 port with
  Unitree's L1 config values, fed from `real_sensors`.
- **FAST-LIO2:** official ROS 2 branch, about one CPU core. Fallback if Point-LIO won't port.
- **nvblox:** NVIDIA's GPU mapper, part of Isaac ROS: a 3D map, mesh and distance field from lidar or depth,
  with a Nav2 costmap plugin. It only maps; the robot's position must come from elsewhere, like Point-LIO.
  Needs the [GPU Docker](#gpu-on-the-robot-ros-with-cuda-torch).
- **LIO-SAM:** adds loop closure; heavier to tune.
- **KISS-ICP:** lidar only, simplest, drifts more.
- **RTAB-Map:** full 3D SLAM, can also use the camera.

Start with Point-LIO for the robot's position and keep `slam_toolbox` for Nav2's 2D map. Then add nvblox on
top of Point-LIO's position, once the GPU Docker works, for the 3D map and low obstacles.

**Setup:** `third_party/point_lio_ros2` (dfloreaa's ROS 2 port of Unitree's `point_lio_unilidar`, pinned at a8e2d0d),
linked as `src/point_lio`. `real_sensors` bridges the L1 IMU to `/lidar/imu`, and `lio.launch.py` (launcher [L]) starts it
with `config/point_lio_go2.yaml`, publishing `/lio/odom`, `/lio/cloud`, `/lio/map` in `lio_odom → lio_imu`.
`lio_map_stream` turns `/lio/cloud` into 10 cm voxels for the app: `/lio/map_voxels` (snapshot, every 10 s)
and `/lio/map_voxels/delta` (new voxels, every 1 s); about 1 kB/s on the test bag. The floor (within
`--ground_band` 10 cm of the floor under the robot) is a height map, one voxel per column: the L1's floor is
2-8 cm thick (thicker at grazing range), which stacked it 2-3 voxels deep. Floor-only columns with one voxel:
73% → 92% on the walking bag, 97% with a 15 cm band (but then lower obstacles merge into the floor).
Known port bug: `standard_pcl_cbk` keeps only whole seconds of `last_timestamp_lidar` (Unitree's uses `toSec()`).

**Prerequisites:**
- [x] **Matching timestamps.** `real_sensors` keeps the robot's stamps on cloud and IMU, shifted by one offset.
- [x] **IMU rate:** 250 Hz, `imu_time_inte: 0.004`.
- [x] **IMU rotation.** `ros2 run quadruped_perception lidar_imu_calib` fits the lidar IMU's gyro to the body
      IMU's while the robot turns and walks, and prints `extrinsic_R`. The first try (mount rotation without
      tilt) made forward walking drift sideways and diverge. Measured: the gyro is in the cloud's axes
      (identity, 2 deg).
- [x] **Accelerometer.** On this (newer) firmware the L1 IMU's acceleration isn't a measurement: z stays ~9.8
      and y ~0 in any pose while x climbs ~0.6 m/s^2 per second, and its orientation turned 28 deg for a 90 deg
      tilt. Same pattern reported in autonomy_stack_go2 issue #27 (old firmware: usable, in the cloud's axes). It
      turned Point-LIO's map upside down and made it drift standing still. `/lidar/imu` is the L1's gyro, turned
      into the body's axes, with the body IMU's accelerometer (`rt/lowstate`, newest sample at 50 Hz), so its
      extrinsic is the lidar mount and `lio_odom` is level with x forward. Gyro-only (no gravity, as CMU's Go2
      stack runs) worked too, but its world was the body's lean at start (~1-2 deg), tilting the floor.
- [x] **Start spin.** With `init_map_size: 10` the first map was one scan, ~90% floor within 1 m: yaw was
      barely held, the first match was ~14 deg off, and while the lidar pulled it back the filter learned a
      0.3 rad/s gyro bias, so the map kept turning (2 of 3 starts on 2026-10-08). `init_map_size: 3000`
      builds it from ~0.3 s of scans: all three starts hold within 1.5 deg. Start with the robot still.
- [ ] **Per-point times.** Check that the cloud's `time` field holds each point's time within the scan;
      it's what undoes the smear while walking.
- [ ] **Raw scans.** Unitree's clouds carry about 59k points/s against the L1's 21.6k, so they overlap or are
      preprocessed. LIO may need the raw scans instead.
- [ ] **Same data as Unitree's driver.** Point-LIO's L1 config expects the cloud and IMU as `unilidar_sdk`
      publishes them. Compare the Go2's DDS cloud with that: frame, point-time units, IMU rate.
- [ ] **Start with the real driver, not Nav2.** LIO supplies `odom → base` (an `odometry.source: lio` switch in
      `config.yaml`, with telemetry then dropping its own), so the pose never jumps and the app's view works
      without [N]. An adapter turns LIO's `camera_init → body` into `odom → base`, and leg odometry fills
      `/odom`'s velocity if LIO leaves it empty. Gazebo keeps leg odometry, since its lidar has no point times.

**What builds on it:**
1. **Odometry for Nav2: fuse, don't replace.** A filter blends legs + IMU with Point-LIO when it runs, and
   carries on with legs + IMU when it doesn't:
   ```
   map --(SLAM)--> odom --(filter: legs + IMU, + Point-LIO when present)--> base
   ```
   - **Point-LIO goes in as motion**, the change between consecutive `/lio/odom` poses, not as a position:
     `lio_odom` starts elsewhere than `odom` and resets to zero when Point-LIO restarts. It mostly fixes the
     legs' yaw drift. The output stays smooth, with no jumps for Nav2's controller.
   - **SLAM stays on top**, correcting `map → odom` as now. It reads `odom → base` to match scans, so feeding
     its answer back into the filter would loop, and its jumps would reach the controller.
   - **Build:** a `robot_localization` EKF node for navigation only (legs, IMU, Point-LIO as relative motion
     in; `odom → base` out). The LKF stays as is for the policy's velocity, so walking is untouched and
     turning the filter off gives today's behaviour. Check that `robot_localization` is in the robot's Docker.
   - **Only one publisher of `odom → base`:** turn the telemetry's TF off while the filter runs.
   - **A fixed `base → lio_imu` offset**, measured from where the IMU sits. Without it, `/lio/odom`'s motion
     is the IMU's, not the base's, and the app draws the robot floating.
   - Then the app's Map and 3D tabs agree on where the robot is.
2. **Low obstacles.** A 3D obstacle layer for what the `/scan` slice misses: a low box, a table edge, a step.
3. **Terrain-aware policies.** An elevation map, a grid of ground heights around the robot built from the
   3D map, becomes the policy's height-scan input, so it can see steps and slopes before touching them.
   - **Training:** the policy needs a height-scan observation, a grid of rays like Isaac Lab's
     `RayCasterCfg` (already imported in Walk's env cfg). Use the same grid size and spacing the robot
     will produce, with noise and missing cells added in training, since the L1 can't see under the body.
   - **Robot:** build the grid around the robot from the LIO map, and sample it exactly like the training
     rays. `elevation_mapping_cupy` (ETH, GPU) or a simple CPU binning of the map points near the robot.
   - Watch the timing: the grid has to keep up with the 20 ms policy step, or lag shows up as stumbles.
4. **LLM context.** The LLM can't read point clouds, but the robot can condense the map into text through a
   `describe_surroundings()` tool:
   ```
   open area 4 x 3 m ahead; wall 1.2 m left; low box (0.2 m high) 1 m front-right;
   cone #2 at 3.1 m, 40 deg left, not sniffed yet; unexplored doorway at (5.0, -2.0)
   ```
   - The 3D map adds heights the 2D one can't: a box vs a table vs a wall, what passes under a table, steps.
   - Occasionally, a camera frame or a top-down map render as an image, for bigger decisions.
   - Research angle: a small scene graph (objects, positions, how they relate) as the robot's memory,
     which the LLM queries and updates. Related work: ConceptGraphs, Hydra.
5. **3D map in the app's 3D tab.** The point cloud drawn around the robot model.
   - **Robot:** a small node keeps one point per 5–10 cm cube of the LIO map and publishes it about
     once a second on `/map_cloud`. The raw map and scans are too heavy for Wi-Fi next to the heartbeat.
   - **App:** read `PointCloud2` in the native DDS code, best-effort and only while the tab is open. Draw
     the points colored by height, placed with the TF from the LIO frame (`camera_init`) to the base.

## GPU on the robot: ROS with CUDA torch

The Go2's Jetson Orin Nano (40 TOPS) has a GPU with 1024 CUDA cores and 32 Tensor cores. Our Docker can't
use it:
- the image is based on `ros:humble-ros-base`, which has no CUDA;
- pip's `torch` for ARM is CPU-only;
- the container starts without NVIDIA's runtime.

The laptop's image stays as it is; only the robot's ARM image changes.

1. **Find the robot's JetPack version.** On the robot itself, not in Docker:
   `cat /etc/nv_tegra_release`. For example, `R35` means JetPack 5 on Ubuntu 20.04. The container's L4T major
   version must match the robot's, or CUDA won't load.
2. **Pick a base image for that version.** dusty-nv's jetson-containers images already bundle ROS Humble
   with CUDA torch, e.g. `dustynv/ros:humble-pytorch-l4t-r35.x.x`. Pick the tag closest to the robot's L4T.
   The alternative is NVIDIA's `nvcr.io/nvidia/l4t-jetpack:r35.x.x` with ROS installed on top; it takes
   longer to build.
3. **Make the Dockerfile's base an argument:** `ARG BASE=ros:humble-ros-base` and `FROM ${BASE}`.
   - The ARM build passes `--build-arg BASE=dustynv/ros:...`.
   - Our own layers then go on top: CycloneDDS, the Unitree SDK, Nav2, `twist_mux`.
   - Don't let pip install `torch` over the CUDA one. Skip it in the ARM build, or pin to the one already
     in the image.
   - Some apt packages may need building from source on that base, because dusty-nv builds ROS from source.
4. **Run with the GPU.** Add `--runtime nvidia` to the robot's `quaddocker` alias. JetPack already ships
   `nvidia-container-toolkit`.
5. **Check:** `python3 -c "import torch; print(torch.cuda.is_available())"` prints `True` inside the
   container.
6. **YOLO on TensorRT.** Train the model on the laptop, then build the engine on the Jetson itself, since
   engines don't move between GPUs: `yolo export model=cones.pt format=engine half=True imgsz=416`.
   Expect about 5–10 ms per image.
7. **Leave the policy on the CPU.** It runs in about 6 ms there, and the GPU would only add launch latency.
   The GPU is for perception.

Things to watch:
- **Size and build time.** The image grows to roughly 10+ GB and takes hours to build. Build it once on the
  robot and push it to GHCR (`:arm64`).
- **Memory.** The GPU shares the 8 GB of RAM with the CPU, so watch it with a usage monitor (backlog).

## Backlog

- **CPU, GPU, RAM and temperature in the app.** A `system_monitor` node reads `/proc/stat`, `/proc/meminfo`,
  the GPU load file in `/sys` (find it with `find /sys/devices -name load | grep -i ga10b`) and
  `/sys/class/thermal`. It publishes JSON on `/system_stats` at 1 Hz, and the app's Robot tab shows it.

- **SSH terminal in the app: built, not yet tried on the robot.** Shell tab, see `Android/README.md`.
  Still to check: `sshd` and the login user on the Go2's Jetson (the app defaults to `unitree@10.42.0.1`).
  The terminal can press ENTER on the driver, which releases an e-stop from the phone; not blocked.
- **Save Map** button in the launcher or the app.
- **A custom rviz config.**
- **Flags to ROS parameters.**
