# Roadmap: robot having fun autonomously

Goal: the Go2 explores on its own, finds cones, sniffs them and pees on them, driven by an LLM over MCP.

Each phase ends with a test on the real robot. Phases are in order; each one needs the one before.
Each lists what is left **to build** (code) and what is left **to test on the robot**.

## Phase 1: Real-robot navigation (now)

Working: the L1 lidar (mount, yaw, range offset, leg filter), SLAM in a room, and Nav2 goals from the app.

**To build:**
- [x] **Save a map.** "Save map" on the app's map tab while SLAM builds one ([N], "Build a new map"): `map_save`
      writes `maps/<name>.pgm/.yaml` on the robot (map_saver's format, checked byte for byte against
      `maps/nav.pgm`), and [N] lists it next time. Without the app:
      `ros2 topic pub --once /map/save std_msgs/msg/String "data: room1"`.
- [x] **Keepout zones:** `scripts/make_keepout_mask.py` on a saved map.

**To test on the robot:**
- [ ] **Save a map** of a room from the app, and find it in [N].
- [ ] **Big loop.** Walk one large loop (a corridor, or around a building) and check:
  - when the robot gets back to the start, the map closes without a doubled wall;
  - how far the pose drifted before SLAM corrected it.
  If leg odometry drifts badly, compare it with Point-LIO's (`/lio/odom`).
- [ ] **Keepout zones** on the saved map: Nav2 plans around them.
- [ ] **Localize on the saved map:**
  1. AMCL with a known start pose.
  2. Unknown start pose plus the `localize` skill. Never confirmed, not even in Gazebo, so try Gazebo first.
- [ ] **Goals and obstacles:** several goals on the saved map from the app, then someone stepping into the path.
- [ ] **Jetson load:** the app's CPU/GPU/RAM with the policy, SLAM or AMCL, and Nav2 all running.
- [ ] **Lidar checks:**
  - the 12 cm range offset, against walls to the side and behind at tape-measured distances;
  - narrowing the body box back from ±0.35 m toward ±0.25 m, now that the leg filter exists.

**Done when:** the robot reaches 10 goals in a row across the saved map with no help.

## Phase 2: Camera

**To build:**
- [x] **Camera bridge.** `camera_bridge` starts every camera under `cameras:` in `config.yaml`, so a second
      one (a thermal camera, say) is a new entry, not new code. Sources: `unitree` (the front camera's
      `VideoClient` JPEGs, passed on without re-encoding), `v4l2` (any USB/UVC camera; `kind: thermal` keeps
      16-bit frames, sent as PNG) and `ros` (another driver's image topic; Gazebo uses it). Each camera
      `<name>` gets `/camera/<name>/image_raw/compressed`, `image_raw` (decoded), `camera_info`, the mount TF,
      and `/camera/<name>/preview/compressed` (320 px, 2 Hz, ~2-15 kB). `/cameras` lists them. Nothing is read
      or encoded while nobody subscribes. Its own launcher option, [I] (`camera.launch.py`), on top of the
      real driver or Gazebo's nav world.
- [x] **App camera tab:** the preview only, one camera at a time, nothing once the tab is left. Preview rate
      (1-10 Hz) and width (160-640 px) are set from the app (`/camera/<name>/preview/settings`); the robot
      echoes them in `/cameras`, and the tab shows the frame size and kB/s.
- [x] **Heartbeat drops:** the robot counts gaps between heartbeats (a gap of k periods = k - 1 lost) and puts
      them in `/robot_state` (`heartbeat`: Hz, lost in 10 s, total, received, longest gap). Shown in the camera
      tab and the System tab's link card, red once a gap reaches the watchdog; the header's Live pill has a
      bar per second of the last minute and the % lost over it.
- [x] **Camera mount check tool:** `ros2 run quadruped_perception camera_lidar_check` draws the last 20 lidar
      clouds on the image, coloured by distance, on `/camera/front/lidar_check/compressed` (2 Hz); view it in
      `rqt_image_view`. Checked in sim on a synthetic cloud: points land on their projected pixels.

**To test on the robot:**
- [ ] **Bridge:** the front camera's resolution and rate, and the Jetson CPU with it running. Needs the Docker
      image rebuilt (OpenCV, `camera_calibration`, `image_transport_plugins`, `rqt_image_view`).
- [ ] **Calibration:** checkerboard with `camera_calibration`; its Commit button saves `calibration/front.yaml`
      through the bridge's `set_camera_info`. Run it on a laptop on the robot's Ethernet (full-rate JPEGs over
      Wi-Fi starve the heartbeat), decoding there rather than pulling the robot's raw `image_raw`:
      `ros2 run image_transport republish compressed raw --ros-args -r in/compressed:=/camera/front/image_raw/compressed -r out:=/calib/image`,
      then `ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025 --ros-args -r image:=/calib/image -p camera:=/camera/front`.
      If the lens is too wide for `plumb_bob`, try camera_calibration's fisheye model; the bridge stores it and
      `camera_lidar_check` handles `equidistant`.
- [ ] **Camera mount:** the lidar points land on the objects in the image.
- [ ] **Preview rate:** the app's preview doesn't load the Wi-Fi: 0 heartbeats lost in the camera tab at each
      rate and width, to find the largest that is safe.

**Done when:** a lidar point on a cone lands on the cone in the image.

## Phase 3: Objects

YOLOE, an open-vocabulary YOLO: its prompt-free model names objects from a built-in 4,585-word vocabulary
(cones included), so nothing is trained; with `classes:` in `config.yaml` it looks only for those names.
3a finds objects in the image; 3b puts them on the map.

### 3a: Detection

**To build:**
- [x] **GPU in the robot's Docker:** [Docker/Dockerfile.jetson](Docker/Dockerfile.jetson), see
      [GPU on the robot](#gpu-on-the-robot-ros-with-cuda-torch) below. Built on the laptop under QEMU (about 1.5 h,
      7 GB): everything builds and imports. Not run on the robot yet (no GPU or DDS under QEMU).
- [x] **`object_detector` node:** `/camera/<name>/image_raw/compressed` → `/camera/<name>/detections`
      (JSON boxes) and `/camera/<name>/detections/image/compressed`; `--export` builds the TensorRT engine.
      `camera.launch.py detect:=true`, or [I] in the launcher. About 12 ms per image on the laptop's GPU.
- [x] **Laptop image with YOLO:** ultralytics and torch's CUDA 12.6 build (runs on NVIDIA driver 525+) in
      [Docker/Dockerfile](Docker/Dockerfile); same pip list as the Jetson image, OpenCV 5 in both.
- [x] **Tried in Gazebo** (nav world): it sees the cones from far away, but up close it gives odd labels
      ("extrude"): Gazebo's flat renders are far from the real photos it learned from. In Gazebo, a `*-seg.pt`
      model with `classes:` limits it to names we choose.
- [x] **Boxes in the app:** the camera tab draws `/camera/<name>/detections` over the preview ("Objects").
- Only if YOLOE misses cones:
  - [ ] **Dataset:** Gazebo images, labelled automatically from the known cone poses; 100–200 real photos,
        labelled by hand.
  - [ ] **Model:** fine-tune YOLOE on them, exported to TensorRT on the Jetson.

**To test on the robot:**
- [ ] **Build the GPU image** on the robot; `torch.cuda.is_available()` is `True` inside it.
- [ ] **Everything else still works** in the new image: driver, Nav2, SLAM, Point-LIO, the app.
- [ ] **Detection** in different light, and with other orange things around.
- [ ] **Speed:** FPS of the TensorRT engine on the Jetson, and CPU/GPU/RAM while Nav2 runs.

**Done when:** the camera tab shows boxes on the real robot's objects, at the detector's rate.

### 3b: Objects on the map

`object_mapper` in `quadruped_perception` (config.yaml's `object_mapper:`); launcher [I] → detect → "put them
on the map", with [N] running. Needs the camera calibrated (`calibration/front.yaml`; Gazebo publishes its own
`camera_info`) and the mount checked with `camera_lidar_check` (Phase 2).

**To build:**
- [x] **Box → direction:** the camera's intrinsics turn a pixel into a ray (`pixel_ray`, shared with
      `camera_lidar_check`). For objects on the floor, the box's bottom-centre.
- [x] **Direction → distance from the lidar:**
  - the last ~1 s of `/lidar/points_filtered`, kept in the map frame, projected onto the image through TF;
  - the points inside the object's outline (object_detector now sends YOLOE's mask as a polygon), and the
    nearest cluster of them, so the wall behind it does not count;
  - pushed back by half the object's width (how far its points spread sideways), since the lidar sees only
    its front: cone error 11 → 4 cm;
  - fallback when no point hits it: the bottom-centre ray crossed with the floor.
- [x] **Camera → map:** TF at the image's stamp.
- [x] **Merging sightings:** within 0.5 m (0.25 m for a label it never had), reported after 3 sightings,
      labels voted. Boxes wider than 90% of the image or over half its area (floor, "room", "sky") and
      `ignore:` labels (people, surfaces) are not mapped.
- [ ] **Forgetting:** each object already has `last_seen`, but time alone would forget everything out of view.
      For each camera frame, a known object that should have been seen counts a miss:
  - in view (its position projects inside the image), in range and not cut off;
  - not hidden (no lidar points in front of it along that direction);
  - not detected there.

  After ~10 misses in a row (~2 s at 5 Hz) it is dropped; a sighting resets the count. Also add
  `first_seen`. Test in Gazebo by mapping a cone, then removing it from the world. For now
  `/objects/clear` (or "Clear objects" in the app) forgets them all.
- [x] **Output:** `/objects` (JSON, latched: id, label, x, y, z, sightings, last seen, label votes) and
      `/objects/markers` for rviz.
- [x] **Show them** in rviz (markers) and on the app's map tab (dots in the label's colour).
- [x] **Gazebo check:** `ros2 run quadruped_perception object_check [--start=x,y,yaw]` compares `/objects`
      with the cones' poses in `nav.sdf`. Standing 2–4.5 m from each of the 4 cones: 0–4 cm. Labels up close
      are odd with the prompt-free model ("extrude", "stupa"); with `classes: [traffic cone, box, pillar, wall]`
      it is "traffic cone" and nothing else gets mapped. A cone behind an obstacle is not mapped (correct).

**To test on the robot:**
- [ ] **Positions:** objects on the map against tape-measured positions.
- [ ] **Moving robot:** positions stay put while it walks and turns.

**Done when:** every cone in a room appears on the map within about 20 cm.

## Phase 4: Skills

All skills go through `/skill`. Two kinds:

- **Tricks** (movements): one trained policy each, in `IsaacLab_Tasks/Tricks`. The skill switches to the trick's
  policy, holds its switch on `/pipeline/trick` for the skill's duration, gives it 2 s to stand back up, then
  switches back to the walker. If the switch stops arriving, the policy stands by itself. The app does the same
  by hand: pick the trick policy, then hold its button (Left/Right for Pee).
- **Behaviours**: code in `skill_server` on top of Nav2 and the walker.

**To build:**
- [x] **Tricks module:** `Go2-Pee` (two switches, left and right: rear foot 12 cm up and out, calf folded, the
      front foot on that side forward), `Go2-Sniff` (nose 0.3 rad down, base at 26 cm, nose bobbing on a 1 Hz
      clock), `Go2-Wiggle` (back end swinging ±0.08 rad about the front, feet planted). Trained inside the
      robot's tilt and joint checks, so they keep all protections. `pee_left`, `pee_right`, `sniff`, `wiggle`
      skills; `policies.json` `commands: "trick"`.
- [x] **Trained** (`Pee1` 1000 iterations, `Sniff1` and `Wiggle1` 600) and in `policies.json` as `Pee`, `Sniff`,
      `Wiggle`. Measured in Isaac: Sniff bobs ±0.058 rad every ~1.05 s, Wiggle swings ±0.095 rad every ~0.6 s.
- [x] **App:** a Trick card in the Drive tab while a trick policy runs, hold to do it (`trick_switches` in
      `/robot_state`).
- [ ] **MuJoCo check:** each trick through `skill_server` and through the app, walker → trick → walker.
- [ ] **`go_to_cone`:** a Nav2 goal about 0.5 m from the cone, facing it.
- [ ] **`explore`:** frontier exploration, so the robot finds cones in an unknown place.
- [ ] **`follow_me`:** track the person's legs in the lidar beside the robot, published on `/cmd_vel/skill`.

**To test on the robot:**
- [ ] **Tricks**, after MuJoCo: the walker → trick → walker switches are smooth, and `pee` holds the leg up
      without tipping.
- [ ] **`go_to_cone`** from different starts and angles.
- [ ] **`explore`** in a room it has never mapped.
- [ ] **`follow_me`** with one person, then with others nearby.

**Done when:** go to cone → sniff → pee works on the real robot.

## Phase 5: LLM brain over MCP

**To build:**
- [ ] **MCP server** with these tools:
  - `get_state`, `list_cones`, `go_to(x, y)`, `skill(name)`, `explore`, `stop`;
  - `describe_surroundings()`: the map condensed to a few lines of text, see [3D mapping](#3d-mapping-lidar-inertial-odometry);
  - `look_for(words)`: object_detector looks only for those names until cleared. YOLOE can change its words
    while running, but only with the PyTorch model (a TensorRT engine has them built in), so it is slower on the
    Jetson; the prompt-free engine stays the default.
- [ ] **LLM on the laptop.** Calls cross Wi-Fi over TCP or HTTP, not DDS.
- [ ] **Safety:**
  - the LLM only reaches skills and `/cmd_vel/llm`, the lowest priority;
  - the app's e-stop and the joystick always win.
- [ ] **A "having fun" prompt:** explore, pick a cone, go sniff it, sometimes pee, then repeat.

**To test on the robot:**
- [ ] **Override:** the app's e-stop and the joystick take over from the LLM mid-skill.
- [ ] **Wi-Fi drops:** the robot stops safely when the laptop goes out of range.

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
`lio_map_stream` turns `/lio/cloud` into voxels for the app (5 cm default, [L] asks): `/lio/map_voxels` (snapshot, on a
Point-LIO restart or a new receiver) and `/lio/map_voxels/delta` (new voxels, every 1 s), never more than 8000 voxels
(96 kB) a message: a whole map at once held the Wi-Fi long enough to starve the heartbeat (>2 s, robot cut twice).
About 7 kB/s on the test bag at 5 cm. Voxels keep a hit/miss score as in OctoMap (numpy, ~3 ms a scan on the
laptop): beams passing through clear them, so an opened door goes in ~5 s and the L1's range noise no longer
thickens walls over minutes (standing still, growth down ~60%); a hit just behind a voxel already solid on its
beam counts for that voxel. Deltas carry removals too. The floor (within
`--ground_band` 10 cm of the floor under the robot) is a height map, one voxel per column: the L1's floor is
2-8 cm thick (thicker at grazing range), which stacked it 2-3 voxels deep. Floor-only columns with one voxel:
73% → 92% on the walking bag, 97% with a 15 cm band (but then lower obstacles merge into the floor).
Known port bug: `standard_pcl_cbk` keeps only whole seconds of `last_timestamp_lidar` (Unitree's uses `toSec()`).

**Done:**
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
- [x] **Per-point times.** The cloud's `time` field is each point's time in the scan, in seconds: 0 to ~63 ms,
      rising, one value per point, ~4200 points per scan at 15.5 Hz (`timestamp_unit: 0`). Checked on both bags.
- [x] **Body crop.** `real_sensors` blanks the robot's own hips and legs (`real_lidar.body_box`), so Point-LIO's
      `blind` dropped from 0.5 to 0.1 m: the floor within ~0.4 m of the lidar is mapped (34% → 98% covered).
- [x] **Voxel size from the robot** (`/lio/map_voxels/size`), paced deltas, voxel clearing, and removals in the app.

**To build:**
- [ ] **Start with the real driver, not Nav2.** LIO supplies `odom → base` (an `odometry.source: lio` switch in
      `config.yaml`, with telemetry then dropping its own), so the pose never jumps and the app's view works
      without [N]. An adapter turns LIO's `lio_odom → lio_imu` into `odom → base`, and leg odometry fills
      `/odom`'s velocity if LIO leaves it empty. Gazebo keeps leg odometry, since its lidar has no point times.
- [ ] Items 1-4 of **What builds on it**, below.

**To test (bags or robot):**
- [ ] **Raw scans.** Unitree's clouds carry about 59k points/s against the L1's 21.6k, so they overlap or are
      preprocessed. LIO may need the raw scans instead.
- [ ] **Same data as Unitree's driver.** Point-LIO's L1 config expects the cloud and IMU as `unilidar_sdk`
      publishes them. Compare the Go2's DDS cloud with that: frame, point-time units, IMU rate.

**To test on the robot** (everything from 2026-10-08 on was checked on replayed bags only):
- [ ] **Starts:** several [L] starts in a row, robot still for ~1 s each: no spinning map.
- [ ] **Level floor:** the floor is flat in the app's 3D tab (body accelerometer), and the floor in front of
      the robot is filled in (body crop).
- [ ] **Doors and people:** open a door or walk past; their voxels clear in ~5 s.
- [ ] **Long run:** 10 minutes at 5 cm voxels: the voxel count levels off, the heartbeat never drops (no
      "Supervisor heartbeat lost"), and `lio_map_stream` stays around 10-15% of a core.
- [ ] **Save the 3D map** ([L], save on exit) and look at the PCD.
- [ ] **Big loop with Point-LIO:** how far `/lio/odom` is off when back at the start.

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
5. **3D map in the app's 3D tab: done.** `lio_map_stream` sends the LIO map as voxels (see Setup), and the
   app draws them around the robot model, colored by height.

## GPU on the robot: ROS with CUDA torch

The Go2's Jetson Orin Nano (40 TOPS) has a GPU with 1024 CUDA cores and 32 Tensor cores. The `arm64` image
can't use it: pip's `torch` for ARM is CPU-only, and `ros:humble-ros-base` has no CUDA.

JetPack 5 is Ubuntu 20.04, and NVIDIA ships the R35 CUDA libraries and CUDA torch only for 20.04 (Python 3.8).
ROS 2 Humble has no apt packages for 20.04, so [Docker/Dockerfile.jetson](Docker/Dockerfile.jetson) starts
from dusty-nv's `dustynv/ros:humble-desktop-pytorch-l4t-r35.4.1` (Humble built from source, CUDA 11.4,
TensorRT, torch 2.0) and builds on top:
- the robot's ROS packages: Nav2 (with MPPI, gcc 11), slam_toolbox, pointcloud_to_laserscan, twist_mux,
  joy, teleop_twist_joy, the MCAP plugin and pcl_ros;
- torchvision with CUDA, CycloneDDS 0.10 for the Unitree SDK, and the laptop image's pip list (ultralytics,
  OpenCV 5 headless instead of NVIDIA's 4.5 Python binding, whose C++ libraries stay for cv_bridge);
- pip is held to the base's torch, so nothing replaces it with the CPU one.

Still different from the laptop because of 20.04: Python 3.8, torch 2.0, numpy 1.24, mujoco 3.2.3.
How to build and run it: [Docker/README.md](Docker/README.md), section 1d.

- **Check the L4T version** on the robot (`cat /etc/nv_tegra_release`); the base's must be R35 too.
- **Leave the policy on the CPU.** It runs in about 6 ms there, and the GPU would only add launch latency.
  The GPU is for perception.
- **Size and build time.** About 7 GB, and 1.5 h on the laptop under QEMU (most of it Nav2). Push it to GHCR
  (`:jetson`) so the robot only pulls it.
- **Memory.** The GPU shares the 8 GB of RAM with the CPU; watch `/system_stats`.
- **Later, JetPack 6** (Ubuntu 22.04, "Super" mode, ~67 TOPS) would allow Jazzy from apt with dusty-nv's
  `jazzy-ros-base-r36.4.0-cu128-24.04`, with no source builds. That needs reflashing the Jetson.

## Policies: switching, safety lie-down and numb

Built and tried in MuJoCo (2026-10-11); see `PROJECT_OVERVIEW.md` (Robot interface). Policies are listed in
`config/policies.json` with the checks each one wants; after a trip the `Lie2` policy takes over with no
checks and the robot goes numb once lying; the app and console pick policies and have a Numb button.

- [x] **Policy list** with per-policy `protections` (heartbeat, tilt, pitch, rom) and `commands`.
- [x] **Load on demand** in a background thread, off the policy's real-time core; the safety one is always loaded.
- [x] **Switching** on `/pipeline/policy`, blended over 0.5 s, refused during an e-stop, a trip or numb.
- [x] **Safety lie-down**, then numb (lying: level, near `lie_flat`, still for 1 s; or after 10 s).
- [x] **Numb** on `/safety/numb`, released from the app or console into pose mode.
- [x] **App and console**: policy picker, Numb / Release numb, banner.
- [ ] **Real robot**, on the rope first: a heartbeat cut while standing, then while walking; Numb and release.
- [ ] **Load time and RAM** per policy on the Jetson, and whether the background load shows up as control-loop jitter.
- [ ] **Per-policy action scale and default pose** from each run's `params/deploy.yaml` (today the same for all,
      which the current ones share).
- [ ] **Skills** (`skill_server`): use `Reco2` after a fall, and `Lie2` for `lie`.

## Backlog

- **CPU, GPU, RAM and temperature in the app: done.** `system_monitor` publishes `/system_stats` at 1 Hz,
  shown in the app's Robot tab.

- **SSH terminal in the app: built, not yet tried on the robot.** Shell tab, see `Android/README.md`.
  Still to check: `sshd` and the login user on the Go2's Jetson (the app defaults to `unitree@10.42.0.1`).
  The terminal can press ENTER on the driver, which releases an e-stop from the phone; not blocked.
- **A custom rviz config.**
- **Flags to ROS parameters.**
