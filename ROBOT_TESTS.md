# Robot tests

What is left to check on the real Go2, taken out of [ROADMAP.md](ROADMAP.md). Grouped by roadmap phase; each
group needs the ones before it. Tests for things not built yet are at the end.

## Retest after the 2026-10-14 fixes

- [ ] **ENTER at the driver** releases a lost-heartbeat trip and numb, and an e-stop. The driver now reads the
      terminal directly; before, `ros2 launch` never passed ENTER on. Try it from the launcher's terminal.
- [ ] **Light colors:** blue in policy mode, green in pose mode, red during an e-stop, numb or a trip. Uses the
      Go2 app's undocumented light call; if the robot refuses it, the driver logs it once and leaves the
      light alone.
- [ ] **Policy switch and Lie2** with the colcon install rebuilt (repo path fix).

## Navigation (Phase 1)

- [ ] **Save a map** of a room from the app, and find it in [N].
- [ ] **Big loop.** One large loop (a corridor, or around a building):
  - back at the start, the map closes without a doubled wall;
  - how far the pose drifted before SLAM corrected it, and the same for Point-LIO (`/lio/odom`).
- [ ] **Keepout zones** on the saved map: Nav2 plans around them.
- [ ] **Localize on the saved map:** AMCL with a known start pose, then an unknown start with the `localize`
      skill (works in Gazebo).
- [ ] **Goals and obstacles:** several goals on the saved map from the app, then someone stepping into the path.
- [ ] **Lidar:**
  - the 12 cm range offset, against walls to the side and behind at tape-measured distances;
  - narrowing the body box from ±0.35 m toward ±0.25 m.

## Camera (Phase 2)

- [ ] **Calibration:** checkerboard with `camera_calibration`, from a laptop on the robot's Ethernet. Its Commit
      saves `calibration/front.yaml` through the bridge. Decode on the laptop:
      `ros2 run image_transport republish compressed raw --ros-args -r in/compressed:=/camera/front/image_raw/compressed -r out:=/calib/image`,
      then `ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025 --ros-args -r image:=/calib/image -p camera:=/camera/front`
      (`--size` is the inner corners). If the lens is too wide for `plumb_bob`, try the fisheye model.
- [ ] **Camera mount:** `camera_lidar_check`, the lidar points land on the objects in the image.

## Detection (Phase 3a)

- [ ] **Speed:** FPS of the TensorRT engine (`object_detector --export`) on the Jetson.
- [ ] **Detection** in different light, and with other orange things around, at the raised `conf`.

## Objects on the map (Phase 3b)

Needs the calibration above.

- [ ] **Positions:** objects on the map against tape-measured positions (done when every cone is within ~20 cm).
- [ ] **Moving robot:** positions stay put while it walks and turns.

## Skills (Phase 4)

- [ ] **Tricks through `skill_server`** (`pee_left`, `pee_right`, `sniff`, `wiggle`): walker → trick → walker.
      Through the app they all work on the robot.

## Point-LIO

- [ ] **Raw scans.** Unitree's clouds carry about 59k points/s against the L1's 21.6k, so they overlap or are
      preprocessed. LIO may need the raw scans instead.
- [ ] **Same data as Unitree's driver.** Compare the Go2's DDS cloud with what `unilidar_sdk` publishes, which
      Point-LIO's L1 config expects: frame, point-time units, IMU rate.

## App

- [ ] **SSH terminal:** `sshd` and the login user on the Go2's Jetson (the app defaults to `unitree@10.42.0.1`).

## Once built

- [ ] **`go_to_cone`** from different starts and angles.
- [ ] **`explore`** in a room it has never mapped.
- [ ] **`follow_me`** with one person, then with others nearby.
- [ ] **LLM override:** the app's e-stop and the joystick take over from the LLM mid-skill.
- [ ] **Wi-Fi drops:** the robot stops safely when the laptop goes out of range.

## Results so far

- **Jetson load** (2026-10-14), all cores, with the policy running:
  - policy only: ~10% (the policy's core slightly more);
  - plus Point-LIO: ~25%;
  - plus camera and YOLO: ~30%, GPU 0-25% with a spike to 90%, RAM 5.7 of 7.2 GB, 65 °C;
  - camera, YOLO, LIO, policy, SLAM and Nav2: ~40%, RAM 5.8 GB.
- **Camera preview** at its highest rate and width: no heartbeats lost.
- **Point-LIO:** starts hold without spinning, doors and people clear, 10 min run fine, 3D map saves. Floor ~5 cm
  off level, maybe the real floor.
- **Policies:** switching, heartbeat cut standing and walking (Lie2, then numb), numb and release, load time and
  RAM all fine. Tricks and `Reco2` work from the app.
- **Kp/Kd** from the console work, but the policies were trained with `config.yaml`'s values; change them only
  for experiments.
