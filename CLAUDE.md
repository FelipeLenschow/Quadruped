# CLAUDE.md

Write all text (replies, docs, comments, commit messages) in ASD-STE100 Simplified Technical English.

This file orients an agent working in this repo. For the deployment/runtime architecture (drivers, telemetry, controller), see `README.md` and `PROJECT_OVERVIEW.md` — this file covers environment setup and the Isaac Lab training side, which those docs don't get into.

## Environment setup

Two separate Python environments are in play; picking the wrong one is the most common cause of "command not found" or import errors (e.g. `tensorboard`, Isaac Lab modules):

- **Isaac Lab / training / MuJoCo bridges** (Python 3.12): `source ~/env_isaacsim/bin/activate`
  Needed for `launcher.py` train/play actions, `tensorboard`, the MuJoCo driver/twin scripts, `eval_mujoco.py`.
  Run `unset PYTHONPATH` first: a sourced ROS 2 Humble setup puts its Python 3.10 `site-packages`
  on `PYTHONPATH`, and those leak into this venv and break resolution.
- **ROS 2 Humble tooling / Gazebo / real-robot deploy** (Python 3.10, system): `source venv_robot/bin/activate`, or run inside `Docker/` if the host isn't natively Python 3.10. `launcher.py` auto-detects this (`IS_DOCKER`, `IS_ROBOT`, interpreter version) and disables menu options that don't apply to the current environment.

### Isaac Sim / Isaac Lab versions

Full migration write-up, including every 2.x->3.0 break and its fix: `isaaclab3_migration_notes.md`.

Rebuilt 2026-09-18. `~/env_isaacsim` is Python 3.12 with Isaac Sim **6.1.0.0**; Isaac Lab is
**v3.0.0-EA**, cloned to `~/IsaacLab30` and pip-installed editable. The older `~/IsaacLab`
(on `main`) is no longer what gets imported, and Isaac Sim 5.1 is gone — there is no fallback.

Two things that surprise people:

- **PhysX is still the default backend.** `SimulationCfg.physx` was renamed to
  `SimulationCfg.physics`, and it defaults to `None`, which means `PhysxCfg()`. `PhysxCfg`
  itself moved out of `isaaclab.sim` into `isaaclab_physx.physics`, so the old
  `sim_utils.PhysxCfg` raises `AttributeError: No isaaclab.sim attribute PhysxCfg`.
  All the `gpu_*` capacity fields survived the move unchanged.
- What *is* newly on by default is `use_newton_actuators = True`: Newton executes
  `DCMotorCfg` / `IdealPDActuatorCfg` actuators (PhysX runs them via a host adapter) instead of
  the old Isaac Lab path. This changes actuator behaviour, so it is the flag to set `False` if
  a run needs to match the `Final*` / `ph*` results — not the physics backend.
- Isaac Lab 3.0 is a **uv workspace**. `pip install -e source/<pkg>` pulls only that package's
  own dependencies, not the root project's ~52 externals, so imports fail one at a time
  (`rich`, `hydra-core`, `lazy_loader` first). Prefer `uv sync`; if using pip, install the
  workspace members in topological order with `--no-deps`, then the root dependency list.

`Command_list.txt` has the exact activation lines and other commands run day to day — check it before assuming a tool isn't installed.

## `launcher.py` — the single entry point

`python launcher.py` is a CLI menu that drives everything: train, play (Isaac Lab / Isaac Sim / MuJoCo / Gazebo), deploy to the real robot, evaluate a policy, and ROS 2 tools (teleop, PlotJuggler, MCAP record/replay, rqt/tf2). Flow:

1. Detects environment (Docker, ARM64 robot, `env_isaacsim` venv) and hides menu options that don't apply.
2. Lists task modules under `IsaacLab_Tasks/` (see below) and prompts for one.
3. Finds checkpoints under `<module>/logs/**/*.pt` or `<module>/checkpoints/*.pt`.
4. For `train`, reads `<module>/source/**/training_phases.yaml` to list phases and can auto-chain a curriculum sequence (e.g. phase1 → phase6), feeding each segment's best checkpoint into the next `train.py` invocation.
5. Dispatches to a subprocess, passing robot/terrain/phase selection as **environment variables** (`QUADRUPED_TRAINING_PHASE`, `QUADRUPED_ROBOT_CFG`, `QUADRUPED_TERRAIN`, `PYTHONPATH=<module>/source/Quadruped`) rather than CLI flags — task code reads these from `os.environ`.
   The ROS-side actions run from the ROS 2 packages in `src/` (see below): `ros2 launch quadruped_bringup <mode>.launch.py key:=value ...` for drivers, eval, twins, gamepad teleop, eval sweep, navigation (`nav.launch.py`, menu [N]), cameras (`camera.launch.py`, menu [I]), DLS2 low level (menu [DLS2], `dls2.launch.py`: pick any of `mock_go2` (on `lo` by default; another interface only on a cable from the PC straight to the Jetson, with the Go2 cable removed, after a warning), `twist_mux`, `skill_server`, `system_monitor`, `real_sensors`, `mcp_server`, the nodes of option [6] other than `real_driver`, while DLS2's `go2_hal` and `ros2_translator` have the robot), bag record and rviz/plotjuggler/rqt/foxglove; `ros2 run` for the interactive ones that read stdin (console, keyboard teleop, joint tester, MCAP replay). Before those actions the launcher runs `colcon build --symlink-install --base-paths src` when the install is missing or stale, and sources `install/setup.bash`.
6. For sim/deploy actions, can also start the reward estimator (`reward:=true`; off by default, `START_REWARD_ESTIMATOR` in `launcher.py`).
7. For `train`, optionally starts `Tools/auto_eval.py` in the background (one watcher per curriculum
   segment, stopped before the run folder is renamed). It watches the run's `checkpoints/` and puts
   every Nth checkpoint (default 50k) through `quadruped_drivers.eval_mujoco` (run with `-m` from `src/`; only `quadruped_control` comes from the colcon install), so the evaluation half of the
   dashboard fills in while the run is still going. The sweep is spawned through a ROS 2 interpreter
   (sourcing `/opt/ros/<distro>/setup.bash` and stripping the Isaac venv off `PATH`) because
   `eval_mujoco.py` imports `rclpy`, which the Isaac venv does not have.

## ROS 2 packages (`src/`)

Runtime code (everything that is not training) is a colcon workspace; build only `src/`, since `IsaacLab_Tasks/` also contains `setup.py` packages:

    colcon build --symlink-install --base-paths src && source install/setup.bash

In Docker run colcon as `python3 -m colcon ...` from the venv, so node scripts get the venv's Python (torch, mujoco).

- `src/`: the ROS 2 packages (colcon workspace, `colcon build --symlink-install --base-paths src`):
  - `quadruped_common`: Python helpers shared by every node, no nodes: `paths.py`, `config_loader.py`, `realtime.py` (keeps everything off the policy's core), `robot_defaults.py` (default stance), `kinematics.py` (leg kinematics and base rotation helpers, as `quadruped_control/src/kinematics.cpp`). The control decision is `quadruped_control`'s (C++).
  - `quadruped_drivers`: Python nodes: MuJoCo (`mujoco_driver`, `eval_mujoco`) and Gazebo (`gazebo_driver`, `gazebo_sensors`: lidar and camera to ROS), `camera_bridge` (every camera in `config.yaml`'s `cameras:`, real or Gazebo, to `/camera/<name>/...`), and `gps_driver` (the u-blox RTK GPS on USB to `/gps/fix`, `config.yaml`'s `gps:`).
  - `quadruped_operator`: console, supervisor, teleops, twins, reward estimator, MCAP tool, `skill_server`.
  - `quadruped_perception`: `lidar_filter` (drops the robot's own body from the lidar cloud), `camera_lidar_check` (lidar points drawn on a camera image, to check the mount), `object_detector` (YOLOE open-vocabulary detection on a camera, config.yaml's `detector:`; on the robot it needs the GPU image, `Docker/Dockerfile.jetson`), `object_mapper` (those detections placed on the map with the lidar, merged, on `/objects`), `object_check` (Gazebo: `/objects` against the nav world's cone poses).
  - `quadruped_control` (branch `cpp-core`): the control core in C++: kinematics, the LKF estimator, the policy runner (observation layouts, history, WTW/speed clocks, MLP) and `Controller`, the ROS-free pipeline (safety processor, pose generator, policy switching, safety-policy takeover and numb, slew, blends; clocks passed in, side effects drained as events). pybind11 bindings (`import quadruped_control`; pybind11 is the `third_party/pybind11` submodule, 2.13, numpy 2 safe); Each driver has its ROS side in a `driver_node` file: `quadruped_real/src/driver_node.cpp` on the robot, `quadruped_control/driver_node.py` (`Pipeline`: subscriptions, telemetry, joint commands, `/robot_state`) for the sim drivers (MuJoCo, eval, Gazebo, Isaac). `quadruped_control/export.py` reads training checkpoints (skrl, rsl_rl; what the archive lacks comes from the run's `params/`) into the `.qpol` files the C++ side loads; `scripts/export_policy.py` is its command line. The Python pipeline it replaced is gone: `test/estimator_parity.py`, `test/policy_parity.py` and `test/pipeline_parity.py` (`--via-ros` for Pipeline) replay recordings of it (`test/reference/*.npz`, with the config files and checkpoint hashes they were made with) step by step against the C++ side.
  - `quadruped_real` (branch `cpp-core`): C++ `real_driver`, launched by `real.launch.py`. The control loop (LowState, `Controller.step`, LowCmd at 200 Hz, policy every 4th tick) runs on the reserved core at SCHED_FIFO with the Unitree SDK2's DDS threads (`third_party/unitree_sdk2` submodule, linked by RPATH, not installed); an rclcpp node on the other cores trades commands and snapshots with it through lock-free mailboxes and publishes the telemetry. Policies load from `.qpol` files beside the checkpoints, exported by `quadruped_control`'s `export_policy.py` when missing or older than the `.pt`. Everything else that talks to the robot's DDS is here too, all on the C++ SDK (there is no Python SDK): `real_sensors` (L1 lidar cloud and IMU to `/lidar/points`, `/lidar/imu` for Point-LIO; started by `real.launch.py`), `go2_camera` (the Go2 camera's JPEGs to `/go2_camera/image/compressed` while subscribed; `camera.launch.py` starts it and `camera_bridge` reads it for `source: unitree` cameras), `test_joints` (joint tester, launcher [T]), `helios_driver` (no SDK: the RoboSense Helios 16P over UDP, decoded by the header-only `third_party/rs_driver` submodule, to `/helios/points` in the Velodyne layout FAST-LIO reads, robot body cropped by `config.yaml`'s `helios: body_box`; `--from_topic` reads an old rslidar_sdk `/rslidar_points` bag instead) and `mock_go2` (`test/mock_go2.cpp`, built where the `mujoco` wheel is installed: a MuJoCo Go2 on rt/lowstate / rt/lowcmd, CRC-checked, to run `real_driver` on a PC; it shows a MuJoCo window where GLFW is installed, `--headless` for none: use `--interface lo` on both and set `QUADRUPED_ROS_DOMAIN_ID` (overrides config.yaml's domain) to stay off a running stack).
  - `quadruped_interfaces`: `srv/Skill.srv` (ament_cmake).
  - `quadruped_description`: MuJoCo menagerie, Gazebo worlds (`scene`: flat, `nav`: walls, obstacles, 4 cones; `scripts/sdf_to_map.py` makes its Nav2 map), Go2/Go1/A1 models (the Go2 carries an L1-style lidar and a front camera), kinematics yaml, bundled policies.
  - `quadruped_bringup`: `launch/*.launch.py` and `config/` (`config.yaml`, `joy_f710.config.yaml`, `twist_mux.yaml`, `nav2.yaml`, `slam.yaml`).
  - `point_lio`: symlink to the `third_party/point_lio_ros2` submodule (Point-LIO, C++), config `point_lio_go2.yaml`.
  - `fast_lio`: symlink to the `third_party/FAST_LIO` submodule (upstream FAST-LIO, branch ROS2, with its ikd-Tree submodule: `git submodule update --init --recursive`), config `fast_lio_helios.yaml`, started by `fast_lio.launch.py` (launcher [L], choice 2) on `/helios/points` and `/lidar/imu`, outputs on `/helios_lio/*`. The extrinsic comes from `config.yaml`'s `helios: xyz, rpy`. FAST-LIO forces C++14 and ROS 2 Humble needs C++17: `colcon.meta` (repo root, so build from there) injects `third_party/fast_lio_cxx17.cmake` into its configure, which sets C++17 at the end of its CMakeLists (CMake 3.19+); the submodule stays unpatched.
  - `livox_ros_driver2`: only Livox's two messages (`CustomMsg`, `CustomPoint`), which FAST-LIO needs to build; no driver, no Livox SDK.
- `Android/`: phone app replacing the console, a native Fast DDS participant (no bridge). See `Android/README.md`.
- `IsaacSim/isaac_driver.py`: stays outside the packages (Isaac Sim's Python 3.12); imports `quadruped_common` from `src/`. Broken for now: it needs `quadruped_control` built for Python 3.12, which nothing builds yet.

Velocity sources publish on `/cmd_vel/<source>` and `twist_mux` picks one for `/cmd_vel`; the pipeline publishes `/robot_state` (JSON) and `skill_server` serves `/skill`. See `PROJECT_OVERVIEW.md` (Robot interface).

Scripts still take command-line flags (not ROS parameters); `rclpy.utilities.remove_ros_args` strips the `--ros-args` that `Node` appends. Data is found through `quadruped_common.paths` (ament share dir, falling back to `src/`), and `config.yaml` lives at `src/quadruped_bringup/config/config.yaml`.

## `IsaacLab_Tasks/` structure

Each subfolder is a fully independent copy of the Isaac Lab task package (own `source/Quadruped/Quadruped/tasks/direct/quadruped/...`, own `logs/`, own `training_phases.yaml`) — nothing is shared between them, so a fix or reward tweak made in one does not propagate to the others. Confirm which module is actually meant before editing:

- **`Walk/`** — the main, actively developed task. Default assumption unless told otherwise.
- **`Walk_GO2/`** — a Go2-only simplification made when a possible internship lab also had a Go2; that plan changed and work went back to `Walk` (3 robots). Likely stale.
- **`Stairs/`** — experiment adding a terrain/height-scan sensor for stair climbing.
- **`Handstand/`** — handstand task.
- **`Simple/`** — unitree_rl_lab cut down to the Go2 (package `simple`, manager-based, rsl_rl): the Sigma-Vel-Foot-Rough and Deploy tasks. Trains through its own `scripts/rsl_rl/train.py`.
- **`WalkTheseWays/`** — Walk These Ways for the Go2 (package `walk_these_ways`), cut from Simple to the single `Unitree-Go2-Velocity-WTW` task. Same rsl_rl layout as Simple.
- **`Recovery/`** — Go2 fall recovery (package `recovery`), cut from Simple: `Go2-Recovery` (get back to the standing pose slowly from drops, rolls and random poses) and `Go2-LieDown` (a safety controller that lies the robot down on its belly so the motors can be switched off). Same rsl_rl layout as Simple.
- **`Tricks/`** — one policy per trick for `skill_server` (package `tricks`), cut from Recovery: `Go2-Pee`, `Go2-Sniff`, `Go2-Wiggle`. Each reads an on/off flag in the velocity command's place and stands when off. Same rsl_rl layout as Simple.

## Reward/training config pattern (Walk)

Rewards and curriculum for the main task live in `IsaacLab_Tasks/Walk/source/Quadruped/Quadruped/tasks/direct/quadruped/training_phases.yaml`: a `default` block plus `phases.phaseN` entries that `inherits` from a parent phase and deep-merge over it (see `resolve_phase_launcher` in `launcher.py`). Each phase configures `env`, `domain_randomization`, `events`, `rewards`, and `commands`. The yaml only supplies scale factors/thresholds — the actual reward math is in `quadruped_env.py` (`_get_rewards` and the JIT-compiled reward function it calls).
