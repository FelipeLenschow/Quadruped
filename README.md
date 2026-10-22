# Quadruped RL Deployment Framework

A unified suite for training and deploying reinforcement learning policies for Unitree quadruped robots (Go2, Go1, A1) across multiple simulators and physical hardware.

## 🚀 Unified Launcher
Everything starts at `launcher.py`. Run it to access a simplified, professional menu:

- **Train Policy**: High-parallelism reinforcement learning in Isaac Lab.
- **Play Policy**: Rapid evaluation of trained checkpoints in Isaac Lab (Sim2Sim).
- **Play IsaacSim / MuJoCo / Gazebo**: High-fidelity verification using specialized **Drivers** that mirror real-robot firmware behavior.

## 🏗️ Technical Architecture
The framework is built on a **Hardware-Agnostic Core** to ensure zero-gap sim-to-real transfer.

- **Unified Drivers**: Simulation bridges have been refactored into intelligent drivers (`mujoco_driver.py`, `gazebo_driver.py`, `isaac_driver.py`).
- **TelemetryManager**: A centralized state standardizer in `telemetry.py` that converts raw simulator data into a `StandardState` object.
- **CommandProcessor**: A safety-first action pipeline in `policy_bridge.py` that handles hardware-aware scaling and 90% saturation limits for the Go2.

### High-Performance Drivers
Each backend (MuJoCo, Gazebo, Isaac Sim, Real Robot) has a dedicated driver that handles physics and policy inference locally. This bypasses ROS 2 network overhead, achieving sub-millisecond control latency.

## 🧬 Project Structure
- `src/`: the ROS 2 packages (colcon workspace, `colcon build --symlink-install --base-paths src`):
  - `quadruped_common`: Python helpers shared by every node, no nodes: `paths.py`, `config_loader.py`, `realtime.py` (keeps everything off the policy's core), `robot_defaults.py` (default stance), `kinematics.py` (leg kinematics and base rotation helpers, as `quadruped_control/src/kinematics.cpp`). The control decision is `quadruped_control`'s (C++).
  - `quadruped_drivers`: Python nodes: MuJoCo (`mujoco_driver`, `eval_mujoco`) and Gazebo (`gazebo_driver`, `gazebo_sensors`: lidar and camera to ROS), `camera_bridge` (every camera in `config.yaml`'s `cameras:`, real or Gazebo, to `/camera/<name>/...`), and `gps_driver` (the u-blox RTK GPS on USB to `/gps/fix`, `config.yaml`'s `gps:`; the RTK pair is two ZED-F9P with XBee SX 900 MHz radios between their UART2s, and `Tools/ublox_tool.py` shows a receiver's setup, writes the base's (survey-in 0.5 m / 300 s, MSM4 GPS, GLONASS, Galileo, BeiDou on UART2) and checks a rover's corrections).
  - `quadruped_control`: the control core in C++ (estimator, policies, safety, poses), shared by every driver; `driver_node.py` is the simulators' ROS side, `export.py` turns checkpoints into the `.qpol` files it loads.
  - `quadruped_real`: C++ nodes for the real Go2 on the Unitree SDK2: `real_driver` (the control loop alone on its core), `real_sensors` (L1 lidar and IMU to ROS), `go2_camera` (the Go2 camera, for `camera_bridge`), `test_joints`, `mock_go2` (a MuJoCo Go2 to run `real_driver` on a PC), `helios_driver` (the Helios 16P lidar to `/helios/points` for FAST-LIO, on the `third_party/rs_driver` submodule; `--from_topic` replays an old `/rslidar_points` bag), and `lidar_merge` (the Helios and the L1 as one cloud in the base frame, `/lidars/points`, for FAST-LIO on both: launcher [L] choice 3).
  - `quadruped_operator`: console, supervisor, teleops, twins, reward estimator, MCAP tool and `skill_server` (`/skill`).
  - `quadruped_perception`: `lidar_filter` (drops the robot's own body from the lidar cloud), `camera_lidar_check` (lidar points drawn on a camera image, to check the mount).
  - `quadruped_interfaces`: `srv/Skill.srv`.
  - `quadruped_description`: MuJoCo menagerie, Gazebo worlds (`scene`: flat, `nav`: walls, obstacles, 4 cones; `scripts/sdf_to_map.py` makes its Nav2 map), Go2/Go1/A1 models (the Go2 carries an L1-style lidar and a front camera), kinematics yaml, bundled policies.
  - `quadruped_bringup`: `launch/*.launch.py` and `config/` (`config.yaml`, `joy_f710.config.yaml`, `twist_mux.yaml`, `nav2.yaml`, `slam.yaml`).
- `IsaacSim/isaac_driver.py`: stays outside the packages (Isaac Sim's Python 3.12); imports `quadruped_common` from `src/`. Broken for now: it needs `quadruped_control` built for Python 3.12, which nothing builds yet.
- `IsaacLab_Tasks/`: RL task definitions and Isaac Lab configurations. Each subfolder is an independent task package (own source tree, own logs, own `training_phases.yaml`):
  - `Walk/`: the main, actively developed locomotion task (3 robots: Go2/Go1/A1).
  - `Walk_GO2/`: a Go2-only simplification, superseded by going back to `Walk`; likely stale.
  - `Stairs/`: experiment adding a terrain/height-scan sensor for stair climbing.
  - `Handstand/`: handstand task.

## 🛠️ Typical Workflow

1. **Train Policy**: `python launcher.py` -> Select Task -> **[1] Train Policy**.
2. **Play Policy**: Verify logic immediately in Isaac Lab via **[2] Play Policy**.
3. **High-Fidelity Verification**: Use the **MuJoCo** or **Gazebo** drivers to validate physics-dependent behaviors (e.g., foot friction, actuator dynamics).
4. **Deploy**: The same `quadruped_control` C++ controller used in simulation runs on the Jetson Orin on the physical robot.

## ⚠️ Requirements

- **Isaac Sim Environment**: Python 3.12 (for Isaac Lab, training, and MuJoCo drivers). Activate with `source ~/env_isaacsim/bin/activate`.
- **System ROS 2**: Python 3.10 (for Gazebo drivers and monitoring). Activate with `source venv_robot/bin/activate`, or use `Docker/` if the host isn't natively Python 3.10.
- The launcher automatically handles environment switching between `env_isaacsim` and `/usr/bin/python3`, and disables menu options incompatible with the currently active environment.

See `CLAUDE.md` for more on environment setup, the `launcher.py` flow, and how the `IsaacLab_Tasks/` task modules and training curriculum (`training_phases.yaml`) are structured.
