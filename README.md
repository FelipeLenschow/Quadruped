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
  - `quadruped_core`: library, no nodes: `pipeline.py`, `controller/` (policy runner, safety processor, pose generator), `telemetry/` (TelemetryManager, LKF estimator, kinematics), `config_loader.py`, `paths.py`.
  - `quadruped_drivers`: nodes for the real Go2 (`real_driver`, `test_joints`), MuJoCo (`mujoco_driver`, `eval_mujoco`) and Gazebo (`gazebo_driver`).
  - `quadruped_operator`: console, supervisor, teleops, twins, reward estimator, MCAP tool.
  - `quadruped_description`: MuJoCo menagerie, Gazebo world, Go2/Go1/A1 models, kinematics yaml, bundled policies.
  - `quadruped_bringup`: `launch/*.launch.py` and `config/` (`config.yaml`, `joy_f710.config.yaml`).
  - `unitree_sdk2py`: wraps the `third_party/unitree_sdk2_python` submodule.
- `IsaacSim/isaac_driver.py`: stays outside the packages (Isaac Sim's Python 3.12); imports `quadruped_core` from `src/`.
- `IsaacLab_Tasks/`: RL task definitions and Isaac Lab configurations. Each subfolder is an independent task package (own source tree, own logs, own `training_phases.yaml`):
  - `Walk/`: the main, actively developed locomotion task (3 robots: Go2/Go1/A1).
  - `Walk_GO2/`: a Go2-only simplification, superseded by going back to `Walk`; likely stale.
  - `Stairs/`: experiment adding a terrain/height-scan sensor for stair climbing.
  - `Handstand/`: handstand task.

## 🛠️ Typical Workflow

1. **Train Policy**: `python launcher.py` -> Select Task -> **[1] Train Policy**.
2. **Play Policy**: Verify logic immediately in Isaac Lab via **[2] Play Policy**.
3. **High-Fidelity Verification**: Use the **MuJoCo** or **Gazebo** drivers to validate physics-dependent behaviors (e.g., foot friction, actuator dynamics).
4. **Deploy**: The same `quadruped_core` package used in simulation is deployed directly to the Jetson Orin on the physical robot.

## ⚠️ Requirements

- **Isaac Sim Environment**: Python 3.12 (for Isaac Lab, training, and MuJoCo drivers). Activate with `source ~/env_isaacsim/bin/activate`.
- **System ROS 2**: Python 3.10 (for Gazebo drivers and monitoring). Activate with `source venv_robot/bin/activate`, or use `Docker/` if the host isn't natively Python 3.10.
- The launcher automatically handles environment switching between `env_isaacsim` and `/usr/bin/python3`, and disables menu options incompatible with the currently active environment.

See `CLAUDE.md` for more on environment setup, the `launcher.py` flow, and how the `IsaacLab_Tasks/` task modules and training curriculum (`training_phases.yaml`) are structured.
