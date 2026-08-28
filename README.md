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
- `Controller/`: The brain of the robot.

  - `policy_runner.py`: The cross-platform inference engine.
  - `policy_bridge.py`: Contains the `CommandProcessor` for safety and scaling.
  - `Utils/telemetry.py`: Standardizes data from any source (Sim or Real).
- `IsaacLab_Tasks/`: RL task definitions and Isaac Lab configurations. Each subfolder is an independent task package (own source tree, own logs, own `training_phases.yaml`):
  - `Walk/`: the main, actively developed locomotion task (3 robots: Go2/Go1/A1).
  - `Walk_GO2/`: a Go2-only simplification, superseded by going back to `Walk`; likely stale.
  - `Stairs/`: experiment adding a terrain/height-scan sensor for stair climbing.
  - `Handstand/`: handstand task.
- `Mujoco/`, `Gazebo/`, `IsaacSim/`: Simulator-specific drivers and assets.

## 🛠️ Typical Workflow

1. **Train Policy**: `python launcher.py` -> Select Task -> **[1] Train Policy**.
2. **Play Policy**: Verify logic immediately in Isaac Lab via **[2] Play Policy**.
3. **High-Fidelity Verification**: Use the **MuJoCo** or **Gazebo** drivers to validate physics-dependent behaviors (e.g., foot friction, actuator dynamics).
4. **Deploy**: The same `Controller/` module used in simulation is deployed directly to the Jetson Orin on the physical robot.

## ⚠️ Requirements

- **Isaac Sim Environment**: Python 3.11 (for Isaac Lab, training, and MuJoCo drivers). Activate with `source ~/env_isaacsim/bin/activate`.
- **System ROS 2**: Python 3.10 (for Gazebo drivers and monitoring). Activate with `source venv_robot/bin/activate`, or use `Docker/` if the host isn't natively Python 3.10.
- The launcher automatically handles environment switching between `env_isaacsim` and `/usr/bin/python3`, and disables menu options incompatible with the currently active environment.

See `CLAUDE.md` for more on environment setup, the `launcher.py` flow, and how the `IsaacLab_Tasks/` task modules and training curriculum (`training_phases.yaml`) are structured.
