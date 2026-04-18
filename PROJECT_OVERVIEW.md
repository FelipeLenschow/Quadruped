# Project Overview: High-Performance Quadruped Locomotion

This document serves as the ground truth for agents and developers working on the Unitree Go2 Locomotion project.

## Core Philosophy: Unified Driver Architecture
The system utilizes a **decentralized, high-speed control architecture** designed to eliminate ROS 2 communication latency (typically 5-10ms) from the critical control loop.

- **Internal Inference**: Each Driver (Sim or Real) locally instantiates a `PolicyRunner` and `CommandProcessor`.
- **Latency Target**: < 1ms internal jitter.
- **Symmetric Code Path**: The logic used in MuJoCo, Gazebo, and the real robot is bit-perfect. If it works in the simulator, it works on the real robot.

## Project Structure
- `launcher.py`: High-level entry point. Handles environment variables and process orchestration.
- `Controller/`: The core brain.
  - `policy_runner.py`: Handles ONNX/PyTorch inference.
  - `policy_bridge.py`: Contains the `CommandProcessor` for hardware safety.
  - `config/config.yaml`: Static configuration for the robot.
  - `Utils/telemetry.py`: Standardizes disparate sensor data into a `StandardState`.
- `Unitree/`: Hardware-specific SDKs and drivers for the physical robot.
- `Mujoco/`, `Gazebo/`, `IsaacSim/`: Simulator-specific drivers (autonomous nodes).

## Key Architectural Patterns

### 1. State Standardization
All sensor inputs MUST pass through the `TelemetryManager.standardize()` method. This ensures that the policy always sees:
- **Joint Order**: Type-Grouped (FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, ...).
- **Coordinate Frames**: Body-frame for IMU, world-frame or body-frame for linear velocity depending on training.

### 2. Command Processing (Safety)
All actions output by the policy MUST pass through the `CommandProcessor`.
- **Saturation**: Clamped to 90% (by default) of physical joint limits.
- **Scaling**: Actions are scaled by `action_scale` (typically 0.25) before being added to prime joint positions.

### 3. "Hybrid ROS" Monitoring
While the control loop is internal (non-ROS), the system publishes telemetry to ROS 2 topics for monitoring:
- `/sensors/joint_states`
- `/sensors/imu`
- `/odom`
Internal drivers subscribe to `/cmd_vel` for steering and high-level commands.

## Important Constants (Unitree Go2)
- **Control Frequency**: 50Hz (Policy), 200-500Hz (Actuator Loops).
- **Communication**: CycloneDDS (SDK2).
- **Hardware Link**: 192.168.123.161 (Robot default).

## For Future AI Agents
When adding a new feature:
1. **Don't add ROS Subscribers for joint commands**. Control must stay internal to the Driver.
2. **Always update `config.yaml`** instead of hardcoding limits.
3. **Use the `StandardState`** object for any new observation logic to maintain backend-agnosticism.
