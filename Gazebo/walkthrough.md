# Gazebo sim2sim and ROS 2 Bridge Walkthrough

I have implemented a high-fidelity Gazebo simulation environment for the Unitree Go1, matching the architecture of the existing MuJoCo setup.

## Changes Made

### 1. ROS 2 Gazebo Bridge
Implemented [ros2_gazebo_bridge.py](file:///home/05680435969@corp.udesc.br/Quadruped/Gazebo/ros2_gazebo_bridge.py) which handles:
- **ActuatorNet Integration**: Converts joint position targets to torques at 200Hz using the same core logic as MuJoCo.
- **Sim-Time Sync**: Synchronizes the control loop with Gazebo's simulation clock.
- **Odometry & Sensors**: Publishes body-frame linear velocity, IMU data, and joint states to ROS 2 topics.

### 2. Standalone Sim2Sim
Implemented [gazebo_sim2sim.py](file:///home/05680435969@corp.udesc.br/Quadruped/Gazebo/gazebo_sim2sim.py) which provides:
- **Direct Control Loop**: Runs the policy and actuator loop within a single script for minimal latency.
- **Gazebo Transport Integration**: Communicates directly with Gazebo Harmonic without requiring the ROS 2 bridge for core control.
- **Interactive Teleop**: Supports keyboard control (WASD) for testing.

### 3. Simulation Assets
Prepared a clean `Gazebo` directory with:
- [scene.sdf](file:///home/05680435969@corp.udesc.br/Quadruped/Gazebo/scene.sdf): Configured for Gazebo Harmonic with physics and sensor plugins.
- [go1.urdf](file:///home/05680435969@corp.udesc.br/Quadruped/Gazebo/go1.urdf): Robot definition optimized for torque control.
- [models/](file:///home/05680435969@corp.udesc.br/Quadruped/Gazebo/models): Local model resources for high-performance loading.

## Verification Results

- **Syntax & Execution**: Verified that both scripts execute and correctly handle environment setup (including ROS 2 Humble sourcing).
- **Parity**: I/O and joint mapping verified to match MuJoCo and Isaac Lab standards.
- **Transport**: Verified connection to Gazebo Transport topics for sensor feedback and torque command.

## How to Run

### Standalone Sim2Sim
```bash
python3 /home/05680435969@corp.udesc.br/Quadruped/Gazebo/gazebo_sim2sim.py --checkpoint /path/to/policy.pt
```

### ROS 2 Bridge (for external policy)
```bash
source /opt/ros/humble/setup.bash
python3 /home/05680435969@corp.udesc.br/Quadruped/Gazebo/ros2_gazebo_bridge.py
```
