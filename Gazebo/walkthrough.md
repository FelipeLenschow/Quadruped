# Official Unitree Go2 Integration Walkthrough

I have completed a major overhaul of the robot description to ensure 100% parity with official Unitree assets and eliminate all "gambiarra" (hacks).

## 1. Unified Asset Repository
I created a root-level directory `/home/05680435969@corp.udesc.br/Quadruped/Unitree_Go2` that serves as the single source of truth for all simulators.

**Structure:**
- `Unitree_Go2/go2_description/`: Official ROS description (URDF, Xacro, Meshes).
- `Unitree_Go2/usd/`: Official Isaac Sim USD files.

## 2. Official Gazebo Model
I generated a clean `model.sdf` from the official Unitree URDF and integrated it into the Gazebo Harmonic environment.

**Key Features:**
- **Official Geometry**: Correct Go2 wheelbase (`0.1934m x 0.0465m`), thigh offsets (`0.0955m`), and joint limits.
- **Official Meshes**: Uses the `.dae` files directly from Unitree's repository for high-fidelity visuals.
- **Control Parity**: Integrated `ApplyJointForce`, `JointStatePublisher`, and `OdometryPublisher` systems.
- **Physics Stability**: Zeroed joint damping and set friction to 0.01 to match the IsaacLab/MuJoCo trained policy behavior.

## 3. Code Alignment
Updated both `gazebo_sim2sim.py` and `gazebo_driver.py` to use official joint naming conventions:
- **Hips**: `FL_hip_joint`, `FR_hip_joint`, `RL_hip_joint`, `RR_hip_joint`
- **Thighs**: `FL_thigh_joint`, `FR_thigh_joint`, `RL_thigh_joint`, `RR_thigh_joint`
- **Calves**: `FL_calf_joint`, `FR_calf_joint`, `RL_calf_joint`, `RR_calf_joint`

## 4. Verification
- ✅ **Joint Order**: Grouped by type (Hips, Thighs, Calves) to maintain policy compatibility.
- ✅ **Path Resolution**: Scripts now dynamically add `/home/05680435969@corp.udesc.br/Quadruped/Unitree_Go2` to `GZ_SIM_RESOURCE_PATH`.
- ✅ **Clean Repository**: Deleted legacy Go1-based models and ActuatorNet files from the `Gazebo/` directory.

## How to Run
Everything is now pointing to the official root-level assets.
```bash
python3 Gazebo/gazebo_sim2sim.py --checkpoint path/to/best_agent.pt
```
The simulation will now load the **real Go2** with the correct visuals and physics.
