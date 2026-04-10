# Quadruped RL Deployment Framework

A unified suite for training and deploying reinforcement learning policies for Unitree quadruped robots across multiple simulators and physical hardware.

## 🚀 Unified Launcher
Everything starts at `launcher.py`. Run it to access:
- **Isaac Lab**: Training and high-parallelism evaluation.
- **MuJoCo**: High-fidelity physics validation (Sim-to-Real).
- **Gazebo Sim (Harmonic)**: Standalone ROS-compatible simulation.
- **Physical Hardware**: Direct deployment to Unitree Go1/A1.

## 🏗️ Project Structure

- `IsaacLab_Tasks/`: Contains task definitions (Walk, Stairs, Handstand, etc.). The launcher automatically detects new tasks here.
- `Mujoco/`: MuJoCo simulation bridge and robot model (using `unitree_sdk_mock`).
- `Gazebo/`: Standalone Gazebo Harmonic SDF assets and the Python bridge (`gazebo_sim2sim.py`).
- `Deployment/`: The core of the Sim-to-Real pipeline.
  - `policy_runner.py`: Modular inference engine shared by all platforms.
  - `export_jit.py`: Tool to convert PyTorch checkpoints into optimized **TorchScript** models.
  - `real_deploy.py`: Final script for the robot's onboard computer.

## 🧬 Multi-Python Architecture (Gazebo)
> [!NOTE]
> Gazebo Harmonic mode uses a hybrid Python approach.

Because the official **Gazebo Python bindings (`gz-transport`)** are built for the system's default Python version (usually **3.10** on Ubuntu 22.04), they are incompatible with the `env_isaacsim` environment which uses **Python 3.11**. 

The launcher automatically handles this by:
1. Executing Isaac Lab and MuJoCo scripts using the `env_isaacsim` (3.11) interpreter.
2. Executing Gazebo bridge scripts using the system `/usr/bin/python3` (3.10) interpreter.

To support this, lightweight versions of `torch` and `numpy` should be installed for the system Python:
```bash
python3 -m pip install --user torch numpy
```

## 🛠️ Typical Workflow
1. **Train** in Isaac Lab: `python launcher.py` -> Select Task -> Train.
2. **Export** to JIT: 
   ```bash
   python Deployment/export_jit.py --checkpoint path/to/best_agent.pt --out Deployment/policy_jit.pt
   ```
3. **Verify** in MuJoCo or Gazebo via the launcher.
4. **Deploy** to a physical robot using the `Deployment/` folder.
