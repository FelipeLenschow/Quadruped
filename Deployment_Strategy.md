# Real Robot Deployment Strategy

To deploy your trained policies from Isaac Lab to a real Unitree robot (A1, Go1, or Go2), we need to bridge the gap between the `skrl` model and the real-world controls.

## 1. Export the Model (TorchScript)
Real-world deployment should be lightweight. We will create a script to export the trained `skrl` policy into **TorchScript**. This allows the robot's onboard computer to run inference without needing `skrl`, `IsaacLab`, or complex dependencies.

## 2. Deployment Script (Unitree SDK)
We will implement a Python deployment script that runs on the robot. It will follow the architecture of `mujoco_sim2sim.py` but interface with the real hardware:
- **Input**: Sensor data from the Unitree SDK (IMU, Encoder).
- **Processing**: Build the identical 49-dim observation vector (since we don't have height scans in the real world yet).
- **Inference**: Run the TorchScript model.
- **Output**: Send joint targets (position/torque) back to the Unitree SDK.

## 3. Communication Bridge
Depending on your model, we can use:
- **Unitree Legged SDK (C++/Python)**: For A1 and Go1.
- **Unitree SDK2 (C++/Python)**: For Go2.

## Proposed Implementation Steps

### [NEW] export_policy.py
A utility to convert `best_agent.pt` to a standalone `policy_jit.pt`.

### [NEW] Deployment Hub
Create a `Deployment/` directory containing:
- **`unitree_wrapper.py`**: A class to handle SDK communication.
- **`real_play.py`**: The main execution loop (Sensor -> Obs -> Model -> Act).

---
**Would you like me to start by creating the Export script and a Deployment template?**
