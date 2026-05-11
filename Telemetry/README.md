# Telemetry & State Estimation

This package handles the standardization of sensor data and the estimation of the robot's physical state (velocity, position, orientation) for the high-level control policy.

## Components

### 1. Telemetry Manager (`telemetry.py`)
Bridges the gap between backend-specific data (MuJoCo, SDK2, Gazebo) and the unified `StandardState` used by the controller.
- **Publishers**: `/sensors/joint_states`, `/sensors/imu`, `/odom`.
- **Standardization**: Reorders joints and converts coordinate frames to match the training environment.
- **Passes** `joint_pos`, `joint_vel`, and `gyro_body` into the state estimator for leg-odometry corrections.

### 2. State Estimator (`estimator.py`)
Implements a **6-state Linear Kalman Filter** estimating body velocity and accelerometer bias.

| Step | Description |
|------|-------------|
| **Predict** | Gravity-compensated IMU integration with bias subtraction: `v += (a_imu - bias + R^T g) * dt` |
| **Correct** | Per-grounded-foot leg-odometry measurement: `z = -J(q) dq - ω × r_foot` |
| **Bias** | Accelerometer bias co-estimated and cancelled each step |

State: `x = [vx, vy, vz, bax, bay, baz]`

### 3. Forward Kinematics (`kinematics.py`)
Pure-NumPy FK and Jacobians for the Unitree Go2, geometry from `Configs/go2_kinematics.yaml`.
- **`foot_position_body(leg_idx, q_leg)`**: foot position in body frame.
- **`foot_jacobian_body(leg_idx, q_leg)`**: 3×3 geometric Jacobian `J` such that `v_foot ≈ J @ dq_leg`.
- Passes a numerical vs. analytical Jacobian self-check (error < 1e-9).

### 4. Kinematics Config (`Configs/go2_kinematics.yaml`)
Go2 robot geometry extracted from the official MuJoCo model (`go2.xml`):
- Hip origins, thigh/calf link lengths, foot X-offset.

## Topics Published

| Topic | Type | Description |
|-------|------|-------------|
| `/sensors/joint_states` | `JointState` | Raw joint positions and velocities |
| `/sensors/imu` | `Imu` | Raw IMU (quaternion, gyro, accel) |
| `/odom` | `Odometry` | Base velocity (estimated or GT) |
| `/estimator/projected_gravity` | `Vector3` | Gravity in body frame — used by Supervisor |
| `/estimator/base_lin_vel` | `Vector3` | Estimated/GT body velocity |
| `/estimator/base_ang_vel` | `Vector3` | Gyroscope angular velocity |
| `/estimator/base_height` | `Float32` | Z position |
| `/estimator/feet_contact` | `Float32MultiArray` | Binary contact flags [FL, FR, RL, RR] |
| `/estimator/leg_odometry` | `Float32MultiArray` | Per-leg FK velocity norms (debug) |

## Configuration (`Configs/config.yaml`)

```yaml
state_estimator:
  use_estimator: false       # Enable LKF (false = pass-through GT velocity)
  dt: 0.02
  kalman:
    process_noise_vel:  0.01   # Q: trust IMU prediction less → more FK weight
    process_noise_bias: 0.001  # Q: bias random-walk rate
    measurement_noise:  0.05   # R: trust FK less → smoother but slower
```

## Tuning Guide

- **Increase `process_noise_vel`**: faster response to real acceleration, more noise.
- **Decrease `measurement_noise`**: trust leg odometry more, penalizes slip events.
- **Decrease `process_noise_bias`**: bias estimated more slowly (use for stable IMUs).

## Roadmap

- [x] Contact-aided linear velocity estimator (removed — replaced by LKF)
- [x] Forward kinematics (`kinematics.py`) + geometric Jacobians
- [x] Leg odometry velocity estimate from no-slip constraint
- [x] Linear Kalman Filter with accelerometer bias estimation
- [ ] Dynamic contact probability from FSR force data (soft weighting)
- [ ] Terrain inclination estimation from foot positions
- [ ] EKF upgrade for nonlinear bias/orientation coupling
