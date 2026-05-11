# Telemetry & State Estimation

This package handles the standardization of sensor data and the estimation of the robot's physical state (velocity, position, orientation) for the high-level control policy.

## Components

### 1. Telemetry Manager (`telemetry.py`)
Bridges the gap between backend-specific data (MuJoCo, SDK2, Gazebo) and the unified `StandardState` used by the controller.
- **Publishers**: `/sensors/joint_states`, `/sensors/imu`, `/odom`.
- **Standardization**: Reorders joints and converts coordinate frames to match the training environment.

### 2. State Estimator (`estimator.py`)
Currently implements a **Contact-Aided Linear Velocity Estimator**.
- **Gravity Compensation**: $a_{lin} = f_{imu} + R^T g_{world}$.
- **Integration**: $v += a_{lin} \cdot dt$.
- **Contact Decay**: Velocity is decayed toward zero based on the number of feet in contact (no-slip assumption).

## Roadmap for Improvements

To improve the reliability of sim-to-real locomotion, the following upgrades are planned:

### Leg Odometry Integration
Use forward kinematics to calculate velocity directly from joint encoders:
$v_{body} = - (R \cdot J_i \cdot \dot{q}_i + \omega \times r_i)$.
This provides an absolute reference that eliminates integration drift.

### Linear Kalman Filter (LKF)
Transition from simple decay logic to a Kalman Filter fusing:
- **Prediction**: IMU-based acceleration integration.
- **Correction**: Leg odometry and height estimation.
- **Bias Estimation**: Explicitly track and compensate for accelerometer bias.

### Dynamic Contact Probability
Use FSR (foot force) data to calculate a soft contact probability, allowing the estimator to trust leg odometry less during impacts or slips.

### Terrain Inclination
Use foot positions to estimate the local ground slope for better navigation in complex environments.
