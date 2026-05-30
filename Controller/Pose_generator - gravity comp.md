# Future Plan: Pose Generator Gravity Compensation

## Overview
This document outlines the planned architectural upgrades to the `PoseGenerator` and `LocomotionPipeline` to support **Gravity Compensation** using the Pinocchio rigid-body dynamics library.

Currently, the `PoseGenerator` relies entirely on a high-gain PD loop to force the robot into a desired stance. By implementing whole-body gravity compensation, we can calculate the feedforward torques (`tau_ff`) needed to support the mass of the robot's trunk and legs dynamically. This allows the robot to stand, sit, or pose compliantly with much softer PD gains, increasing safety and stability.

---

## 1. Theoretical Approach: Link-Mass vs. Whole-Body Compensation

### Leg-Only (Link-Mass) Compensation
*   **Concept**: Compensates only for the physical mass of the individual leg segments (thigh, calf, foot) acting under gravity.
*   **Limitation**: Assumes the base is fixed or flat. It completely ignores the heavy trunk mass. If used while standing, the robot would immediately collapse under its own weight unless fought by extremely stiff `Kp` gains.
*   **Use Case**: Ideal for swing-leg tracking in mid-air (e.g., foot trajectory tracking during locomotion), but insufficient for static posing on the ground.

### Whole-Body Gravity Compensation (Floating Base)
*   **Concept**: Treats the robot as an interconnected "Free-Flyer" system in 3D space (19 DoF: 7 for base, 12 for joints).
*   **Implementation**: 
    1.  Uses IMU data (quaternions) to determine the true gravity vector relative to the robot's body.
    2.  Computes the **Ground Reaction Forces (GRF)** needed to support the trunk's mass (e.g., $Mg/4$ per leg).
    3.  Maps those forces to joint torques using the leg Jacobian matrices ($\tau = J^T F$).
*   **Use Case**: Allows the robot to stand and pose compliantly with soft `Kp` gains, as the feedforward torques carry the majority of the trunk's weight. 

### The Hybrid Approach (Standard Whole-Body Control)
*   **Concept**: A dynamic blend of both methods based on real-time foot contact sensors. This is the industry standard for quadruped locomotion.
*   **Implementation**:
    1.  **Always apply Link-Mass Gravity**: Run Pinocchio's `computeGeneralizedGravity` every step and apply this baseline torque to **all** 12 joints. This ensures that any leg lifted in the air is perfectly weightless and tracks smoothly.
    2.  **Add Contact Forces conditionally**: Read the foot force sensors (`state.contact`). 
    3.  **Dynamic Force Distribution (CoM)**: For feet *in contact* with the ground, you do NOT simply divide the weight equally. Because the robot's Center of Mass (CoM) shifts continuously, the weight distribution is uneven. 
        *   You compute the robot's CoM position using Pinocchio.
        *   You use a simple optimization solver (like a Quadratic Program / QP) or a pseudo-inverse math operation to calculate the exact Ground Reaction Force ($F_{contact}$) needed at each foot to keep the robot balanced. A foot closer to the CoM will automatically be assigned a higher force than a foot further away.
    4.  Map that precise force through the Jacobian ($\tau = J^T F$) and **add** it to the baseline gravity torque for that leg.
*   **Result**: 
    - **Swing Leg (Contact = 0)**: $\tau_{ff} = \tau_{leg\_gravity}$ (Perfect tracking, weightless feel).
    - **Stance Leg (Contact = 1)**: $\tau_{ff} = \tau_{leg\_gravity} + J^T F$ (Supports the heavy body, compliant stance).
    - **This hybrid model is the target approach for the Pose Generator and future locomotion pipelines.**

---

## 2. Implementation Roadmap

### Phase 1: Dependencies and Model Loading
1.  **Pinocchio Integration**: Add `pinocchio` to the project dependencies (`pip install pinocchio`).
2.  **URDF Provision**: Ensure the standard Go2 URDF is available in the `Configs` or `Mujoco` directories (Pinocchio cannot parse MJCF XMLs for full dynamic models).
3.  **Initialization**: Update `Controller/pose_generator.py` to instantiate the model:
    ```python
    import pinocchio as pin
    self.model = pin.buildModelFromUrdf("path/to/go2.urdf", pin.JointModelFreeFlyer())
    self.data = self.model.createData()
    ```

### Phase 2: Compute Feedforward Torques in `PoseGenerator`
Modify the `step` method in `PoseGenerator` to output both position targets and feedforward torques.
1.  Construct the 19-dimensional generalized coordinate vector `q_full`:
    *   **Base Pos**: `[0,0,0]` or `state.pos`
    *   **Base Quat**: `state.quat` (convert from `[w,x,y,z]` telemetry format to Pinocchio's expected `[x,y,z,w]`)
    *   **Joint Pos**: `state.q`
2.  Compute Jacobians and gravity forces.
3.  Calculate the Ground Reaction Forces (GRF) for feet in contact.
4.  Map GRF to joint torques: $\tau = J^T F + \tau_{leg\_gravity}$.
5.  Return a structured command: `{"q_des": target_positions, "tau_ff": computed_torques}`.

### Phase 3: Update `LocomotionPipeline` & `PolicyManager`
Currently, the pipeline only routes position targets (`latest_targets`). 
1.  **`PolicyManager`**: Update `step_single` to return the new dictionary format. If the active neural network policy doesn't output feedforward torques, default to `tau_ff = np.zeros(12)`.
2.  **`LocomotionPipeline`**: Update `step` to accept the `tau_ff` array and pass it to the `Distributor` and drivers.
3.  **`Distributor`**: Update the ROS 2 publication logic (if desired) to log `tau_ff`. (e.g., packing it into the `effort` field of the `JointState` message, or creating a custom `MotorCommand.msg`).

### Phase 4: Apply Torques in Low-Level Drivers
The drivers must be updated to apply the feedforward torques in their PD loops, rather than defaulting to `0.0`.

*   **`Unitree/real_driver.py`**:
    ```python
    # In send_to_sdk()
    self.low_cmd.motor_cmd[i].q = float(joint_targets[ros_idx])
    self.low_cmd.motor_cmd[i].tau = float(tau_ff[ros_idx]) # Inject feedforward torque
    ```

*   **`Mujoco/mujoco_driver.py`**:
    ```python
    # In _pd_torques()
    torques = kp * pos_err + kd * (0 - v) + tau_ff
    return np.clip(torques, min_torque, max_torque)
    ```

---

## 4. Future Enhancements (Post-Implementation)
*   **Dynamic Force Distribution**: Adjust GRF mapping dynamically as the robot shifts its center of mass (e.g., during a pushup or sit motion) using a QP solver.
*   **Friction Compensation**: Integrate the newly developed Neural Network actuator friction compensation alongside the gravity compensation to achieve near-zero perceived impedance in the legs.

---

## 5. Beyond Gravity: Full Inertia & Coriolis Compensation
While Gravity Compensation is sufficient for holding static poses or moving slowly, fast or highly dynamic movements require compensating for the robot's **Inertia (Mass Matrix)** and **Coriolis/Centrifugal forces**.

The full rigid body dynamics equation is:
$$ M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) = \tau $$

*   **Gravity Comp ($G$)**: Assumes velocity ($\dot{q}$) and acceleration ($\ddot{q}$) are zero. 
*   **Full Inverse Dynamics**: If your `PoseGenerator` produces not just target positions, but also target velocities ($\dot{q}_{des}$) and target accelerations ($\ddot{q}_{des}$), you can compensate for inertia.

**Implementation with Pinocchio**:
Instead of `pin.computeGeneralizedGravity`, you use the **Recursive Newton-Euler Algorithm (RNEA)**:
```python
# Compute full feedforward torque including Inertia, Coriolis, and Gravity
tau_ff = pin.rnea(self.model, self.data, q_full, dq_full, ddq_full)
```
If you pass $\dot{q} = 0$ and $\ddot{q} = 0$ into `rnea`, the output is exactly equal to the gravity compensation vector! By adding the desired accelerations into the math, the feedforward torque will actively "push" the heavy leg segments to accelerate them, resulting in incredibly snappy and precise dynamic tracking.
