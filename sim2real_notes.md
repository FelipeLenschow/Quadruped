# Sim2Real Deployment Notes

> [!TIP]
> **Observation Scaling is now AUTOMATIC!**
> You no longer need to apply manual multipliers (`* 2.0`, `* 0.25`, etc.) in your C++ driver. 

We have reverted the policy to use `skrl`'s built-in `RunningStandardScaler` for state preprocessing. This matches the fast-learning reference example. 

When you export your trained policy to ONNX, `skrl` automatically embeds the mean and variance of the `RunningStandardScaler` directly into the ONNX computational graph. 

## C++ Implementation
In your C++ driver (`DLS2`), simply pack the raw, unscaled sensor data directly into the 49-dimensional observation tensor and pass it to the ONNX model. The model will normalize it internally.

```cpp
// Pseudocode for DLS2 State Builder (RAW VALUES ONLY)
for (int i = 0; i < 3; i++) {
    obs_tensor[i] = raw_base_lin_vel[i];         // Linear Vel (m/s)
    obs_tensor[i+3] = raw_base_ang_vel[i];       // Angular Vel (rad/s)
    obs_tensor[i+6] = projected_gravity[i];      // Gravity unit vector
}

obs_tensor[9] = cmd_lin_x;
obs_tensor[10] = cmd_lin_y;
obs_tensor[11] = cmd_ang_z;

for (int i = 0; i < 12; i++) {
    obs_tensor[12 + i] = (raw_joint_pos[i] - default_joint_pos[i]); // Pos error (rad)
    obs_tensor[24 + i] = raw_joint_vel[i];                          // Vel (rad/s)
    obs_tensor[36 + i] = previous_actions[i];                       // Prev Action
}
```
