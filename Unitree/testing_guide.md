# Safe Deployment Guide: Unitree Go2

Following these steps ensures that you can test the new `real_driver.py` while minimizing risk to the hardware.

## Phase 1: Physical Preparation
1. **Safety Harness**: Support the robot with a gantry or a sturdy harness. The legs should be able to move freely without hitting the floor initially.
2. **Clear Area**: Ensure a 2-meter radius around the robot is clear of obstacles and people.
3. **Remote Control**: Keep the Unitree physical remote or the mobile app open and ready to trigger an **Emergency Stop**.

## Phase 2: Software Setup
1. **Source the Environment**:
   ```bash
   source /opt/ros/humble/setup.bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/Unitree/unitree_sdk2_python
   ```
2. **Disable Sport Mode**: Use the Unitree App to **Disable High-Level Motion** (Sport Service). This ensures our low-level commands don't conflict with the internal controller.
3. **Verify Config**: Check [config.yaml](file:///home/05680435969@corp.udesc.br/Quadruped/Controller/config/config.yaml) to ensure `saturation_limit` is set to `0.9` (default for safety).

## Phase 3: Graduated Testing Sequence

### Step 1: Telemetry-Only Run
Launch the driver **without** a policy first to verify data flow.
```bash
python3 Unitree/real_driver.py --robot=go2
```
- **Check**: Open a new terminal and run `ros2 topic echo /sensors/joint_states`.
- **Verify**: Rotate the robot slightly; ensure the IMU and joint angles update correctly in the telemetry.

### Step 2: Policy Dry-Run (On Harness)
Launch with a trained checkpoint while the robot is suspended.
```bash
python3 launcher.py
# Select [6] Deploy to Robot
```
- **Observe**: The legs should move to the "start" position and begin small oscillations (idling).
- **Control**: Start the **Remote Teleop** (Option [7] in launcher) and give small forward/backward commands. Verify the leg phasing looks correct.

### Step 3: First Ground Contact
1. Lower the robot until its feet just touch the ground while still supported by the harness.
2. Start the driver and policy.
3. If the robot stabilizes, gradually release the tension on the harness.
4. **Immediate E-Stop** if:
   - High-frequency vibration (oscillation) occurs.
   - The robot "leaps" unexpectedly.
   - Any leg reaches a physical limit and stays there.

## Phase 4: Full Deployment
Once stabilized on the ground, use the **Remote Teleop** to verify walking, turning, and side-stepping.

> [!WARNING]
> **Battery**: Low-level control consumes more power than the optimized high-level sport mode. Keep an eye on battery voltage in the telemetry `/sensors/joint_states`.
