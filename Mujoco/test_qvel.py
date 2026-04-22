import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_go2/go2.xml")
data = mujoco.MjData(model)

# Set the robot to be pitched 90 degrees around Y axis (facing straight down to floor)
# Euler to quat (pitch = 90 deg)
pitch = np.pi / 2
# w = cos(pi/4), y = sin(pi/4)
w = np.cos(pitch/2)
y = np.sin(pitch/2)
data.qpos[3:7] = [w, 0, y, 0]

# Give world angular velocity of 1.0 around Z axis (turning horizontally in world)
# If qvel is world frame, we should set qvel[5] = 1.0, and locally this would mean rotating around local X axis (roll) because the robot is pitched down.
data.qvel[3:6] = [0, 0, 1.0]

mujoco.mj_forward(model, data)
print("qvel[3:6]:", data.qvel[3:6])
print("cvel (body object) angvel:", data.cvel[1][3:6])

