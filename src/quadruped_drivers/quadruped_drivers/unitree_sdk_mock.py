import numpy as np

def quat_to_rot_matrix(q):
    """(w, x, y, z) -> [3,3] matrix"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])

class MotorState:
    def __init__(self):
        self.q = 0.0         # position
        self.dq = 0.0        # velocity
        self.tauEst = 0.0    # estimated torque

class IMU:
    def __init__(self):
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.gyroscope = np.array([0.0, 0.0, 0.0])
        self.accelerometer = np.array([0.0, 0.0, 0.0])
        self.rpy = np.array([0.0, 0.0, 0.0])

class LowState:
    def __init__(self):
        self.motorState = [MotorState() for _ in range(20)]
        self.imu = IMU()
        # Non-standard additions for sim2sim abstraction
        self.base_pos = np.array([0.0, 0.0, 0.0])
        self.base_lin_vel = np.array([0.0, 0.0, 0.0]) # Linear velocity in body frame

class MotorCmd:
    def __init__(self):
        self.q = 0.0
        self.dq = 0.0
        self.kp = 0.0
        self.kd = 0.0
        self.tau = 0.0

class LowCmd:
    def __init__(self):
        self.motorCmd = [MotorCmd() for _ in range(20)]

class MockUDP:
    """Mimics the UDP communication in unitree_legged_sdk."""
    def __init__(self, model, data, mapping: dict):
        self.model = model
        self.data = data
        self.mapping = mapping

    def Recv(self):
        """Mock receiving data from the robot (MuJoCo)."""
        state = LowState()
        
        # Base state
        state.base_pos = self.data.qpos[0:3].copy()
        
        # IMU logic (MuJoCo data.qpos[3:7] is base quat [w,x,y,z])
        state.imu.quaternion = self.data.qpos[3:7].copy()
        
        # Calculate Rotation Matrix for Body Frame conversion
        R = quat_to_rot_matrix(state.imu.quaternion)
        
        # Angular Velocity (Rotate World Frame qvel[3:6] to Body Frame)
        state.imu.gyroscope = R.T @ self.data.qvel[3:6]
        
        # Calculate Body Frame Linear Velocity (as the real robot KF would provide)
        state.base_lin_vel = R.T @ self.data.qvel[0:3]

        # Motor states (Isaac Order)
        for i in range(12):
            q_addr = self.mapping["qpos_addr"][i]
            v_addr = self.mapping["qvel_addr"][i]
            state.motorState[i].q = self.data.qpos[q_addr]
            state.motorState[i].dq = self.data.qvel[v_addr]

        return state

    def Send(self, cmd: LowCmd):
        """Mock sending commands to the robot (MuJoCo)."""
        for i in range(12):
            act_idx = self.mapping["ctrl_idx"][i]
            if act_idx != -1:
                # Applying torque directly (matching mujoco_sim2sim logic)
                self.data.ctrl[act_idx] = cmd.motorCmd[i].tau
