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
    def __init__(self, model, data, mj_to_isaac, isaac_to_mj):
        self.model = model
        self.data = data
        self.mj_to_isaac = mj_to_isaac
        self.isaac_to_mj = isaac_to_mj
        self.num_joints = len(mj_to_isaac)

    def Recv(self):
        """Mock receiving data from the robot (MuJoCo)."""
        state = LowState()
        
        # Base state
        state.base_pos = self.data.qpos[0:3].copy()
        
        # IMU logic (MuJoCo data.qpos[3:7] is base quat [w,x,y,z])
        state.imu.quaternion = self.data.qpos[3:7].copy()
        state.imu.gyroscope = self.data.qvel[3:6].copy()
        
        # Calculate Body Frame Linear Velocity (as the real robot KF would provide)
        R = quat_to_rot_matrix(state.imu.quaternion)
        state.base_lin_vel = R.T @ self.data.qvel[0:3]

        # Motor states
        mj_qpos = self.data.qpos[7:7+self.num_joints]
        mj_qvel = self.data.qvel[6:6+self.num_joints]
        
        for i in range(self.num_joints):
            state.motorState[i].q = mj_qpos[i]
            state.motorState[i].dq = mj_qvel[i]

        return state

    def Send(self, cmd: LowCmd):
        """Mock sending commands to the robot (MuJoCo)."""
        for i in range(self.num_joints):
            # Applying torque directly for now (matching mujoco_sim2sim logic)
            self.data.ctrl[i] = cmd.motorCmd[i].tau
