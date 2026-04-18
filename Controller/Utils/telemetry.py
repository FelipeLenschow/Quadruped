import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3
import numpy as np

class StandardState:
    """Standardized state object used as input for the PolicyRunner."""
    def __init__(self):
        self.imu = type('obj', (object,), {
            'quaternion': [1.0, 0.0, 0.0, 0.0],
            'gyroscope': [0.0, 0.0, 0.0]
        })
        self.base_lin_vel = [0.0, 0.0, 0.0]
        self.base_pos = [0.0, 0.0, 0.5]
        self.motorState = [type('obj', (object,), {'q': 0.0, 'dq': 0.0}) for _ in range(12)]

class TelemetryManager:
    """
    Centralized component to handle ROS 2 telemetry and state standardization.
    Bridges the gap between backend-specific data and the unified Policy Runner.
    """
    def __init__(self, node: Node, joint_names: list = None):
        self.node = node
        self.joint_names = joint_names or [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ]
        
        # 1. Publishers
        self.joint_pub = self.node.create_publisher(JointState, '/sensors/joint_states', 10)
        self.imu_pub   = self.node.create_publisher(Imu, '/sensors/imu', 10)
        self.odom_pub  = self.node.create_publisher(Odometry, '/odom', 10)

    def standardize(self, raw_data, backend="generic", **kwargs):
        """Standardizes raw data from various backends into a StandardState."""
        if backend == "mujoco":
            return self.parse_mujoco(raw_data, **kwargs)
        elif backend == "isaac":
            return self.parse_isaac(raw_data, **kwargs)
        else:
            # For SDK2, we need to extract the data first. 
            # In real_driver.py we pass the LowState object.
            # SDK2 LowState format (approximate based on examples):
            # q, dq = extracted from msg.motor_state[i].q
            if hasattr(raw_data, 'motor_state'):
                q = [float(raw_data.motor_state[i].q) for i in range(12)]
                dq = [float(raw_data.motor_state[i].dq) for i in range(12)]
                quat = raw_data.imu_state.quaternion
                ang_vel = raw_data.imu_state.gyroscope
                # SDK2 LowState doesn't always provide body velocity directly in the same spot, 
                # but we'll use zeros or kwargs if available.
                lin_vel = [0.0, 0.0, 0.0]
                return self.parse_bridge_data(q, dq, quat, ang_vel, lin_vel)
            
            return self.parse_bridge_data(**kwargs)

    def publish(self, sim_time, state: StandardState):
        """Publishes the standardized state to ROS 2 topics."""
        msg_time = rclpy.time.Time(seconds=sim_time).to_msg()
        
        # 1. Joint States
        js = JointState()
        js.header.stamp = msg_time
        js.name = self.joint_names
        js.position = [float(m.q) for m in state.motorState]
        js.velocity = [float(m.dq) for m in state.motorState]
        self.joint_pub.publish(js)
        
        # 2. IMU
        imu = Imu()
        imu.header.stamp = msg_time
        imu.header.frame_id = 'imu_link'
        q = state.imu.quaternion
        imu.orientation = Quaternion(w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3]))
        gv = state.imu.gyroscope
        imu.angular_velocity = Vector3(x=float(gv[0]), y=float(gv[1]), z=float(gv[2]))
        self.imu_pub.publish(imu)
        
        # 3. Odometry
        odom = Odometry()
        odom.header.stamp = msg_time
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base'
        lv = state.base_lin_vel
        odom.twist.twist.linear = Vector3(x=float(lv[0]), y=float(lv[1]), z=float(lv[2]))
        self.odom_pub.publish(odom)

    # --- Backend Specific Parsers ---

    def parse_mujoco(self, data, qpos_addr, qvel_addr):
        """Converts raw MuJoCo data into StandardState."""
        state = StandardState()
        state.imu.quaternion = data.qpos[3:7].tolist()
        state.imu.gyroscope = data.qvel[3:6].tolist()
        state.base_pos = data.qpos[:3].tolist()
        
        # Body frame velocity rotation
        w, x, y, z = state.imu.quaternion
        R = np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x**2-2*z**2, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x**2-2*y**2]
        ])
        state.base_lin_vel = (R.T @ data.qvel[:3]).tolist()
        
        for i, (p_addr, v_addr) in enumerate(zip(qpos_addr, qvel_addr)):
            state.motorState[i].q = data.qpos[p_addr]
            state.motorState[i].dq = data.qvel[v_addr]
        return state

    def parse_isaac(self, robot_data, mapped_idx):
        """Converts Isaac Sim ArticulationData into StandardState."""
        state = StandardState()
        state.imu.quaternion = robot_data.root_quat_w[0].tolist()
        state.imu.gyroscope = robot_data.root_ang_vel_b[0].tolist()
        state.base_lin_vel = robot_data.root_lin_vel_b[0].tolist()
        state.base_pos = robot_data.root_pos_w[0].tolist()
        
        for i, idx in enumerate(mapped_idx):
            state.motorState[i].q = robot_data.joint_pos[0, idx].item()
            state.motorState[i].dq = robot_data.joint_vel[0, idx].item()
        return state

    def parse_bridge_data(self, q, dq, quat, ang_vel, lin_vel_b, base_pos=None):
        """Generic parser for bridges that already have processed attributes (like Gazebo)."""
        state = StandardState()
        state.imu.quaternion = quat.tolist() if hasattr(quat, 'tolist') else list(quat)
        state.imu.gyroscope = ang_vel.tolist() if hasattr(ang_vel, 'tolist') else list(ang_vel)
        state.base_lin_vel = lin_vel_b.tolist() if hasattr(lin_vel_b, 'tolist') else list(lin_vel_b)
        if base_pos is not None:
             state.base_pos = base_pos.tolist() if hasattr(base_pos, 'tolist') else list(base_pos)
             
        for i in range(12):
            state.motorState[i].q = q[i]
            state.motorState[i].dq = dq[i]
        return state
