import rclpy
import os
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3
import numpy as np
from .estimator import StateEstimator, rot_from_quat
from Configs.config_loader import load_config


# ---------------------------------------------------------------------------
# Standard State
# ---------------------------------------------------------------------------

class StandardState:
    """Standardized state object used as input for the PolicyRunner."""
    def __init__(self):
        self.imu = type('obj', (object,), {
            'quaternion':    [1.0, 0.0, 0.0, 0.0],
            'gyroscope':     [0.0, 0.0, 0.0],
            'accelerometer': [0.0, 0.0, 9.81],   # body-frame specific force
        })
        self.base_lin_vel  = [0.0, 0.0, 0.0]
        self.base_pos      = [0.0, 0.0, 0.5]
        self.feet_contact  = [0.0, 0.0, 0.0, 0.0]  # FL, FR, RL, RR binary
        self.motorState    = [
            type('obj', (object,), {'q': 0.0, 'dq': 0.0}) for _ in range(12)
        ]


# ---------------------------------------------------------------------------
# Telemetry Manager
# ---------------------------------------------------------------------------

class TelemetryManager:
    """
    Centralized component to handle ROS 2 telemetry and state standardization.
    Bridges the gap between backend-specific data and the unified Policy Runner.
    
    This class is now simulator-agnostic and only deals with standardized 
    numpy/primitive types.
    """

    def __init__(self, node: Node, joint_names: list = None, estimator_dt: float = 0.02,
                 use_estimator: bool = None):
        self.node = node
        
        # Load centralized configuration
        self.config = load_config()
        est_cfg = self.config.get("state_estimator", {})
        
        # Priority: Constructor arg > ENV variable > YAML config > Default (False)
        if use_estimator is not None:
            self.use_estimator = use_estimator
        else:
            env_val = os.environ.get("USE_ESTIMATOR")
            if env_val is not None:
                self.use_estimator = (env_val == "1")
            else:
                self.use_estimator = est_cfg.get("use_estimator", False)

        self.joint_names = joint_names or [
            "FL_hip_joint",  "FR_hip_joint",  "RL_hip_joint",  "RR_hip_joint",
            "FL_thigh_joint","FR_thigh_joint","RL_thigh_joint","RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        ]

        # State estimator setup
        dt = est_cfg.get("dt", estimator_dt)
        decay_cfg = est_cfg.get("decay", {})
        decay_dict = None
        if decay_cfg:
            # Map YAML keys to foot counts
            decay_dict = {
                0: decay_cfg.get("air", 0.999),
                2: decay_cfg.get("trot", 0.980),
                4: decay_cfg.get("standing", 0.900)
            }
            
        self.estimator = StateEstimator(dt=dt, decay_dict=decay_dict)

        # ROS 2 Publishers
        self.joint_pub = self.node.create_publisher(JointState, '/sensors/joint_states', 10)
        self.imu_pub   = self.node.create_publisher(Imu,        '/sensors/imu',          10)
        self.odom_pub  = self.node.create_publisher(Odometry,   '/odom',                 10)

    # ------------------------------------------------------------------
    def process_state(self, q, dq, quat, gyro, accel=None, pos=None, vel=None, contact=None):
        """
        Creates a StandardState from raw vectors and applies estimation if enabled.
        
        Args:
            q, dq: 12-dim joint positions and velocities.
            quat: [w, x, y, z] orientation.
            gyro: [wx, wy, wz] angular velocity in body frame.
            accel: [ax, ay, az] specific force in body frame (default [0,0,9.81]).
            pos: [x, y, z] global position (optional).
            vel: [vx, vy, vz] body-frame linear velocity (ground truth).
            contact: [FL, FR, RL, RR] binary contacts (optional).
        """
        state = StandardState()
        
        # 1. Populate basic IMU and Joints
        state.imu.quaternion = quat.tolist() if hasattr(quat, 'tolist') else list(quat)
        state.imu.gyroscope = gyro.tolist() if hasattr(gyro, 'tolist') else list(gyro)
        
        if accel is not None:
            state.imu.accelerometer = accel.tolist() if hasattr(accel, 'tolist') else list(accel)
            
        for i in range(12):
            state.motorState[i].q = q[i]
            state.motorState[i].dq = dq[i]
            
        if pos is not None:
            state.base_pos = pos.tolist() if hasattr(pos, 'tolist') else list(pos)
            
        if contact is not None:
            state.feet_contact = contact.tolist() if hasattr(contact, 'tolist') else list(contact)

        # 2. Velocity logic
        # If ground truth velocity is provided, we use it by default
        if vel is not None:
            state.base_lin_vel = vel.tolist() if hasattr(vel, 'tolist') else list(vel)

        # 3. Estimator override
        if self.use_estimator:
            # We need quat, accel, and contact for the estimator.
            # If accel wasn't provided, StandardState defaults to [0,0,9.81] (gravity compensation only).
            v_est = self.estimator.update(state.imu.quaternion, 
                                          state.imu.accelerometer, 
                                          state.feet_contact)
            state.base_lin_vel = v_est.tolist()
            
        return state

    # ------------------------------------------------------------------
    def publish(self, sim_time, state: StandardState):
        """Publishes the standardized state to ROS 2 topics."""
        msg_time = rclpy.time.Time(seconds=sim_time).to_msg()

        # 1. Joint States
        js = JointState()
        js.header.stamp = msg_time
        js.name         = self.joint_names
        js.position     = [float(m.q)  for m in state.motorState]
        js.velocity     = [float(m.dq) for m in state.motorState]
        self.joint_pub.publish(js)

        # 2. IMU
        imu = Imu()
        imu.header.stamp    = msg_time
        imu.header.frame_id = 'imu_link'
        q = state.imu.quaternion
        imu.orientation     = Quaternion(w=float(q[0]), x=float(q[1]),
                                         y=float(q[2]), z=float(q[3]))
        gv = state.imu.gyroscope
        imu.angular_velocity = Vector3(x=float(gv[0]), y=float(gv[1]), z=float(gv[2]))
        if hasattr(state.imu, 'accelerometer'):
            ac = state.imu.accelerometer
            imu.linear_acceleration = Vector3(x=float(ac[0]), y=float(ac[1]), z=float(ac[2]))
        self.imu_pub.publish(imu)

        # 3. Odometry (estimated or measured velocity)
        odom = Odometry()
        odom.header.stamp      = msg_time
        odom.header.frame_id   = 'odom'
        odom.child_frame_id    = 'base'
        lv = state.base_lin_vel
        odom.twist.twist.linear = Vector3(x=float(lv[0]), y=float(lv[1]), z=float(lv[2]))
        self.odom_pub.publish(odom)
