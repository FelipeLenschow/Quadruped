import rclpy
import os
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3
from std_msgs.msg import Float32, Float32MultiArray
import numpy as np
from .estimator import StateEstimator, rot_from_quat, projected_gravity_b
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

    This class is simulator-agnostic and only deals with standardized
    numpy/primitive types.

    When use_estimator=True, the 6-state Linear Kalman Filter in estimator.py
    fuses IMU prediction with leg-odometry corrections (requires joint_pos and
    joint_vel to be passed into process_state).
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

        # ── Kalman Filter setup ────────────────────────────────────────────
        dt         = est_cfg.get("dt", estimator_dt)
        kal_cfg    = est_cfg.get("kalman", {})
        q_vel      = float(kal_cfg.get("process_noise_vel",  0.01))
        q_bias     = float(kal_cfg.get("process_noise_bias", 1e-3))
        r_meas     = float(kal_cfg.get("measurement_noise",  0.05))

        self.estimator = StateEstimator(dt=dt, q_vel=q_vel, q_bias=q_bias, r_meas=r_meas)

        # ── ROS 2 Publishers — raw sensors ────────────────────────────────
        self.joint_pub = self.node.create_publisher(JointState, '/sensors/joint_states', 10)
        self.imu_pub   = self.node.create_publisher(Imu,        '/sensors/imu',          10)
        self.odom_pub  = self.node.create_publisher(Odometry,   '/odom',                 10)

        # ── ROS 2 Publishers — derived / estimated state ──────────────────
        # Computed once here so all downstream nodes (Supervisor, Controller,
        # Digital Twin) get a consistent, geometry-free view of the robot state.
        self.proj_gravity_pub  = self.node.create_publisher(Vector3,          '/estimator/projected_gravity', 10)
        self.base_lin_vel_pub  = self.node.create_publisher(Vector3,          '/estimator/base_lin_vel',      10)
        self.base_ang_vel_pub  = self.node.create_publisher(Vector3,          '/estimator/base_ang_vel',      10)
        self.base_height_pub   = self.node.create_publisher(Float32,          '/estimator/base_height',       10)
        self.feet_contact_pub  = self.node.create_publisher(Float32MultiArray,'/estimator/feet_contact',      10)

        # Debug: per-leg odometry velocity norms [|v_FL|, |v_FR|, |v_RL|, |v_RR|]
        self.leg_odom_pub = self.node.create_publisher(
            Float32MultiArray, '/estimator/leg_odometry', 10
        )

    # ------------------------------------------------------------------
    def process_state(self, q, dq, quat, gyro, accel=None, pos=None, vel=None, contact=None, update_estimator=True):
        """
        Creates a StandardState from raw vectors and applies LKF estimation if enabled.

        Args:
            q, dq   : 12-dim joint positions and velocities.
            quat    : [w, x, y, z] orientation.
            gyro    : [wx, wy, wz] angular velocity in body frame.
            accel   : [ax, ay, az] specific force in body frame (default [0,0,9.81]).
            pos     : [x, y, z] global position (optional).
            vel     : [vx, vy, vz] body-frame linear velocity (ground truth).
            contact : [FL, FR, RL, RR] binary contacts (optional).
            update_estimator : Whether to step the Kalman Filter forward in time.


        Note
        ----
        When use_estimator=True, both `dq` (joint velocities) and `gyro` are
        forwarded to the LKF for leg-odometry corrections.  Ensure the caller
        always provides real joint-velocity data in that mode.
        """
        state = StandardState()

        # 1. Populate basic IMU and Joints
        state.imu.quaternion = quat.tolist() if hasattr(quat, 'tolist') else list(quat)
        state.imu.gyroscope  = gyro.tolist() if hasattr(gyro, 'tolist') else list(gyro)

        if accel is not None:
            state.imu.accelerometer = accel.tolist() if hasattr(accel, 'tolist') else list(accel)

        for i in range(12):
            state.motorState[i].q  = q[i]
            state.motorState[i].dq = dq[i]

        if pos is not None:
            state.base_pos = pos.tolist() if hasattr(pos, 'tolist') else list(pos)

        if contact is not None:
            state.feet_contact = contact.tolist() if hasattr(contact, 'tolist') else list(contact)

        # 2. Default to ground-truth velocity when provided (sim / debug mode)
        if vel is not None:
            state.base_lin_vel = vel.tolist() if hasattr(vel, 'tolist') else list(vel)

        # 3. LKF override — runs when use_estimator=True
        if self.use_estimator:
            if update_estimator:
                v_est = self.estimator.update(
                    quat_wxyz   = state.imu.quaternion,
                    accel_body  = state.imu.accelerometer,
                    feet_contact= state.feet_contact,
                    joint_pos   = [m.q  for m in state.motorState],
                    joint_vel   = [m.dq for m in state.motorState],
                    gyro_body   = state.imu.gyroscope,
                )
            else:
                v_est = self.estimator.velocity
            state.base_lin_vel = v_est.tolist()

        return state

    # ------------------------------------------------------------------
    def publish(self, sim_time, state: StandardState):
        """Publishes raw sensor data and all derived/estimated state to ROS 2 topics."""
        msg_time = rclpy.time.Time(seconds=sim_time).to_msg()

        # ── 1. Raw sensors ────────────────────────────────────────────────

        # 1a. Joint States
        js          = JointState()
        js.header.stamp = msg_time
        js.name     = self.joint_names
        js.position = [float(m.q)  for m in state.motorState]
        js.velocity = [float(m.dq) for m in state.motorState]
        self.joint_pub.publish(js)

        # 1b. IMU
        imu = Imu()
        imu.header.stamp    = msg_time
        imu.header.frame_id = 'imu_link'
        q = state.imu.quaternion
        imu.orientation          = Quaternion(w=float(q[0]), x=float(q[1]),
                                              y=float(q[2]), z=float(q[3]))
        gv = state.imu.gyroscope
        imu.angular_velocity     = Vector3(x=float(gv[0]), y=float(gv[1]), z=float(gv[2]))
        if hasattr(state.imu, 'accelerometer'):
            ac = state.imu.accelerometer
            imu.linear_acceleration = Vector3(x=float(ac[0]), y=float(ac[1]), z=float(ac[2]))
        self.imu_pub.publish(imu)

        # 1c. Odometry (estimated or measured velocity)
        odom = Odometry()
        odom.header.stamp       = msg_time
        odom.header.frame_id    = 'odom'
        odom.child_frame_id     = 'base'
        lv = state.base_lin_vel
        odom.twist.twist.linear = Vector3(x=float(lv[0]), y=float(lv[1]), z=float(lv[2]))
        self.odom_pub.publish(odom)

        # ── 2. Derived / Estimated state (/estimator/*) ───────────────────

        # 2a. Projected gravity (gravity vector in body frame)
        pg = projected_gravity_b(state.imu.quaternion)
        self.proj_gravity_pub.publish(
            Vector3(x=float(pg[0]), y=float(pg[1]), z=float(pg[2]))
        )

        # 2b. Linear base velocity in body frame (estimated or GT)
        self.base_lin_vel_pub.publish(
            Vector3(x=float(lv[0]), y=float(lv[1]), z=float(lv[2]))
        )

        # 2c. Angular base velocity (from IMU gyroscope)
        self.base_ang_vel_pub.publish(
            Vector3(x=float(gv[0]), y=float(gv[1]), z=float(gv[2]))
        )

        # 2d. Base height (z from odom position, or default if not available)
        height_msg      = Float32()
        height_msg.data = float(state.base_pos[2]) if state.base_pos else 0.35
        self.base_height_pub.publish(height_msg)

        # 2e. Feet contact flags [FL, FR, RL, RR] — binary floats
        fc_msg      = Float32MultiArray()
        fc_msg.data = [float(c) for c in state.feet_contact]
        self.feet_contact_pub.publish(fc_msg)

        # 2f. Leg odometry debug — velocity norm per leg from FK (if estimator active)
        if self.use_estimator:
            from .kinematics import Go2Kinematics
            _kin = Go2Kinematics()
            omega = np.asarray(state.imu.gyroscope, dtype=np.float64)
            norms = []
            for leg_idx in range(4):
                idx     = [leg_idx, leg_idx + 4, leg_idx + 8]
                q_leg   = np.array([state.motorState[i].q  for i in idx])
                dq_leg  = np.array([state.motorState[i].dq for i in idx])
                r_foot  = _kin.foot_position_body(leg_idx, q_leg)
                J       = _kin.foot_jacobian_body(leg_idx, q_leg)
                v_leg   = -J @ dq_leg - np.cross(omega, r_foot)
                norms.append(float(np.linalg.norm(v_leg)))
            lo_msg      = Float32MultiArray()
            lo_msg.data = norms
            self.leg_odom_pub.publish(lo_msg)
        else:
            lo_msg      = Float32MultiArray()
            lo_msg.data = [0.0, 0.0, 0.0, 0.0]
            self.leg_odom_pub.publish(lo_msg)
