import os
import sys
# Ensure absolute path of the repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import mujoco
from mujoco import viewer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Quaternion, Vector3, Twist
from nav_msgs.msg import Odometry
import argparse
import threading
from Controller.policy_runner import PolicyRunner

class Ros2MujocoBridge(Node):
    def __init__(self, robot_type="go2", checkpoint=None, obs_dim=49):
        super().__init__('mujoco_bridge_node')
        self.robot_type = robot_type
        self.cmd_vel = [0.0, 0.0, 0.0, 0.0]
        
        # Internal Policy Runner (Turbo Mode)
        self.runner = None
        if checkpoint:
            print(f"[MujocoBridge] Loading internal policy runner (Turbo Mode): {checkpoint}")
            self.runner = PolicyRunner(checkpoint, obs_dim=obs_dim, robot_type=robot_type)
            self.last_actions = np.zeros(12, dtype=np.float32)
            self.inference_counter = 0
            self.inference_decimation = 4 # Policy at 50Hz (200Hz / 4)
            self.mj_to_isaac = list(range(12)) # Identity mapping for our standardized bridge

        # 1. Load Go2 MuJoCo Scene
        mjcf_path = os.path.join(os.path.dirname(__file__), "mujoco_menagerie", "unitree_go2", "scene.xml")
        if not os.path.exists(mjcf_path):
             mjcf_path = os.path.join(os.path.dirname(__file__), "scene.xml")

        print(f"[MujocoBridge] Initializing for Go2. Model: {mjcf_path}")
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.001

        self._init_physics()

        # 2. ROS 2 Publishers
        self.joint_pub = self.create_publisher(JointState, '/sensors/joint_states', 10)
        self.imu_pub   = self.create_publisher(Imu,        '/sensors/imu', 10)
        self.odom_pub  = self.create_publisher(Odometry,   '/odom', 10)

        # 3. Subscriptions
        self.create_subscription(JointState, '/commands/joint_commands', self.joint_command_cb, 10)
        self.create_subscription(Twist, '/cmd_vel', self.teleop_cb, 10)

        # 4. Physics Thread
        self.physics_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.physics_thread.start()

        print(f"[MujocoBridge] Initialized for {self.robot_type}. Physics running at 200Hz.")

    def _init_physics(self):
        """Initialize MuJoCo physics and resolve joint addresses."""
        # PD Decimation: 200 Hz PD loop from 1000 Hz physics
        self.PD_DECIMATION = 5

        # Resolve joint addresses once
        self.isaac_names = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        ]
        self.isaac_qpos_addr = np.zeros(12, dtype=int)
        self.isaac_qvel_addr = np.zeros(12, dtype=int)
        self.isaac_ctrl_idx = np.zeros(12, dtype=int)

        for i, name in enumerate(self.isaac_names):
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.isaac_qpos_addr[i] = self.model.jnt_qposadr[j_id]
            self.isaac_qvel_addr[i] = self.model.jnt_dofadr[j_id]
            act_name = name.replace("_joint", "")
            self.isaac_ctrl_idx[i] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)

        # Default Pose
        self.desired_qpos = np.array([
            0.1, -0.1, 0.1, -0.1,  # hips
            0.8, 0.8, 1.0, 1.0,    # thighs
            -1.5, -1.5, -1.5, -1.5, # calves
        ], dtype=np.float32)

        # Buffer for smooth targets
        self.current_targets = self.desired_qpos.copy()

        # Match training damping (KD=0.5, we apply via PD loop, so set dof_damping to 0)
        for i in range(self.model.njnt):
            if self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
                self.model.dof_damping[self.model.jnt_dofadr[i]] = 0.0
                self.model.dof_frictionloss[self.model.jnt_dofadr[i]] = 0.01

        # History for PD deriv
        self.pos_err_hist = np.zeros((1, 12), dtype=np.float32)
        self.vel_hist = np.zeros((1, 12), dtype=np.float32)

    def joint_command_cb(self, msg):
        """Receive joint targets from the ROS Controller."""
        if len(msg.position) == 12:
            self.current_targets = np.array(msg.position, dtype=np.float32)

    def teleop_cb(self, msg):
        """Teleop passed through to sensors for the policy runner to see."""
        self.cmd_vel = [msg.linear.x, msg.linear.y, msg.angular.z, 0.0]

    def _reset_robot(self):
        mujoco.mj_resetData(self.model, self.data)
        for i, addr in enumerate(self.isaac_qpos_addr):
            self.data.qpos[addr] = self.desired_qpos[i]
        self.data.qpos[2] = 0.50
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        print("[MujocoBridge] Robot reset to standing pose.")

    def _pd_torques(self, targets):
        """Compute DCMotor PD torques matching training (Kp=25, Kd=0.5)."""
        q = self.data.qpos[self.isaac_qpos_addr]
        v = self.data.qvel[self.isaac_qvel_addr]
        
        pos_err = targets - q  # Target - Actual
        kp, kd = 25.0, 0.5
        effort_limit, sat_effort, vel_lim = 23.5, 23.5, 30.0
        
        torques = kp * pos_err + kd * (0 - v)
        
        vel_at_lim = vel_lim * (1 + effort_limit / sat_effort)
        v_clamp = np.clip(v, -vel_at_lim, vel_at_lim)
        t_top = sat_effort * (1.0 - v_clamp / vel_lim)
        t_bot = sat_effort * (-1.0 - v_clamp / vel_lim)
        return np.clip(torques, np.minimum(t_bot, -effort_limit), np.minimum(t_top, effort_limit))

    def _physics_loop(self):
        """Primary simulation thread running PD at 200 Hz and Physics at 1000 Hz."""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            track_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
            if track_id == -1:
                track_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
            viewer.cam.trackbodyid = track_id
            
            self._reset_robot()
            next_time = time.time()
            self.step_counter = 0

            while rclpy.ok() and viewer.is_running():
                # --- Turbo Mode Inference (50 Hz) ---
                if self.runner:
                    if self.inference_counter % self.inference_decimation == 0:
                        # Build state proxy
                        class StateProxy:
                            def __init__(self, q, dq, qpos_addr, qvel_addr):
                                self.imu = type('obj', (object,), {
                                    'quaternion': q[3:7].tolist(),
                                    'gyroscope': dq[3:6].tolist()
                                })
                                # Compute lin_vel_b
                                w, x, y, z = q[3:7]
                                R = np.array([
                                    [1-2*y**2-2*z**2, 2*x*y-2*w*z, 2*x*z+2*w*y],
                                    [2*x*y+2*w*z, 1-2*x**2-2*z**2, 2*y*z-2*w*x],
                                    [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x**2-2*y**2]
                                ])
                                self.base_lin_vel = (R.T @ dq[:3]).tolist()
                                self.motorState = [type('obj', (object,), {
                                    'q': q[addr],
                                    'dq': dq[v_addr]
                                }) for addr, v_addr in zip(qpos_addr, qvel_addr)]
                        
                        state = StateProxy(self.data.qpos, self.data.qvel, self.isaac_qpos_addr, self.isaac_qvel_addr)
                        obs = self.runner.build_obs(state, self.cmd_vel, self.last_actions, self.desired_qpos, self.mj_to_isaac)
                        actions = self.runner.get_action(obs)
                        self.last_actions[:] = actions
                        self.current_targets = actions * 0.25 + self.desired_qpos
                    
                    self.inference_counter += 1

                # --- PD step (200 Hz) ---
                torques = self._pd_torques(self.current_targets)
                for i, act_idx in enumerate(self.isaac_ctrl_idx):
                    self.data.ctrl[act_idx] = torques[i]

                # --- Physics steps ---
                for _ in range(self.PD_DECIMATION):
                    mujoco.mj_step(self.model, self.data)

                # --- Publish sensors ---
                self._publish_sensors(self.get_clock().now().to_msg())
                viewer.sync()

                self.step_counter += 1

                # Sync with real time
                next_time += 0.005 # 200 Hz
                sleep_dur = next_time - time.time()
                if sleep_dur > 0:
                    time.sleep(sleep_dur)
                elif sleep_dur < -0.1: # Catch up if significantly behind
                    next_time = time.time()

    def _publish_sensors(self, now):
        # Joint States
        js = JointState()
        js.header.stamp = now
        js.name = self.isaac_names
        js.position = self.data.qpos[self.isaac_qpos_addr].tolist()
        js.velocity = self.data.qvel[self.isaac_qvel_addr].tolist()
        self.joint_pub.publish(js)

        q = self.data.qpos[3:7]        # IMU
        imu = Imu()
        imu.header.stamp = now
        imu.orientation = Quaternion(w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3]))
        gyro_b = self.data.qvel[3:6]
        imu.angular_velocity.x = float(gyro_b[0])
        imu.angular_velocity.y = float(gyro_b[1])
        imu.angular_velocity.z = float(gyro_b[2])
        self.imu_pub.publish(imu)

        # Odom (Base Velocity)
        w, x, y, z = q
        R = np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x**2-2*z**2, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x**2-2*y**2]
        ])
        lin_vel_b = R.T @ self.data.qvel[:3]

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base'
        odom.twist.twist.linear = Vector3(x=float(lin_vel_b[0]), y=float(lin_vel_b[1]), z=float(lin_vel_b[2]))
        self.odom_pub.publish(odom)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="go2")
    parser.add_argument("--internal_policy", type=str, default=None, help="Path to policy checkpoint (Turbo Mode)")
    parser.add_argument("--obs_dim", type=int, default=49)
    args = parser.parse_args()
    
    rclpy.init()
    bridge = Ros2MujocoBridge(robot_type=args.robot, checkpoint=args.internal_policy, obs_dim=args.obs_dim)
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    bridge.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
