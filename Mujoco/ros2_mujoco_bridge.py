import os
import sys
import time
import numpy as np
import torch
import argparse
import threading
import mujoco

# ROS 2 Standard Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3, Pose, Twist

# Project Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Mujoco.unitree_sdk_mock import quat_to_rot_matrix

class Ros2MujocoBridge(Node):
    def __init__(self, scene_path, checkpoint=None, obs_dim=49, robot_type="go2"):
        super().__init__('mujoco_bridge_node')
        self.robot_type = robot_type

        # 1. Load Go2 MuJoCo Scene
        mjcf_path = os.path.join(os.path.dirname(__file__), "mujoco_menagerie", "unitree_go2", "scene.xml")
        if not os.path.exists(mjcf_path):
             mjcf_path = os.path.join(os.path.dirname(__file__), "scene.xml")

        print(f"[MujocoBridge] Initializing for Go2. Model: {mjcf_path}")
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.001

        # 2. Joint resolution (Type-Grouped order → MuJoCo indices)
        self.isaac_joint_names = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        ]
        self.isaac_qpos_addr = np.zeros(12, dtype=np.int32)
        self.isaac_qvel_addr = np.zeros(12, dtype=np.int32)
        self.isaac_ctrl_idx = np.zeros(12, dtype=np.int32)

        for i, name in enumerate(self.isaac_joint_names):
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id == -1: j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name.replace("_joint", ""))
            if j_id == -1: raise RuntimeError(f"Joint {name} not found in MJCF")
            self.isaac_qpos_addr[i] = self.model.jnt_qposadr[j_id]
            self.isaac_qvel_addr[i] = self.model.jnt_dofadr[j_id]
            act_name = name.replace("_joint", "")
            a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if a_id == -1: a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if a_id == -1: raise RuntimeError(f"Actuator for {name} not found in MJCF")
            self.isaac_ctrl_idx[i] = a_id

        # 3. Default pose and command buffers
        self.desired_qpos = np.array([
            0.1, -0.1, 0.1, -0.1,   # hips
            0.8,  0.8, 1.0, 1.0,    # thighs
           -1.5, -1.5,-1.5,-1.5,    # calves
        ], dtype=np.float32)

        # Teleop command buffer (protected by lock, updated from ROS callback)
        self._cmd_lock = threading.Lock()
        self.cmd_vel = [0.0, 0.0, 0.0, 0.0]   # [vx, vy, wz, 0]

        # 4. Load policy inline (eliminates ROS roundtrip latency)
        self.runner = None
        if checkpoint:
            from Deployment.policy_runner import PolicyRunner
            self.runner = PolicyRunner(checkpoint, obs_dim=obs_dim, robot_type=robot_type)
            print(f"[MujocoBridge] Policy loaded from {checkpoint}")

        # 5. ROS 2 Publishers (for visualization / debug only)
        self.joint_pub = self.create_publisher(JointState, '/sensors/joint_states', 10)
        self.imu_pub   = self.create_publisher(Imu,        '/sensors/imu', 10)
        self.odom_pub  = self.create_publisher(Odometry,   '/odom', 10)

        # Teleop subscription
        self.create_subscription(Twist, '/cmd_vel', self._teleop_cb, 10)

        # Legacy: external command subscription (used when no checkpoint)
        self.latest_command = np.zeros(12, dtype=np.float32)
        self.has_command = False
        self.create_subscription(JointState, '/commands/joint_commands', self._command_cb, 10)

        # 6. Physics Thread
        self.physics_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.physics_thread.start()

        print(f"[MujocoBridge] Initialized for {self.robot_type}. Physics running at 200Hz.")

    def _teleop_cb(self, msg):
        with self._cmd_lock:
            self.cmd_vel[0] = float(np.clip(msg.linear.x,  -1.0, 1.0))
            self.cmd_vel[1] = float(np.clip(msg.linear.y,  -1.0, 1.0))
            self.cmd_vel[2] = float(np.clip(msg.angular.z, -1.0, 1.0))

    def _command_cb(self, msg):
        """Legacy: accepts joint targets from external policy_bridge (no-checkpoint mode)."""
        self.latest_command[:] = msg.position
        self.has_command = True

    def reset_robot(self):
        mujoco.mj_resetData(self.model, self.data)
        for i, addr in enumerate(self.isaac_qpos_addr):
            self.data.qpos[addr] = self.desired_qpos[i]
        self.data.qpos[2] = 0.50
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        print("[MujocoBridge] Robot reset to standing pose.")

    def _init_physics(self):
        self.model.opt.timestep = 0.001

        for i in range(self.model.nu):
            self.model.actuator_gainprm[i, 0] = 1.0
            self.model.actuator_biasprm[i, 1] = 0.0
            self.model.actuator_ctrllimited[i] = 0
            self.model.actuator_forcerange[i, :2] = [-23.7, 23.7]

        for i in range(self.model.njnt):
            if self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
                self.model.dof_damping[self.model.jnt_dofadr[i]] = 0.0
                self.model.dof_frictionloss[self.model.jnt_dofadr[i]] = 0.01

        # Match IsaacLab ground sliding friction (1.0)
        for i in range(self.model.ngeom):
            name = self.model.geom(i).name
            if "floor" in name or "ground" in name:
                self.model.geom_friction[i, 0] = 1.0

        self.step_counter = 0
        self.pos_err_hist = np.zeros((3, 12), dtype=np.float32)
        self.vel_hist = np.zeros((3, 12), dtype=np.float32)

    def _pd_torques(self, targets):
        """Compute DCMotor PD torques matching IsaacLab's DCMotorCfg."""
        q = self.data.qpos[self.isaac_qpos_addr]
        v = self.data.qvel[self.isaac_qvel_addr]
        pos_err = q - targets
        self.pos_err_hist = np.roll(self.pos_err_hist, 1, 0); self.pos_err_hist[0] = pos_err
        self.vel_hist     = np.roll(self.vel_hist,     1, 0); self.vel_hist[0]     = v

        kp, kd = 25.0, 0.5
        effort_limit, sat_effort, vel_lim = 23.5, 23.5, 30.0
        torques = -kp * self.pos_err_hist[0] - kd * self.vel_hist[0]

        # DCMotor velocity-dependent saturation
        vel_at_lim = vel_lim * (1 + effort_limit / sat_effort)
        v_clamp = np.clip(v, -vel_at_lim, vel_at_lim)
        t_top = sat_effort * (1.0 - v_clamp / vel_lim)
        t_bot = sat_effort * (-1.0 - v_clamp / vel_lim)
        return np.clip(torques, np.minimum(t_bot, -effort_limit), np.minimum(t_top, effort_limit))

    def _read_obs_components(self):
        """Read current simulation state and return obs components."""
        q = self.data.qpos[3:7]  # [w, x, y, z]
        w, x, y, z = q
        R = np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x**2-2*z**2, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x**2-2*y**2]
        ])
        lin_vel_b = R.T @ self.data.qvel[:3]
        # In MuJoCo, qvel[3:6] for a free joint is already in the body frame
        ang_vel_b = self.data.qvel[3:6].astype(np.float32)
        proj_grav = R.T @ np.array([0.0, 0.0, -1.0])
        jpos = self.data.qpos[self.isaac_qpos_addr].astype(np.float32)
        jvel = self.data.qvel[self.isaac_qvel_addr].astype(np.float32)
        return lin_vel_b, ang_vel_b, proj_grav, jpos, jvel, R

    def _physics_loop(self):
        self._init_physics()
        self.reset_robot()

        last_actions = np.zeros(12, dtype=np.float32)
        targets      = self.desired_qpos.copy()   # hold default pose until first policy step

        ACTION_SCALE = 0.25
        POLICY_DECIMATION = 4   # call policy every 4 PD steps = 50Hz
        PD_DECIMATION = 5       # 5 physics steps per PD = 200Hz

        try:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                track_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
                if track_id == -1:
                    track_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
                viewer.cam.trackbodyid = track_id

                next_time = time.time()
                while rclpy.ok() and viewer.is_running():

                    # --- PD step (200 Hz) ---
                    torques = self._pd_torques(targets)
                    for i, act_idx in enumerate(self.isaac_ctrl_idx):
                        self.data.ctrl[act_idx] = torques[i]

                    for _ in range(PD_DECIMATION):
                        mujoco.mj_step(self.model, self.data)

                    self.step_counter += 1

                    # --- Policy step (50 Hz = every POLICY_DECIMATION PD steps) ---
                    if self.step_counter % POLICY_DECIMATION == 0:
                        lin_vel_b, ang_vel_b, proj_grav, jpos, jvel, _ = self._read_obs_components()

                        if self.runner is not None:
                            # INLINE policy inference — zero latency
                            with self._cmd_lock:
                                commands = list(self.cmd_vel)
                            obs = np.concatenate([
                                lin_vel_b, ang_vel_b, proj_grav, commands,
                                jpos - self.desired_qpos, jvel, last_actions
                            ]).astype(np.float32)
                            actions = self.runner.get_action(obs)
                            targets = actions * ACTION_SCALE + self.desired_qpos
                            last_actions[:] = actions

                            if self.step_counter % 200 == 0:
                                print(f"\r[MujocoBridge] h={self.data.qpos[2]:.3f} "
                                      f"cmd={self.cmd_vel[:3]} vx={lin_vel_b[0]:+.2f} "
                                      f"act_mean={np.mean(np.abs(actions)):.3f}   ",
                                      end="", flush=True)
                        else:
                            # Legacy mode: use externally provided targets via ROS
                            if self.has_command:
                                targets = self.latest_command.copy()

                        # Still publish sensors for teleop feedback / debugging
                        self.publish_sensors()
                        viewer.sync()

                    next_time += 0.005
                    sleep_dur = next_time - time.time()
                    if sleep_dur > 0:
                        time.sleep(sleep_dur)
                    else:
                        next_time = time.time()

        except Exception as e:
            import traceback
            print(f"[MujocoBridge] Error in physics loop: {e}")
            traceback.print_exc()

    def publish_sensors(self):
        now = self.get_clock().now().to_msg()

        js = JointState()
        js.header.stamp = now
        js.name = self.isaac_joint_names
        js.position = self.data.qpos[self.isaac_qpos_addr].tolist()
        js.velocity  = self.data.qvel[self.isaac_qvel_addr].tolist()
        self.joint_pub.publish(js)

        q = self.data.qpos[3:7]
        w, x, y, z = q
        R = np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x**2-2*z**2, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x**2-2*y**2]
        ])

        imu = Imu()
        imu.header.stamp = now
        imu.orientation = Quaternion(w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3]))
        # In MuJoCo, qvel[3:6] for a free joint is already in the body frame
        gyro_b = self.data.qvel[3:6]
        imu.angular_velocity = Vector3(x=float(gyro_b[0]), y=float(gyro_b[1]), z=float(gyro_b[2]))
        self.imu_pub.publish(imu)

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base'
        lin_vel_b = R.T @ self.data.qvel[:3]
        odom.twist.twist.linear = Vector3(x=float(lin_vel_b[0]), y=float(lin_vel_b[1]), z=float(lin_vel_b[2]))
        self.odom_pub.publish(odom)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot',      type=str, default='go2')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to policy checkpoint. If provided, policy runs inline (recommended).')
    parser.add_argument('--obs_dim',    type=int, default=49)
    args = parser.parse_args()

    rclpy.init()
    node = Ros2MujocoBridge(None, checkpoint=args.checkpoint, obs_dim=args.obs_dim, robot_type=args.robot)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == "__main__":
    main()
