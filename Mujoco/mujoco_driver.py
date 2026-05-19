import os
import sys

# Ensure absolute path of the repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np

"""
MuJoCo Driver for Quadruped Locomotion. 
Handles internal policy inference, physics stepping, 
and standardizes telemetry for ROS 2 monitoring.
"""

import mujoco
from mujoco import viewer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Quaternion, Vector3, Twist
from nav_msgs.msg import Odometry
import argparse
import threading
from pipeline import LocomotionPipeline
from Configs.config_loader import load_config


class Ros2MujocoDriver(Node):
    def __init__(self, robot_type="go2", checkpoint=None, obs_dim=49, use_estimator=False, headless=False):
        super().__init__("mujoco_bridge_node")
        self.robot_type = robot_type
        self.cmd_vel = [0.0, 0.0, 0.0, 0.0]
        self.headless = headless

        # 0. Load Central Config
        self.config = load_config()
        self.ctrl_cfg = self.config.get("control", {})

        # 1. Load Go2 MuJoCo Scene
        mjcf_path = os.path.join(
            os.path.dirname(__file__), "mujoco_menagerie", "unitree_go2", "scene.xml"
        )
        if not os.path.exists(mjcf_path):
            mjcf_path = os.path.join(os.path.dirname(__file__), "scene.xml")

        print(f"[MujocoDriver] Initializing for Go2. Model: {mjcf_path}")
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.001

        self._init_physics()

        # 2. Locomotion Pipeline
        self.pipeline = LocomotionPipeline(
            node=self,
            robot_type=robot_type,
            checkpoint=checkpoint,
            obs_dim=obs_dim,
            use_estimator=use_estimator,
            joint_names=self.isaac_names
        )

        # 3. Subscriptions
        self.create_subscription(Twist, "/cmd_vel", self.teleop_cb, 10)

        # 4. Physics Thread
        self.physics_thread = threading.Thread(target=self._physics_loop, args=(self.headless,), daemon=True)
        self.physics_thread.start()

        print(
            f"[MujocoDriver] Initialized for {self.robot_type.upper()}. Physics running at 200Hz."
        )

    def _init_physics(self):
        """Initialize MuJoCo physics and resolve joint addresses."""
        # PD Decimation: 1000 Hz PD loop from 1000 Hz physics
        self.PD_DECIMATION = 1

        # Resolve joint addresses once
        self.isaac_names = [
            "FL_hip_joint",
            "FR_hip_joint",
            "RL_hip_joint",
            "RR_hip_joint",
            "FL_thigh_joint",
            "FR_thigh_joint",
            "RL_thigh_joint",
            "RR_thigh_joint",
            "FL_calf_joint",
            "FR_calf_joint",
            "RL_calf_joint",
            "RR_calf_joint",
        ]
        self.isaac_qpos_addr = np.zeros(12, dtype=int)
        self.isaac_qvel_addr = np.zeros(12, dtype=int)
        self.isaac_ctrl_idx = np.zeros(12, dtype=int)

        for i, name in enumerate(self.isaac_names):
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.isaac_qpos_addr[i] = self.model.jnt_qposadr[j_id]
            self.isaac_qvel_addr[i] = self.model.jnt_dofadr[j_id]
            act_name = name.replace("_joint", "")
            self.isaac_ctrl_idx[i] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
            )

        # Default Pose
        self.desired_qpos = np.array(
            [
                0.1,
                -0.1,
                0.1,
                -0.1,  # hips
                0.8,
                0.8,
                1.0,
                1.0,  # thighs
                -1.5,
                -1.5,
                -1.5,
                -1.5,  # calves
            ],
            dtype=np.float32,
        )

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

    def _get_raw_sensor_data(self):
        """Extracts raw state vectors from MuJoCo data."""
        q = self.data.qpos[self.isaac_qpos_addr]
        dq = self.data.qvel[self.isaac_qvel_addr]
        quat = self.data.qpos[3:7]  # [w, x, y, z]
        pos = self.data.qpos[:3]

        # Body frame rotation
        w, x, y, z = quat
        R = np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*w*z,      2*x*z+2*w*y],
            [2*x*y+2*w*z,     1-2*x**2-2*z**2,  2*y*z-2*w*x],
            [2*x*z-2*w*y,     2*y*z+2*w*x,      1-2*x**2-2*y**2],
        ])

        # Body frame velocities
        global_ang_vel = self.data.cvel[1][:3]
        gyro = R.T @ global_ang_vel
        vel_b = R.T @ self.data.qvel[:3]

        # Accelerometer specific force
        try:
            accel = self.data.sensor('accelerometer').data.copy()
        except KeyError:
            accel = np.array([0.0, 0.0, 9.81])

        # Contacts
        contact = [0.0, 0.0, 0.0, 0.0]
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            if g1 == 0 or g2 == 0:
                for foot_idx, foot_name in enumerate(["FL", "FR", "RL", "RR"]):
                    name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
                    name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2)
                    if (name1 and foot_name.lower() in name1.lower()) or \
                       (name2 and foot_name.lower() in name2.lower()):
                        contact[foot_idx] = 1.0

        return {
            'q': q, 'dq': dq, 'quat': quat, 'gyro': gyro, 
            'accel': accel, 'pos': pos, 'vel': vel_b, 'contact': contact
        }

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
        print("[MujocoDriver] Robot reset to standing pose.")

    def _pd_torques(self, targets):
        """Compute DCMotor PD torques matching training (Kp=25, Kd=0.5)."""
        q = self.data.qpos[self.isaac_qpos_addr]
        v = self.data.qvel[self.isaac_qvel_addr]

        pos_err = targets - q  # Target - Actual
        kp = self.ctrl_cfg.get("kp", 25.0)
        kd = self.ctrl_cfg.get("kd", 0.5)
        
        # Override with safety watchdog torque
        effort_limit = self.pipeline.command_processor.active_max_torque
        sat_effort, vel_lim = 23.5, 30.0
        
        if effort_limit <= 0.1:
            # Go limp
            kp = 0.0
            kd = 0.0

        torques = kp * pos_err + kd * (0 - v)

        vel_at_lim = vel_lim * (1 + effort_limit / sat_effort)
        v_clamp = np.clip(v, -vel_at_lim, vel_at_lim)
        t_top = effort_limit * (1.0 - v_clamp / vel_lim)
        t_bot = effort_limit * (-1.0 - v_clamp / vel_lim)
        return np.clip(
            torques, np.minimum(t_bot, -effort_limit), np.minimum(t_top, effort_limit)
        )

    def _physics_loop(self, headless=False):
        """Primary simulation thread running PD at 200 Hz and Physics at 1000 Hz."""
        
        # Helper to handle viewer synchronization if not headless
        def sync_viewer(v):
            if v:
                v.sync()

        # Context manager for optional viewer
        class DummyViewer:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def is_running(self): return True
            def sync(self): pass

        viewer_ctx = mujoco.viewer.launch_passive(self.model, self.data) if not headless else DummyViewer()

        with viewer_ctx as viewer:
            if not headless:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                track_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
                if track_id == -1:
                    track_id = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
                    )
                viewer.cam.trackbodyid = track_id

            self._reset_robot()
            next_time = time.time()
            self.step_counter = 0
            
            # Use a slightly different loop condition for headless
            while rclpy.ok():
                if not headless and not viewer.is_running():
                    break
                
                # --- Centralized Pipeline (handles inference & telemetry) ---
                raw_data = self._get_raw_sensor_data()
                self.current_targets = self.pipeline.step(
                    raw_state_kwargs=raw_data,
                    cmd_vel=self.cmd_vel,
                    sim_time=self.data.time
                )

                # --- PD step (200 Hz) ---
                torques = self._pd_torques(self.current_targets)
                for i, act_idx in enumerate(self.isaac_ctrl_idx):
                    self.data.ctrl[act_idx] = torques[i]

                # --- Physics steps ---
                for _ in range(self.PD_DECIMATION):
                    mujoco.mj_step(self.model, self.data)
                
                if not headless:
                    viewer.sync()

                self.step_counter += 1

                # Logging for diagnosis (every 200 steps ~ 0.2s)
                if self.step_counter % 200 == 0:
                    inf_ms = 0.0
                    if self.pipeline.runner and hasattr(self.pipeline.runner, "inf_times") and self.pipeline.runner.inf_times:
                        inf_ms = self.pipeline.runner.inf_times[-1] * 1000
                    print(
                        f"\r[Bridge] t={self.data.time:7.2f} h={raw_data['pos'][2]:.2f} vx={raw_data['vel'][0]:+5.2f} | inf={inf_ms:4.1f}ms   ",
                        end="",
                        flush=True,
                    )

                # Sync with real time
                next_time += 0.001  # 1000 Hz
                sleep_dur = next_time - time.time()
                if sleep_dur > 0:
                    time.sleep(sleep_dur)
                elif sleep_dur < -0.1:  # Catch up if significantly behind
                    next_time = time.time()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="go2")
    parser.add_argument(
        "--internal_policy", type=str, default=None, help="Path to policy checkpoint"
    )
    parser.add_argument("--obs_dim", type=int, default=49)
    parser.add_argument("--use_estimator", action="store_true", help="Use IMU-based state estimation instead of ground truth")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    args = parser.parse_args()

    rclpy.init()
    node = Ros2MujocoDriver(
        robot_type=args.robot, 
        checkpoint=args.internal_policy, 
        obs_dim=args.obs_dim,
        use_estimator=args.use_estimator,
        headless=args.headless
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
