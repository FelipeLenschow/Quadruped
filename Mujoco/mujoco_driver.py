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
from std_msgs.msg import Bool, Float32
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
        self.motor_cfg = self.config.get("motor", {})
        self.kp = float(self.ctrl_cfg.get("kp", 0.0))
        self.kd = float(self.ctrl_cfg.get("kd", 0.0))



        # 1. Load MuJoCo Scene
        robot_folder = f"unitree_{self.robot_type.lower()}"
        mjcf_path = os.path.join(
            os.path.dirname(__file__), "mujoco_menagerie", robot_folder, "scene.xml"
        )
        if not os.path.exists(mjcf_path):
            mjcf_path = os.path.join(os.path.dirname(__file__), "scene.xml")

        print(f"[MujocoDriver] Initializing for {self.robot_type.upper()}. Model: {mjcf_path}")
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
            joint_names=self.isaac_names,
            sim_dt=0.001
        )

        # 3. Subscriptions
        self.create_subscription(Twist, "/cmd_vel", self.teleop_cb, 10)
        self.create_subscription(Bool, "/base/freeze", self._freeze_base_cb, 10)

        # Freeze-base state
        self._freeze_base_active = False
        self._freeze_base_pose = None  # np.array of shape (7,): xyz + quat [w,x,y,z]

        self.create_subscription(Float32, "/control/kp", self.kp_cb, 10)
        self.create_subscription(Float32, "/control/kd", self.kd_cb, 10)
        self._startup_console_check = False

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

        # Convert position actuators into pure torque motors (Go1/A1 use position tags in their XML)
        for i in range(self.model.nu):
            self.model.actuator_gainprm[i, 0] = 1.0
            self.model.actuator_biasprm[i, 1] = 0.0
            self.model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE
            self.model.actuator_ctrllimited[i] = 0

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
                    is_match1 = name1 and name1.lower() != "floor" and foot_name.lower() in name1.lower()
                    is_match2 = name2 and name2.lower() != "floor" and foot_name.lower() in name2.lower()
                    if is_match1 or is_match2:
                        contact[foot_idx] = 1.0

        return {
            'q': q, 'dq': dq, 'quat': quat, 'gyro': gyro, 
            'accel': accel, 'pos': pos, 'vel': vel_b, 'contact': contact
        }

    def teleop_cb(self, msg):
        """Teleop passed through to sensors for the policy runner to see."""
        self.cmd_vel = [msg.linear.x, msg.linear.y, msg.angular.z, 0.0]

    def _freeze_base_cb(self, msg: Bool):
        """Enable/disable mid-air base freeze at Z=1.0m."""
        if msg.data and not self._freeze_base_active:
            # Capture current XY and orientation, but force Z = 1.0 m
            pos = self.data.qpos[0:3].copy()
            quat = self.data.qpos[3:7].copy()  # [w, x, y, z]
            self._freeze_base_pose = np.array([
                pos[0], pos[1], 1.0,          # X, Y from current pose; Z fixed
                quat[0], quat[1], quat[2], quat[3],
            ], dtype=np.float64)
            self._freeze_base_active = True
            self.get_logger().info(
                f"[MujocoDriver] Base FROZEN at XY=({pos[0]:.2f}, {pos[1]:.2f}) Z=1.0m")
        elif not msg.data and self._freeze_base_active:
            self._freeze_base_active = False
            self._freeze_base_pose = None
            self.get_logger().info("[MujocoDriver] Base freeze RELEASED.")

    def kp_cb(self, msg):
        new_kp = float(msg.data)
        if new_kp != self.kp:
            self.kp = new_kp
            self.get_logger().info(f"[MujocoDriver] Dynamic Kp updated to: {self.kp:.1f}")

    def kd_cb(self, msg):
        new_kd = float(msg.data)
        if new_kd != self.kd:
            self.kd = new_kd
            self.get_logger().info(f"[MujocoDriver] Dynamic Kd updated to: {self.kd:.2f}")


    @property
    def act_effort_limit(self) -> float:
        """Peak actuator torque, N*m. From the checkpoint's env.yaml via launcher.py."""
        return float(os.environ.get("QUADRUPED_EFFORT_LIMIT", 23.7))

    @property
    def act_saturation(self) -> float:
        """Stall torque of the DC motor model, N*m -- the intercept of the torque-speed line."""
        return float(os.environ.get("QUADRUPED_SATURATION_EFFORT", 23.7))

    @property
    def act_velocity_limit(self) -> float:
        """No-load joint speed, rad/s -- where available torque reaches zero."""
        return float(os.environ.get("QUADRUPED_VELOCITY_LIMIT", 30.0))

    @property
    def barrier_stiffness(self) -> float:
        """Stiffness of the joint-limit spring in torque mode, in N*m/rad.

        Exported by launcher.py from the checkpoint's env.yaml
        (joint_limit_barrier_stiffness). Training picks it to match position mode's Kp so the
        boundary feels the same in both modes; a policy trained against it will drive into the
        limits without it.
        """
        return float(os.environ.get("QUADRUPED_BARRIER_STIFFNESS", 25.0))

    @property
    def spawn_height(self) -> float:
        """Base height the robot is reset to, in metres.

        QUADRUPED_SPAWN_HEIGHT wins if set (launcher.py exports it from the checkpoint's
        env.yaml). Otherwise falls back to the same mode-dependent defaults the training config
        uses, so a checkpoint is dropped from the height it learned to land from. Lower it
        further if the robot still cannot recover -- a Go2 stands at ~0.32 m, so 0.35 is a 3 cm
        settle rather than a fall.
        """
        env_val = os.environ.get("QUADRUPED_SPAWN_HEIGHT")
        if env_val:
            return float(env_val)
        torque = os.environ.get("QUADRUPED_CONTROL_MODE", "position").lower() == "torque"
        return 0.35 if torque else 0.50

    def _reset_robot(self):
        mujoco.mj_resetData(self.model, self.data)
        for i, addr in enumerate(self.isaac_qpos_addr):
            self.data.qpos[addr] = self.desired_qpos[i]
        # Drop height, matched to what the policy was trained with rather than hardcoded.
        # 0.50 was the position-mode number and assumes the PD loop holds the default stance
        # through the ~0.2 m fall so the robot lands on its feet. A torque policy has no PD
        # underneath it -- the legs are only as stiff as the torques it happens to command --
        # so the same drop lands it on its belly. The training cfg already accounts for this
        # (quadruped_env_cfg.spawn_height: 0.35 for torque, 0.50 for position); this reads the
        # value the checkpoint actually trained at, exported by launcher.py from its env.yaml.
        self.data.qpos[2] = self.spawn_height
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        print("[MujocoDriver] Robot reset to standing pose.")

    def _pd_torques(self, targets):
        """Compute DCMotor PD torques matching training (Kp=25, Kd=0.5)."""
        q = self.data.qpos[self.isaac_qpos_addr]
        v = self.data.qvel[self.isaac_qvel_addr]

        pos_err = targets - q  # Target - Actual
        kp = self.kp
        kd = self.kd
        
        # Override with safety watchdog torque
        effort_limit = self.pipeline.safety_processor.active_max_torque
        
        if self.robot_type.lower() == "a1":
            sat_effort, vel_lim = 33.5, 21.0
        elif self.robot_type.lower() == "go1":
            sat_effort, vel_lim = 23.7, 30.0
        else: # go2
            sat_effort = float(self.motor_cfg.get("max_torque", 45.0))
            vel_lim = float(self.motor_cfg.get("max_velocity", 30.0))
        
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

                # Check if console was opened before this pipeline
                if hasattr(self, "_startup_console_check") and self._startup_console_check:
                    if not hasattr(self, "_startup_ticks"):
                        self._startup_ticks = 0
                    self._startup_ticks += 1
                    if self._startup_ticks >= 200: # 200ms at 1000Hz physics loop rate
                        self._startup_console_check = False
                        if self.count_publishers("/safety/heartbeat") > 0:
                            self.get_logger().error("[Safety] Console was detected running before driver! Exiting simulator for safety.")
                            import sys
                            sys.exit(0)
                
                # --- Centralized Pipeline (handles inference & telemetry) ---
                raw_data = self._get_raw_sensor_data()
                raw_data['tau'] = getattr(self, '_last_applied_torque', np.zeros(12))
                self.current_targets = self.pipeline.step(
                    raw_state_kwargs=raw_data,
                    cmd_vel=self.cmd_vel,
                    sim_time=self.data.time
                )

                # --- Actuation (200 Hz) ---
                if getattr(self.pipeline, "output_is_torque", False):
                    # Torque-mode policy: the pipeline handed us joint EFFORTS in N*m, already
                    # scaled by torque_scale. Applied straight to the actuators, bypassing the PD
                    # loop entirely -- running them through _pd_torques would treat newton-metres
                    # as radians of position error and multiply them by Kp.
                    # Clipped to the same effort limit the PD path is bounded by, which is also
                    # what the training env's actuator model enforces.
                    torques = np.asarray(self.current_targets, dtype=np.float64)
                    # JOINT-LIMIT BARRIER, mirroring QuadrupedEnv._joint_limit_barrier. Position
                    # mode makes limit violation structurally impossible by clamping the target;
                    # torque mode has no target, so training re-establishes the guarantee in the
                    # action path with a one-sided spring. The policy learned WITH that safety
                    # net underneath it, so leaving it out here changes behaviour exactly at the
                    # limits -- the joints get driven into their stops instead of pushed back.
                    # Same two parts, same order as training:
                    #   1. drop any commanded torque still pushing the joint further out
                    #   2. add a spring proportional to the overshoot, pulling it back
                    sp = self.pipeline.safety_processor
                    q = self.data.qpos[self.isaac_qpos_addr]
                    over_hi = np.clip(q - sp.soft_max, 0.0, None)
                    over_lo = np.clip(sp.soft_min - q, 0.0, None)
                    torques = np.where(over_hi > 0.0, np.minimum(torques, 0.0), torques)
                    torques = np.where(over_lo > 0.0, np.maximum(torques, 0.0), torques)
                    torques = torques + self.barrier_stiffness * (over_lo - over_hi)

                    # ACTUATOR MODEL, mirroring Isaac's DCMotorCfg. Training does not clip to
                    # a flat effort limit: available torque falls off with joint speed (back-EMF),
                    #   max_eff =  sat * (1 - v/v_lim), clipped to [0, +effort_limit]
                    #   min_eff =  sat * (-1 - v/v_lim), clipped to [-effort_limit, 0]
                    # so a joint at 15 rad/s can only pull 11.8 N*m and at 25 rad/s only 3.9,
                    # against MuJoCo's flat ctrlrange of 23.7. A trot peaks at 10-20 rad/s in
                    # swing, so without this the policy gets 2-6x the torque it was ever trained
                    # to have, exactly during the fastest part of the stride. Braking torque
                    # stays available at full, which is the physically correct asymmetry.
                    v = self.data.qvel[self.isaac_qvel_addr]
                    sat, v_lim, eff = self.act_saturation, self.act_velocity_limit, self.act_effort_limit
                    max_eff = np.clip(sat * (1.0 - v / v_lim), 0.0, eff)
                    min_eff = np.clip(sat * (-1.0 - v / v_lim), -eff, 0.0)
                    torques = np.clip(torques, min_eff, max_eff)

                    # Safety watchdog limit still applies on top.
                    limit = sp.active_max_torque
                    torques = np.clip(torques, -limit, limit)
                    # Fed back as the "previous applied torque" observation. Training feeds the
                    # torque the articulation ACTUALLY applied -- after this barrier and after
                    # the actuator clamp -- not what the policy asked for. They differ precisely
                    # when the barrier or the effort limit is active, which is where a torque
                    # policy most needs to know what really happened.
                    self._last_applied_torque = torques.copy()
                else:
                    torques = self._pd_torques(self.current_targets)
                for i, act_idx in enumerate(self.isaac_ctrl_idx):
                    self.data.ctrl[act_idx] = torques[i]

                # --- Physics steps ---
                for _ in range(self.PD_DECIMATION):
                    mujoco.mj_step(self.model, self.data)

                # --- Freeze Base (override free-joint after every step) ---
                if self._freeze_base_active and self._freeze_base_pose is not None:
                    self.data.qpos[0:7] = self._freeze_base_pose
                    self.data.qvel[0:6] = 0.0   # zero linear + angular base velocity
                    mujoco.mj_forward(self.model, self.data)  # resync kinematics
                
                if not headless:
                    viewer.sync()

                self.step_counter += 1

                # Logging for diagnosis (every 200 steps ~ 0.2s)
                if self.step_counter % 200 == 0:
                    inf_ms = 0.0
                    runner = self.pipeline.policy_manager.policies.get("main")
                    if runner:
                        if hasattr(runner, "inf_times") and runner.inf_times:
                            inf_ms = runner.inf_times[-1] * 1000
                    print(
                        f"\r[Bridge] t={self.data.time:7.2f} h={raw_data['pos'][2]:.2f} vx={raw_data['vel'][0]:+5.2f} vy={raw_data['vel'][1]:+5.2f} wz={raw_data['gyro'][2]:+5.2f} | inf={inf_ms:4.1f}ms   ",
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
