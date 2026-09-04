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
from std_msgs.msg import Bool, Float32, Float32MultiArray
import argparse
import threading
from pipeline import LocomotionPipeline
from Configs.config_loader import load_config
from Controller.robot_defaults import DEFAULT_STANCE_QPOS
from Mujoco.foot_contact_overlay import FootContactOverlay
from Mujoco.velocity_arrow_overlay import VelocityArrowOverlay
from Mujoco import terrain as terrain_mod
from Mujoco import robot_mods


class Ros2MujocoDriver(Node):
    def __init__(self, robot_type="go2", checkpoint=None, obs_dim=49, use_estimator=False, headless=False,
                 no_ground_truth=False):
        super().__init__("mujoco_bridge_node")
        self.robot_type = robot_type
        self.cmd_vel = [0.0, 0.0, 0.0, 0.0]
        self.headless = headless

        # 0. Load Central Config
        self.config = load_config()
        # Withhold the simulator's ground-truth base pose and velocity so the sim
        # sees exactly what hardware sees. real_driver cannot supply either (the
        # SDK's LowState has no global pose), so leaving them on lets the sim pass
        # tests the robot would fail - which is how the diverging estimator went
        # unnoticed until it hit the real robot.
        self.no_ground_truth = no_ground_truth
        if no_ground_truth and not use_estimator:
            # Without 'vel' the sim velocity is zero, so the policy would be fed
            # zeros. Ground truth off only makes sense with the estimator on.
            use_estimator = True
            self.get_logger().warn(
                "[MujocoDriver] --no_ground_truth forces the state estimator on.")

        self.ctrl_cfg = self.config.get("control", {})
        self.motor_cfg = self.config.get("motor", {})
        self.kp = float(self.ctrl_cfg.get("kp", 0.0))
        self.kd = float(self.ctrl_cfg.get("kd", 0.0))
        self.emergency_kd = float(
            self.config.get("safety", {}).get("emergency_kd", 5.0))

        # Simulated FSR. The real foot sensor is a raw integer with a big no-load
        # offset (~16 in the air, ~30 loaded) that differs from foot to foot, so
        # the gate is counts above each foot's own offset rather than an absolute
        # reading. Feeding the policy MuJoCo's ground-truth contact boolean
        # instead hides that failure mode completely, which is why it only ever
        # showed up on hardware. Map the touch sensor's newtons back onto the
        # robot's raw scale and apply the SAME threshold the robot applies.
        _est_cfg = self.config.get("state_estimator", {})
        self.contact_threshold = float(_est_cfg.get("contact_threshold", 10.0))
        self.fsr_bias = float(_est_cfg.get("fsr_bias", 16.0))
        self.fsr_scale = float(_est_cfg.get("fsr_scale", 2.66))
        self.fsr_noise = float(_est_cfg.get("fsr_noise", 0.7))
        self.foot_names = ["FL", "FR", "RL", "RR"]
        # The robot measures these at startup; here the simulated sensor is built
        # from fsr_bias, so that IS its calibration - one value, known exactly.
        self.fsr_offset = np.full(4, self.fsr_bias, dtype=np.float64)
        # Latest readings, kept for the viewer overlay.
        self.foot_force_raw = np.zeros(4, dtype=np.float32)
        self.foot_force_n = np.zeros(4, dtype=np.float32)
        self.foot_contact = np.zeros(4, dtype=np.float32)



        # 1. Load MuJoCo Scene
        robot_folder = f"unitree_{self.robot_type.lower()}"
        mjcf_path = os.path.join(
            os.path.dirname(__file__), "mujoco_menagerie", robot_folder, "scene.xml"
        )
        if not os.path.exists(mjcf_path):
            mjcf_path = os.path.join(os.path.dirname(__file__), "scene.xml")

        # Terrain comes from the launcher via the environment, same channel as
        # the robot and phase selection. "rough" rewrites the scene onto a
        # heightfield; the generated file is only needed until the model is
        # compiled, so it is deleted immediately afterwards.
        self.terrain = os.environ.get("QUADRUPED_TERRAIN", "flat").strip() or "flat"
        self.terrain_params = terrain_mod.resolve(self.config)
        _tmp_scenes = []
        try:
            scene_path = terrain_mod.scene_path(
                mjcf_path, self.terrain, self.terrain_params, _tmp_scenes)
        except (ValueError, RuntimeError) as exc:
            print(f"[MujocoDriver] Terrain '{self.terrain}' unavailable ({exc}); "
                  f"falling back to flat.")
            self.terrain, scene_path = "flat", mjcf_path

        print(f"[MujocoDriver] Initializing for {self.robot_type.upper()}. "
              f"Model: {mjcf_path}  Terrain: {self.terrain}")
        try:
            self.model = mujoco.MjModel.from_xml_path(scene_path)
        finally:
            for f in _tmp_scenes:
                try:
                    os.remove(f)
                except OSError:
                    pass

        mods = robot_mods.add_foot_mass(
            self.model, robot_mods.resolve(self.config))
        if mods is not None:
            print(f"[MujocoDriver] Foot mass: +{mods['per_foot']:.3f} kg per foot "
                  f"({mods['total_added']:.2f} kg total). Knee swing inertia "
                  f"{mods['swing_inertia_before']:.5f} -> "
                  f"{mods['swing_inertia_after']:.5f} kg m^2 "
                  f"(+{100*(mods['swing_inertia_after']/mods['swing_inertia_before']-1):.0f}%)")

        slide, torsion, n_feet = terrain_mod.apply_foot_friction(
            self.model, self.terrain_params)
        if n_feet:
            print(f"[MujocoDriver] Foot friction: {slide:.2f} sliding, "
                  f"{torsion:.3f} torsional ({n_feet} feet)")
        else:
            print("[MujocoDriver] WARNING: no foot geoms named "
                  f"{terrain_mod.FOOT_GEOMS} - stock friction left in place.")

        stats = terrain_mod.randomize(self.model, self.terrain_params)
        if stats is not None:
            span = self.terrain_params["extent"] * 2
            print(f"[MujocoDriver] Rough terrain: {stats['ptp']*1000:.0f} mm "
                  f"peak-to-peak over {span:.0f}x{span:.0f} m, "
                  f"{stats['cell']*100:.1f} cm cells "
                  f"(seed={self.terrain_params['seed']})")
            # The step between neighbouring cells is what a toe catches on, so
            # it is the number to compare against the policy's swing clearance.
            print(f"[MujocoDriver] Adjacent-cell step: {stats['step']*1000:.0f} mm "
                  f"max, {stats['step_p95']*1000:.0f} mm p95")

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
        self.foot_force_pub = self.create_publisher(
            Float32MultiArray, "/sensors/foot_force", 10)
        # Same 1 Hz [threshold, offset x4] broadcast the real driver sends, so the
        # twin viewer draws its bars from the driver in front of it instead of
        # from its own copy of config.yaml (which, off-robot, is not the robot's).
        self.foot_cal_pub = self.create_publisher(
            Float32MultiArray, "/sensors/foot_force_calibration", 10)
        self.create_timer(1.0, self._publish_contact_calibration)
        # The viewer colours its markers from the published flags, exactly like
        # the twin does, so both windows show one source of truth instead of two
        # look-alike derivations of it.
        self.create_subscription(
            Float32MultiArray, "/estimator/feet_contact", self._est_contact_cb, 10)
        self.published_contact = None
        self._startup_console_check = False

        # 4. Physics Thread
        self.physics_thread = threading.Thread(target=self._physics_loop, args=(self.headless,), daemon=True)
        self.physics_thread.start()

        print(
            f"[MujocoDriver] Initialized for {self.robot_type.upper()}. Physics running at 200Hz."
        )
        print(
            f"[MujocoDriver] Simulated FSR: bias={self.fsr_bias:.0f} "
            f"scale={self.fsr_scale:.2f} N/count noise={self.fsr_noise:.1f} "
            f"-> contact above {self.contact_threshold:.0f} counts over the "
            f"{self.fsr_bias:.0f} offset ({self.contact_threshold * self.fsr_scale:.1f} N)"
        )

    def _publish_contact_calibration(self):
        """Broadcast the FSR gate so viewers show what the driver actually applies."""
        msg = Float32MultiArray()
        msg.data = [float(self.contact_threshold)] + [float(o) for o in self.fsr_offset]
        self.foot_cal_pub.publish(msg)

    def _est_contact_cb(self, msg: Float32MultiArray):
        if len(msg.data) >= 4:
            self.published_contact = np.array(msg.data[:4], dtype=np.float64)

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
        self.desired_qpos = DEFAULT_STANCE_QPOS.copy()

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

        # Per-foot touch sensors (<touch> on a site enclosing the foot geom).
        # A scene without them still works - _read_foot_fsr falls back to summing
        # the contact list - but the sensor is the path the real robot has.
        self.touch_sensor_adr = []
        for foot in self.foot_names:
            s_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"{foot}_touch")
            if s_id == -1:
                self.touch_sensor_adr = None
                print(f"[MujocoDriver] No '{foot}_touch' sensor in the model; "
                      f"falling back to contact-list forces.")
                break
            self.touch_sensor_adr.append(self.model.sensor_adr[s_id])

        # Viewer overlay: shared with the passive twin so both draw the same
        # bar on the same scale.
        self.contact_overlay = FootContactOverlay(
            self.model, config=self.config, foot_names=self.foot_names)
        # Commanded vs measured base velocity, drawn over the trunk.
        self.velocity_overlay = VelocityArrowOverlay(self.model)
        self.base_lin_vel_b = np.zeros(3)

        # History for PD deriv
        self.pos_err_hist = np.zeros((1, 12), dtype=np.float32)
        self.vel_hist = np.zeros((1, 12), dtype=np.float32)

    def _read_foot_fsr(self):
        """Simulated foot FSR: (raw counts, newtons, binary contact) per foot.

        The touch sensor reports the normal force in newtons summed over the
        contacts inside the foot site. The robot's sensor reports a raw integer
        with a large no-load offset instead, so convert with

            raw = fsr_offset + N / fsr_scale + noise    (rounded, clamped >= 0)

        and gate on the margin over that offset, with the same config
        contact_threshold the real driver applies to its measured offsets.
        """
        if self.touch_sensor_adr is not None:
            force_n = np.array(
                [self.data.sensordata[a] for a in self.touch_sensor_adr],
                dtype=np.float64)
        else:
            # Fallback for scenes with no touch sensors: sum the normal
            # component of every contact involving a foot geom.
            acc = {name: 0.0 for name in self.foot_names}
            f6 = np.zeros(6)
            for i in range(self.data.ncon):
                con = self.data.contact[i]
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                for g in (con.geom1, con.geom2):
                    nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
                    if nm in acc:
                        acc[nm] += abs(f6[0])
            force_n = np.array([acc[n] for n in self.foot_names], dtype=np.float64)

        force_n = np.maximum(force_n, 0.0)
        raw = self.fsr_offset + force_n / max(self.fsr_scale, 1e-6)
        if self.fsr_noise > 0.0:
            raw += np.random.normal(0.0, self.fsr_noise, size=4)
        raw = np.maximum(np.round(raw), 0.0)

        contact = (raw - self.fsr_offset > self.contact_threshold).astype(np.float32)
        return raw.astype(np.float32), force_n.astype(np.float32), contact

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
        self.base_lin_vel_b = vel_b

        # Accelerometer specific force
        try:
            accel = self.data.sensor('accelerometer').data.copy()
        except KeyError:
            accel = np.array([0.0, 0.0, 9.81])

        # Contacts, via the simulated FSR. Both the raw reading and the binary
        # flag come from _read_foot_fsr, so the policy is gated by a numeric
        # sensor crossing contact_threshold exactly as it is on the robot,
        # instead of by the physics engine's contact list (which is ground truth
        # and never fails - that is why the real robot's marginal-threshold
        # contact bug never showed up here).
        raw_fsr, force_n, contact = self._read_foot_fsr()
        self.foot_force_raw = raw_fsr
        self.foot_force_n = force_n
        self.foot_contact = contact

        # 1000 Hz loop -> publish at 50 Hz, in the same raw units real_driver
        # publishes, so the two can be compared on one PlotJuggler plot.
        self._ff_tick = getattr(self, "_ff_tick", 0) + 1
        if self._ff_tick % 20 == 0:
            _msg = Float32MultiArray()
            _msg.data = [float(v) for v in raw_fsr]
            self.foot_force_pub.publish(_msg)

        raw = {
            'q': q, 'dq': dq, 'quat': quat, 'gyro': gyro,
            'accel': accel, 'contact': contact
        }
        if not self.no_ground_truth:
            raw['pos'] = pos
            raw['vel'] = vel_b
        return raw

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
        
        emergency = effort_limit <= 0.1
        if emergency:
            # Damping only, no position hold - mirrors real_driver.send_to_sdk,
            # which sends kp=0 with a damping kd so the robot sinks under control
            # instead of collapsing.
            kp = 0.0
            kd = self.emergency_kd

        torques = kp * pos_err + kd * (0 - v)

        if emergency:
            # On the real robot the motor controller applies this damping itself,
            # bounded by the motor, NOT by the safety torque budget - which is
            # zero here. Clipping to effort_limit like the normal path would
            # cancel the damping entirely and make the robot go limp again.
            return np.clip(torques, -sat_effort, sat_effort)

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

                # --- Freeze Base (override free-joint after every step) ---
                if self._freeze_base_active and self._freeze_base_pose is not None:
                    self.data.qpos[0:7] = self._freeze_base_pose
                    self.data.qvel[0:6] = 0.0   # zero linear + angular base velocity
                    mujoco.mj_forward(self.model, self.data)  # resync kinematics
                
                if not headless:
                    if viewer.user_scn is not None:
                        with viewer.lock():
                            # contact_overlay clears the scene, so it goes first and
                            # the arrows are appended to it.
                            self.contact_overlay.draw(
                                viewer.user_scn, self.data, self.foot_force_raw,
                                self.published_contact
                                if self.published_contact is not None
                                else self.foot_contact)
                            self.velocity_overlay.draw(
                                viewer.user_scn, self.data,
                                self.cmd_vel, self.base_lin_vel_b, reset=False)
                    viewer.sync()

                self.step_counter += 1

                # Logging for diagnosis (every 200 steps ~ 0.2s)
                if self.step_counter % 200 == 0:
                    inf_ms = 0.0
                    runner = self.pipeline.policy_manager.policies.get("main")
                    if runner:
                        if hasattr(runner, "inf_times") and runner.inf_times:
                            inf_ms = runner.inf_times[-1] * 1000
                    # With --no_ground_truth there is no 'pos'/'vel' to show, so
                    # fall back to the height from the physics state and the
                    # ESTIMATED velocity - which is what the policy is reading in
                    # that mode anyway. Marked "est" so the two are not confused.
                    if "pos" in raw_data and "vel" in raw_data:
                        h = raw_data["pos"][2]
                        vx, vy = raw_data["vel"][0], raw_data["vel"][1]
                        tag = ""
                    else:
                        h = float(self.data.qpos[2])
                        v_est = self.pipeline.telemetry.estimator.velocity
                        vx, vy = float(v_est[0]), float(v_est[1])
                        tag = " est"
                    fsr = " ".join(
                        f"{n}{int(r):3d}{'*' if c > 0.5 else ' '}"
                        for n, r, c in zip(
                            self.foot_names, self.foot_force_raw, self.foot_contact))
                    print(
                        f"\r[Bridge] t={self.data.time:7.2f} h={h:.2f} vx={vx:+5.2f} vy={vy:+5.2f}{tag} wz={raw_data['gyro'][2]:+5.2f} | fsr {fsr} | inf={inf_ms:4.1f}ms   ",
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
        "--no_ground_truth", action="store_true",
        help="Withhold simulator pos/vel so the sim matches what hardware can measure"
    )
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
        no_ground_truth=args.no_ground_truth,
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
