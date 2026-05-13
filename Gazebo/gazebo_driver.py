"""
Gazebo Driver for Quadruped Locomotion.
Manages high-frequency physics stepping, ActuatorNet simulation,
and deterministic policy deployment (Turbo Mode).
"""

import os
import sys

# Ensure absolute path of the repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import torch
import argparse
import signal
import threading
import subprocess
from pathlib import Path

from Controller.policy_runner import PolicyRunner
from Controller.policy_bridge import CommandProcessor
from Telemetry.telemetry import TelemetryManager
from Configs.config_loader import load_config

# ROS 2 Standard Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3, Twist

# Gazebo Transport & Msgs
try:
    from gz.transport13 import Node as GzTransportNode
    from gz.msgs10 import (
        double_pb2,
        model_pb2,
        imu_pb2,
        pose_pb2,
        world_stats_pb2,
        odometry_pb2,
    )
except ImportError:
    print("[ERROR] Gazebo (Harmonic) Python bindings not found.")
    sys.exit(1)


# MuJoCo/Isaac order (Grouped by Joint Type): FL, FR, RL, RR
JOINT_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]

# Sign corrections to match MuJoCo's outward-positive behavior in Gazebo.
# FL=-1, RL=-1 for hips (indices 0 and 2 in Isaac order)
HAA_SIGN = np.array([-1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)


class Ros2GazeboDriver(Node):
    def __init__(
        self, robot_type, world_name="quadruped_world", checkpoint=None, obs_dim=49,
        use_estimator=False
    ):
        super().__init__("gazebo_bridge_node")
        self.robot_type = robot_type
        
        # 0. Load Central Config
        self.config = load_config()
        self.ctrl_cfg = self.config.get("control", {})
        self.motor_cfg = self.config.get("motor", {})
        est_cfg = self.config.get("state_estimator", {})
        
        self.cmd_vel = [0.0, 0.0, 0.0, 0.0]  # [vx, vy, wz, unused_height_cmd]
        
        # Priority: CLI arg (if explicitly True) > YAML config
        effective_use_estimator = use_estimator
        if not effective_use_estimator:
            effective_use_estimator = est_cfg.get("use_estimator", False)

        # Handles internal policy inference, physics stepping,
        # and standardizes telemetry for ROS 2 monitoring.
        self.runner = None
        if checkpoint:
            print(f"[GazeboDriver] Loading internal policy runner: {checkpoint}")
            self.runner = PolicyRunner(
                checkpoint, obs_dim=obs_dim, robot_type=robot_type
            )
            self.runner.decimation = self.ctrl_cfg.get("decimation", 4)
            self.mj_to_isaac = list(range(12))  # Identity
        self.world_name = world_name

        # 4. ROS 2 Telemetry Manager
        self.telemetry = TelemetryManager(self, JOINT_NAMES, use_estimator=effective_use_estimator)
        self.command_processor = CommandProcessor(
            self, robot_type=robot_type, joint_names=JOINT_NAMES
        )

        self.create_subscription(Twist, "/cmd_vel", self._teleop_cb, 10)

        # Nominal standing pose (Matches MuJoCo Driver exactly in Isaac order)
        self.desired_qpos = np.array(
            [
                0.1, -0.1, 0.1, -0.1,  # hips
                0.8, 0.8, 1.0, 1.0,    # thighs
                -1.5, -1.5, -1.5, -1.5  # calves
            ],
            dtype=np.float32,
        )
        self.latest_torques = np.zeros(12, dtype=np.float32)
        self.latest_targets = self.desired_qpos.copy()
        self.sim_time = 0.0
        self.q = np.zeros(12)
        self.dq = np.zeros(12)
        self.base_pos = np.zeros(3)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self.base_ang_vel = np.zeros(3)
        self.base_lin_vel_b = np.zeros(3)
        self.base_accel = np.array([0., 0., 9.81])  # body-frame specific force (m/s^2)

        # 3. Gazebo Transport (Deferred until physics loop for partitioning)
        self.gz_node = None
        self.joint_pubs = []

        self.new_data_event = threading.Event()
        self._stop = threading.Event()
        self.world_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "scene.sdf")
        )

        # 4. Starting Threads
        self.physics_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.repeater_thread = threading.Thread(target=self._repeater_loop, daemon=True)
        self.physics_thread.start()
        self.repeater_thread.start()

        print(
            f"[GazeboDriver] Initialized for {robot_type.upper()}. Physics at 500Hz (Slave)."
        )

    def _teleop_cb(self, msg):
        # Update velocities, keeping 4th command 0.0 to match training distribution
        self.cmd_vel[0] = msg.linear.x
        self.cmd_vel[1] = msg.linear.y
        self.cmd_vel[2] = msg.angular.z
        self.cmd_vel[3] = 0.0

    # --- Control & Gains ---
    def _stats_cb(self, msg):
        self.sim_time = msg.sim_time.sec + msg.sim_time.nsec * 1e-9

    def _imu_cb(self, msg):
        # msg.orientation is [x, y, z, w] in protobuf
        q = msg.orientation
        self.base_quat = np.array([q.w, q.x, q.y, q.z])
        self.base_ang_vel = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        )
        if hasattr(msg, 'linear_acceleration'):
            la = msg.linear_acceleration
            self.base_accel = np.array([la.x, la.y, la.z])

    def _joint_cb(self, msg):
        if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
            self.sim_time = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
            
        for joint in msg.joint:
            if joint.name in JOINT_NAMES:
                idx = JOINT_NAMES.index(joint.name)
                try:
                    # Apply HAA axis sign correction so that Gazebo joint readings
                    # match the IsaacLab convention (right-side HAA joints are negated).
                    self.q[idx] = joint.axis1.position * HAA_SIGN[idx]
                    self.dq[idx] = joint.axis1.velocity * HAA_SIGN[idx]
                except AttributeError:
                    pass
        # Signal the physics loop to step
        self.new_data_event.set()

    def _odom_cb(self, msg):
        self.base_pos = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        )
        
        # Ground truth orientation and angular velocity from simulator
        q = msg.pose.orientation
        self.base_quat = np.array([q.w, q.x, q.y, q.z])
        w_body = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])

        # gz::sim::systems::OdometryPublisher (dimensions=3) reports the twist
        # in the BODY frame (child frame), NOT the world frame.
        v_body = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])

        try:
            self.base_lin_vel_b[:] = v_body
            self.base_ang_vel[:] = w_body
        except Exception:
            self.base_lin_vel_b[:] = v_body
            self.base_ang_vel[:] = w_body

        # Log Suspected Drift
        if np.abs(self.base_lin_vel_b[1]) > 0.5 and self.sim_time > 5.0:
            print(
                f"\r[Bridge] WARNING: Lateral vel vy={self.base_lin_vel_b[1]:.2f} at t={self.sim_time:.1f}",
                end="", flush=True,
            )

    def _repeater_loop(self):
        """
        Background thread to repeat the latest torques at 1000Hz.
        Gazebo clears forces every step; this ensures the robot stays powered.
        """
        while not self._stop.is_set():
            if hasattr(self, "joint_pubs") and len(self.joint_pubs) == 12:
                torch_torques = self.latest_torques.copy()
                for i, torque in enumerate(torch_torques):
                    msg = double_pb2.Double()
                    msg.data = float(torque)
                    self.joint_pubs[i].publish(msg)
            time.sleep(0.001)

    def _physics_loop(self):
        # 1. Load ActuatorNet
        act_net_path = Path(__file__).parent.parent / "Mujoco" / "unitree_quadruped.pt"
        if not act_net_path.exists():
            print(f"[Ros2GazeboBridge] ERROR: ActuatorNet missing at {act_net_path}")
            return
        self.act_net = torch.jit.load(str(act_net_path), map_location="cpu").eval()

        # 2. Resource & Global Cleanup
        root_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(
            ["pkill", "-9", "-f", "gz-sim-server"], stderr=subprocess.DEVNULL
        )
        time.sleep(1.0)

        # 3. Environment Setup (Crucial for Resource Path and Partition)
        env = os.environ.copy()

        # Ensure a truly unique partition for THIS specific bridge instance
        unique_id = os.getpid() % 100000
        partition = f"quadruped_sim_{unique_id}"
        env["GZ_PARTITION"] = partition

        # Resource paths for model loading
        env["GZ_SIM_RESOURCE_PATH"] = (
            root_dir
            + os.pathsep
            + os.path.join(root_dir, "models")
            + os.pathsep
            + os.path.abspath(os.path.join(root_dir, "..", "Unitree_Go2", "models"))
            + os.pathsep
            + env.get("GZ_SIM_RESOURCE_PATH", "")
        )

        # VDI Rendering Fixes
        if os.environ.get("FORCE_SOFTWARE_RENDER", "1") == "1":
            env["LIBGL_ALWAYS_SOFTWARE"] = "1"
            env["QT_X11_NO_MITSHM"] = "1"

        gz_args = ["gz", "sim", self.world_path]
        if os.environ.get("GZ_HEADLESS", "0") == "1":
            gz_args.append("-s")

        print(f"[GazeboDriver] Partition: {partition}")
        print(f"[GazeboDriver] Launching: {' '.join(gz_args)}")

        # Launch Gazebo in a new process group with the ENRICHED env
        self.gz_proc = subprocess.Popen(gz_args, env=env, preexec_fn=os.setsid)

        # Match our own process partition for topic discovery
        os.environ["GZ_PARTITION"] = partition
        print(f"[GazeboDriver] Initializing GzTransportNode on partition: {partition}")
        self.gz_node = GzTransportNode()

        # Advertise joint topics
        self.joint_pubs = []
        for jname in JOINT_NAMES:
            topic = f"/model/{self.robot_type}/joint/{jname}/cmd_force"
            self.joint_pubs.append(self.gz_node.advertise(topic, double_pb2.Double))

        time.sleep(5.0)

        # 4. Bind Subscriptions
        self.gz_node.subscribe(
            imu_pb2.IMU,
            f"/model/{self.robot_type}/link/base/sensor/imu/imu",
            self._imu_cb,
        )
        self.gz_node.subscribe(
            model_pb2.Model, f"/model/{self.robot_type}/joint_state", self._joint_cb
        )
        self.gz_node.subscribe(
            odometry_pb2.Odometry, f"/model/{self.robot_type}/odometry", self._odom_cb
        )
        self.gz_node.subscribe(
            world_stats_pb2.WorldStatistics,
            f"/world/{self.world_name}/stats",
            self._stats_cb,
        )

        # 5. Wait for First State
        print("[GazeboDriver] Waiting for simulation to start and first joint state...")
        while (self.sim_time == 0) and not self._stop.is_set():
            time.sleep(0.1)

        print(
            f"[GazeboDriver] Simulation started at t={self.sim_time:.2f}. Initializing targets."
        )
        self.latest_targets[:] = (
            self.q if not np.all(self.q == 0) else self.latest_targets
        )



        print(
            "\n=======================================================\n"
            "[GazeboDriver] Activating Control Loop..."
            "\n=======================================================\n"
        )
        actuator_count = 0
        count = 0
        pos_err_hist = np.zeros((6, 12), dtype=np.float32)
        vel_hist = np.zeros((6, 12), dtype=np.float32)

        while not self._stop.is_set():
            self.new_data_event.wait(timeout=0.1)
            self.new_data_event.clear()

            # We process a new PD/NN step every time we get a joint state callback (200 Hz).
            # Do NOT skip steps based on sim_time, as it can cause massive latency
            # if timestamps are coarse or not updated fast enough.
            
            # --- Internal Inference (50 Hz) ---
            if self.runner:
                if self.runner.should_step():
                    # 1. Use centralized parser for Standardization
                    state = self.telemetry.process_state(
                        q=self.q,
                        dq=self.dq,
                        quat=self.base_quat,
                        gyro=self.base_ang_vel,
                        vel=self.base_lin_vel_b,
                        pos=self.base_pos,
                        accel=self.base_accel,
                    )

                    # 2. Feed Policy (Unified Inference with Timing)
                    actions, _ = self.runner.infer(
                        state, self.cmd_vel, self.desired_qpos, self.mj_to_isaac
                    )

                    # 3. Use CommandProcessor for Sequenced Pipelining (Limit -> Sim -> ROS)
                    self.latest_targets[:] = self.command_processor.process(
                        actions, self.desired_qpos
                    )



            actuator_count += 1

            # Motor model matching MuJoCo.
            # NOTE: self.q and self.dq are already in Isaac convention (HAA signs corrected).
            # Targets from the policy are also in Isaac convention.
            # After computing torques in Isaac space, apply HAA_SIGN again to convert
            # back to Gazebo convention before publishing.
            targets = self.latest_targets
            kp = self.ctrl_cfg.get("kp", 25.0)
            kd = self.ctrl_cfg.get("kd", 0.5)
            
            # Override with safety watchdog torque
            effort_limit = self.command_processor.active_max_torque
            sat_effort = self.motor_cfg.get("max_torque", 45.0)
            vel_lim = self.motor_cfg.get("max_velocity", 30.0)
            
            if effort_limit <= 0.1:
                kp = 0.0
                kd = 0.0
            
            pos_err = targets - self.q
            raw_torques = kp * pos_err - kd * self.dq
            
            vel_at_lim = vel_lim * (1 + effort_limit / sat_effort)
            v_clamp = np.clip(self.dq, -vel_at_lim, vel_at_lim)
            t_top = effort_limit * (1.0 - v_clamp / vel_lim)
            t_bot = effort_limit * (-1.0 - v_clamp / vel_lim)
            
            pd_torques = np.clip(
                raw_torques, np.minimum(t_bot, -effort_limit), np.minimum(t_top, effort_limit)
            )

            # Convert torques from Isaac convention back to Gazebo convention
            # (negate right-side HAA torques to match Gazebo's +1 0 0 axis).
            self.latest_torques[:] = pd_torques * HAA_SIGN

            # LOGGING FOR DIAGNOSIS (every 100 physics steps ~ 0.2s)
            # Silenced debug print to keep terminal clean
            pass

            if count % 4 == 0:
                state = self.telemetry.process_state(
                    q=self.q,
                    dq=self.dq,
                    quat=self.base_quat,
                    gyro=self.base_ang_vel,
                    vel=self.base_lin_vel_b,
                    pos=self.base_pos,
                    accel=self.base_accel,
                )
                self.telemetry.publish(sim_time=self.sim_time, state=state)
            count += 1
            if count % 200 == 0:
                # Access latest inference time if available from runner
                inf_ms = 0.0
                if self.runner and hasattr(self.runner, "inf_times") and self.runner.inf_times:
                    inf_ms = self.runner.inf_times[-1] * 1000
                print(
                    f"\r[Bridge] t={self.sim_time:7.2f} h={self.base_pos[2]:.2f} vx={self.base_lin_vel_b[0]:+5.2f} | inf={inf_ms:4.1f}ms   ",
                    end="",
                    flush=True,
                )

    def _cleanup(self):
        """Clean up Gazebo server and simulator processes."""
        self._stop.set()
        if hasattr(self, "gz_proc") and self.gz_proc:
            print(f"[GazeboDriver] Terminating simulator (PID {self.gz_proc.pid})...")
            try:
                os.killpg(os.getpgid(self.gz_proc.pid), signal.SIGTERM)
                self.gz_proc.wait(timeout=5)
            except Exception:
                subprocess.run(
                    ["pkill", "-9", "-f", "gz-sim-server"], stderr=subprocess.DEVNULL
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="go2")
    parser.add_argument("--world", type=str, default="quadruped_world")
    parser.add_argument(
        "--internal_policy",
        type=str,
        default=None,
        help="Path to policy checkpoint (Turbo Mode)",
    )
    parser.add_argument("--obs_dim", type=int, default=49)
    parser.add_argument(
        "--use_estimator", action="store_true", default=False,
        help="Replace perfect odometry with contact-aided IMU velocity estimator (for sim2real testing)"
    )
    args = parser.parse_args()
    rclpy.init()
    node = Ros2GazeboDriver(
        args.robot, args.world, checkpoint=args.internal_policy,
        obs_dim=args.obs_dim, use_estimator=args.use_estimator,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._cleanup()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
