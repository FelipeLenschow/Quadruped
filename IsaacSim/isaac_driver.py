"""
Isaac Sim Driver for Quadruped Locomotion.
Integrates the simulator's Articulation interface with the central
Policy Runner and Command Processor (Turbo Mode).
"""

import argparse
import sys
import os
import os
import sys

# Ensure absolute path of the repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import torch

# Launcher needs to happen before any other Omniverse imports
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="IsaacSim ROS 2 Bridge for Quadruped.")
parser.add_argument(
    "--robot", type=str, default="go2", help="Robot type (go2, go1, a1)"
)
parser.add_argument(
    "--internal_policy",
    type=str,
    default=None,
    help="Path to policy checkpoint (Turbo Mode)",
)
parser.add_argument("--obs_dim", type=int, default=49, help="Observation dimension")
parser.add_argument("--use_estimator", action="store_true", help="Use IMU-based state estimation instead of ground truth")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- ROS 2 Compatibility Fix for Isaac Sim (Python 3.11) ---
import sys

isaac_ros_path = "/home/05680435969@corp.udesc.br/env_isaacsim/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/rclpy"
if os.path.exists(isaac_ros_path):
    if isaac_ros_path not in sys.path:
        sys.path.insert(0, isaac_ros_path)
    # Also ensure the rclpy directory inside it is found
    # The structure is humble/rclpy/rclpy/...
    # But usually just adding 'humble' to sys.path is enough if it contains 'rclpy' folder
# ---------------------------------------------------------

# --- Rest of imports ---
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Quaternion, Vector3, Twist
from nav_msgs.msg import Odometry

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from Controller.policy_runner import PolicyRunner
from Controller.policy_bridge import CommandProcessor
from Controller.Utils.telemetry import TelemetryManager
from configs.config_loader import load_config

from isaaclab_assets.robots.unitree import (
    UNITREE_A1_CFG,
    UNITREE_GO1_CFG,
    UNITREE_GO2_CFG,
)


@configclass
class BridgeSceneCfg(InteractiveSceneCfg):
    ground = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="average",
            restitution_combine_mode="average",
        ),
    )
    robot: ArticulationCfg = None  # To be set dynamically
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


class Ros2IsaacDriver(Node):
    def __init__(self, robot, robot_type="go2", checkpoint=None, obs_dim=49, use_estimator=False):
        super().__init__("isaac_driver")
        self.robot = robot
        self.robot_type = robot_type

        # 0. Load Central Config
        self.config = load_config()
        self.ctrl_cfg = self.config.get("control", {})

        # Handles internal policy inference, physics stepping,
        # and standardizes telemetry for ROS 2 monitoring.
        self.runner = None
        if checkpoint:
            print(f"[IsaacDriver] Loading internal policy runner: {checkpoint}")
            self.runner = PolicyRunner(
                checkpoint, obs_dim=obs_dim, robot_type=robot_type
            )
            self.runner.decimation = self.ctrl_cfg.get("decimation", 4)
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
            # Joint index mapping: ISAAC to Type-Grouped (identity in our bridge)
            self.mj_to_isaac = list(range(12))

        # Joint Configuration
        # (Internal names are what Isaac Sim uses, joint_names are what we group into ROS)
        self.internal_names = self.robot.data.joint_names
        # 1. Join Mapping (Strict Type-Grouped order for compatibility)
        self.joint_names = [
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

        # Map our expected names to internal indices from ArticulationData
        self.mapped_dof_idx = []
        all_internal_names = self.robot.data.joint_names
        for name in self.joint_names:
            if name in all_internal_names:
                self.mapped_dof_idx.append(all_internal_names.index(name))
            else:
                # Pattern match fallback
                found = False
                for i, inter_name in enumerate(all_internal_names):
                    if name.lower() in inter_name.lower():
                        self.mapped_dof_idx.append(i)
                        found = True
                        break
                if not found:
                    print(
                        f"[IsaacDriver] ERROR: Could not find joint {name} in simulation!"
                    )

        if len(self.mapped_dof_idx) != 12:
            print(
                f"[IsaacBridge] WARNING: Found only {len(self.mapped_dof_idx)}/12 joints!"
            )

        # ROS 2 Managers
        self.telemetry = TelemetryManager(self, self.joint_names, use_estimator=use_estimator)
        self.command_processor = CommandProcessor(
            self, robot_type=robot_type, joint_names=self.joint_names
        )
        self.cmd_echo_pub = self.create_publisher(
            JointState, "/commands/joint_commands", 10
        )

        # 4. State Buffers
        self.latest_targets = torch.zeros(
            (1, 12), device=self.robot.data.joint_pos.device, dtype=torch.float32
        )
        if hasattr(self, "desired_qpos"):
            self.latest_targets[:] = torch.from_numpy(self.desired_qpos).to(
                self.latest_targets
            )

        # 3. Subscriptions
        self.create_subscription(Twist, "/cmd_vel", self.teleop_cb, 10)

        # 4. Buffers
        self.latest_targets = self.robot.data.default_joint_pos[
            0, self.mapped_dof_idx
        ].clone()
        self.cmd_vel = [0.0, 0.0, 0.0, 0.0]
        self.step_counter = 0

        print(
            f"[IsaacDriver] Initialized for {self.robot_type.upper()} with {len(self.mapped_dof_idx)} joints."
        )
        defaults = self.robot.data.default_joint_pos[0, self.mapped_dof_idx].tolist()
        for i, name in enumerate(self.joint_names):
            print(
                f"  - Joint [{i}] (Idx {self.mapped_dof_idx[i]}): {name} | Default: {defaults[i]:.3f}"
            )

    def _get_standard_state(self):
        """Extracts raw state vectors from Isaac Sim data."""
        robot_data = self.robot.data
        q = robot_data.joint_pos[0, self.mapped_dof_idx].cpu().numpy()
        dq = robot_data.joint_vel[0, self.mapped_dof_idx].cpu().numpy()
        quat = robot_data.root_quat_w[0].cpu().numpy()  # [w, x, y, z]
        gyro = robot_data.root_ang_vel_b[0].cpu().numpy()
        vel_b = robot_data.root_lin_vel_b[0].cpu().numpy()
        pos = robot_data.root_pos_w[0].cpu().numpy()

        return self.telemetry.process_state(
            q=q, dq=dq, quat=quat, gyro=gyro, pos=pos, vel=vel_b
        )

    def teleop_cb(self, msg):
        # We just store it; if using policy_bridge, this is mostly for completeness
        # as policy_bridge subscribes to /cmd_vel directly.
        self.cmd_vel = [msg.linear.x, msg.linear.y, msg.angular.z, 0.0]

    def step(self):
        # --- Internal Inference (50 Hz) ---
        if self.runner:
            if self.runner.should_step():
                # 1. Use standardized parser
                state = self._get_standard_state()

                # 2. Feed Policy (Unified Inference with Timing)
                actions, _ = self.runner.infer(
                    state, self.cmd_vel, self.desired_qpos, self.mj_to_isaac
                )

                # 3. Use CommandProcessor for Sequenced Pipelining (Limit -> Sim -> ROS)
                targets = self.command_processor.process(actions, self.desired_qpos)
                self.latest_targets[:] = torch.from_numpy(targets).to(
                    device=self.robot.data.joint_pos.device, dtype=torch.float32
                )

        # 1. PD Effort Calculation (Matching training: Kp=25.0, Kd=0.5)
        curr_jpos = self.robot.data.joint_pos[0, self.mapped_dof_idx]
        curr_jvel = self.robot.data.joint_vel[0, self.mapped_dof_idx]

        pos_err = self.latest_targets - curr_jpos

        kp = self.ctrl_cfg.get("kp", 25.0)
        kd = self.ctrl_cfg.get("kd", 0.5)
        effort_limit = 23.5

        torques = kp * pos_err + kd * (0 - curr_jvel)
        torques = torch.clamp(torques, -effort_limit, effort_limit).to(torch.float32)

        # Apply Actions via Effort
        self.robot.set_joint_effort_target(torques, joint_ids=self.mapped_dof_idx)

        if self.runner and self.runner.counter % 10 == 0:
            # Publish echo of internal commands for PlotJuggler
            js_echo = JointState()
            js_echo.header.stamp = self.get_clock().now().to_msg()
            js_echo.name = self.joint_names
            js_echo.position = self.latest_targets.tolist()
            self.cmd_echo_pub.publish(js_echo)

        # Telemetry every 1s
        now = time.time()
        if not hasattr(self, "_last_telemetry"):
            self._last_telemetry = 0
        if now - self._last_telemetry > 1.0:
            root_v_w = self.robot.data.root_com_lin_vel_w[0].tolist()
            root_v_b = self.robot.data.root_lin_vel_b[0].tolist()

            # Per-joint error check
            curr_jpos = self.robot.data.joint_pos[0, self.mapped_dof_idx]
            j_errs = torch.abs(curr_jpos - self.latest_targets)
            max_err_idx = torch.argmax(j_errs).item()

            print(
                f"[IsaacDriver] V_body: {[round(x,3) for x in root_v_b]} | Max Error: {j_errs[max_err_idx]:.3f} on {self.joint_names[max_err_idx]}"
            )
            self._last_telemetry = now

        # 2. Publish Standardized Telemetry (Downsampled to 20Hz for network efficiency)
        if self.step_counter % 5 == 0:
            state = self._get_standard_state()
            self.telemetry.publish(
                sim_time=float(self.step_counter * 0.005), state=state
            )

        self.step_counter += 1


def main():
    # 1. Initialize Simulation Context
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        render_interval=1,
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            max_position_iteration_count=255,
            max_velocity_iteration_count=255,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
        ),
    )
    sim_context = sim_utils.SimulationContext(sim_cfg)

    # 2. Setup Scene
    scene_cfg = BridgeSceneCfg(num_envs=1, env_spacing=2.0)

    # 2. Extract Robot Config
    if args_cli.robot == "a1":
        scene_cfg.robot = UNITREE_A1_CFG.replace(prim_path="/World/envs/env_0/Robot")
    elif args_cli.robot == "go1":
        scene_cfg.robot = UNITREE_GO1_CFG.replace(prim_path="/World/envs/env_0/Robot")
    else:
        scene_cfg.robot = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_0/Robot")

    # Neutralize internal gains so bridge manual PID takes over
    for actuator_cfg in scene_cfg.robot.actuators.values():
        if hasattr(actuator_cfg, "stiffness"):
            actuator_cfg.stiffness = 0.0
            actuator_cfg.damping = 0.0

    scene_cfg.robot.init_state.pos = (0.0, 0.0, 0.5)

    scene = InteractiveScene(scene_cfg)
    robot = scene.articulations["robot"]

    # 4. ROS 2 Init
    rclpy.init()

    # 5. Play Sim
    sim_context.reset()
    scene.reset()
    scene.update(0.0)

    # 6. Initialize Driver (After reset so .data is available)
    node = Ros2IsaacDriver(
        robot,
        args_cli.robot,
        checkpoint=args_cli.internal_policy,
        obs_dim=args_cli.obs_dim,
        use_estimator=args_cli.use_estimator,
    )

    print("[IsaacDriver] Starting simulation loop...")
    next_time = time.time()
    count = 0
    while simulation_app.is_running():
        # 1. Sync ROS 2 (Drain all pending messages)
        while rclpy.ok():
            if not rclpy.spin_once(node, timeout_sec=0.0):
                break

        # 2. Driver Logic (Manual PD)
        node.step()

        # 3. Write data to sim (APPLY EFFORTS TO PHYSX)
        scene.write_data_to_sim()

        # 4. Physics Step (Adv 5ms)
        sim_context.step(render=args_cli.headless is False)

        # 5. Refresh buffers by updating scene (GET SENSORS FROM PHYSX)
        scene.update(0.005)

        count += 1
        if count % 100 == 0:
            pos = robot.data.root_pos_w[0]
            print(
                f"\r[IsaacDriver] Sim Time Step: {count} | Robot Height: {pos[2]:.3f}m | T: {sim_context.current_time:.2f}s\n",
                end="",
                flush=True,
            )

        # 6. Real-time sync
        while time.time() < next_time:
            time.sleep(0.0001)
        next_time += 0.005

    # Cleanup
    node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
