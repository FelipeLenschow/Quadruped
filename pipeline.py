import os
import numpy as np
from Telemetry.telemetry import TelemetryManager
from Controller.policy_manager import PolicyManager
from Controller.command_safety_processor import CommandSafetyProcessor
from Controller.distributor import Distributor
from Configs.config_loader import load_config


class LocomotionPipeline:
    """
    Centralized pipeline encapsulating Telemetry, Policy Selection/Inference,
    Safety Arbitration, and Command Distribution.

    Ensures identical execution across MuJoCo, Gazebo, Isaac Sim, and Physical Hardware.
    """
    def __init__(self, node, robot_type="go2", checkpoint=None, obs_dim=49,
                 use_estimator=False, joint_names=None):
        self.node = node
        self.robot_type = robot_type

        self.config = load_config()
        self.ctrl_cfg = self.config.get("control", {})
        safety_cfg = self.config.get("safety", {})

        # 1. Telemetry Manager
        self.telemetry = TelemetryManager(node, joint_names, use_estimator=use_estimator)

        # 2. Policy Manager (Unified registry for policy runners)
        self.policy_manager = PolicyManager(node, robot_type=robot_type, obs_dim=obs_dim)

        # Register Main Policy (policy under test selected by the launcher)
        if checkpoint:
            self.policy_manager.load_policy("main", checkpoint)
        else:
            self.node.get_logger().warn(
                "[LocomotionPipeline] No main policy checkpoint provided. Policy runner disabled.")

        # Register Safety Policy (backup recovery policy configured in config.yaml)
        safety_policy_path = safety_cfg.get("safety_policy_path", "")
        if safety_policy_path:
            loaded = self.policy_manager.load_policy("safety", safety_policy_path)
            if not loaded:
                self.node.get_logger().warn(
                    "[LocomotionPipeline] Safety policy failed to load. "
                    "Robot will DISABLE directly on safety violations.")
        else:
            self.node.get_logger().info(
                "[LocomotionPipeline] No safety policy configured. "
                "Robot will DISABLE directly on safety violations.")

        self.mj_to_isaac = list(range(12))  # Standard mapping
        self.decimation = self.ctrl_cfg.get("decimation", 4)

        # 3. Command Safety Processor (safety checking + arbitration)
        self.safety_processor = CommandSafetyProcessor(
            node, robot_type=robot_type, joint_names=joint_names)

        # 4. Distributor (hardware/ROS command output)
        self.distributor = Distributor(node, joint_names=joint_names)

        # Nominal standing pose (default fallback)
        self.desired_qpos = np.array([
            0.1, -0.1, 0.1, -0.1,  # hips
            0.8, 0.8, 1.0, 1.0,    # thighs
            -1.5, -1.5, -1.5, -1.5  # calves
        ], dtype=np.float32)

        self.latest_targets = self.desired_qpos.copy()
        self.step_counter = 0

    def step(self, raw_state_kwargs, cmd_vel, sim_time):
        """
        Executes one step of the pipeline.

        Args:
            raw_state_kwargs: dict containing q, dq, quat, gyro, accel, pos, vel, contact, etc.
            cmd_vel: list/array of velocity commands [vx, vy, wz, unused]
            sim_time: current simulation or physical time

        Returns:
            latest_targets (np.ndarray): The target joint positions to send to the motors.
        """
        # Determine if this is a policy inference step (e.g., 50Hz)
        is_policy_step = (self.step_counter % self.decimation) == 0
        self.step_counter += 1

        # 1. Standardize State
        raw_state_kwargs['update_estimator'] = is_policy_step
        state = self.telemetry.process_state(**raw_state_kwargs)

        # 2. Policy Inference & Command Processing
        if is_policy_step and "main" in self.policy_manager.policies:
            # 2a. Pre-evaluate safety to determine which policy is actually needed
            is_safe, _ = self.safety_processor.evaluate_safety(state)
            active_policy = "main" if is_safe else "safety"

            # 2b. Only query the policy runner that is actually needed
            proposed_targets = {}
            if active_policy == "main":
                proposed_targets["main"] = self.policy_manager.step_single(
                    "main", state, cmd_vel, self.mj_to_isaac
                )
            elif active_policy == "safety" and "safety" in self.policy_manager.policies:
                proposed_targets["safety"] = self.policy_manager.step_single(
                    "safety", state, cmd_vel, self.mj_to_isaac
                )

            # 2c. Safety gate & select winner
            final_targets, max_torque = self.safety_processor.process(
                proposed_targets=proposed_targets,
                state=state
            )

            self.latest_targets = final_targets

            # Distribute final command to ROS 2/robot drivers
            self.distributor.send(final_targets, max_torque)

        # 3. Telemetry Publishing
        if is_policy_step:
            self.telemetry.publish(sim_time=sim_time, state=state)

        return self.latest_targets
