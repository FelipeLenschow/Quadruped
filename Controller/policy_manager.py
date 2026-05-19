import os
import sys
import numpy as np
from rclpy.node import Node

# Ensure project root is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Controller.policy_runner import PolicyRunner
from Configs.config_loader import load_config


class PolicyManager:
    """
    Centralized Coordinator for Robot Locomotion Policies.

    Manages a registry of policies (e.g., 'main' policy under test, 'safety'
    backup recovery policy, etc.).

    For all registered policies, it:
      1. Builds observations from the standardized StandardState.
      2. Executes model inference (via PolicyRunner).
      3. Scales and centers the actions into absolute joint targets (radians).
      4. Returns a dictionary of absolute targets proposed by each policy.
    """

    def __init__(self, node: Node, robot_type: str = "go2", obs_dim: int = 49):
        self.node = node
        self.robot_type = robot_type
        self.obs_dim = obs_dim
        self.policies = {}

        # Load standard control configs for scaling fallback
        self.config = load_config()
        self.ctrl_cfg = self.config.get("control", {})
        self.global_action_scale = self.ctrl_cfg.get("action_scale", 0.25)

        # Go2 Nominal Standing Pose (Default)
        self.desired_qpos = np.array([
            0.1, -0.1, 0.1, -0.1,  # hips
            0.8, 0.8, 1.0, 1.0,    # thighs
            -1.5, -1.5, -1.5, -1.5  # calves
        ], dtype=np.float32)

    def load_policy(self, name: str, checkpoint_path: str) -> bool:
        """
        Loads a PolicyRunner checkpoint and registers it under the given name.

        Args:
            name:            Unique identifier for the policy (e.g. 'main', 'safety').
            checkpoint_path: Path to the JIT or PyTorch policy checkpoint (.pt / .jit).

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not checkpoint_path:
            return False

        # Resolve relative paths from project root
        if not os.path.isabs(checkpoint_path):
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), ".."))
            checkpoint_path = os.path.join(project_root, checkpoint_path)

        if not os.path.exists(checkpoint_path):
            self.node.get_logger().warn(
                f"[PolicyManager] Policy file not found: {checkpoint_path}")
            return False

        self.node.get_logger().info(
            f"[PolicyManager] Loading policy '{name}' from: {checkpoint_path}")

        try:
            runner = PolicyRunner(
                checkpoint_path,
                obs_dim=self.obs_dim,
                robot_type=self.robot_type
            )
            self.policies[name] = runner
            return True
        except Exception as e:
            self.node.get_logger().error(
                f"[PolicyManager] Failed to load policy '{name}': {e}")
            return False

    def step_single(self, name: str, state, commands, mapping) -> np.ndarray:
        """
        Step inference and compute joint-space targets for a single registered policy.

        Args:
            name:     The registered identifier of the policy (e.g. 'main', 'safety').
            state:    StandardState from TelemetryManager.
            commands: Velocity commands [vx, vy, wz, height_cmd].
            mapping:  Joint mapping (e.g., self.mj_to_isaac).

        Returns:
            np.ndarray: Proposed absolute joint targets in radians.
        """
        if name not in self.policies:
            raise KeyError(f"[PolicyManager] Policy '{name}' is not loaded.")

        runner = self.policies[name]
        # 1. Inference (returns raw action vector in [-1, 1])
        raw_actions, _ = runner.infer(
            state,
            commands,
            runner.desired_qpos if runner.desired_qpos is not None else self.desired_qpos,
            mapping
        )

        # 2. Scale & Center to get absolute targets (radians)
        scale = getattr(runner, "action_scale", self.global_action_scale)
        qpos = runner.desired_qpos if runner.desired_qpos is not None else self.desired_qpos

        targets = raw_actions * scale + qpos
        return targets.astype(np.float32)

    def step_all(self, state, commands, mapping) -> dict:
        """
        Step inference and compute joint-space targets for all loaded policies.
        (Kept for backward compatibility, but step_single should be preferred for performance).
        """
        proposed_targets = {}
        for name in self.policies.keys():
            try:
                proposed_targets[name] = self.step_single(name, state, commands, mapping)
            except Exception as e:
                self.node.get_logger().error(
                    f"[PolicyManager] Error during step on policy '{name}': {e}")
        return proposed_targets

    @property
    def loaded_policy_names(self) -> list:
        return list(self.policies.keys())
