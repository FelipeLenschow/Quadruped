import os
import sys
import threading
import numpy as np
from rclpy.node import Node

# Ensure project root is importable

from quadruped_core.controller.policy_runner import PolicyRunner
from quadruped_core.controller.pose_generator import PoseGenerator
from quadruped_core.config_loader import load_config, load_policies
from quadruped_core import paths
from quadruped_core.controller.robot_defaults import DEFAULT_STANCE_QPOS, PROTECTIONS
from quadruped_core import realtime


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

        # Nominal Standing Pose (Default)
        self.desired_qpos = DEFAULT_STANCE_QPOS.copy()

        # Policies that can be switched to by name (policies.json), and the settings of each
        # loaded one. Only loaded policies are in self.policies.
        policies_cfg = load_policies()
        self.safety_policy = policies_cfg.get("safety_policy") or ""
        self.catalog = self._read_catalog(policies_cfg.get("policies") or {})
        self.specs = {}
        self.load_errors = {}
        self._loading = set()
        self._lock = threading.Lock()

    def _read_catalog(self, policies: dict) -> dict:
        out = {}
        for name, spec in policies.items():
            spec = spec or {}
            protections = spec.get("protections")
            protections = PROTECTIONS if protections is None else tuple(protections)
            unknown = [p for p in protections if p not in PROTECTIONS]
            if unknown:
                self.node.get_logger().error(
                    f"[PolicyManager] Policy '{name}': unknown protections {unknown}, "
                    f"known: {list(PROTECTIONS)}. Using all of them.")
                protections = PROTECTIONS
            out[name] = {
                "path": spec.get("path", ""),
                "protections": protections,
                "commands": bool(spec.get("commands", True)),
            }
        return out

    def spec(self, name: str) -> dict:
        """protections and commands of a policy; all checks and real commands when it has no entry."""
        return self.specs.get(name) or self.catalog.get(name) or {
            "path": "", "protections": PROTECTIONS, "commands": True}

    @property
    def available(self) -> list:
        """Names that can be switched to: "main" when loaded, then policies.json's list."""
        main = self.specs.get("main")
        return (["main"] if "main" in self.policies else []) + [
            n for n, spec in self.catalog.items() if n != "main" and spec is not main]

    def is_loading(self, name: str) -> bool:
        return name in self._loading

    def request(self, name: str) -> bool:
        """Make a policy available, loading it in the background if needed. False if it is unknown."""
        if name in self.policies:
            return True
        if name not in self.catalog:
            return False
        with self._lock:
            if name in self._loading:
                return True
            self._loading.add(name)
            self.load_errors.pop(name, None)
        threading.Thread(target=self._load_in_background, args=(name,), daemon=True).start()
        return True

    def _load_in_background(self, name: str):
        # Started from the driver's thread, so it would inherit the policy's core at SCHED_FIFO and
        # hold up the control loop for as long as the load takes.
        realtime.leave_realtime()
        try:
            if not self.load_policy(name, self.catalog[name]["path"]):
                self.load_errors[name] = "could not load (see the driver's log)"
        except Exception as e:
            self.load_errors[name] = str(e)
        finally:
            with self._lock:
                self._loading.discard(name)

    def unload(self, name: str):
        self.policies.pop(name, None)
        self.specs.pop(name, None)

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
            checkpoint_path = paths.repo(checkpoint_path)

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
                robot_type=self.robot_type,
                obs_dim_from_env=name == "main",
            )
            self.specs[name] = self._spec_for(name, checkpoint_path)
            self.policies[name] = runner
            spec = self.specs[name]
            self.node.get_logger().info(
                f"[PolicyManager] '{name}': protections {list(spec['protections']) or 'none'}"
                f"{', zero velocity command' if not spec['commands'] else ''}.")
            return True
        except Exception as e:
            self.node.get_logger().error(
                f"[PolicyManager] Failed to load policy '{name}': {e}")
            return False

    def _spec_for(self, name: str, checkpoint_path: str) -> dict:
        """A policy's own entry, or the entry whose path is this checkpoint (the launcher's "main")."""
        if name in self.catalog:
            return self.catalog[name]
        real = os.path.realpath(checkpoint_path)
        for spec in self.catalog.values():
            if spec["path"] and os.path.realpath(paths.repo(spec["path"])) == real:
                return spec
        return {"path": checkpoint_path, "protections": PROTECTIONS, "commands": True}

    def register_pose_generator(self, node: Node):
        """
        Create and register a PoseGenerator as a policy provider under 'pose'.
        """
        pg = PoseGenerator(node)
        self.policies["pose"] = pg
        self.node.get_logger().info(
            "[PolicyManager] PoseGenerator registered under 'pose'.")
        return pg

    def step_single(self, name: str, state, commands, mapping,
                    current_time: float = None, dt: float = 0.02) -> np.ndarray:
        """
        Step inference and compute joint-space targets for a single registered policy.

        Args:
            name:     The registered identifier of the policy (e.g. 'main', 'safety').
            state:    StandardState from TelemetryManager.
            commands: Velocity commands [vx, vy, wz, height_cmd].
            mapping:  Joint mapping (e.g., self.mj_to_isaac).
            current_time: Simulation time for pose generator interpolation.
            dt:       Simulation time elapsed between policy steps (default 0.02s / 50Hz).

        Returns:
            np.ndarray: Proposed absolute joint targets in radians.
        """
        if name not in self.policies:
            raise KeyError(f"[PolicyManager] Policy '{name}' is not loaded.")

        provider = self.policies[name]

        # --- Polymorphic dispatch ---
        if isinstance(provider, PoseGenerator):
            # PoseGenerator returns absolute joint targets directly
            return provider.step(state, current_time)

        # --- Standard PolicyRunner path ---
        runner = provider
        if not self.spec(name)["commands"]:
            commands = np.zeros_like(np.asarray(commands, dtype=np.float32))
        # 1. Inference (returns raw action vector in [-1, 1])
        raw_actions, _ = runner.infer(
            state,
            commands,
            runner.desired_qpos if runner.desired_qpos is not None else self.desired_qpos,
            mapping,
            dt=dt
        )

        # 2. Scale & Center to get absolute targets (radians)
        scale = getattr(runner, "action_scale", self.global_action_scale)
        qpos = runner.desired_qpos if runner.desired_qpos is not None else self.desired_qpos

        targets = raw_actions * scale + qpos
        return targets.astype(np.float32)

    def step_all(self, state, commands, mapping, dt: float = 0.02) -> dict:
        """
        Step inference and compute joint-space targets for all loaded policies.
        (Kept for backward compatibility, but step_single should be preferred for performance).
        """
        proposed_targets = {}
        for name in list(self.policies):
            try:
                proposed_targets[name] = self.step_single(name, state, commands, mapping, dt=dt)
            except Exception as e:
                self.node.get_logger().error(
                    f"[PolicyManager] Error during step on policy '{name}': {e}")
        return proposed_targets

    @property
    def loaded_policy_names(self) -> list:
        return list(self.policies.keys())

    def reset_policy_state(self):
        """Clear per-episode state on every loaded runner.

        Nothing used to call this, so an eval sweep carried observation history and last_actions
        from one velocity test into the next -- and once a test produced a NaN, every remaining
        test in the sweep inherited it and reported frozen, identical numbers.
        """
        for name, policy in list(self.policies.items()):
            if hasattr(policy, "reset_history"):
                policy.reset_history()
