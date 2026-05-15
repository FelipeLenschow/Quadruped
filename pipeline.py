import numpy as np
from Telemetry.telemetry import TelemetryManager
from Controller.policy_bridge import CommandProcessor
from Configs.config_loader import load_config

class LocomotionPipeline:
    """
    Centralized pipeline encapsulating Telemetry, Policy Inference, and Command Processing.
    Ensures identical execution across MuJoCo, Gazebo, Isaac Sim, and Physical Hardware.
    """
    def __init__(self, node, robot_type="go2", checkpoint=None, obs_dim=49, use_estimator=False, joint_names=None):
        self.node = node
        self.robot_type = robot_type
        
        self.config = load_config()
        self.ctrl_cfg = self.config.get("control", {})
        
        # 1. Telemetry Manager
        self.telemetry = TelemetryManager(node, joint_names, use_estimator=use_estimator)
        
        # 2. Policy Runner
        self.runner = None
        self.decimation = self.ctrl_cfg.get("decimation", 4)
        if checkpoint:
            self.node.get_logger().info(f"[LocomotionPipeline] Loading internal policy runner: {checkpoint}")
            try:
                from Controller.policy_runner import PolicyRunner
                self.runner = PolicyRunner(checkpoint, obs_dim=obs_dim, robot_type=robot_type)
                self.runner.decimation = self.decimation
                self.mj_to_isaac = list(range(12))  # Identity mapping standard
            except ImportError:
                self.node.get_logger().error("[LocomotionPipeline] PyTorch not found. Policy runner disabled.")
            
        # 3. Command Processor
        self.command_processor = CommandProcessor(node, robot_type=robot_type, joint_names=joint_names)
        
        # Nominal standing pose
        self.desired_qpos = np.array([
            0.1, -0.1, 0.1, -0.1,  # hips
            0.8, 0.8, 1.0, 1.0,    # thighs
            -1.5, -1.5, -1.5, -1.5 # calves
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
        # Only update the Kalman Filter at the policy frequency (e.g., 50Hz)
        raw_state_kwargs['update_estimator'] = is_policy_step
        state = self.telemetry.process_state(**raw_state_kwargs)
        
        # 2. Policy Inference & Command Processing
        if is_policy_step and self.runner:
            # Bypass runner.should_step() since we manage decimation in the pipeline
            actions, _ = self.runner.infer(state, cmd_vel, self.desired_qpos, self.mj_to_isaac)
            self.latest_targets = self.command_processor.process(actions, self.desired_qpos)
            
        # 3. Telemetry Publishing
        # Throttle telemetry to policy frequency to save bandwidth
        if is_policy_step:
            self.telemetry.publish(sim_time=sim_time, state=state)
            
        return self.latest_targets
