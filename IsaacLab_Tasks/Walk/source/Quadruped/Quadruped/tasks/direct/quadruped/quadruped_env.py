# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import copy
import random
from collections.abc import Sequence
from typing import Dict, Tuple

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import sample_uniform

from .quadruped_env_cfg import QuadrupedEnvCfg


class QuadrupedEnv(DirectRLEnv):
    """
    A simplified environment for getting started with Reinforcement Learning on a quadruped robot (Unitree QUADRUPED).
    This environment focuses on the basics: controlling joint positions to keep the robot upright.
    """

    cfg: QuadrupedEnvCfg

    def __init__(self, cfg: QuadrupedEnvCfg, render_mode: str | None = None, **kwargs):
        # Initialize base environment (calls _setup_scene)
        super().__init__(cfg, render_mode, **kwargs)

        # 4. Finalize buffers (Simulation has been reset by super())
        if getattr(self, "is_heterogeneous", False):
            # Global Aggregation Buffers
            self.joint_pos = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel = torch.zeros((self.num_envs, 12), device=self.device)
            self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
            self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
            self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device)
            self.body_pos_w = torch.zeros(
                (self.num_envs, self.robot_views[0].num_bodies, 3), device=self.device
            )
            self.root_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
            self.root_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
            self.applied_torque = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_acc = torch.zeros((self.num_envs, 12), device=self.device)

            self.desired_joint_pos = torch.zeros(
                (self.num_envs, 12), device=self.device
            )
            self.robot_feet_ids = []
            for i, view in enumerate(self.robot_views):
                indices = self.robot_view_indices[i]
                self.desired_joint_pos[indices] = view.data.default_joint_pos[
                    0, :12
                ].clone()
                # Find feet for this specific view (relative to Articulation)
                f_ids, _ = view.find_bodies(".*_foot")
                # Normalize order: FL, FR, RL, RR
                if len(f_ids) >= 4:
                    self.robot_feet_ids.append([f_ids[2], f_ids[3], f_ids[0], f_ids[1]])
                else:
                    self.robot_feet_ids.append(f_ids)

            # Contact sensor mapping (relative to sensor matched bodies)
            c_feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")
            if len(c_feet_ids) >= 4:
                self._feet_ids = [c_feet_ids[2], c_feet_ids[3], c_feet_ids[0], c_feet_ids[1]]
            else:
                self._feet_ids = c_feet_ids
        else:
            self.joint_pos = self.robot.data.joint_pos
            self.joint_vel = self.robot.data.joint_vel
            self.base_lin_vel = self.robot.data.root_lin_vel_b
            self.base_ang_vel = self.robot.data.root_ang_vel_b
            self.projected_gravity = self.robot.data.projected_gravity_b
            self.body_pos_w = self.robot.data.body_pos_w
            self.root_pos_w = self.robot.data.root_pos_w
            self.root_quat_w = self.robot.data.root_quat_w
            self.applied_torque = self.robot.data.applied_torque
            self.joint_acc = self.robot.data.joint_acc
            self.desired_joint_pos = self.robot.data.default_joint_pos[:, :12].clone()
            feet_ids, _ = self.robot.find_bodies(".*_foot")
            # Articulation ordering: FL(2), FR(3), RL(0), RR(1)
            self._feet_ids_articulation = [
                feet_ids[2],
                feet_ids[3],
                feet_ids[0],
                feet_ids[1],
            ]
            # Contact sensor mapping
            c_feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")
            if len(c_feet_ids) >= 4:
                self._feet_ids = [c_feet_ids[2], c_feet_ids[3], c_feet_ids[0], c_feet_ids[1]]
            else:
                self._feet_ids = c_feet_ids
            
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*_thigh|.*_calf|trunk")

        self.net_contact_forces = torch.zeros(self.num_envs, 20, 3, device=self.device)
        self._joint_dof_idx, _ = self.robot.find_joints(
            ".*_hip_joint|.*_thigh_joint|.*_calf_joint"
        )
        if getattr(self, "is_heterogeneous", False):
            self._view_joint_dof_idx = []
            for v in self.robot_views:
                idx, _ = v.find_joints(".*_hip_joint|.*_thigh_joint|.*_calf_joint")
                self._view_joint_dof_idx.append(torch.tensor(idx, dtype=torch.long, device=self.device))
        
        self.actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        self.previous_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        self.commands = torch.zeros(self.num_envs, 4, device=self.device)
        self.target_commands = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_joint_vel = torch.zeros(self.num_envs, 12, device=self.device)
        self.last_base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.ref_pos_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.ref_yaw = torch.zeros(self.num_envs, device=self.device)
        self.pos_deviation_val = torch.zeros(self.num_envs, device=self.device)
        self.yaw_deviation_val = torch.zeros(self.num_envs, device=self.device)
        self.feet_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_feet_contact = torch.zeros(
            self.num_envs, 4, dtype=torch.bool, device=self.device
        )
        self.feet_air_time_reward_val = torch.zeros(self.num_envs, device=self.device)
        self.foot_height_reward_val = torch.zeros(self.num_envs, device=self.device)
        self.feet_air_penalty_val = torch.zeros(self.num_envs, device=self.device)
        self.feet_air_penalty_static_val = torch.zeros(self.num_envs, device=self.device)
        self.joint_vel_l2_static_val = torch.zeros(self.num_envs, device=self.device)
        self.grf_balance_val = torch.zeros(self.num_envs, device=self.device)
        self.grf_target_val = torch.zeros(self.num_envs, device=self.device)
        self.max_contact_force_val = torch.zeros(self.num_envs, device=self.device)
        self.robot_total_weight = torch.zeros(self.num_envs, device=self.device)  # mg per env, updated on reset
        # Strike-time buffers for contact-driven gait phase symmetry reward
        # last_strike_time: simulation time of last touchdown per foot (N, 4) [FL, FR, RL, RR]
        # stride_duration:  time between the last two consecutive touchdowns per foot (N, 4)
        self.last_strike_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.stride_duration  = torch.ones(self.num_envs, 4, device=self.device)  # init to 1.0 to avoid div-by-zero
        self.gait_phase_sym_val = torch.zeros(self.num_envs, device=self.device)
        self.command_timer = torch.full(
            (self.num_envs,), 100.0, device=self.device
        )  # Force immediate resample

        if self.cfg.obs_history_len > 0:
            self.obs_history_buf = torch.zeros(
                self.num_envs, self.cfg.obs_history_len * 49, device=self.device
            )

        # Internal Curriculum Sequence
        self.agent_steps = 0
        self.curriculum_phase_idx = 0
        self.curriculum_phases = getattr(self.cfg, "curriculum_phases", [])
        
        self.curriculum_thresholds = []
        if self.curriculum_phases:
            cumulative_steps = getattr(self.cfg, "base_max_timesteps", 500000)
            for p in self.curriculum_phases:
                self.curriculum_thresholds.append(cumulative_steps)
                cumulative_steps += p["max_timesteps"]

        # Domain Randomization Buffers
        max_delay = self.cfg.action_latency_range_steps[1]
        for p in self.curriculum_phases:
            dr_cfg = p["cfg"].get("domain_randomization", {})
            if "action_latency_range_steps" in dr_cfg:
                max_delay = max(max_delay, dr_cfg["action_latency_range_steps"][1])
                
        self.action_history = torch.zeros(
            (self.num_envs, max_delay + 1, self.cfg.action_space), device=self.device
        )
        self.env_latencies = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )
        self.backlash_state = torch.zeros(
            (self.num_envs, 12), device=self.device
        )
        self.env_backlash_sizes = torch.zeros(
            (self.num_envs, 12), device=self.device
        )
        self.last_targets = self.desired_joint_pos.clone()

    def _setup_scene(self):
        import os
        from .quadruped_env_cfg import ROBOT_VARIANTS
        import copy
        import torch

        selection = self.cfg.robot_choice.upper()
        num_envs = self.scene.cfg.num_envs
        
        # Guard clause for Heterogeneous + Replicate Physics
        if (selection == "RANDOM" or not selection) and self.scene.cfg.replicate_physics:
            raise ValueError("Heterogeneous multi-robot training requires replicate_physics=False! You cannot use GPU instancing with mixed robot models. Please select a single robot model in the launcher, or set replicate_physics=False.")

        if selection == "RANDOM" or not selection:
            # MIXED MODE: Partition and Spawn
            self.a1_indices = list(range(0, num_envs, 3))
            self.quadruped_indices = list(range(1, num_envs, 3))
            self.go2_indices = list(range(2, num_envs, 3))

            # Use nested namespaces to isolate USD assets while preserving "Robot" name context
            for i in self.a1_indices:
                ROBOT_VARIANTS[0].spawn.func(
                    f"/World/envs/env_{i}/A1/Robot", ROBOT_VARIANTS[0].spawn
                )
            for i in self.quadruped_indices:
                ROBOT_VARIANTS[1].spawn.func(
                    f"/World/envs/env_{i}/Quadruped/Robot", ROBOT_VARIANTS[1].spawn
                )
            for i in self.go2_indices:
                ROBOT_VARIANTS[2].spawn.func(
                    f"/World/envs/env_{i}/Go2/Robot", ROBOT_VARIANTS[2].spawn
                )

            # Create views for each partition using the nested paths
            a1_cfg = copy.deepcopy(ROBOT_VARIANTS[0])
            a1_cfg.spawn = None
            a1_cfg.prim_path = "/World/envs/env_.*/A1/Robot"
            self.a1_view = Articulation(a1_cfg)

            quadruped_cfg = copy.deepcopy(ROBOT_VARIANTS[1])
            quadruped_cfg.spawn = None
            quadruped_cfg.prim_path = "/World/envs/env_.*/Quadruped/Robot"
            self.quadruped_view = Articulation(quadruped_cfg)

            go2_cfg = copy.deepcopy(ROBOT_VARIANTS[2])
            go2_cfg.spawn = None
            go2_cfg.prim_path = "/World/envs/env_.*/Go2/Robot"
            self.go2_view = Articulation(go2_cfg)

            # Update sensor paths for nested namespaces
            self.cfg.contact_sensor.prim_path = (
                "/World/envs/env_.*/(A1|Quadruped|Go2)/Robot/(.*_foot|.*_calf|.*_thigh)"
            )

            # Register in scene (needed for Event Manager and base class consistency)
            self.scene.articulations["robot_a1"] = self.a1_view
            self.scene.articulations["robot_quadruped"] = self.quadruped_view
            self.scene.articulations["robot_go2"] = self.go2_view
            self.scene.articulations["robot"] = self.quadruped_view

            self.robot = self.quadruped_view
            self.robot_views = [self.a1_view, self.quadruped_view, self.go2_view]
            self.robot_view_indices = [
                torch.tensor(self.a1_indices, device=self.device),
                torch.tensor(self.quadruped_indices, device=self.device),
                torch.tensor(self.go2_indices, device=self.device),
            ]

            self.is_heterogeneous = True
        else:
            # Homogeneous Mode
            self.is_heterogeneous = False
            variant_cfg = ROBOT_VARIANTS[1]  # Default Quadruped
            if "A1" in selection:
                variant_cfg = ROBOT_VARIANTS[0]
            elif "GO2" in selection:
                variant_cfg = ROBOT_VARIANTS[2]
            elif "QUADRUPED" in selection:
                variant_cfg = ROBOT_VARIANTS[1]

            if self.scene.cfg.replicate_physics:
                variant_cfg.spawn.func("/World/envs/env_0/Robot", variant_cfg.spawn)
            else:
                for i in range(num_envs):
                    variant_cfg.spawn.func(f"/World/envs/env_{i}/Robot", variant_cfg.spawn)

            robot_cfg = copy.deepcopy(variant_cfg)
            robot_cfg.spawn = None
            robot_cfg.prim_path = "/World/envs/env_.*/Robot"
            self.robot = Articulation(robot_cfg)
            # Register as 'robot' (default) and also as aliases for Event Manager
            self.scene.articulations["robot_a1"] = self.robot
            self.scene.articulations["robot_quadruped"] = self.robot
            self.scene.articulations["robot_go2"] = self.robot

        # Common sensors and setup
        if isinstance(self.cfg.contact_sensor.prim_path, list):
            self.cfg.contact_sensor.prim_path = self.cfg.contact_sensor.prim_path[0]
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        print(f"DEBUG_CONTACT_SENSOR_CFG_TYPE: {type(self._contact_sensor.cfg.prim_path)}")
        print(f"DEBUG_CONTACT_SENSOR_CFG_VAL: {self._contact_sensor.cfg.prim_path}")
        self.scene.sensors["contact_sensor"] = self._contact_sensor


        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Clone environments if replicate_physics is enabled
        if self.scene.cfg.replicate_physics:
            self.scene.clone_environments(copy_from_source=False)

    def _resample_commands(self, env_ids: Sequence[int]):
        """Resamples the velocity commands for the specified environments."""
        # Sample x velocity
        self.target_commands[env_ids, 0] = sample_uniform(
            self.cfg.command_x_range[0],
            self.cfg.command_x_range[1],
            (len(env_ids),),
            device=self.device,
        )
        # Sample y velocity
        self.target_commands[env_ids, 1] = sample_uniform(
            self.cfg.command_y_range[0],
            self.cfg.command_y_range[1],
            (len(env_ids),),
            device=self.device,
        )
        # Sample yaw velocity
        self.target_commands[env_ids, 2] = sample_uniform(
            self.cfg.command_yaw_range[0],
            self.cfg.command_yaw_range[1],
            (len(env_ids),),
            device=self.device,
        )
        # Heading (unused for now, kept zero)
        self.target_commands[env_ids, 3] = 0.0

        # Apply special command modes: zero, x-only, y-only, yaw-only
        n_envs = len(env_ids)
        rand_vals = torch.rand(n_envs, device=self.device)

        # Cumulative thresholds for mutually exclusive assignment
        p_zero = self.cfg.zero_command_fraction
        p_x = p_zero + getattr(self.cfg, "x_only_command_fraction", 0.0)
        p_y = p_x + getattr(self.cfg, "y_only_command_fraction", 0.0)
        p_yaw = p_y + getattr(self.cfg, "yaw_only_command_fraction", 0.0)

        # Zero-command case
        zero_mask = rand_vals < p_zero
        self.target_commands[env_ids[zero_mask], :3] = 0.0

        # X-only command case
        x_only_mask = (rand_vals >= p_zero) & (rand_vals < p_x)
        self.target_commands[env_ids[x_only_mask], 1:3] = 0.0

        # Y-only command case
        y_only_mask = (rand_vals >= p_x) & (rand_vals < p_y)
        self.target_commands[env_ids[y_only_mask], 0] = 0.0
        self.target_commands[env_ids[y_only_mask], 2] = 0.0

        # Yaw-only command case
        yaw_only_mask = (rand_vals >= p_y) & (rand_vals < p_yaw)
        self.target_commands[env_ids[yaw_only_mask], 0:2] = 0.0

        # Reset timer
        self.command_timer[env_ids] = 0.0

    def _transition_to_next_phase(self):
        if self.curriculum_phase_idx >= len(self.curriculum_phases):
            return
            
        next_phase = self.curriculum_phases[self.curriculum_phase_idx]
        p_cfg = next_phase["cfg"]
        
        print(f"\n{'='*50}\n[Curriculum] Transitioning to Phase: {next_phase['name']}\n{'='*50}\n")
        
        # Update Rewards
        r_cfg = p_cfg.get("rewards", {})
        if r_cfg:
            for k, v in r_cfg.items():
                if hasattr(self.cfg, k):
                    setattr(self.cfg, k, v)
                    
        # Update Domain Randomization
        dr_cfg = p_cfg.get("domain_randomization", {})
        if dr_cfg:
            for k, v in dr_cfg.items():
                if hasattr(self.cfg, k):
                    setattr(self.cfg, k, tuple(v) if isinstance(v, list) else v)
            
        # Commands
        c_cfg = p_cfg.get("commands", {})
        if c_cfg:
            for k, v in c_cfg.items():
                if hasattr(self.cfg, k):
                    setattr(self.cfg, k, v)
            
        self.curriculum_phase_idx += 1

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Called before the physics step. Here we just store the action."""
        
        # Internal Curriculum Logic
        self.agent_steps += 1
        if self.curriculum_phases and self.curriculum_phase_idx < len(self.curriculum_thresholds):
            if self.agent_steps >= self.curriculum_thresholds[self.curriculum_phase_idx]:
                self._transition_to_next_phase()
        self.previous_actions = self.actions.clone()
        self.last_joint_vel = self.joint_vel.clone()
        self.last_base_lin_vel = self.base_lin_vel.clone()
        self.actions = actions.clone()

        # Update action history for latency simulation
        self.action_history = torch.roll(self.action_history, shifts=1, dims=1)
        self.action_history[:, 0, :] = self.actions.clone()

        # Update command timer
        self.command_timer += self.step_dt
        # Resample commands if timer exceeded
        resample_env_ids = (
            (self.command_timer >= self.cfg.command_resampling_time)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(resample_env_ids) > 0:
            self._resample_commands(resample_env_ids)

        import os
        if os.environ.get("QUADRUPED_TELEOP", "0") != "1":
            standby_duration = getattr(self.cfg, "standby_duration_s", 0.5)
            if standby_duration > 0.0:
                # Apply zero velocity standby ONLY at the start of the episode to allow stable standing.
                standby_mask = (self.episode_length_buf * self.step_dt) < standby_duration
                self.commands = torch.where(
                    standby_mask.unsqueeze(1),
                    torch.zeros_like(self.target_commands),
                    self.target_commands
                )
            else:
                # No standby — forward commands directly
                self.commands = self.target_commands.clone()

        # Teleoperation Hook via Environment Variable
        import os

        if os.environ.get("QUADRUPED_TELEOP", "0") == "1":
            if not hasattr(self, "keyboard"):
                import numpy as np
                from isaaclab.devices.keyboard.se2_keyboard import (
                    Se2Keyboard,
                    Se2KeyboardCfg,
                )

                class WasdKeyboard(Se2Keyboard):
                    def __init__(self, cfg):
                        self.speed_multiplier = 1.0
                        super().__init__(cfg)
                        print(
                            "\n[Teleop] Controls: W/S=fwd, A/D=strafe, Q/E=turn"
                            " | +/= to speed up, - to slow down"
                            f" | Current speed: {self.speed_multiplier:.1f}x\n"
                        )

                    def _create_key_bindings(self):
                        super()._create_key_bindings()
                        self._INPUT_KEY_MAPPING.update(
                            {
                                "W": np.asarray([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
                                "S": np.asarray([-1.0, 0.0, 0.0])
                                * self.v_x_sensitivity,
                                "A": np.asarray([0.0, 1.0, 0.0]) * self.v_y_sensitivity,
                                "D": np.asarray([0.0, -1.0, 0.0])
                                * self.v_y_sensitivity,
                                "Q": np.asarray([0.0, 0.0, 1.0])
                                * self.omega_z_sensitivity,
                                "E": np.asarray([0.0, 0.0, -1.0])
                                * self.omega_z_sensitivity,
                            }
                        )

                    def _on_keyboard_event(self, event, *args, **kwargs):
                        import carb.input as carb_input

                        if event.type == carb_input.KeyboardEventType.KEY_PRESS:
                            if event.input in (
                                carb_input.KeyboardInput.EQUAL,  # = / + key
                                carb_input.KeyboardInput.NUMPAD_ADD,
                            ):
                                self.speed_multiplier = round(
                                    min(3.0, self.speed_multiplier + 0.1), 1
                                )
                                print(f"[Teleop] Speed: {self.speed_multiplier:.1f}x")
                            elif event.input in (
                                carb_input.KeyboardInput.MINUS,
                                carb_input.KeyboardInput.NUMPAD_SUBTRACT,
                            ):
                                self.speed_multiplier = round(
                                    max(0.1, self.speed_multiplier - 0.1), 1
                                )
                                print(f"[Teleop] Speed: {self.speed_multiplier:.1f}x")
                        return super()._on_keyboard_event(event, *args, **kwargs)

                    def advance(self):
                        cmd = super().advance()
                        return cmd * self.speed_multiplier

                kb_cfg = Se2KeyboardCfg(
                    v_x_sensitivity=1.0, v_y_sensitivity=1.0, omega_z_sensitivity=1.2
                )
                kb_cfg.class_type = WasdKeyboard
                kb_cfg.sim_device = self.device
                self.keyboard = kb_cfg.class_type(kb_cfg)

            teleop_cmd = self.keyboard.advance()
            self.commands[:, 0] = teleop_cmd[0]
            self.commands[:, 1] = teleop_cmd[1]
            self.commands[:, 2] = teleop_cmd[2]
            self.commands[:, 3] = 0.0

        # Integrate virtual reference yaw and position using current commands
        self.ref_yaw += self.commands[:, 2] * self.step_dt
        cos_yaw = torch.cos(self.ref_yaw)
        sin_yaw = torch.sin(self.ref_yaw)
        vx_world = self.commands[:, 0] * cos_yaw - self.commands[:, 1] * sin_yaw
        vy_world = self.commands[:, 0] * sin_yaw + self.commands[:, 1] * cos_yaw
        self.ref_pos_xy[:, 0] += vx_world * self.step_dt
        self.ref_pos_xy[:, 1] += vy_world * self.step_dt

    def _apply_action(self) -> None:
        """
        Applies the neural network action to the robot joints.
        Mode: Absolute Position Control (PD)
        """
        # Fetch delayed action
        env_indices = torch.arange(self.num_envs, device=self.device)
        delayed_actions = self.action_history[env_indices, self.env_latencies, :]

        # 1. Compute Targets
        targets = delayed_actions * self.cfg.action_scale + self.desired_joint_pos

        # Apply backlash deadband
        diff = targets - self.last_targets
        self.backlash_state = torch.clamp(
            self.backlash_state + diff,
            -self.env_backlash_sizes / 2.0,
            self.env_backlash_sizes / 2.0
        )
        effective_targets = targets - self.backlash_state
        self.last_targets = targets.clone()

        targets = effective_targets

        if getattr(self, "is_heterogeneous", False):
            # DISTRIBUTE to partitioned views
            for i, view in enumerate(self.robot_views):
                indices = self.robot_view_indices[i]
                if len(indices) == 0: continue
                v_idx = self._view_joint_dof_idx[i]
                # Clamp per-robot (they all have same limits usually, but good practice)
                lower = view.data.soft_joint_pos_limits[0, v_idx, 0]
                upper = view.data.soft_joint_pos_limits[0, v_idx, 1]
                view_targets = torch.clamp(targets[indices], lower, upper)

                view.set_joint_position_target(
                    view_targets, joint_ids=v_idx
                )
                view.set_joint_velocity_target(
                    torch.zeros_like(view_targets), joint_ids=v_idx
                )
        else:
            # 2. Safety limits (Standard)
            lower_limits = self.robot.data.soft_joint_pos_limits[
                0, self._joint_dof_idx, 0
            ]
            upper_limits = self.robot.data.soft_joint_pos_limits[
                0, self._joint_dof_idx, 1
            ]
            targets = torch.clamp(targets, lower_limits, upper_limits)

            # 3. Apply to Simulation
            self.robot.set_joint_position_target(targets, joint_ids=self._joint_dof_idx)
            zeros = torch.zeros_like(targets)
            self.robot.set_joint_velocity_target(zeros, joint_ids=self._joint_dof_idx)

    def _refresh_joint_and_base_vel(self) -> None:
        """Refresh self.joint_vel / self.base_lin_vel from the live sim state.

        DirectRLEnv.step() calls _get_dones() then _get_rewards() BEFORE _get_observations()
        (see isaaclab/envs/direct_rl_env.py), even though the physics substeps (and the
        scene.update() that refreshes self.robot.data.*) already ran earlier in step(). So the
        self.joint_vel / self.base_lin_vel attributes below are only refreshed by
        _get_observations(), which means at reward-computation time they still hold the value
        from the END of the PREVIOUS step -- identical to last_joint_vel/last_base_lin_vel
        (captured from the same stale attributes in _pre_physics_step), making the dof_acc_l2/
        base_acc_l2 finite-difference always evaluate to exactly zero. Calling this at the top
        of _get_rewards() (before last_joint_vel/last_base_lin_vel get overwritten downstream)
        pulls the genuinely fresh, current-step value straight from self.robot.data instead.
        """
        if getattr(self, "is_heterogeneous", False):
            for i, view in enumerate(self.robot_views):
                indices = self.robot_view_indices[i]
                self.joint_vel[indices] = view.data.joint_vel[:, self._joint_dof_idx]
                self.base_lin_vel[indices] = view.data.root_lin_vel_b
        else:
            self.joint_vel = self.robot.data.joint_vel[:, self._joint_dof_idx]
            self.base_lin_vel = self.robot.data.root_lin_vel_b

    def _get_observations(self) -> dict:
        """
        Collects data from the simulation to feed into the neural network.
        """
        if getattr(self, "is_heterogeneous", False):
            # AGGREGATE state from partitioned views
            for i, view in enumerate(self.robot_views):
                indices = self.robot_view_indices[i]
                self.joint_pos[indices] = view.data.joint_pos[:, self._joint_dof_idx]
                self.joint_vel[indices] = view.data.joint_vel[:, self._joint_dof_idx]
                self.base_lin_vel[indices] = view.data.root_lin_vel_b
                self.base_ang_vel[indices] = view.data.root_ang_vel_b
                self.projected_gravity[indices] = view.data.projected_gravity_b
                self.root_pos_w[indices] = view.data.root_pos_w
                self.root_quat_w[indices] = view.data.root_quat_w
                self.applied_torque[indices] = view.data.applied_torque[
                    :, self._joint_dof_idx
                ]
                self.joint_acc[indices] = view.data.joint_acc[:, self._joint_dof_idx]

                # Handle possible body count differences
                num_bodies = min(
                    self.body_pos_w.shape[1], view.data.body_pos_w.shape[1]
                )
                self.body_pos_w[indices, :num_bodies] = view.data.body_pos_w[
                    :, :num_bodies
                ]
        else:
            self.joint_pos = self.robot.data.joint_pos[:, self._joint_dof_idx]
            self.joint_vel = self.robot.data.joint_vel[:, self._joint_dof_idx]
            self.base_lin_vel = self.robot.data.root_lin_vel_b
            self.base_ang_vel = self.robot.data.root_ang_vel_b
            self.projected_gravity = self.robot.data.projected_gravity_b
            self.body_pos_w = self.robot.data.body_pos_w
            self.root_pos_w = self.robot.data.root_pos_w
            self.root_quat_w = self.robot.data.root_quat_w
            self.applied_torque = self.robot.data.applied_torque
            self.joint_acc = self.robot.data.joint_acc

        self.net_contact_forces = self._contact_sensor.data.net_forces_w
        if len(self._undesired_contact_body_ids) > 0:
            self.net_undesired_contact_forces = self.net_contact_forces[:, self._undesired_contact_body_ids, :]
        else:
            self.net_undesired_contact_forces = torch.zeros((self.num_envs, 1, 3), device=self.device)

        # -- Update feet air time logic --
        # Check contact (force > threshold, e.g. 1.0)
        contact = (
            torch.norm(self.net_contact_forces[:, self._feet_ids, :], dim=-1) > 1.0
        )
        # First contact this step: currently contact AND NOT previously contact
        first_contact = contact & ~self.last_feet_contact
        # Increment air time
        self.feet_air_time += self.step_dt
        # Potential-based shaping: treat phi(t) = exp(-(air_time-target)^2/sigma) as a potential and
        # reward its rate of change dphi/dt every step while airborne, instead of paying phi(t) once
        # at landing. Summed over a swing this telescopes back to phi(landing)-phi(0) (same total as
        # the old lump-sum design for a well-timed swing), but gives dense, directional feedback:
        # positive while air_time is approaching target, zero at the peak, negative past it -- so
        # there's no way to "camp" near the target, the signal pushes toward landing right around it.
        air_time_err = self.feet_air_time - self.cfg.target_feet_air_time
        phi = torch.exp(-torch.square(air_time_err) / self.cfg.feet_air_time_sigma)
        dphi_dt = -2.0 * air_time_err / self.cfg.feet_air_time_sigma * phi
        # Multiply the rate by step_dt: this is what actually makes per-step rewards sum (Riemann
        # sum) to phi(landing)-phi(0) over a swing -- the raw rate alone would overcount by 1/step_dt.
        rew_air_time = torch.sum(
            dphi_dt * self.step_dt * (~contact).float(), dim=1
        ) * (torch.norm(self.commands[:, :3], dim=1) > self.cfg.static_velocity_threshold)
        self.feet_air_time_reward_val = rew_air_time

        # -- Update foot height reward logic --
        if getattr(self, "is_heterogeneous", False):
            # Multi-robot foot height aggregation
            all_feet_heights = torch.zeros((self.num_envs, 4), device=self.device)
            for i, view in enumerate(self.robot_views):
                indices = self.robot_view_indices[i]
                feet_ids = self.robot_feet_ids[
                    i
                ]  # Relative to Articulation (FL, FR, RL, RR order)
                all_feet_heights[indices] = view.data.body_pos_w[:, feet_ids, 2]
            feet_heights = all_feet_heights
        else:
            # Homogeneous case
            feet_heights = self.body_pos_w[:, self._feet_ids_articulation, 2]

        # Reward for reaching target height during swing
        # (exp(-square(height - target) / sigma) * ~contact)
        # Masked by command norm to avoid lifting feet when standing still
        rew_foot_height = torch.sum(
            torch.exp(-torch.square(feet_heights - self.cfg.target_foot_height) / self.cfg.foot_height_sigma)
            * (~contact).float(),
            dim=1,
        )
        # Apply command mask (x, y, yaw commands)
        rew_foot_height *= (torch.norm(self.commands[:, :3], dim=1) > self.cfg.static_velocity_threshold).float()

        self.foot_height_reward_val = rew_foot_height

        # Penalty grows with how long each foot has been continuously airborne (self.feet_air_time,
        # already tracked above -- resets to 0 on landing), instead of a flat per-airborne-foot cost.
        # Keeps a normal step cheap while discouraging a foot getting stuck hovering; see
        # target_feet_air_time/feet_air_time_sigma comment in training_phases.yaml for the balance
        # against rew_scale_feet_air_time so this penalty doesn't outweigh completing a real step.
        self.feet_air_penalty_val = torch.sum(self.feet_air_time * (~contact).float(), dim=1)
        # Extra penalty when standing still (commands == 0)
        static_mask = (torch.norm(self.commands[:, :3], dim=1) < self.cfg.static_velocity_threshold).float()
        self.feet_air_penalty_static_val = self.feet_air_penalty_val * static_mask
        self.joint_vel_l2_static_val = (
            torch.sum(torch.square(self.joint_vel), dim=1) * static_mask
        )

        # GRF computations: two separate penalties
        feet_forces_z = self.net_contact_forces[:, self._feet_ids, 2].abs()  # (N, 4)
        contact_float = contact.float()
        n_contact = contact_float.sum(dim=1).clamp(min=1.0)

        # 1) GRF balance (CV²): penalize relative unevenness among contacting feet
        mean_force = (feet_forces_z * contact_float).sum(dim=1) / n_contact
        force_var = ((feet_forces_z - mean_force.unsqueeze(1)).square() * contact_float).sum(dim=1) / n_contact
        self.grf_balance_val = force_var / mean_force.square().clamp(min=1.0)

        # 2) GRF target (mg/n): penalize deviation from physics-based weight share
        # Uses cached robot weight so force spikes can't inflate their own target.
        target_force_per_foot = (self.robot_total_weight / n_contact).unsqueeze(1)  # mg/n
        force_deviation = ((feet_forces_z - target_force_per_foot).square() * contact_float).sum(dim=1) / n_contact
        self.grf_target_val = force_deviation / target_force_per_foot.squeeze(1).square().clamp(min=1.0)

        # Max contact force penalty: penalize per-foot forces exceeding a fraction of robot weight
        # Threshold = robot_total_weight * max_contact_force_pct (e.g., 0.75 = 75% of mg)
        max_force_pct = getattr(self.cfg, "max_contact_force_pct", 0.75)
        per_foot_thresh = self.robot_total_weight * max_force_pct  # (N,)
        excess = (feet_forces_z - per_foot_thresh.unsqueeze(1)).clamp(min=0.0)  # (N, 4)
        # Normalize by mg² to make dimensionless (scale-invariant across robot masses)
        self.max_contact_force_val = torch.sum(excess.square(), dim=1) / self.robot_total_weight.square().clamp(min=1.0)

        # Reset air time for feet in contact
        self.feet_air_time[contact] = 0.0
        self.last_feet_contact = contact

        # -- Contact-driven gait phase symmetry reward --
        # Mirrors the strike-time logic used in eval_mujoco.py.
        # On each foot's touchdown we record the event time and compute the stride duration
        # (time between consecutive touchdowns). The phase of foot B relative to foot A is:
        #   phase = ((t_B - last_strike_A) % stride_A) / stride_A  → [0, 1]
        # We then score how close this is to the target offset via cosine distance.
        if self.cfg.rew_scale_gait_phase_sym != 0.0:
            # Current simulation time for all envs  (N,)
            t_now = self.episode_length_buf.float() * self.step_dt  # proxy: steps * dt

            # Use the first_contact already computed above (before last_feet_contact was updated)
            # Update stride_duration and last_strike_time for feet that just landed
            for foot in range(4):
                landing = first_contact[:, foot]  # (N,) bool
                if landing.any():
                    prev_t = self.last_strike_time[landing, foot]
                    new_dur = t_now[landing] - prev_t
                    # Only update stride if the duration looks physically plausible (> 0.1s)
                    valid = new_dur > 0.1
                    if valid.any():
                        update_mask = landing.clone()
                        update_mask[landing] = valid
                        self.stride_duration[update_mask, foot] = new_dur[valid]
                    self.last_strike_time[landing, foot] = t_now[landing]

            # Compute phase of foot B relative to foot A for each of the 6 pairs
            # phase_rel(A, B) = ((t_now - last_strike_A) % stride_A) / stride_A  → [0, 1]
            # Only valid when stride_A is known (> 0.1s) and robot is moving
            moving = (torch.norm(self.commands[:, :3], dim=1) > self.cfg.static_velocity_threshold)  # (N,) bool

            def _phase_rel(ref_foot, other_foot):
                """Phase of other_foot relative to ref_foot. Returns score in [0,1] and validity mask."""
                dur = self.stride_duration[:, ref_foot]          # (N,)
                
                # Check if the foot has struck recently (prevent reward farming by standing still)
                time_since_strike = t_now - self.last_strike_time[:, ref_foot]
                active_stepping = time_since_strike < (dur * 1.5)
                
                valid = (dur > 0.1) & moving & active_stepping
                time_diff = self.last_strike_time[:, other_foot] - self.last_strike_time[:, ref_foot]
                phase = (time_diff % dur.clamp(min=1e-4)) / dur.clamp(min=1e-4)            # [0, 1]
                return phase, valid

            def _cosine_score(phase, target_offset):
                """Cosine score: 1.0 when phase==target, 0.0 when phase==target+0.5."""
                diff = (phase - target_offset) * 2.0 * torch.pi
                return 0.5 * (1.0 + torch.cos(diff))

            # All 6 unique pairs  [FL=0, FR=1, RL=2, RR=3]
            pairs = [
                (0, 1, self.cfg.gait_phase_offset_front),  # FL vs FR
                (2, 3, self.cfg.gait_phase_offset_rear),   # RL vs RR
                (0, 2, self.cfg.gait_phase_offset_left),   # FL vs RL
                (1, 3, self.cfg.gait_phase_offset_right),  # FR vs RR
                (0, 3, self.cfg.gait_phase_offset_diag1),  # FL vs RR
                (1, 2, self.cfg.gait_phase_offset_diag2),  # FR vs RL
            ]

            total_score = torch.zeros(self.num_envs, device=self.device)
            total_weight = torch.zeros(self.num_envs, device=self.device)
            for ref, other, target in pairs:
                phase, valid = _phase_rel(ref, other)
                score = _cosine_score(phase, target)
                total_score += score * valid.float()
                total_weight += valid.float()

            # Average over valid pairs; if no pair is valid (robot just spawned), score = 0
            self.gait_phase_sym_val = total_score / total_weight.clamp(min=1.0) * (total_weight > 0).float()
        else:
            self.gait_phase_sym_val = torch.zeros(self.num_envs, device=self.device)

        # Compute leashed virtual reference position deviation
        pos_error = self.ref_pos_xy - self.root_pos_w[:, :2]
        error_dist = torch.norm(pos_error, dim=1)
        max_leash = getattr(self.cfg, "max_pos_leash", 0.4)
        exceeds_leash = error_dist > max_leash
        if exceeds_leash.any():
            scale = max_leash / error_dist[exceeds_leash]
            self.ref_pos_xy[exceeds_leash] = (
                self.root_pos_w[exceeds_leash, :2]
                + pos_error[exceeds_leash] * scale.unsqueeze(1)
            )
            error_dist[exceeds_leash] = max_leash
        self.pos_deviation_val = error_dist

        # Compute leashed virtual reference yaw deviation
        w = self.root_quat_w[:, 0]
        x = self.root_quat_w[:, 1]
        y = self.root_quat_w[:, 2]
        z = self.root_quat_w[:, 3]
        current_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        yaw_error = self.ref_yaw - current_yaw
        yaw_error = (yaw_error + torch.pi) % (2.0 * torch.pi) - torch.pi
        
        max_yaw_leash = self.cfg.max_yaw_leash
        exceeds_yaw_leash = torch.abs(yaw_error) > max_yaw_leash
        if exceeds_yaw_leash.any():
            yaw_error_clamped = torch.sign(yaw_error[exceeds_yaw_leash]) * max_yaw_leash
            self.ref_yaw[exceeds_yaw_leash] = current_yaw[exceeds_yaw_leash] + yaw_error_clamped
            yaw_error[exceeds_yaw_leash] = yaw_error_clamped
        self.yaw_deviation_val = torch.abs(yaw_error)

        # Observations (unscaled)
        obs = torch.cat(
            (
                self.base_lin_vel,
                self.base_ang_vel,
                self.projected_gravity,
                self.commands,
                self.joint_pos - self.desired_joint_pos,
                self.joint_vel,
                self.actions,
            ),
            dim=-1,
        )

        # Add observation noise (Sim2Real)
        obs_noise = torch.randn_like(obs) * self.cfg.observation_noise_scale
        obs = obs + obs_noise

        if self.cfg.obs_history_len > 0:
            full_obs = torch.cat([obs, self.obs_history_buf], dim=-1)
            self.obs_history_buf = torch.cat([obs, self.obs_history_buf[:, :-49]], dim=-1)
            obs = full_obs

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        Computes the reward (score) for the current step.
        The goal is to teach the robot to stand up and retain balance.
        """
        # Pull fresh joint_vel/base_lin_vel before computing dof_acc_l2/base_acc_l2 below -- see
        # _refresh_joint_and_base_vel's docstring for why this can't just rely on self.joint_vel.
        self._refresh_joint_and_base_vel()

        # Calculate undesired contacts penalty
        # If any thigh/calf/trunk sensor registers > 1.0 N force, it's a contact
        undesired_contacts = (torch.norm(self.net_undesired_contact_forces, dim=-1).max(dim=1)[0] > 1.0).float()
        
        total_reward, reward_log = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_undesired_contacts,
            self.cfg.rew_scale_track_lin_vel_xy_exp,
            self.cfg.rew_scale_track_ang_vel_z_exp,
            self.cfg.rew_scale_lin_vel_z_l2,
            self.cfg.rew_scale_ang_vel_xy_l2,
            self.cfg.rew_scale_dof_pos_l2,
            self.cfg.rew_scale_dof_torques_l2,
            self.cfg.rew_scale_dof_acc_l2,
            self.cfg.rew_scale_action_rate_l2,
            self.cfg.rew_scale_feet_air_time,
            self.cfg.rew_scale_flat_orientation_l2,
            self.cfg.rew_scale_foot_height_exp,
            self.cfg.rew_scale_feet_air_penalty,
            self.cfg.rew_scale_feet_air_penalty_static,
            self.cfg.rew_scale_joint_vel_l2_static,
            self.cfg.rew_scale_base_height_l2,
            self.cfg.rew_scale_grf_balance,
            self.cfg.rew_scale_grf_target,
            self.cfg.rew_scale_max_contact_force,
            self.cfg.rew_scale_base_acc_l2,
            self.cfg.rew_scale_pos_deviation_l1,
            self.cfg.rew_scale_yaw_deviation_l1,
            self.cfg.rew_scale_gait_phase_sym,
            self.cfg.target_base_height,
            self.cfg.static_velocity_threshold,
            self.cfg.command_lin_vel_std,
            self.cfg.command_ang_vel_std,
            self.cfg.vel_tracking_sigma_exp,
            self.commands,
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity,
            self.joint_pos,
            self.desired_joint_pos,
            self.joint_vel,
            self.last_joint_vel,
            self.last_base_lin_vel,
            self.applied_torque,
            self.joint_acc,
            self.actions,
            self.previous_actions,
            self.feet_air_time_reward_val,
            self.foot_height_reward_val,
            self.feet_air_penalty_val,
            self.feet_air_penalty_static_val,
            self.joint_vel_l2_static_val,
            self.grf_balance_val,
            self.grf_target_val,
            self.max_contact_force_val,
            self.pos_deviation_val,
            self.yaw_deviation_val,
            self.gait_phase_sym_val,
            self.root_pos_w[:, 2] - self.scene.env_origins[:, 2],
            undesired_contacts,
            self.reset_terminated,
            self.step_dt,
        )
        self.extras.setdefault("log", {})
        self.extras["log"].update(reward_log)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Determines if the episode is over.
        1. Died: Base hit the ground.
        2. Timeout: Episode duration exceeded limit.
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Check if base is too tilted (not upright)
        upright_check = (
            self.projected_gravity[:, 2] > -self.cfg.base_angle_termination_thresh
        )

        # Fall detection: if the robot's body is lower than 15cm, it likely fell.
        base_height = self.root_pos_w[:, 2] - self.scene.env_origins[:, 2]

        # Suppress termination during the standby/landing phase so the robot is not
        # killed mid-bounce when it drops from spawn height.
        standby_duration = getattr(self.cfg, "standby_duration_s", 0.5)
        past_standby = (self.episode_length_buf * self.step_dt) > standby_duration

        died = past_standby & ((base_height < 0.15) | upright_check)


        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if getattr(self, "is_heterogeneous", False):
            # RESET partitioned views independently
            for i, view in enumerate(self.robot_views):
                view_global_indices = self.robot_view_indices[i]
                mask = torch.isin(env_ids, view_global_indices)
                subset_env_ids = env_ids[mask]

                if len(subset_env_ids) > 0:
                    # Vectorized local index finding
                    local_indices = (
                        torch.isin(view_global_indices, subset_env_ids)
                        .nonzero()
                        .squeeze(-1)
                    )
                    view.reset(local_indices)
                    # Use existing randomization logic but point to specific view/indices
                    self._randomize_view_state(subset_env_ids, view, local_indices, view_idx=i)

            # CRITICAL: Reset the base environment buffers (which we bypassed)
            self.episode_length_buf[env_ids] = 0
            self.reset_buf[env_ids] = 0
            self.feet_air_time[env_ids] = 0.0
            self.last_joint_vel[env_ids] = 0.0
            self.last_base_lin_vel[env_ids] = 0.0
            self.previous_actions[env_ids] = 0.0
            self.last_strike_time[env_ids] = 0.0
            self.stride_duration[env_ids] = 1.0
            self.gait_phase_sym_val[env_ids] = 0.0
            if self.cfg.obs_history_len > 0:
                self.obs_history_buf[env_ids] = 0.0
        else:
            super()._reset_idx(env_ids)
            # Standard Mass/Friction/State randomization
            self._randomize_view_state(env_ids, self.robot)

    def _randomize_view_state(
        self,
        env_ids: torch.Tensor,
        view: Articulation,
        local_ids: torch.Tensor | None = None,
        view_idx: int | None = None,
    ):
        v_idx = self._view_joint_dof_idx[view_idx] if view_idx is not None else self._joint_dof_idx
        # 0. Randomize Base Mass (Sim2Real)
        env_ids_cpu = env_ids.cpu()
        local_ids_cpu = local_ids.cpu() if local_ids is not None else env_ids_cpu

        masses = view.root_physx_view.get_masses().clone()
        mass_noise = sample_uniform(
            self.cfg.payload_mass_range[0],
            self.cfg.payload_mass_range[1],
            (len(env_ids_cpu), 1),
            "cpu",
        )
        masses[local_ids_cpu, 0] = (
            view.data.default_mass[local_ids_cpu, 0] + mass_noise[:, 0]
        )
        view.root_physx_view.set_masses(masses, local_ids_cpu)

        # Cache total robot weight (mg) for physics-based GRF penalty
        total_mass_per_env = masses[local_ids_cpu].sum(dim=1)  # sum all body masses
        self.robot_total_weight[env_ids] = total_mass_per_env.to(self.device) * 9.81

        # 0.5 Randomize Center of Mass (Sim2Real)
        if self.cfg.com_displacement_range[0] != 0.0 or self.cfg.com_displacement_range[1] != 0.0:
            coms = view.root_physx_view.get_coms().clone()
            if not hasattr(view, "default_coms"):
                view.default_coms = coms.clone()

            com_noise_x = sample_uniform(
                self.cfg.com_displacement_range[0],
                self.cfg.com_displacement_range[1],
                (len(env_ids_cpu), 1),
                "cpu",
            )
            com_noise_y = sample_uniform(
                self.cfg.com_displacement_range[0],
                self.cfg.com_displacement_range[1],
                (len(env_ids_cpu), 1),
                "cpu",
            )
            coms[local_ids_cpu, 0, 0] = view.default_coms[local_ids_cpu, 0, 0] + com_noise_x[:, 0]
            coms[local_ids_cpu, 0, 1] = view.default_coms[local_ids_cpu, 0, 1] + com_noise_y[:, 0]
            view.root_physx_view.set_coms(coms, local_ids_cpu)

        # Use correct ID set for shape (local_ids if heterogeneous, else env_ids)
        ids = local_ids if local_ids is not None else env_ids

        # 0.1 Randomize internal joint friction (viscous drag, usually very small)
        friction_noise = sample_uniform(
            self.cfg.joint_friction_range[0],
            self.cfg.joint_friction_range[1],
            (len(ids), len(v_idx)),
            self.device,
        )
        base_friction = view.data.default_joint_friction_coeff[ids][:, v_idx]
        randomized_friction = torch.clamp(base_friction + friction_noise, min=0.0)
        
        view.write_joint_friction_coefficient_to_sim(
            randomized_friction,
            joint_ids=v_idx,
            env_ids=ids,
        )

        # 0.2 Randomize PD gains (Kp = stiffness, Kd = damping) for DCMotorCfg actuators.
        # NOTE: These writes have no effect when using ActuatorNetMLPCfg (Go1-style),
        # because the net computes torques directly and bypasses PhysX PD.
        # They ARE effective with DCMotorCfg (A1-style), which uses PhysX's PD drive.
        if self.cfg.joint_stiffness_range[0] != 0.0 or self.cfg.joint_stiffness_range[1] != 0.0:
            kp_noise = sample_uniform(
                self.cfg.joint_stiffness_range[0],
                self.cfg.joint_stiffness_range[1],
                (len(ids), len(v_idx)),
                self.device,
            )
            view.write_joint_stiffness_to_sim(
                kp_noise,
                joint_ids=v_idx,
                env_ids=ids,
            )

        if self.cfg.joint_pd_damping_range[0] != 0.0 or self.cfg.joint_pd_damping_range[1] != 0.0:
            kd_noise = sample_uniform(
                self.cfg.joint_pd_damping_range[0],
                self.cfg.joint_pd_damping_range[1],
                (len(ids), len(v_idx)),
                self.device,
            )
            view.write_joint_damping_to_sim(
                kd_noise,
                joint_ids=v_idx,
                env_ids=ids,
            )

        # 0.3 Randomize Latency and Backlash
        self.env_latencies[env_ids] = torch.randint(
            self.cfg.action_latency_range_steps[0],
            self.cfg.action_latency_range_steps[1] + 1,
            (len(env_ids),),
            device=self.device,
        )
        self.env_backlash_sizes[env_ids] = sample_uniform(
            self.cfg.motor_backlash_range[0],
            self.cfg.motor_backlash_range[1],
            (len(env_ids), 12),
            self.device,
        )
        # Reset backlash states and action history
        self.backlash_state[env_ids] = 0.0
        self.action_history[env_ids] = 0.0
        self.last_targets[env_ids] = self.desired_joint_pos[env_ids].clone()

        # 1. Reset Joint States (Use Default Pose + Noise on controlled joints)
        # Use full joint arrays (all joints, not just controlled ones)
        joint_pos = view.data.default_joint_pos[ids].clone()
        joint_vel = view.data.default_joint_vel[ids].clone()

        # Add small random noise to initial joint positions and velocities
        pos_noise = sample_uniform(
            -0.2, 0.2, (len(ids), len(v_idx)), joint_pos.device
        )
        vel_noise = sample_uniform(
            -0.5, 0.5, (len(ids), len(v_idx)), joint_vel.device
        )

        # Apply noise only to controlled joints
        joint_pos[:, v_idx] += pos_noise
        joint_vel[:, v_idx] += vel_noise

        # 2. Reset Base State (Position + Velocity)
        default_root_state = view.data.default_root_state[ids].clone()
        # Offset the base to the environment origin (so robots don't spawn on top of each other)
        # env_origins is global (32 rows)
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, 2] = (
            self.scene.env_origins[env_ids][:, 2] + self.cfg.spawn_height
        )

        # 3. Write to Simulator
        view.write_root_pose_to_sim(default_root_state[:, :7], ids)
        view.write_root_velocity_to_sim(default_root_state[:, 7:], ids)
        view.write_joint_state_to_sim(joint_pos, joint_vel, None, ids)

        # 4. Reset Action Buffer
        self.actions[env_ids] = 0.0
        w, x, y, z = default_root_state[:, 3], default_root_state[:, 4], default_root_state[:, 5], default_root_state[:, 6]
        current_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        self.ref_pos_xy[env_ids] = default_root_state[:, :2].clone()
        self.ref_yaw[env_ids] = current_yaw

        # 5. Resample Commands
        self._resample_commands(env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_undesired_contacts: float,
    rew_scale_track_lin_vel_xy_exp: float,
    rew_scale_track_ang_vel_z_exp: float,
    rew_scale_lin_vel_z_l2: float,
    rew_scale_ang_vel_xy_l2: float,
    rew_scale_dof_pos_l2: float,
    rew_scale_dof_torques_l2: float,
    rew_scale_dof_acc_l2: float,
    rew_scale_action_rate_l2: float,
    rew_scale_feet_air_time: float,
    rew_scale_flat_orientation_l2: float,
    rew_scale_foot_height_exp: float,
    rew_scale_feet_air_penalty: float,
    rew_scale_feet_air_penalty_static: float,
    rew_scale_joint_vel_l2_static: float,
    rew_scale_base_height_l2: float,
    rew_scale_grf_balance: float,
    rew_scale_grf_target: float,
    rew_scale_max_contact_force: float,
    rew_scale_base_acc_l2: float,
    rew_scale_pos_deviation_l1: float,
    rew_scale_yaw_deviation_l1: float,
    rew_scale_gait_phase_sym: float,
    target_base_height: float,
    static_velocity_threshold: float,
    command_lin_vel_std: float,
    command_ang_vel_std: float,
    vel_tracking_sigma_exp: float,
    commands: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    joint_pos: torch.Tensor,
    desired_joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    last_joint_vel: torch.Tensor,
    last_base_lin_vel: torch.Tensor,
    joint_torques: torch.Tensor,
    joint_acc: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    feet_air_time_reward_val: torch.Tensor,
    foot_height_reward_val: torch.Tensor,
    feet_air_penalty_val: torch.Tensor,
    feet_air_penalty_static_val: torch.Tensor,
    joint_vel_l2_static_val: torch.Tensor,
    grf_balance_val: torch.Tensor,
    grf_target_val: torch.Tensor,
    max_contact_force_val: torch.Tensor,
    pos_deviation_val: torch.Tensor,
    yaw_deviation_val: torch.Tensor,
    gait_phase_sym_val: torch.Tensor,
    base_height_val: torch.Tensor,
    undesired_contacts: torch.Tensor,
    reset_terminated: torch.Tensor,
    step_dt: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    log: Dict[str, torch.Tensor] = {}

    # 1. Alive (Optional, usually 0)
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    
    # Undesired contacts penalty
    rew_undesired_contacts = rew_scale_undesired_contacts * undesired_contacts

    # 2. Tracking Linear Velocity XY (Exponential)
    # Target is commands[:, 0:2] (x, y)
    # Local velocity is base_lin_vel[:, :2]
    # commands is [vx, vy, wz, heading]
    # Sigma scales with command speed so the robot is held to a tighter relative
    # tolerance at low speeds (must actually stop) and a looser absolute tolerance
    # at high speeds (normal velocity variance).
    command_speed_xy = torch.norm(commands[:, :2], dim=1)
    dynamic_lin_vel_denom = torch.clamp(command_lin_vel_std * command_speed_xy**vel_tracking_sigma_exp, min=0.005)

    lin_vel_error = torch.sum(
        torch.square(base_lin_vel[:, :2] - commands[:, :2]), dim=1
    )
    rew_track_lin_vel_xy_exp = rew_scale_track_lin_vel_xy_exp * torch.exp(
        -lin_vel_error / dynamic_lin_vel_denom
    )

    # 3. Tracking Angular Velocity Z (Exponential)
    # Target is commands[:, 2] (wz)
    command_speed_w = torch.abs(commands[:, 2])
    dynamic_ang_vel_denom = torch.clamp(command_ang_vel_std * command_speed_w**vel_tracking_sigma_exp, min=0.005)

    ang_vel_error = torch.square(base_ang_vel[:, 2] - commands[:, 2])
    rew_track_ang_vel_z_exp = rew_scale_track_ang_vel_z_exp * torch.exp(
        -ang_vel_error / dynamic_ang_vel_denom
    )

    # 4. Linear Velocity Z L2 Penalty
    rew_lin_vel_z_l2 = rew_scale_lin_vel_z_l2 * torch.square(base_lin_vel[:, 2])

    # 5. Angular Velocity XY L2 Penalty
    rew_ang_vel_xy_l2 = rew_scale_ang_vel_xy_l2 * torch.sum(
        torch.square(base_ang_vel[:, :2]), dim=1
    )

    # 6. DOF Torques L2 (Penalty)
    rew_dof_torques_l2 = rew_scale_dof_torques_l2 * torch.sum(
        torch.square(joint_torques), dim=1
    )

    # 7. DOF Acceleration L2 (Penalty)
    # self.robot.data.joint_acc is always zero in this DirectRLEnv setup (confirmed via
    # scripts/check_joint_acc.py), so compute via finite difference instead, same as base_acc below.
    joint_acc = (joint_vel - last_joint_vel) / step_dt
    rew_dof_acc_l2 = rew_scale_dof_acc_l2 * torch.sum(torch.square(joint_acc), dim=1)

    # Base Acceleration Penalty (calculated via finite difference)
    base_acc = (base_lin_vel - last_base_lin_vel) / step_dt
    rew_base_acc_l2 = rew_scale_base_acc_l2 * torch.sum(torch.square(base_acc), dim=1)

    # 8. Action Rate L2 (Penalty)
    # Penalize large changes in action
    rew_action_rate_l2 = rew_scale_action_rate_l2 * torch.sum(
        torch.square(actions - previous_actions), dim=1
    )

    # 9. Feet Air Time Reward
    # Computed in _get_observations
    rew_feet_air_time = rew_scale_feet_air_time * feet_air_time_reward_val

    # 10. DOF Position L2 Penalty
    rew_dof_pos_l2 = rew_scale_dof_pos_l2 * torch.sum(
        torch.square(joint_pos - desired_joint_pos), dim=1
    )

    # 11. Flat Orientation Penalty (Penalize Pitch/Roll)
    rew_flat_orientation_l2 = rew_scale_flat_orientation_l2 * torch.sum(
        torch.square(projected_gravity[:, :2]), dim=1
    )

    # 12. Foot Height Reward
    rew_foot_height = rew_scale_foot_height_exp * foot_height_reward_val

    # 13. Base Height Penalty
    rew_base_height_l2 = rew_scale_base_height_l2 * torch.square(base_height_val - target_base_height)

    # 14. Integrated Position Deviation L1 Penalty
    rew_pos_deviation = rew_scale_pos_deviation_l1 * pos_deviation_val
    rew_yaw_deviation = rew_scale_yaw_deviation_l1 * yaw_deviation_val

    # 15. Gait Phase Symmetry (cosine-based, all 6 leg pairs, configured offsets)
    rew_gait_phase_sym = rew_scale_gait_phase_sym * gait_phase_sym_val

    rew_feet_air_penalty = rew_scale_feet_air_penalty * feet_air_penalty_val
    rew_feet_air_penalty_static = rew_scale_feet_air_penalty_static * feet_air_penalty_static_val
    rew_joint_vel_l2_static = rew_scale_joint_vel_l2_static * joint_vel_l2_static_val
    rew_grf_balance = rew_scale_grf_balance * grf_balance_val
    rew_grf_target = rew_scale_grf_target * grf_target_val
    rew_max_contact_force = rew_scale_max_contact_force * max_contact_force_val

    # Per-term breakdown, mean across all parallel envs -- shows up in TensorBoard under
    # "Info / <key>" (skrl's agent config has environment_info: log wired up already).
    log["reward/alive"] = rew_alive.mean()
    log["reward/undesired_contacts"] = rew_undesired_contacts.mean()
    log["reward/track_lin_vel_xy_exp"] = rew_track_lin_vel_xy_exp.mean()
    log["reward/track_ang_vel_z_exp"] = rew_track_ang_vel_z_exp.mean()
    log["reward/lin_vel_z_l2"] = rew_lin_vel_z_l2.mean()
    log["reward/ang_vel_xy_l2"] = rew_ang_vel_xy_l2.mean()
    log["reward/dof_torques_l2"] = rew_dof_torques_l2.mean()
    log["reward/dof_pos_l2"] = rew_dof_pos_l2.mean()
    log["reward/dof_acc_l2"] = rew_dof_acc_l2.mean()
    log["reward/base_acc_l2"] = rew_base_acc_l2.mean()
    log["reward/action_rate_l2"] = rew_action_rate_l2.mean()
    log["reward/feet_air_time"] = rew_feet_air_time.mean()
    log["reward/flat_orientation_l2"] = rew_flat_orientation_l2.mean()
    log["reward/foot_height"] = rew_foot_height.mean()
    log["reward/base_height_l2"] = rew_base_height_l2.mean()
    log["reward/feet_air_penalty"] = rew_feet_air_penalty.mean()
    log["reward/feet_air_penalty_static"] = rew_feet_air_penalty_static.mean()
    log["reward/joint_vel_l2_static"] = rew_joint_vel_l2_static.mean()
    log["reward/grf_balance"] = rew_grf_balance.mean()
    log["reward/grf_target"] = rew_grf_target.mean()
    log["reward/max_contact_force"] = rew_max_contact_force.mean()
    log["reward/pos_deviation"] = rew_pos_deviation.mean()
    log["reward/yaw_deviation"] = rew_yaw_deviation.mean()
    log["reward/gait_phase_sym"] = rew_gait_phase_sym.mean()
    # Raw (unscaled) diagnostics -- useful to sanity-check a term is actually receiving live,
    # nonzero physical data before worrying about whether its reward *scale* is well tuned.
    log["diag/joint_acc_sum_sq_mean"] = torch.sum(torch.square(joint_acc), dim=1).mean()
    log["diag/base_acc_sum_sq_mean"] = torch.sum(torch.square(base_acc), dim=1).mean()

    total_reward = (
        rew_alive
        + rew_undesired_contacts
        + rew_track_lin_vel_xy_exp
        + rew_track_ang_vel_z_exp
        + rew_lin_vel_z_l2
        + rew_ang_vel_xy_l2
        + rew_dof_torques_l2
        + rew_dof_pos_l2
        + rew_dof_acc_l2
        + rew_base_acc_l2
        + rew_action_rate_l2
        + rew_feet_air_time
        + rew_flat_orientation_l2
        + rew_foot_height
        + rew_base_height_l2
        + rew_feet_air_penalty
        + rew_feet_air_penalty_static
        + rew_joint_vel_l2_static
        + rew_grf_balance
        + rew_grf_target
        + rew_max_contact_force
        + rew_pos_deviation
        + rew_yaw_deviation
        + rew_gait_phase_sym
    )
    return total_reward, log
