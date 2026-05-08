# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import copy
import random
from collections.abc import Sequence

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
            c_f_ids, _ = self._contact_sensor.find_bodies(".*_foot")
            self._feet_ids = c_f_ids
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
            # Contact sensor ordering (relative index [0,1,2,3])
            self._feet_ids = [1, 0, 3, 2]

        self.net_contact_forces = torch.zeros(self.num_envs, 20, 3, device=self.device)
        self._joint_dof_idx, _ = self.robot.find_joints(
            ".*_hip_joint|.*_thigh_joint|.*_calf_joint"
        )

        self.actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        self.previous_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        self.commands = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_joint_vel = torch.zeros(self.num_envs, 12, device=self.device)
        self.feet_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_feet_contact = torch.zeros(
            self.num_envs, 4, dtype=torch.bool, device=self.device
        )
        self.feet_air_time_reward_val = torch.zeros(self.num_envs, device=self.device)
        self.foot_height_reward_val = torch.zeros(self.num_envs, device=self.device)
        self.feet_air_penalty_val = torch.zeros(self.num_envs, device=self.device)
        self.feet_air_penalty_static_val = torch.zeros(self.num_envs, device=self.device)
        self.joint_vel_l2_static_val = torch.zeros(self.num_envs, device=self.device)
        self.command_timer = torch.full(
            (self.num_envs,), 100.0, device=self.device
        )  # Force immediate resample

    def _setup_scene(self):
        import os
        from .quadruped_env_cfg import ROBOT_VARIANTS
        import copy
        import torch

        selection = os.environ.get(
            "QUADRUPED_ROBOT", os.environ.get("FORCE_ROBOT", "")
        ).upper()
        num_envs = self.scene.cfg.num_envs

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
                "/World/envs/env_.*/(A1|Quadruped|Go2)/Robot/.*_foot"
            )
            self.cfg.height_scanner.prim_path = (
                "/World/envs/env_.*/(A1|Quadruped|Go2)/Robot/base"
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
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _resample_commands(self, env_ids: Sequence[int]):
        """Resamples the velocity commands for the specified environments."""
        # Sample x velocity
        self.commands[env_ids, 0] = sample_uniform(
            self.cfg.command_x_range[0],
            self.cfg.command_x_range[1],
            (len(env_ids),),
            device=self.device,
        )
        # Sample y velocity
        self.commands[env_ids, 1] = sample_uniform(
            self.cfg.command_y_range[0],
            self.cfg.command_y_range[1],
            (len(env_ids),),
            device=self.device,
        )
        # Sample yaw velocity
        self.commands[env_ids, 2] = sample_uniform(
            self.cfg.command_yaw_range[0],
            self.cfg.command_yaw_range[1],
            (len(env_ids),),
            device=self.device,
        )
        # Heading (unused for now, kept zero)
        self.commands[env_ids, 3] = 0.0

        # Add zero velocity case for 25% of the resampled environments
        zero_mask = torch.rand(len(env_ids), device=self.device) < 0.25
        self.commands[env_ids[zero_mask], :3] = 0.0

        # Reset timer
        self.command_timer[env_ids] = 0.0

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Called before the physics step. Here we just store the action."""
        self.previous_actions = self.actions.clone()
        self.last_joint_vel = self.joint_vel.clone()
        self.actions = actions.clone()

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

    def _apply_action(self) -> None:
        """
        Applies the neural network action to the robot joints.
        Mode: Absolute Position Control (PD)
        """
        # 1. Compute Targets
        targets = self.actions * self.cfg.action_scale + self.desired_joint_pos

        if getattr(self, "is_heterogeneous", False):
            # DISTRIBUTE to partitioned views
            for i, view in enumerate(self.robot_views):
                indices = self.robot_view_indices[i]
                # local_targets = targets[indices]
                # Clamp per-robot (they all have same limits usually, but good practice)
                lower = view.data.soft_joint_pos_limits[0, self._joint_dof_idx, 0]
                upper = view.data.soft_joint_pos_limits[0, self._joint_dof_idx, 1]
                view_targets = torch.clamp(targets[indices], lower, upper)

                view.set_joint_position_target(
                    view_targets, joint_ids=self._joint_dof_idx
                )
                view.set_joint_velocity_target(
                    torch.zeros_like(view_targets), joint_ids=self._joint_dof_idx
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

        # -- Update feet air time logic --
        # Check contact (force > threshold, e.g. 1.0)
        contact = (
            torch.norm(self.net_contact_forces[:, self._feet_ids, :], dim=-1) > 1.0
        )
        # First contact this step: currently contact AND NOT previously contact
        first_contact = contact & ~self.last_feet_contact
        # Increment air time
        self.feet_air_time += self.step_dt
        # Calculate reward for feet that just landed: (air_time - threshold) * first_contact
        # Threshold from config is 0.5 (based on reference params), but commonly 0.5s or similar.
        # Reference: params={"threshold": 0.5}.
        rew_air_time = torch.sum(
            (self.feet_air_time - 0.5) * first_contact.float(), dim=1
        )
        # Clip negative rewards? Usually we only reward > threshold.
        # But (0.1 - 0.5) is negative. The reward usually is (air_time - threshold).clamp(min=0) OR just raw.
        # Reference implementation `feet_air_time` usually clips or guards.
        # "RewTerm(func=mdp.feet_air_time... threshold=0.5)"
        # Let's assume we want to reward simply if > 0.5.
        # Safe implementation: mask with command norm to avoid farming air time while standing still
        rew_air_time = torch.sum(
            (self.feet_air_time - 0.5).clamp(min=0.0) * first_contact.float(), dim=1
        ) * (torch.norm(self.commands[:, :2], dim=1) > 0.1)
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
            torch.exp(-torch.square(feet_heights - self.cfg.target_foot_height) / 0.005)
            * (~contact).float(),
            dim=1,
        )
        # Apply command mask (x, y, yaw commands)
        rew_foot_height *= (torch.norm(self.commands[:, :3], dim=1) > 0.1).float()

        self.foot_height_reward_val = rew_foot_height

        # Penalty for each foot in the air (constant per-step)
        self.feet_air_penalty_val = torch.sum((~contact).float(), dim=1)
        # Extra penalty when standing still (commands == 0)
        static_mask = (torch.norm(self.commands[:, :3], dim=1) < 0.1).float()
        self.feet_air_penalty_static_val = self.feet_air_penalty_val * static_mask
        self.joint_vel_l2_static_val = (
            torch.sum(torch.square(self.joint_vel), dim=1) * static_mask
        )

        # Reset air time for feet in contact
        self.feet_air_time[contact] = 0.0
        self.last_feet_contact = contact

        # Height Scan Processing
        if self.cfg.observation_space != 49:
            height_scan = self._height_scanner.data.pos_w[:, :, 2] - self.root_pos_w[:, 2].unsqueeze(1)
            # Clip to a reasonable range and potentially add noise later if needed
            # (Already adding noise to the whole 'obs' tensor below)
            height_scan = torch.clip(height_scan, -1.0, 1.0)

        obs_list = [
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity,
            self.commands,
            self.joint_pos - self.desired_joint_pos,
            self.joint_vel,
            self.actions,
        ]
        if self.cfg.observation_space != 49:
            obs_list.append(height_scan)

        obs = torch.cat(obs_list, dim=-1)

        # Add observation noise (Sim2Real)
        obs_noise = torch.randn_like(obs) * self.cfg.observation_noise_scale
        obs = obs + obs_noise

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        Computes the reward (score) for the current step.
        The goal is to teach the robot to stand up and retain balance.
        """
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
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
            self.cfg.command_lin_vel_std,
            self.cfg.command_ang_vel_std,
            self.commands,
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity,
            self.joint_pos,
            self.desired_joint_pos,
            self.joint_vel,
            self.last_joint_vel,
            self.applied_torque,
            self.joint_acc,
            self.actions,
            self.previous_actions,
            self.feet_air_time_reward_val,
            self.foot_height_reward_val,
            self.feet_air_penalty_val,
            self.feet_air_penalty_static_val,
            self.joint_vel_l2_static_val,
            self.reset_terminated,
            self.step_dt,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Determines if the episode is over.
        1. Died: Base height is too low (fell over).
        2. Timeout: Episode duration exceeded limit.
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Check if base is too tilted (not upright)
        upright_check = (
            self.projected_gravity[:, 2] > -self.cfg.base_angle_termination_thresh
        )

        # Fall detection: if the robot's body is lower than 15cm, it likely fell.
        base_height = self.root_pos_w[:, 2]

        died = (base_height < 0.15) | upright_check

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
                    self._randomize_view_state(subset_env_ids, view, local_indices)

            # CRITICAL: Reset the base environment buffers (which we bypassed)
            self.episode_length_buf[env_ids] = 0
            self.reset_buf[env_ids] = 0
            self.feet_air_time[env_ids] = 0.0
            self.last_joint_vel[env_ids] = 0.0
            self.previous_actions[env_ids] = 0.0
        else:
            super()._reset_idx(env_ids)
            # Standard Mass/Friction/State randomization
            self._randomize_view_state(env_ids, self.robot)

    def _randomize_view_state(
        self,
        env_ids: torch.Tensor,
        view: Articulation,
        local_ids: torch.Tensor | None = None,
    ):
        # 0. Randomize Base Mass (Sim2Real)
        env_ids_cpu = env_ids.cpu()
        local_ids_cpu = local_ids.cpu() if local_ids is not None else env_ids_cpu

        masses = view.root_physx_view.get_masses().clone()
        mass_noise = sample_uniform(-1.0, 3.0, (len(env_ids_cpu), 1), "cpu")
        masses[local_ids_cpu, 0] = (
            view.data.default_mass[local_ids_cpu, 0] + mass_noise[:, 0]
        )
        view.root_physx_view.set_masses(masses, local_ids_cpu)

        # Use correct ID set for shape (local_ids if heterogeneous, else env_ids)
        ids = local_ids if local_ids is not None else env_ids

        # 0.1 Randomize Joint Friction and Damping (Enforce Config)
        friction_noise = sample_uniform(
            self.cfg.joint_friction_range[0],
            self.cfg.joint_friction_range[1],
            (len(ids), len(self._joint_dof_idx)),
            self.device,
        )
        view.write_joint_friction_coefficient_to_sim(
            friction_noise,
            joint_ids=self._joint_dof_idx,
            env_ids=ids,
        )

        damping_noise = sample_uniform(
            self.cfg.joint_damping_range[0],
            self.cfg.joint_damping_range[1],
            (len(ids), len(self._joint_dof_idx)),
            self.device,
        )
        view.write_joint_damping_to_sim(
            damping_noise,
            joint_ids=self._joint_dof_idx,
            env_ids=ids,
        )

        # 1. Reset Joint States (Use Default Pose + Noise on controlled joints)
        # Use full joint arrays (all joints, not just controlled ones)
        joint_pos = view.data.default_joint_pos[ids].clone()
        joint_vel = view.data.default_joint_vel[ids].clone()

        # Add small random noise to initial joint positions and velocities
        pos_noise = sample_uniform(
            -0.2, 0.2, (len(ids), len(self._joint_dof_idx)), joint_pos.device
        )
        vel_noise = sample_uniform(
            -0.5, 0.5, (len(ids), len(self._joint_dof_idx)), joint_vel.device
        )

        # Apply noise only to controlled joints
        joint_pos[:, self._joint_dof_idx] += pos_noise
        joint_vel[:, self._joint_dof_idx] += vel_noise

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

        # 5. Resample Commands
        self._resample_commands(env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
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
    command_lin_vel_std: float,
    command_ang_vel_std: float,
    commands: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    joint_pos: torch.Tensor,
    desired_joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    last_joint_vel: torch.Tensor,
    joint_torques: torch.Tensor,
    joint_acc: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    feet_air_time_reward_val: torch.Tensor,
    foot_height_reward_val: torch.Tensor,
    feet_air_penalty_val: torch.Tensor,
    feet_air_penalty_static_val: torch.Tensor,
    joint_vel_l2_static_val: torch.Tensor,
    reset_terminated: torch.Tensor,
    step_dt: float,
):
    # 1. Alive (Optional, usually 0)
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())

    # 2. Tracking Linear Velocity XY (Exponential)
    # Target is commands[:, 0:2] (x, y)
    # Local velocity is base_lin_vel[:, 0:2]
    # commands is [vx, vy, wz, heading]
    lin_vel_error = torch.sum(
        torch.square(base_lin_vel[:, :2] - commands[:, :2]), dim=1
    )
    rew_track_lin_vel_xy_exp = rew_scale_track_lin_vel_xy_exp * torch.exp(
        -lin_vel_error / (command_lin_vel_std**2)
    )

    # 3. Tracking Angular Velocity Z (Exponential)
    # Target is commands[:, 2] (wz)
    ang_vel_error = torch.square(base_ang_vel[:, 2] - commands[:, 2])
    rew_track_ang_vel_z_exp = rew_scale_track_ang_vel_z_exp * torch.exp(
        -ang_vel_error / (command_ang_vel_std**2)
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
    rew_dof_acc_l2 = rew_scale_dof_acc_l2 * torch.sum(torch.square(joint_acc), dim=1)
    # Note: If joint_acc is not readily available or reliable in DirectRLEnv simplifications,
    # we might need to approximate it from (joint_vel - last_joint_vel)/dt.
    # However, Isaac Sim usually provides it. We passed joint_acc.
    # If joint_acc is zero (because no sensor?), check implementation.
    # For now assuming it works.

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

    total_reward = (
        rew_alive
        + rew_track_lin_vel_xy_exp
        + rew_track_ang_vel_z_exp
        + rew_lin_vel_z_l2
        + rew_ang_vel_xy_l2
        + rew_dof_torques_l2
        + rew_dof_acc_l2
        + rew_action_rate_l2
        + rew_feet_air_time
        + rew_dof_pos_l2
        + rew_flat_orientation_l2
        + rew_foot_height
        + rew_scale_feet_air_penalty * feet_air_penalty_val
        + rew_scale_feet_air_penalty_static * feet_air_penalty_static_val
        + rew_scale_joint_vel_l2_static * joint_vel_l2_static_val
    )
    return total_reward
