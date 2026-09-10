# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
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
        self._joint_dof_idx, _joint_names = self.robot.find_joints(
            ".*_hip_joint|.*_thigh_joint|.*_calf_joint"
        )
        # Column positions of the four hip (abduction) joints WITHIN _joint_dof_idx, i.e.
        # within self.joint_pos / self.desired_joint_pos. Resolved from the name list the
        # same call returns rather than assuming a DOF ordering, so it holds for whatever
        # order the USD happens to expose and for all three robot models.
        self._hip_local_idx = torch.tensor(
            [i for i, n in enumerate(_joint_names) if n.endswith("_hip_joint")],
            dtype=torch.long,
            device=self.device,
        )
        if getattr(self, "is_heterogeneous", False):
            self._view_joint_dof_idx = []
            for v in self.robot_views:
                idx, _ = v.find_joints(".*_hip_joint|.*_thigh_joint|.*_calf_joint")
                self._view_joint_dof_idx.append(torch.tensor(idx, dtype=torch.long, device=self.device))

        # Per-env soft joint limits for the joint-limit penalty. Built here rather
        # than up with the other per-env tensors because it needs _joint_dof_idx /
        # _view_joint_dof_idx, which are only resolved just above. In the mixed-robot
        # case each view carries its own ranges, so they are gathered per view.
        self.joint_limit_lower = torch.zeros((self.num_envs, 12), device=self.device)
        self.joint_limit_upper = torch.zeros((self.num_envs, 12), device=self.device)
        if getattr(self, "is_heterogeneous", False):
            for i, view in enumerate(self.robot_views):
                indices = self.robot_view_indices[i]
                v_idx = self._view_joint_dof_idx[i]
                self.joint_limit_lower[indices] = view.data.soft_joint_pos_limits[
                    0, v_idx, 0
                ].clone()
                self.joint_limit_upper[indices] = view.data.soft_joint_pos_limits[
                    0, v_idx, 1
                ].clone()
        else:
            self.joint_limit_lower[:] = self.robot.data.soft_joint_pos_limits[
                :, self._joint_dof_idx, 0
            ]
            self.joint_limit_upper[:] = self.robot.data.soft_joint_pos_limits[
                :, self._joint_dof_idx, 1
            ]
        
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
        # Mirror of what ContactSensor would give with track_air_time=True. The sensor here runs
        # with it OFF (it exists for forces), so the durations are tracked by hand:
        #   feet_contact_time      -- how long each foot has been continuously grounded
        #   last_feet_air_time     -- duration of the most recently COMPLETED swing, latched at touchdown
        #   last_feet_contact_time -- duration of the most recently COMPLETED stance, latched at lift-off
        # air_time_variance and the ported feet_air_time both read the latched pair.
        self.feet_contact_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_feet_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_feet_contact_time = torch.zeros(self.num_envs, 4, device=self.device)
        # Peak height reached so far in the current swing, per foot. Monotonic within a swing and
        # reset on landing -- so on the step a foot lands it still holds that swing's apex, which is
        # what the foot-height penalty is charged on (see _compute_reward_terms).
        self.feet_height_max = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_feet_contact = torch.zeros(
            self.num_envs, 4, dtype=torch.bool, device=self.device
        )
        # Vertical foot velocity from the PREVIOUS control step, per foot. The landing-impact
        # penalty is charged on this, not on the current step's value -- see _compute_reward_terms.
        self.last_feet_vel_z = torch.zeros(self.num_envs, 4, device=self.device)
        self.feet_air_time_reward_val = torch.zeros(self.num_envs, device=self.device)
        # Swing-apex term: (foot_height_bias - gaussian_match), summed over feet that landed.
        # One buffer covering what used to be a separate penalty and reward -- see the block
        # comment in _compute_reward_terms for what the bias does.
        self.foot_height_val = torch.zeros(self.num_envs, device=self.device)
        self.foot_landing_vel_val = torch.zeros(self.num_envs, device=self.device)
        self.feet_air_penalty_val = torch.zeros(self.num_envs, device=self.device)
        self.feet_air_penalty_static_val = torch.zeros(self.num_envs, device=self.device)
        self.joint_vel_l2_val = torch.zeros(self.num_envs, device=self.device)
        self.dof_pos_l2_walk_val = torch.zeros(self.num_envs, device=self.device)
        self.dof_pos_l2_stance_val = torch.zeros(self.num_envs, device=self.device)
        self.hip_dev_l1_val = torch.zeros(self.num_envs, device=self.device)
        self.grf_balance_stance_val = torch.zeros(self.num_envs, device=self.device)
        self.joint_limit_val = torch.zeros(self.num_envs, device=self.device)
        # First-step latency: state for rewarding a prompt step out of a standing
        # start. `pending` is set when the command goes zero -> non-zero and cleared
        # by the first landing (or by the command dropping back to zero).
        self.first_step_val = torch.zeros(self.num_envs, device=self.device)
        self.first_step_pending = torch.zeros(self.num_envs, device=self.device)
        self.first_step_elapsed = torch.zeros(self.num_envs, device=self.device)
        # Latency to the FIRST LIFT after arming, frozen at the moment a foot leaves the
        # ground and used as the payout clock; `lifted` marks that it holds a real value.
        self.first_step_lift_time = torch.zeros(self.num_envs, device=self.device)
        self.first_step_lifted = torch.zeros(self.num_envs, device=self.device)
        self.was_moving = torch.zeros(self.num_envs, device=self.device)
        # How long the command has been at rest. Gates arming so the reward is only
        # offered after a genuine stance, never for the post-reset settle.
        self.static_elapsed = torch.zeros(self.num_envs, device=self.device)
        # Per-env opening hold, redrawn each episode from standby_duration_range_s.
        self.standby_duration = torch.full(
            (self.num_envs,),
            float(self.cfg.standby_duration_range_s[1]),
            device=self.device,
        )
        self.grf_target_val = torch.zeros(self.num_envs, device=self.device)
        self.max_contact_force_val = torch.zeros(self.num_envs, device=self.device)
        self.grf_peak_bw_val = torch.zeros(self.num_envs, device=self.device)
        # Diagnostics for the landing-impact penalty: |vz| summed over the feet that touched down
        # this step, and how many did. Kept as sum+count so the log can report a true mean per
        # LANDING EVENT rather than per env.
        self.foot_landing_speed_sum = torch.zeros(self.num_envs, device=self.device)
        self.foot_landing_count = torch.zeros(self.num_envs, device=self.device)
        # Same sum+count accounting for first-step reaction latency, so the log reports a
        # mean per PAID STEP (in seconds) rather than per env -- the number to compare
        # against first_step_timeout when tuning it.
        self.first_step_latency_sum = torch.zeros(self.num_envs, device=self.device)
        self.first_step_latency_count = torch.zeros(self.num_envs, device=self.device)
        self.robot_total_weight = torch.zeros(self.num_envs, device=self.device)  # mg per env, updated on reset
        # Strike-time buffers for contact-driven gait phase symmetry reward
        # last_strike_time: simulation time of last touchdown per foot (N, 4) [FL, FR, RL, RR]
        # stride_duration:  time between the last two consecutive touchdowns per foot (N, 4)
        self.last_strike_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.stride_duration  = torch.ones(self.num_envs, 4, device=self.device)  # init to 1.0 to avoid div-by-zero
        self.gait_phase_sym_val = torch.zeros(self.num_envs, device=self.device)
        # Ported unitree_rl_lab / Isaac Lab terms (arm U1) -- see _compute_reward_terms.
        self.energy_val = torch.zeros(self.num_envs, device=self.device)
        self.feet_slide_val = torch.zeros(self.num_envs, device=self.device)
        self.air_time_variance_val = torch.zeros(self.num_envs, device=self.device)
        self.joint_pos_limits_val = torch.zeros(self.num_envs, device=self.device)
        self.joint_deviation_val = torch.zeros(self.num_envs, device=self.device)
        self.feet_air_time_thresh_val = torch.zeros(self.num_envs, device=self.device)
        # Live command ranges. Separate from cfg.command_x_range/command_y_range because the level
        # curriculum mutates them during training; the cfg values are the STARTING ranges. Every
        # other command axis (yaw) is read from the cfg directly, matching unitree, whose
        # curriculum widens x and y only.
        self.cmd_x_range = list(self.cfg.command_x_range)
        self.cmd_y_range = list(self.cfg.command_y_range)
        self._apply_command_overrides()
        # Per-env episodic sum of the linear-velocity tracking reward, which is what the level
        # curriculum thresholds on. Zeroed per env in _reset_idx, after the curriculum reads it.
        self.track_lin_vel_episode_sum = torch.zeros(self.num_envs, device=self.device)
        self.command_timer = torch.full(
            (self.num_envs,), 100.0, device=self.device
        )  # Force immediate resample

        if self.cfg.obs_history_len > 0:
            self.obs_history_buf = torch.zeros(
                self.num_envs,
                self.cfg.obs_history_len * self.cfg.obs_dim_single,
                device=self.device,
            )

        self._randomize_body_materials()
        self._build_obs_sensor_model()

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

    # Observation vector layout, 49 dims. Any change here must match _get_observations.
    # Observation term order, per layout. Widths and offsets are derived from this, so the
    # noise/bias groups in the yaml keep working unchanged across all three -- a group simply
    # moves, or (base_lin_vel under the "unitree" layout) stops existing. See _OBS_LAYOUTS in
    # quadruped_env_cfg.py for what each layout is for.
    OBS_LAYOUT_TERMS = {
        "full": [
            ("base_lin_vel", 3), ("base_ang_vel", 3), ("projected_gravity", 3),
            ("commands", 4), ("joint_pos", 12), ("joint_vel", 12), ("actions", 12),
        ],
        "unitree_vel": [
            ("base_lin_vel", 3), ("base_ang_vel", 3), ("projected_gravity", 3),
            ("commands", 3), ("joint_pos", 12), ("joint_vel", 12), ("actions", 12),
        ],
        "unitree": [
            ("base_ang_vel", 3), ("projected_gravity", 3),
            ("commands", 3), ("joint_pos", 12), ("joint_vel", 12), ("actions", 12),
        ],
    }

    def _randomize_body_materials(self) -> None:
        """Sample the robot's contact materials once, at construction.

        Reproduces isaaclab mdp.randomize_rigid_body_material, which unitree_rl_lab runs as a
        startup event: draw `material_buckets` (static friction, dynamic friction, restitution)
        triples once, then assign one at random to every collision shape of every robot. Sampling
        once rather than per reset is upstream's design, not a shortcut -- PhysX caps a scene at
        64000 distinct materials.

        This randomises the ROBOT's material. The terrain is fixed at static/dynamic 1.0 with
        friction_combine_mode "multiply" (see _TERRAIN_PHYSICS_MATERIAL), exactly as in unitree's
        scene, so the effective coefficient under each foot is what is sampled here.

        No-op when all three ranges are [0.0, 0.0], the same "disabled" convention the other
        domain_randomization ranges use. get_material_properties()/set_material_properties() work
        on CPU tensors and are slow, which is the other reason this runs once.
        """
        ranges = [
            tuple(self.cfg.body_static_friction_range),
            tuple(self.cfg.body_dynamic_friction_range),
            tuple(self.cfg.body_restitution_range),
        ]
        if all(lo == 0.0 and hi == 0.0 for lo, hi in ranges):
            return

        n_buckets = max(1, int(self.cfg.material_buckets))
        bounds = torch.tensor(ranges, device="cpu")  # (3, 2)
        buckets = sample_uniform(bounds[:, 0], bounds[:, 1], (n_buckets, 3), device="cpu")

        views = self.robot_views if getattr(self, "is_heterogeneous", False) else [self.robot]
        for view in views:
            physx = view.root_physx_view
            materials = physx.get_material_properties()  # (num_envs, num_shapes, 3), CPU
            n_env, n_shapes = materials.shape[0], materials.shape[1]
            bucket_ids = torch.randint(0, n_buckets, (n_env, n_shapes), device="cpu")
            materials[:] = buckets[bucket_ids]
            physx.set_material_properties(materials, torch.arange(n_env, device="cpu"))

        print(
            f"[DR] body materials: static {ranges[0]}, dynamic {ranges[1]}, "
            f"restitution {ranges[2]}, {n_buckets} buckets"
        )

    def _build_obs_sensor_model(self) -> None:
        """Resolve the observation layout, then expand the noise/bias dicts over it, once.

        Sets three things:
          self.OBS_GROUPS  -- {group name: (lo, hi)} slice into the observation vector
          self.obs_scale   -- per-dimension multiplier, applied AFTER noise (see _get_observations)
          self.obs_noise_std / self.obs_bias_bound / self.obs_bias

        obs_noise_std is left as None when the yaml configures no sensor model, which is the
        signal for _get_observations to fall back to the flat observation_noise_scale over every
        dimension. obs_scale is always built.

        Groups omitted from a dict get zero. `commands` and `actions` are therefore clean unless
        explicitly listed: the command comes from the operator and the action is what this policy
        just emitted, so neither carries sensor error on hardware, and adding some only degrades
        the input.
        """
        layout = self.cfg.obs_layout
        terms = self.OBS_LAYOUT_TERMS[layout]
        width = self.cfg.obs_dim_single

        self.OBS_GROUPS = {}
        offset = 0
        for name, n in terms:
            self.OBS_GROUPS[name] = (offset, offset + n)
            offset += n
        assert offset == width, f"layout '{layout}' sums to {offset}, cfg says {width}"

        # Per-term scales. Unitree's policy observation multiplies base_ang_vel by 0.2 and
        # joint_vel by 0.05; the "full" layout leaves both at 1.0. Scaling here rather than in
        # the yaml's noise numbers keeps every sigma stated in raw physical units.
        self.obs_scale = torch.ones(width, device=self.device)
        for name, scale in (
            ("base_ang_vel", float(self.cfg.obs_ang_vel_scale)),
            ("joint_vel", float(self.cfg.obs_joint_vel_scale)),
        ):
            if name in self.OBS_GROUPS and scale != 1.0:
                lo, hi = self.OBS_GROUPS[name]
                self.obs_scale[lo:hi] = scale
        if bool((self.obs_scale == 1.0).all()):
            self.obs_scale = None

        noise_cfg = getattr(self.cfg, "observation_noise", None)
        bias_cfg = getattr(self.cfg, "observation_bias", None)

        if not noise_cfg and not bias_cfg:
            self.obs_noise_std = None
            self.obs_bias_bound = None
            self.obs_bias = None
            return

        # observation_noise_scale doubles as a global multiplier on the sensor model,
        # so one number per phase still dials the whole thing up or down.
        gain = float(self.cfg.observation_noise_scale)

        known_groups = {g for terms_ in self.OBS_LAYOUT_TERMS.values() for g, _ in terms_}

        def _expand(cfg_dict):
            vec = torch.zeros(width, device=self.device)
            for name, value in (cfg_dict or {}).items():
                if name not in self.OBS_GROUPS:
                    # A layout can legitimately DROP a group -- "unitree" has no base_lin_vel --
                    # so one sensor model can be shared across the whole ladder without being
                    # rewritten per rung. Only a name that is not a group at all is an error.
                    if name in known_groups:
                        continue
                    raise ValueError(
                        f"unknown observation group '{name}' -- expected one of "
                        f"{sorted(known_groups)}"
                    )
                lo, hi = self.OBS_GROUPS[name]
                vec[lo:hi] = float(value) * gain
            return vec

        self.obs_noise_std = _expand(noise_cfg) if noise_cfg else torch.zeros(width, device=self.device)
        if bias_cfg:
            self.obs_bias_bound = _expand(bias_cfg)
            self.obs_bias = torch.zeros(self.num_envs, width, device=self.device)
        else:
            self.obs_bias_bound = None
            self.obs_bias = torch.zeros(self.num_envs, width, device=self.device)

    def _apply_command_overrides(self) -> None:
        """Environment-variable overrides on the sampled command box, for play and evaluation.

        These exist because command_x_range / command_y_range are the STARTING box whenever the
        level curriculum is on. The `unitree` phase starts at +-0.1 and the curriculum widens it
        to +-1.0 over training -- but that widening is training state and is NOT saved with the
        checkpoint. A play session therefore rebuilds the env at +-0.1, which is entirely inside
        the dead zone, and the policy stands still at every command it is handed. That looks
        exactly like a broken policy and is not one. Same trap as unitree's own play config,
        which is why RobotPlayEnvCfg there overwrites ranges with limit_ranges.

          QUADRUPED_CMD_FULL=1  start at the curriculum LIMITS rather than the starting box,
                                i.e. the distribution the policy actually finished training on.
                                Use this whenever playing or evaluating a curriculum-trained arm.
          PLAY_CMD_X=<v>        pin every env to v m/s forward: no lateral, no yaw, no standing
                                envs. Same name and meaning as the PLAY_CMD_X patch applied to
                                unitree_rl_lab, so one habit covers both baselines. This is how
                                you look at the dead zone on screen -- the full +-1.0 box puts
                                only ~1.4% of draws below 0.3 m/s, so it is nearly invisible.

        Neither is read during training runs, which set neither variable.
        """
        if os.environ.get("QUADRUPED_CMD_FULL", "0") == "1":
            if self.cfg.command_level_curriculum:
                self.cmd_x_range = [float(v) for v in self.cfg.command_level_limit_x]
                self.cmd_y_range = [float(v) for v in self.cfg.command_level_limit_y]
                print("[Commands] QUADRUPED_CMD_FULL: starting at the curriculum limits.")
            else:
                print("[Commands] QUADRUPED_CMD_FULL ignored: no level curriculum in this phase.")

        pinned = os.environ.get("PLAY_CMD_X")
        self._pin_command_x = float(pinned) if pinned else None
        if self._pin_command_x is not None:
            self.cmd_x_range = [self._pin_command_x, self._pin_command_x]
            self.cmd_y_range = [0.0, 0.0]
            print(f"[Commands] PLAY_CMD_X: every env pinned to lin_vel_x = {self._pin_command_x} m/s.")

        extra = ""
        if self.cfg.command_level_curriculum:
            extra = (f"  | level curriculum ON, limits x {tuple(self.cfg.command_level_limit_x)}"
                     f" y {tuple(self.cfg.command_level_limit_y)}")
        print(
            f"[Commands] sampling box: x {tuple(self.cmd_x_range)}  y {tuple(self.cmd_y_range)}"
            f"  yaw {tuple(self.cfg.command_yaw_range)}{extra}"
        )

    def _resample_commands(self, env_ids: Sequence[int]):
        """Resamples the velocity commands for the specified environments."""
        # Sample x velocity
        self.target_commands[env_ids, 0] = sample_uniform(
            self.cmd_x_range[0],
            self.cmd_x_range[1],
            (len(env_ids),),
            device=self.device,
        )
        # Sample y velocity
        self.target_commands[env_ids, 1] = sample_uniform(
            self.cmd_y_range[0],
            self.cmd_y_range[1],
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

        # Apply special command modes: zero, x-only, y-only, yaw-only, slow
        n_envs = len(env_ids)
        rand_vals = torch.rand(n_envs, device=self.device)

        # Cumulative thresholds for mutually exclusive assignment
        p_zero = self.cfg.zero_command_fraction
        p_x = p_zero + getattr(self.cfg, "x_only_command_fraction", 0.0)
        p_y = p_x + getattr(self.cfg, "y_only_command_fraction", 0.0)
        p_yaw = p_y + getattr(self.cfg, "yaw_only_command_fraction", 0.0)
        p_slow = p_yaw + getattr(self.cfg, "slow_command_fraction", 0.0)

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

        # Slow-command case: keep the direction just drawn and rescale it to a magnitude
        # from slow_command_range.
        #
        # Uniform sampling over the command CUBE gives almost no low-speed coverage -- the
        # r^2 volume effect leaves ||cmd|| under 0.3 in ~1.2% of draws (0.1-0.2 in 0.31%)
        # while ~79% land above 0.5. The band where the robot has to choose between holding
        # a stance and taking one slow step is therefore essentially never trained, and
        # whatever it does there is nearly free in the aggregate return. That coverage gap,
        # not the reward shape, is where the low-speed dead zone lives: a policy that walked
        # at 0.05-0.15 m/s re-learned the dead zone within 180k steps even after the
        # foot_height and tracking-sigma changes that made a slow step profitable.
        #
        # Direction is kept from the cube draw rather than resampled, so the mode covers slow
        # walks, slides and turns in the same proportion as the main distribution. The
        # magnitude is the 3-norm of (vx, vy, wz) -- the same quantity static_mask gates on --
        # so slow_command_range is stated directly in the units of that gate: a range starting
        # above static_command_ramp lands entirely in the "commanded to move" regime.
        slow_mask = (rand_vals >= p_yaw) & (rand_vals < p_slow)
        if slow_mask.any():
            slow_ids = env_ids[slow_mask]
            direction = self.target_commands[slow_ids, :3]
            # clamp guards the (vanishingly rare) near-zero draw from blowing up the rescale
            norm = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-6)
            lo, hi = self.cfg.slow_command_range
            magnitude = sample_uniform(lo, hi, (len(slow_ids), 1), device=self.device)
            self.target_commands[slow_ids, :3] = direction / norm * magnitude

        # PLAY_CMD_X pins the whole batch, overriding the zero / slow / axis-only modes above so
        # what is on screen is exactly the speed asked for. See _apply_command_overrides.
        if getattr(self, "_pin_command_x", None) is not None:
            self.target_commands[env_ids, 0] = self._pin_command_x
            self.target_commands[env_ids, 1:] = 0.0

        # Reset timer
        self.command_timer[env_ids] = 0.0

    def _transition_to_next_phase(self):
        if self.curriculum_phase_idx >= len(self.curriculum_phases):
            return
            
        next_phase = self.curriculum_phases[self.curriculum_phase_idx]
        p_cfg = next_phase["cfg"]
        
        print(f"\n{'='*50}\n[Curriculum] Transitioning to Phase: {next_phase['name']}\n{'='*50}\n")
        
        def _apply(section: str, as_tuple: bool = False):
            """Push one yaml section onto self.cfg, failing loudly on unknown keys.

            This used to be guarded by `if hasattr(self.cfg, k)`, which silently dropped any key
            the cfg class doesn't define -- so a typo in a phase override, or a yaml key that was
            never wired into quadruped_env_cfg.py, would just never take effect and never warn.
            """
            for k, v in p_cfg.get(section, {}).items():
                if not hasattr(self.cfg, k):
                    raise AttributeError(
                        f"[Curriculum] phase '{next_phase['name']}' sets {section}.{k}, but "
                        f"QuadrupedEnvCfg has no such attribute. Add it to quadruped_env_cfg.py "
                        f"or remove it from training_phases.yaml."
                    )
                setattr(self.cfg, k, tuple(v) if as_tuple and isinstance(v, list) else v)

        _apply("rewards")
        _apply("domain_randomization", as_tuple=True)
        _apply("commands")
        # Re-seed the LIVE command ranges from the phase that just loaded. Without this a phase
        # that narrows or widens command_x_range would be ignored whenever the level curriculum
        # is in play, because _resample_commands reads cmd_x_range, not the cfg. Re-seeding also
        # restarts the level curriculum from the new phase's starting box, which is what a phase
        # boundary should mean.
        self.cmd_x_range = list(self.cfg.command_x_range)
        self.cmd_y_range = list(self.cfg.command_y_range)
        self._apply_command_overrides()

        # Env block. Only a subset can meaningfully change mid-process: these are re-read from
        # self.cfg every step. The rest are consumed once at construction (buffer sizes, scene,
        # spawned robots, terrain) and cannot be changed without restarting -- which is exactly why
        # launcher.py splits the curriculum across processes at the phase2->phase3 boundary.
        # This whole block used to be skipped entirely, so observation_noise_scale silently stayed
        # at the starting phase's value for the rest of the run (0.05 instead of the 0.1 that
        # phases 4-6 ask for), i.e. a sim2real hardening step that never happened.
        _RUNTIME_SETTABLE_ENV = {
            "observation_noise_scale",
            "observation_noise_uniform",
            "spawn_height",
            "action_clip",
            "obs_clip",
            "base_angle_termination_thresh",
            "action_scale",
        }
        # max_timesteps legitimately differs per phase (it defines the phase length) and is
        # consumed at init to build the curriculum thresholds, so it is not a mismatch to report.
        _EXPECTED_TO_DIFFER = {"max_timesteps"}
        # Startup-only keys, mapped to where the resolved value actually lives on the cfg -- the
        # yaml name and the cfg attribute name differ for most of these, so a naive
        # getattr(self.cfg, k) would read None and warn on every transition even when nothing
        # changed. Used only to decide whether a genuine mismatch is worth reporting.
        _STARTUP_ONLY_ENV = {
            "terrain": lambda c: getattr(c, "_ter", None),
            "robot_cfg": lambda c: getattr(c, "robot_choice", None),
            "num_envs": lambda c: getattr(getattr(c, "scene", None), "num_envs", None),
            "episode_length_s": lambda c: getattr(c, "episode_length_s", None),
            "obs_history_len": lambda c: getattr(c, "obs_history_len", None),
            # Observation layout and its per-term scales are baked into the observation space
            # and into obs_scale/OBS_GROUPS at construction, so a phase cannot change them.
            "obs_layout": lambda c: getattr(c, "obs_layout", None),
            "obs_ang_vel_scale": lambda c: getattr(c, "obs_ang_vel_scale", None),
            "obs_joint_vel_scale": lambda c: getattr(c, "obs_joint_vel_scale", None),
        }
        for k, v in p_cfg.get("env", {}).items():
            if k in _RUNTIME_SETTABLE_ENV:
                if not hasattr(self.cfg, k):
                    raise AttributeError(
                        f"[Curriculum] phase '{next_phase['name']}' sets env.{k}, but "
                        f"QuadrupedEnvCfg has no such attribute."
                    )
                setattr(self.cfg, k, v)
            elif k in _EXPECTED_TO_DIFFER:
                continue
            elif k in _STARTUP_ONLY_ENV:
                current = _STARTUP_ONLY_ENV[k](self.cfg)
                if str(current).upper() != str(v).upper():
                    print(
                        f"[Curriculum] WARNING: phase '{next_phase['name']}' wants env.{k} = {v}, "
                        f"but that is fixed at process start (currently {current}). Split the "
                        f"curriculum across processes if this phase change matters."
                    )
            else:
                raise AttributeError(
                    f"[Curriculum] phase '{next_phase['name']}' sets unrecognised env.{k}. Add it "
                    f"to _RUNTIME_SETTABLE_ENV or _STARTUP_ONLY_ENV in _transition_to_next_phase."
                )

        # Events. The push terms are always registered (with a zero range standing in for
        # "disabled"), so the range can be rewritten live here. Previously this section was not
        # handled at all: push_velocity_range stayed at the starting phase's value for the whole
        # run, and in a phase1_to_phase2 run -- where phase1 disables pushes -- phase 2 ran with no
        # pushes whatsoever despite push hardening being its entire purpose.
        e_cfg = p_cfg.get("events", {})
        if e_cfg and getattr(self, "event_manager", None) is not None:
            enabled = e_cfg.get("enable_pushes", True)
            rng = e_cfg.get("push_velocity_range", [0.0, 0.0]) if enabled else [0.0, 0.0]
            interval = e_cfg.get("push_interval_range_s")
            for term_name in ("push_a1", "push_quadruped", "push_go2"):
                try:
                    term_cfg = self.event_manager.get_term_cfg(term_name)
                except (ValueError, KeyError):
                    continue  # term not registered (e.g. non-heterogeneous setups)
                term_cfg.params["velocity_range"] = {
                    "x": (rng[0], rng[1]),
                    "y": (rng[0], rng[1]),
                }
                if interval is not None:
                    term_cfg.interval_range_s = (float(interval[0]), float(interval[1]))
                self.event_manager.set_term_cfg(term_name, term_cfg)
            print(
                f"[Curriculum] push velocity range -> {tuple(rng)} (enabled={enabled}"
                + (f", every {tuple(interval)} s)" if interval is not None else ")")
            )

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
        # unitree's action clip. Applied to the STORED action, so the clipped value is what
        # reaches both the joint targets and the `last_action` observation channel -- the same
        # thing Isaac Lab's ActionManager does (raw -> clip -> scale + offset). See action_clip
        # in quadruped_env_cfg.py for why this matters more than it looks.
        if self.cfg.action_clip is not None:
            self.actions = self.actions.clamp(-float(self.cfg.action_clip), float(self.cfg.action_clip))

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
            if float(self.cfg.standby_duration_range_s[1]) > 0.0:
                # Apply zero velocity standby ONLY at the start of the episode to allow
                # stable standing. The hold is per-env and redrawn each episode, so the
                # envs do not all start walking on the same step.
                standby_mask = (
                    self.episode_length_buf * self.step_dt
                ) < self.standby_duration
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

    def _refresh_state(self) -> None:
        """Pull every sim-backed state attribute fresh from the live sim.

        DirectRLEnv.step() runs the physics substeps, then _get_dones(), then _get_rewards(),
        and only calls _get_observations() LAST (see isaaclab/envs/direct_rl_env.py). All the
        self.* state attributes below used to be written exclusively by _get_observations, so at
        reward-computation time they still held the values from the END of the PREVIOUS step.

        That was originally patched for just joint_vel/base_lin_vel (whose staleness made the
        dof_acc_l2/base_acc_l2 finite differences evaluate to exactly zero, since last_joint_vel/
        last_base_lin_vel were captured from the same stale attributes in _pre_physics_step). But
        the partial fix left the reward function internally inconsistent: track_lin_vel_xy_exp was
        scored against s_{t+1} while track_ang_vel_z_exp, flat_orientation_l2 and dof_torques_l2
        were still scored against s_t. Refreshing everything in one place keeps every term on the
        same timestep.

        Called at the top of both _get_rewards() and _get_observations() -- the second call is not
        redundant, because _reset_idx() runs between them and teleports the reset envs.
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

        self.net_contact_forces = self._contact_sensor.data.net_forces_w
        if len(self._undesired_contact_body_ids) > 0:
            self.net_undesired_contact_forces = self.net_contact_forces[:, self._undesired_contact_body_ids, :]
        else:
            self.net_undesired_contact_forces = torch.zeros((self.num_envs, 1, 3), device=self.device)

    def _compute_reward_terms(self) -> None:
        """Compute every per-step reward quantity (self.*_val) from freshly refreshed state.

        Must be called exactly once per control step, from _get_rewards() after _refresh_state().
        It owns stateful per-step updates -- feet_air_time accumulation/reset, last_feet_contact,
        last_feet_vel_z, the gait-phase strike bookkeeping and the pos/yaw reference leash -- so
        calling it twice in a step (or from _get_observations) would double-count them (and would
        overwrite last_feet_vel_z with the post-impact velocity the landing penalty exists to
        avoid reading).
        """
        # -- Update feet air time logic --
        # Check contact (force > threshold, e.g. 1.0)
        contact = (
            torch.norm(self.net_contact_forces[:, self._feet_ids, :], dim=-1) > 1.0
        )
        # First contact this step: currently contact AND NOT previously contact
        first_contact = contact & ~self.last_feet_contact
        # Feet that completed a real swing this step. _reset_idx clears last_feet_contact to False,
        # so on step 1 of an episode every foot already standing on the ground reads as
        # first_contact -- a landing that never happened. episode_length_buf is incremented before
        # _get_rewards, so it is exactly 1 on that step. Both per-landing penalties below
        # (foot height, landing velocity) gate on this rather than on first_contact directly.
        landed = first_contact & (self.episode_length_buf > 1).unsqueeze(1)
        # Increment air time
        self.feet_air_time += self.step_dt
        # Stance-duration counterpart, plus the two latched COMPLETED durations. Both counters are
        # cleared at the end of this method for the feet in their other state, the way
        # feet_air_time already was, so a value latched here still includes the step that finished
        # the interval -- which is what ContactSensor.last_air_time reports as well.
        self.feet_contact_time += self.step_dt
        lifted_this_step = (~contact) & self.last_feet_contact
        self.last_feet_air_time = torch.where(
            first_contact, self.feet_air_time, self.last_feet_air_time
        )
        self.last_feet_contact_time = torch.where(
            lifted_this_step, self.feet_contact_time, self.last_feet_contact_time
        )

        # Smooth static/moving gate. This used to be a hard switch at static_velocity_threshold
        # (0.001): at ||cmd||=0 the stepping rewards were off and the static penalties on, and one
        # thousandth above it they swapped completely. Those two inputs are near-identical to the
        # network but demanded opposite behaviour, so it could never represent the cliff sharply and
        # the "keep stepping" mode bled across into exact zero -- the robot marching in place under a
        # zero command. Ramping the weight linearly over [threshold, static_command_ramp] instead
        # makes "near-zero command -> hold still" a learnable, continuous function of the command.
        # The ramp tops out well below the speeds we want real walking at, so full stepping reward is
        # still available everywhere it matters.
        cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        ramp_lo = self.cfg.static_velocity_threshold
        ramp_hi = max(self.cfg.static_command_ramp, ramp_lo + 1e-6)
        moving_mask = ((cmd_norm - ramp_lo) / (ramp_hi - ramp_lo)).clamp(0.0, 1.0)
        static_mask = 1.0 - moving_mask

        # Speed-dependent swing target: ramp linearly from target_feet_air_time_slow (long swing,
        # low cadence) at feet_air_time_speed_lo down to target_feet_air_time (the original fixed
        # 0.25 default) at feet_air_time_speed_hi, using xy command speed only (not yaw -- this is
        # about translational stride, not turning in place).
        command_speed_xy = torch.norm(self.commands[:, :2], dim=1)
        speed_lo = self.cfg.feet_air_time_speed_lo
        speed_hi = max(self.cfg.feet_air_time_speed_hi, speed_lo + 1e-6)
        slow_frac = 1.0 - ((command_speed_xy - speed_lo) / (speed_hi - speed_lo)).clamp(0.0, 1.0)
        target_air_time_dyn = (
            self.cfg.target_feet_air_time
            + (self.cfg.target_feet_air_time_slow - self.cfg.target_feet_air_time) * slow_frac
        ).unsqueeze(1)

        # Potential-based shaping: treat phi(t) = exp(-(air_time-target)^2/sigma) as a potential and
        # reward its rate of change dphi/dt every step while airborne, instead of paying phi(t) once
        # at landing. Summed over a swing this telescopes back to phi(landing)-phi(0) (same total as
        # the old lump-sum design for a well-timed swing), but gives dense, directional feedback:
        # positive while air_time is approaching target, zero at the peak, negative past it -- so
        # there's no way to "camp" near the target, the signal pushes toward landing right around it.
        air_time_err = self.feet_air_time - target_air_time_dyn
        phi = torch.exp(-torch.square(air_time_err) / self.cfg.feet_air_time_sigma)
        dphi_dt = -2.0 * air_time_err / self.cfg.feet_air_time_sigma * phi
        # Multiply the rate by step_dt: this is what actually makes per-step rewards sum (Riemann
        # sum) to phi(landing)-phi(0) over a swing -- the raw rate alone would overcount by 1/step_dt.
        rew_air_time = torch.sum(
            dphi_dt * self.step_dt * (~contact).float(), dim=1
        ) * moving_mask
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

        # body_pos_w is world-frame Z, but target_foot_height means "clearance above the ground
        # underneath this robot". On rough terrain the per-env terrain patch sits at a nonzero
        # height (TerrainGenerator sets each sub-terrain origin's z to the max height of its centre
        # region), so an uncorrected world Z biases the reward by however high that patch happens to
        # be -- with noise_range up to 0.06m against foot_height_sigma=0.01, a correctly-lifted foot
        # can score exp(-0.36)=0.70 instead of 1.0 purely from which patch its env landed on. The
        # policy has no terrain sensor here and cannot compensate. base_height_val already applies
        # exactly this correction; foot height was the inconsistent one.
        # NOTE: this removes the per-env systematic bias, not the local roughness under each
        # individual foot -- correcting that would need a height scan (see the Stairs task module).
        feet_heights = feet_heights - self.scene.env_origins[:, 2].unsqueeze(1)

        # -- Swing-apex height term, charged once per swing at touchdown --
        # ONE measurement (this swing's apex vs target_foot_height, scored by a Gaussian) and ONE
        # scale, with foot_height_bias setting where it crosses zero:
        #
        #     reward = rew_scale_foot_height * (match - foot_height_bias)
        #
        # The scale is POSITIVE and the bias is what the match has to beat to earn anything, so
        # the bias slides the term continuously between the two behaviours that used to be
        # separate scales:
        #
        #   bias = 0.0  pure REWARD. Pays up to +scale per landing at ~target_foot_height and
        #     costs nothing for a bad one -- a lift incentive for early phases where the robot is
        #     still scuffing and needs a reason to pick a foot up. Carries cp7's failure mode: the
        #     optimum drifts ABOVE target because a taller arc also earns, and stepping more often
        #     earns more, so it pushes the policy to step more than the task needs.
        #
        #   bias = 1.0  pure PENALTY. Earns nothing at target, charges down to -scale per foot for
        #     missing it, above target (jumping) and below it (scuffing) alike. Standing still is
        #     free, so feet_air_time is the only thing paying to take a step at all.
        #
        #   bias = 0.5  BOTH AT ONCE. A landing at target earns +0.5*scale, one far from it pays
        #     -0.5*scale. The lift incentive survives without the payout growing with step count,
        #     because a bad landing now costs what a good one earns. This is the default.
        #
        # Because match is bounded in [0, 1], the term is bounded in
        # [-scale*bias, scale*(1-bias)] per foot, so one wild apex costs at most a single unit of
        # scale and cannot swamp the rest of the reward the way an unbounded squared error would.
        #
        # A phase written against the old split pair maps over exactly, and keeps its magnitude:
        # the old lift scale with bias 0.0, the old penalty scale (as a positive) with bias 1.0.
        #
        # feet_height_max is the running peak of the current swing (monotonic within a swing, reset
        # on landing further down), so on the step a foot touches down it holds that swing's apex.
        # Charging there -- rather than every airborne step -- keeps the cost a function of the apex
        # ALONE: a slow high-clearance swing and a quick one with the same apex pay the same, so
        # this term never bids against feet_air_time / target_feet_air_time over swing duration.
        # That is the same one-payment-per-swing accounting the potential-based version telescoped
        # to, minus the payout.
        self.feet_height_max = torch.maximum(self.feet_height_max, feet_heights)
        foot_height_match = torch.exp(
            -torch.square(self.feet_height_max - self.cfg.target_foot_height) / self.cfg.foot_height_sigma
        )
        # Only feet that landed THIS step are counted (`landed`, computed above), and only while the
        # robot is actually commanded to move -- target_foot_height is meaningless when it is not
        # supposed to step, so a robot told to hold still neither pays nor earns for shuffling a foot.
        # This is a HARD gate at static_velocity_threshold, NOT the smooth moving_mask the other
        # stepping terms use. With foot_height_bias >= 0.5 this term is mostly a penalty, and ramping
        # it in over [threshold, static_command_ramp] charged a fraction of that penalty for landings
        # under commands so small the robot is not really expected to walk under them -- a partial
        # cost for a step it was never asked to take. Below the threshold the term is off entirely;
        # above it the full term applies at every commanded speed.
        commanded_moving = (cmd_norm > self.cfg.static_velocity_threshold).float()
        landed_moving = landed.float() * commanded_moving.unsqueeze(1)
        self.foot_height_val = torch.sum(
            (foot_height_match - self.cfg.foot_height_bias) * landed_moving, dim=1
        )

        # -- Landing impact PENALTY: vertical foot speed at touchdown, charged once per landing --
        # Target is a foot set down at zero vertical speed. Mismatch again, so
        # rew_scale_foot_landing_vel must be NEGATIVE.
        #
        # MEASURED ON THE PREVIOUS STEP'S VELOCITY, deliberately. By the time a contact force
        # crosses the threshold, the physics has already resolved the collision over that control
        # step's substeps, so the foot's velocity at the end of the step is POST-impact -- near
        # zero for exactly the hard landings this is meant to catch. Reading it one control step
        # earlier gives the pre-impact approach speed. It runs up to one step early (a free-falling
        # foot gains ~0.2 m/s over a 0.02 s step), so this slightly under-reports the true impact
        # speed, but it is the right quantity: the current-step value would report ~0 for a slam.
        #
        # Complements rew_scale_max_contact_force, which charges the resulting force spike. This
        # one is the cause rather than the effect, and is far better conditioned -- foot speed is
        # smooth in the actions, whereas a contact force spike is a near-discontinuous function of
        # them, sensitive to solver stiffness and to the sensor's history window.
        #
        # NOT masked by moving_mask, unlike foot height / feet_air_time. Those shape a gait that is
        # only wanted when a command is given; "land softly" holds unconditionally, and gating it
        # would make hard landings free at zero command -- exactly the push-recovery case where the
        # feet come down hardest. This follows max_contact_force, which is likewise ungated.
        # All three components in one read: z feeds the landing-impact penalty below, xy feeds
        # the ported feet_slide term further down.
        if getattr(self, "is_heterogeneous", False):
            feet_vel_w = torch.zeros((self.num_envs, 4, 3), device=self.device)
            for i, view in enumerate(self.robot_views):
                indices = self.robot_view_indices[i]
                feet_ids = self.robot_feet_ids[i]  # relative to Articulation (FL, FR, RL, RR)
                feet_vel_w[indices] = view.data.body_lin_vel_w[:, feet_ids, :]
        else:
            feet_vel_w = self.robot.data.body_lin_vel_w[:, self._feet_ids_articulation, :]
        feet_vel_z = feet_vel_w[:, :, 2]

        # Squared, so an upward-moving foot at touchdown (a scuff into a bump, or a skimming
        # re-contact) is charged the same as one dropping. No env-origin correction is needed the
        # way it is for height -- a velocity has no terrain offset.
        landing_mismatch = 1.0 - torch.exp(
            -torch.square(self.last_feet_vel_z) / self.cfg.foot_landing_vel_sigma
        )
        self.foot_landing_vel_val = torch.sum(landing_mismatch * landed.float(), dim=1)
        # Raw touchdown speed in m/s -- what you actually tune foot_landing_vel_sigma and the scale
        # against, since the mismatch itself is a saturating unitless number.
        self.foot_landing_speed_sum = torch.sum(self.last_feet_vel_z.abs() * landed.float(), dim=1)
        self.foot_landing_count = landed.float().sum(dim=1)
        # Stored AFTER the charge above, so the buffer holds the pre-impact value on the step a
        # foot lands. Cleared per-episode in _reset_idx alongside last_feet_contact.
        self.last_feet_vel_z = feet_vel_z.clone()

        # Penalty grows with how long each foot has been continuously airborne (self.feet_air_time,
        # already tracked above -- resets to 0 on landing), instead of a flat per-airborne-foot cost.
        # Keeps a normal step cheap while discouraging a foot getting stuck hovering; see
        # target_feet_air_time/feet_air_time_sigma comment in training_phases.yaml for the balance
        # against rew_scale_feet_air_time so this penalty doesn't outweigh completing a real step.
        self.feet_air_penalty_val = torch.sum(self.feet_air_time * (~contact).float(), dim=1)
        # Extra penalty when standing still (ramped static_mask computed above).
        self.feet_air_penalty_static_val = self.feet_air_penalty_val * static_mask
        # Joint-speed penalty, charged at every command. It used to be multiplied by static_mask
        # to target marching in place ("zero base velocity, large joint velocity", which the
        # velocity-tracking and pos_deviation rewards are both fully happy with). That gating made
        # it inert: a stance that is actually still has sum(joint_vel^2) ~ 0, so the masked term
        # was scoring nothing at rest, and nothing at all while walking. Ungated it still prices
        # the marching case -- joint velocity is exactly what it reads -- and additionally taxes
        # flailing at speed.
        #
        # Measured on the Final7pt2 policy in MuJoCo, sum(joint_vel^2) is ~0 standing, ~45 at
        # 0.25 m/s, ~83 at 0.5, ~151 at 0.75 and ~195 at 1.0. It therefore rises with commanded
        # speed and pushes against track_lin_vel_xy_exp at the top of the range, which is what
        # bounds the scale -- see the rew_scale_joint_vel_l2 comment in training_phases.yaml.
        self.joint_vel_l2_val = torch.sum(torch.square(self.joint_vel), dim=1)

        # DOF position deviation, split by static/moving (same ramp as everything else above) so
        # standing posture and walking posture can be regularized independently -- a joint
        # configuration that's a sensible average across a whole gait cycle isn't necessarily what
        # you want a planted, motionless stance to relax into, and tuning one scale was forcing a
        # compromise between the two.
        dof_pos_err = torch.sum(torch.square(self.joint_pos - self.desired_joint_pos), dim=1)
        self.dof_pos_l2_walk_val = dof_pos_err * moving_mask
        self.dof_pos_l2_stance_val = dof_pos_err * static_mask

        # -- Hip (abduction) deviation, L1, charged only when NOT commanded to turn --
        # The hips are the joints that swing a leg sideways. Turning needs them; holding a
        # heading does not, and hip excursion under a zero yaw command is what produces the
        # splayed, laterally-loaded stance the GRF balance term then has to fight. L1 rather
        # than L2 on purpose: the squared form is nearly flat near the default pose and so
        # ignores exactly the small persistent offsets this is meant to remove, while its
        # gradient grows without bound on the large excursions a real turn needs.
        #
        # Gated on the YAW command alone, not ||cmd||, so it stays active while the robot
        # walks straight -- a forward trot has no reason to abduct.
        #
        # HARD gate at static_velocity_threshold (rad/s), not the ramp the static/moving
        # terms use: either the robot was told to turn or it was not, and there is no
        # partial hip excursion that a fraction of a yaw command makes correct.
        #
        # NOTE: a pure lateral (vy) command is executed largely THROUGH the hips and carries
        # no yaw, so it is charged here too. If side-stepping degrades, gate this on the
        # lateral command as well.
        yaw_cmd = self.commands[:, 2].abs()
        hip_dev = (
            self.joint_pos[:, self._hip_local_idx]
            - self.desired_joint_pos[:, self._hip_local_idx]
        ).abs().sum(dim=1)
        self.hip_dev_l1_val = hip_dev * (
            yaw_cmd <= self.cfg.static_velocity_threshold
        ).float()

        # -- First-step latency, paid once per zero -> non-zero command transition --
        # The failure this targets: out of a static stance the policy can sit with all
        # four feet planted for over a second while commanded to walk, pivoting over a
        # loaded foot instead of stepping (measured on hardware and reproduced in sim).
        # Nothing else in the reward charges for that -- velocity tracking is already
        # partially satisfied by leaning, and every gait-shaping term is either masked
        # to static or only fires once a step is under way.
        #
        # Paid on the first LANDING after the transition, not on the first lift. Paying
        # on lift is collectable by raising a foot and holding it up: foot_height only
        # charges at touchdown, and with rew_scale_feet_air_penalty and feet_air_time
        # both at 0 an airborne foot costs nothing at all. Requiring the step to
        # complete puts the landing back under foot_height, so a shallow scuff taken
        # only to collect this still pays the apex penalty.
        #
        # The PAYOUT, though, is clocked on the first LIFT, not on that landing: it decays
        # linearly from 1.0 at the instant of the command to 0.0 at first_step_timeout,
        # evaluated at the moment a foot first leaves the ground. What this term is meant
        # to charge is reaction latency -- how long the robot stands there before committing
        # to a step -- and clocking it on the landing folded swing duration into that, so a
        # prompt lift followed by a deliberate, slow, well-controlled swing scored the same
        # as a late one. Swing duration already has its own terms (feet_air_time targets it,
        # foot_landing_vel charges arriving fast); paying this one for it too meant the two
        # bid against each other, with first_step at 25.0 easily the louder voice. Splitting
        # them leaves the landing as the gate that proves the step was real and the lift as
        # the clock that prices the hesitation.
        #
        # A transition only counts if the robot had actually been standing first.
        # static_elapsed is zeroed on reset, so an episode whose opening command is
        # already non-zero does not arm: otherwise the term collected a near-full
        # payout every episode for the feet touching down after the spawn drop, which
        # is a landing the policy gets for free rather than a step it chose to take.
        moving_now = (cmd_norm > self.cfg.static_velocity_threshold).float()
        onset = moving_now * (1.0 - self.was_moving)
        self.was_moving = moving_now
        # Evaluated before static_elapsed is advanced, so the test sees the rest time
        # accumulated up to the instant the command changed rather than after it.
        armed = onset * (
            self.static_elapsed >= self.cfg.first_step_min_static_s
        ).float()
        self.static_elapsed = (self.static_elapsed + self.step_dt) * (1.0 - moving_now)
        # Start the clock on an armed transition; keep counting otherwise.
        self.first_step_elapsed = torch.where(
            armed > 0.5,
            torch.zeros_like(self.first_step_elapsed),
            self.first_step_elapsed + self.step_dt,
        )
        # Arm on transition, disarm if the command returns to zero before a step.
        self.first_step_pending = torch.clamp(
            self.first_step_pending + armed, max=1.0
        ) * moving_now
        # A foot leaving the ground: the inverse of `first_contact`, computed off the same
        # last_feet_contact snapshot, so lift and landing are read from one contact history.
        lifted_off = (~contact & self.last_feet_contact).any(dim=1)
        # Freeze the clock at the FIRST lift while pending; later lifts in the same step
        # cycle must not overwrite it. Cleared on arming so each transition times its own.
        self.first_step_lifted = torch.where(
            armed > 0.5, torch.zeros_like(self.first_step_lifted), self.first_step_lifted
        )
        first_lift = (self.first_step_pending > 0.5) & lifted_off & (self.first_step_lifted < 0.5)
        self.first_step_lift_time = torch.where(
            first_lift, self.first_step_elapsed, self.first_step_lift_time
        )
        self.first_step_lifted = torch.where(
            first_lift, torch.ones_like(self.first_step_lifted), self.first_step_lifted
        )
        # Landing still gates the payment, and the recorded lift is required with it: a
        # landing with no lift behind it is a foot that was already airborne when the
        # command arrived, which is not a step this term asked for. Staying pending in
        # that case simply defers payment to the next genuine step.
        paid = (
            (self.first_step_pending > 0.5)
            & landed.any(dim=1)
            & (self.first_step_lifted > 0.5)
        )
        self.first_step_val = torch.where(
            paid,
            (1.0 - self.first_step_lift_time / self.cfg.first_step_timeout).clamp(min=0.0),
            torch.zeros_like(self.first_step_val),
        )
        self.first_step_pending = torch.where(
            paid, torch.zeros_like(self.first_step_pending), self.first_step_pending
        )
        # Diagnostic: reaction latency actually achieved, in seconds, on the steps that paid.
        self.first_step_latency_sum = torch.where(
            paid, self.first_step_lift_time, torch.zeros_like(self.first_step_lift_time)
        )
        self.first_step_latency_count = paid.float()

        # Joint-limit proximity. Costs nothing through the inner joint_limit_margin
        # fraction of each joint's soft range and rises quadratically beyond it, so it
        # is silent during normal motion and only pushes back near the ends.
        #
        # Deliberately NOT dof_pos_l2: that measures distance from the DEFAULT pose,
        # which is not where the limits are. The Go2 calf defaults to -1.5 in a range
        # of [-2.72, -0.84], so it sits 1.22 rad from one end and 0.66 from the other
        # - penalising deviation from default charges just as much for moving toward
        # the middle of the range as toward a limit, and taxes the swing excursion
        # that foot clearance depends on. This term leaves the interior free.
        limit_mid = 0.5 * (self.joint_limit_lower + self.joint_limit_upper)
        limit_half = 0.5 * (self.joint_limit_upper - self.joint_limit_lower)
        limit_excess = (
            (self.joint_pos - limit_mid).abs()
            - self.cfg.joint_limit_margin * limit_half
        ).clamp(min=0.0)
        self.joint_limit_val = torch.sum(torch.square(limit_excess), dim=1)

        # GRF computations: two separate penalties
        feet_forces_z = self.net_contact_forces[:, self._feet_ids, 2].abs()  # (N, 4)
        contact_float = contact.float()
        n_contact = contact_float.sum(dim=1).clamp(min=1.0)

        # 1) GRF balance: squared deviation of each foot's vertical load from the even
        # share, over ALL FOUR feet, with no contact mask.
        #
        # The target is the MEASURED mean, sum(F)/4, not the cached mg/4. In static
        # equilibrium the two are identical -- the feet carry exactly the robot's weight,
        # so sum(F) == mg and the term is unchanged -- but the measured mean keeps the
        # penalty purely about EVENNESS and never charges for a total-load mismatch the
        # policy is not responsible for: a push, a partial-support transient, or a payload
        # the cached weight was sampled before. It also drops the term's dependence on
        # robot_total_weight being right for the target, which matters across the payload
        # randomisation and the three robot models.
        #
        # The contact mask is left off deliberately. This used to be a CV^2 taken over
        # contacting feet only, and that made unloading a foot the cheapest way to satisfy
        # it: push two feet under the 1 N contact threshold and they drop out of both the
        # mean and the variance, so a symmetric two-legged diagonal scored a perfect 0.0
        # while a real four-foot stance scored ~0.21. The penalty ranked the failure mode
        # above the behaviour it was meant to produce. Measured against Final4's zero-command
        # stance (FL 35.6 / FR 58.0 / RL 44.6 / RR 11.1 N, one diagonal carrying 102 N of
        # 149 N): old form 0.21 for that stance vs 0.00 for a clean diagonal; this form
        # 0.053 vs 0.25, ranked the right way round. Dividing by a fixed 4 rather than by
        # the contact count is what keeps that true -- an unloaded foot still contributes
        # its full (0 - mean)^2 to the sum, so a two-legged stance is charged 0.25 here.
        #
        # Normalised by mg^2 so it stays dimensionless and comparable across robots. Runs
        # ~4x smaller than the old CV^2 for the same stance, so a scale tuned against that
        # form needs multiplying by ~4 (the -2.5 that was queued for phase2 becomes -10).
        #
        # Masked to STANCE. Uneven load sharing is a fault when the robot is meant to be
        # standing, and it is what decides whether the first step out of the stance
        # succeeds. While WALKING the same quantity is not a fault at all: a trot transfers
        # weight impulsively by design, so charging for it taxes normal gait mechanics.
        mean_share = feet_forces_z.mean(dim=1, keepdim=True)  # sum(F)/4, (N, 1)
        self.grf_balance_stance_val = (
            (feet_forces_z - mean_share).square().sum(dim=1)
            / self.robot_total_weight.square().clamp(min=1.0)
        ) * static_mask

        # 2) GRF target (mg/n): penalize deviation from physics-based weight share
        # Uses cached robot weight so force spikes can't inflate their own target.
        #
        # NOTE: carries the same defect the balance term above was just fixed for -- the
        # target is mg/n_contact and the sum is contact-masked, so lifting two feet halves
        # the target and a two-legged stance scores perfectly. Inert at rew_scale_grf_target
        # 0.0; give it the fixed mg/4 treatment before ever switching it on.
        target_force_per_foot = (self.robot_total_weight / n_contact).unsqueeze(1)  # mg/n
        force_deviation = ((feet_forces_z - target_force_per_foot).square() * contact_float).sum(dim=1) / n_contact
        self.grf_target_val = force_deviation / target_force_per_foot.squeeze(1).square().clamp(min=1.0)

        # Max contact force penalty: penalize per-foot forces exceeding a fraction of robot weight
        # Threshold = robot_total_weight * max_contact_force_pct (e.g., 0.75 = 75% of mg)
        #
        # Uses the PEAK force across the physics substeps of this control step, not the single
        # instantaneous sample in net_forces_w. DirectRLEnv.step() calls scene.update() inside the
        # decimation loop, so the contact sensor refreshes every sim substep (5ms) while rewards are
        # computed once per control step (20ms). A touchdown impact only lasts 1-2 substeps, so
        # reading net_forces_w alone samples a near-arbitrary phase of the contact cycle and misses
        # most spikes entirely -- which is why this term logged ~1e-5 while the MuJoCo eval showed
        # 2.3x-bodyweight slams. net_forces_w_history keeps the last `history_length` substeps
        # (index 0 = most recent); with history_length == decimation it spans the whole control step.
        # Only the peak-force term uses this: contact detection and the grf_balance_stance/grf_target terms
        # stay on the instantaneous value on purpose, since those describe steady stance-phase load
        # sharing and would be distorted by folding a landing spike into them.
        force_hist = self._contact_sensor.data.net_forces_w_history
        if force_hist is not None and force_hist.dim() == 4:
            feet_forces_z_peak = force_hist[:, :, self._feet_ids, 2].abs().amax(dim=1)  # (N, 4)
        else:
            feet_forces_z_peak = feet_forces_z
        max_force_pct = getattr(self.cfg, "max_contact_force_pct", 0.75)
        per_foot_thresh = self.robot_total_weight * max_force_pct  # (N,)
        excess = (feet_forces_z_peak - per_foot_thresh.unsqueeze(1)).clamp(min=0.0)  # (N, 4)
        # Normalize by mg² to make dimensionless (scale-invariant across robot masses)
        self.max_contact_force_val = torch.sum(excess.square(), dim=1) / self.robot_total_weight.square().clamp(min=1.0)
        # Diagnostic: peak foot force as a multiple of body weight, so the new measurement can be
        # compared directly against the MuJoCo eval's grf_peak_stance_N before tuning the scale.
        self.grf_peak_bw_val = feet_forces_z_peak.amax(dim=1) / self.robot_total_weight.clamp(min=1.0)

        # ╔══════════════════════════════════════════════════════════════════════════════╗
        # ║  PORTED unitree_rl_lab / ISAAC LAB TERMS  (arm U1, Write/Article/plan.md)     ║
        # ╚══════════════════════════════════════════════════════════════════════════════╝
        # Each of these reproduces an upstream reward function so unitree's config can be run
        # inside this env and checked against unitree's own env. They are cheap elementwise ops
        # on (N, 12) / (N, 4), so they are computed unconditionally rather than gated on their
        # scales being non-zero. Every scale defaults to 0.0, so phases predating the port are
        # numerically unaffected.

        # energy -- unitree_rl_lab mdp.energy
        self.energy_val = torch.sum(self.joint_vel.abs() * self.applied_torque.abs(), dim=1)

        # feet_slide -- isaaclab mdp.feet_slide. xy speed of each foot while that foot is loaded.
        # DIFFERENCE FROM UPSTREAM: upstream takes the max contact force over the sensor's history
        # window to decide "loaded"; `contact` here is the instantaneous sample the rest of this
        # method already uses. They differ only for a foot whose force crosses 1 N and falls back
        # inside a single 20 ms control step, which is not a stance.
        self.feet_slide_val = torch.sum(
            feet_vel_w[:, :, :2].norm(dim=-1) * contact.float(), dim=1
        )

        # air_time_variance -- unitree_rl_lab mdp.air_time_variance_penalty. Spread of the four
        # feet's completed swing and stance durations, each clipped at 0.5 s.
        self.air_time_variance_val = torch.var(
            self.last_feet_air_time.clamp(max=0.5), dim=1
        ) + torch.var(self.last_feet_contact_time.clamp(max=0.5), dim=1)

        # joint_pos_limits -- isaaclab mdp.joint_pos_limits. L1 distance outside the SOFT limits;
        # zero everywhere inside them. joint_limit_lower/upper already hold soft_joint_pos_limits.
        # Not the same term as joint_limit_val above, which is quadratic outside a margin fraction
        # of the range -- both exist so U1 can run upstream's version verbatim.
        out_of_limits = -(self.joint_pos - self.joint_limit_lower).clamp(max=0.0)
        out_of_limits = out_of_limits + (self.joint_pos - self.joint_limit_upper).clamp(min=0.0)
        self.joint_pos_limits_val = torch.sum(out_of_limits, dim=1)

        # joint_position_penalty -- unitree_rl_lab mdp.joint_position_penalty. L2 NORM (not the
        # squared sum dof_pos_l2_* uses) of the deviation from the default pose, multiplied by
        # joint_deviation_stand_still_scale when the robot is neither commanded to move nor
        # already moving.
        #
        # Note the upstream gate is cmd_norm STRICTLY > 0, not a threshold: the multiplier only
        # ever bites on the exactly-zero commands (unitree's rel_standing_envs, here
        # zero_command_fraction). A 0.05 m/s command is "moving" as far as this term is
        # concerned, so it does nothing to make a slow walk preferable to standing.
        body_speed_xy = torch.norm(self.base_lin_vel[:, :2], dim=1)
        joint_dev_norm = torch.linalg.norm(self.joint_pos - self.desired_joint_pos, dim=1)
        self.joint_deviation_val = torch.where(
            (cmd_norm > 0.0)
            | (body_speed_xy > self.cfg.joint_deviation_velocity_threshold),
            joint_dev_norm,
            self.cfg.joint_deviation_stand_still_scale * joint_dev_norm,
        )

        # feet_air_time -- isaaclab mdp.feet_air_time. Sum over feet touching down this step of
        # (completed swing duration - threshold), off below a 0.1 m/s xy command.
        #
        # This is the term the paper is about. Unitree's threshold is 0.5 s, longer than any real
        # Go2 swing, so (air_time - threshold) is NEGATIVE at every landing: the one term meant to
        # pay the robot for stepping charges it instead, and standing still avoids the charge
        # entirely. Ported verbatim -- the threshold is the finding, not a bug to fix here.
        #
        # DIFFERENCE FROM UPSTREAM: gated on `landed` rather than raw first contact, so the feet
        # that are already on the ground at episode step 1 are not billed for a swing that never
        # happened. Costs upstream one spurious charge per env per episode; this env spawns from a
        # drop, so it would fire here too.
        self.feet_air_time_thresh_val = torch.sum(
            (self.last_feet_air_time - self.cfg.feet_air_time_threshold) * landed.float(), dim=1
        ) * (command_speed_xy > 0.1).float()

        # Reset air time and swing peak height for feet in contact. Must stay AFTER the reward
        # computations above, so the final increment of the swing is credited before clearing.
        self.feet_air_time[contact] = 0.0
        self.feet_contact_time[~contact] = 0.0
        self.feet_height_max[contact] = 0.0
        self.last_feet_contact = contact

        # -- Contact-driven gait phase symmetry PENALTY --
        # Mirrors the strike-time logic used in eval_mujoco.py.
        # On each foot's touchdown we record the event time and compute the stride duration
        # (time between consecutive touchdowns). The phase of foot B relative to foot A is:
        #   phase = ((t_B - last_strike_A) % stride_A) / stride_A  → [0, 1]
        #
        # This accumulates MISMATCH (1 - cosine score), not match, so rew_scale_gait_phase_sym
        # must be NEGATIVE. As a positive reward it paid the robot for *having* a gait, which
        # is an incentive to step more than the task needs; as a penalty it costs nothing to
        # stand still and only charges for stepping in the wrong pattern.
        #
        # A pair only contributes when it is actually stepping (valid), and the whole term is
        # scaled by moving_mask like every other stepping term -- it used to gate on a bare
        # boolean (cmd_norm > static_velocity_threshold), which put a cliff at the threshold
        # while feet_air_time ramped smoothly over [threshold, static_command_ramp]. (foot_height
        # deliberately keeps the hard gate -- see its comment above.)
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
                    # Plausible gait period only. The upper bound matters: without it, a robot
                    # that steps, pauses, then steps again records the PAUSE as its stride, and
                    # every later phase is normalised against that -- which produced a ~0.96/1.0
                    # spurious mismatch held for 1.5x the pause length. Measured cadence in the
                    # MuJoCo sweeps is 0.5-3.2 Hz, so anything slower than MAX_STRIDE_S is a gap,
                    # not a stride, and the previous good estimate is kept instead.
                    MAX_STRIDE_S = 1.5
                    valid = (new_dur > 0.1) & (new_dur < MAX_STRIDE_S)
                    if valid.any():
                        update_mask = landing.clone()
                        update_mask[landing] = valid
                        self.stride_duration[update_mask, foot] = new_dur[valid]
                    self.last_strike_time[landing, foot] = t_now[landing]

            # Compute phase of foot B relative to foot A for each of the 6 pairs
            # phase_rel(A, B) = ((t_now - last_strike_A) % stride_A) / stride_A  → [0, 1]
            # Coarse pre-filter only: the smooth moving_mask below is what actually shapes the
            # term, so this threshold no longer creates a discontinuity at its own boundary.
            moving = cmd_norm > self.cfg.static_velocity_threshold  # (N,) bool

            def _phase_rel(ref_foot, other_foot):
                """Phase of other_foot relative to ref_foot, in [0,1], plus a validity mask."""
                dur = self.stride_duration[:, ref_foot]          # (N,)
                
                # Both feet must have struck recently. Checking only the reference foot let a
                # pair count as valid while the OTHER foot's strike time was seconds stale, so
                # the "relative phase" was measured against an event from a previous gait.
                time_since_ref   = t_now - self.last_strike_time[:, ref_foot]
                time_since_other = t_now - self.last_strike_time[:, other_foot]
                window = dur * 1.5
                active_stepping = (time_since_ref < window) & (time_since_other < window)

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

            total_mismatch = torch.zeros(self.num_envs, device=self.device)
            total_weight = torch.zeros(self.num_envs, device=self.device)
            for ref, other, target in pairs:
                phase, valid = _phase_rel(ref, other)
                score = _cosine_score(phase, target)          # 1 = on target, 0 = half a cycle off
                total_mismatch += (1.0 - score) * valid.float()
                total_weight += valid.float()

            # Mean mismatch over the pairs that are actually stepping. Zero when no pair is
            # valid -- a robot that is not stepping has no gait to be wrong about, so standing
            # still must cost nothing here. Then ramped by moving_mask, the same smooth
            # static/moving ramp every other stepping reward uses, so the term fades out toward
            # zero command instead of switching off at a threshold. Without that ramp the term
            # kept a hard cliff at static_velocity_threshold and stayed fully active right down
            # to a near-zero command, where it dwarfed the static penalties: the robot marched
            # in place and yawed away under a stand-still command.
            self.gait_phase_sym_val = (
                total_mismatch / total_weight.clamp(min=1.0)
                * (total_weight > 0).float()
                * moving_mask
            )
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

    def _get_observations(self) -> dict:
        """
        Collects data from the simulation to feed into the neural network.
        """
        # Re-refresh: _reset_idx() ran between _get_rewards() and here, so reset envs have been
        # teleported to their spawn state since _compute_reward_terms() last looked.
        self._refresh_state()

        # Observations, raw (unscaled) -- see OBS_LAYOUT_TERMS for the three layouts. The
        # commands slice is what differs between "full" (4 wide, the 4th being the unused
        # heading slot) and the two unitree layouts (3 wide).
        jpos_rel = self.joint_pos - self.desired_joint_pos
        layout = self.cfg.obs_layout
        if layout == "unitree":
            parts = (
                self.base_ang_vel,
                self.projected_gravity,
                self.commands[:, :3],
                jpos_rel,
                self.joint_vel,
                self.actions,
            )
        elif layout == "unitree_vel":
            parts = (
                self.base_lin_vel,
                self.base_ang_vel,
                self.projected_gravity,
                self.commands[:, :3],
                jpos_rel,
                self.joint_vel,
                self.actions,
            )
        else:
            parts = (
                self.base_lin_vel,
                self.base_ang_vel,
                self.projected_gravity,
                self.commands,
                jpos_rel,
                self.joint_vel,
                self.actions,
            )
        obs = torch.cat(parts, dim=-1)

        # Add observation noise (Sim2Real)
        # Per-channel white noise + per-episode constant bias when a sensor model is
        # configured; otherwise the original flat sigma over every dim.
        #
        # NOISE FIRST, SCALE SECOND. That is the order Isaac Lab's ObservationManager applies
        # (func -> noise -> clip -> scale), so a sigma in the yaml always means the error on the
        # raw physical quantity: +-1.5 rad/s of joint-velocity noise stays +-1.5 rad/s whether or
        # not the layout then multiplies the channel by 0.05.
        if self.obs_noise_std is not None:
            if self.cfg.observation_noise_uniform:
                # Uniform in [-w, +w], matching isaaclab AdditiveUniformNoiseCfg -- the yaml
                # numbers are half-widths, not standard deviations.
                draw = torch.rand_like(obs) * 2.0 - 1.0
            else:
                draw = torch.randn_like(obs)
            obs = obs + draw * self.obs_noise_std + self.obs_bias
        else:
            obs = obs + torch.randn_like(obs) * self.cfg.observation_noise_scale

        # Clip BEFORE scale, the order Isaac Lab's ObservationManager uses
        # (func -> noise -> clip -> scale), so the bound is stated in raw physical units.
        if self.cfg.obs_clip is not None:
            obs = obs.clamp(-float(self.cfg.obs_clip), float(self.cfg.obs_clip))

        if self.obs_scale is not None:
            obs = obs * self.obs_scale

        if self.cfg.obs_history_len > 0:
            width = self.cfg.obs_dim_single
            full_obs = torch.cat([obs, self.obs_history_buf], dim=-1)
            self.obs_history_buf = torch.cat([obs, self.obs_history_buf[:, :-width]], dim=-1)
            obs = full_obs

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        Computes the reward (score) for the current step.
        The goal is to teach the robot to stand up and retain balance.
        """
        # Pull all sim state fresh, then derive every per-step reward quantity from it, so the
        # whole reward vector is evaluated on the same (current) timestep -- see _refresh_state's
        # docstring for why neither can rely on _get_observations having run.
        self._refresh_state()
        self._compute_reward_terms()

        # Calculate undesired contacts penalty
        # If any thigh/calf/trunk sensor registers > 1.0 N force, it's a contact
        if self.cfg.undesired_contacts_count:
            # isaaclab mdp.undesired_contacts: the NUMBER of monitored bodies over the threshold,
            # measured on the peak force across the control step, not a single 0/1 flag. Ranges
            # 0..len(monitored bodies), so a weight tuned against the flag form is ~N times too
            # strong here.
            hist = self._contact_sensor.data.net_forces_w_history
            if hist is not None and hist.dim() == 4 and len(self._undesired_contact_body_ids) > 0:
                peak = hist[:, :, self._undesired_contact_body_ids, :].norm(dim=-1).amax(dim=1)
            else:
                peak = torch.norm(self.net_undesired_contact_forces, dim=-1)
            undesired_contacts = (peak > 1.0).float().sum(dim=1)
        else:
            undesired_contacts = (torch.norm(self.net_undesired_contact_forces, dim=-1).max(dim=1)[0] > 1.0).float()

        total_reward, reward_log, track_lin_vel_per_env = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_undesired_contacts,
            self.cfg.rew_scale_track_lin_vel_xy_exp,
            self.cfg.rew_scale_track_ang_vel_z_exp,
            self.cfg.rew_scale_lin_vel_z_l2,
            self.cfg.rew_scale_ang_vel_xy_l2,
            self.cfg.rew_scale_dof_pos_l2_walk,
            self.cfg.rew_scale_dof_pos_l2_stance,
            self.cfg.rew_scale_hip_dev_l1,
            self.cfg.rew_scale_dof_torques_l2,
            self.cfg.rew_scale_dof_acc_l2,
            self.cfg.rew_scale_action_rate_l2,
            self.cfg.rew_scale_feet_air_time,
            self.cfg.rew_scale_flat_orientation_l2,
            self.cfg.rew_scale_foot_height,
            self.cfg.rew_scale_foot_landing_vel,
            self.cfg.rew_scale_feet_air_penalty,
            self.cfg.rew_scale_feet_air_penalty_static,
            self.cfg.rew_scale_joint_vel_l2,
            self.cfg.rew_scale_base_height_l2,
            self.cfg.rew_scale_grf_balance_stance,
            self.cfg.rew_scale_joint_limits,
            self.cfg.rew_scale_first_step,
            self.cfg.rew_scale_grf_target,
            self.cfg.rew_scale_max_contact_force,
            self.cfg.rew_scale_base_acc_l2,
            self.cfg.rew_scale_pos_deviation_l1,
            self.cfg.rew_scale_yaw_deviation_l1,
            self.cfg.rew_scale_gait_phase_sym,
            self.cfg.rew_scale_energy,
            self.cfg.rew_scale_feet_slide,
            self.cfg.rew_scale_air_time_variance,
            self.cfg.rew_scale_joint_pos_limits,
            self.cfg.rew_scale_joint_deviation,
            self.cfg.rew_scale_feet_air_time_thresh,
            self.cfg.target_base_height,
            self.cfg.static_velocity_threshold,
            self.cfg.command_lin_vel_std,
            self.cfg.command_ang_vel_std,
            self.cfg.vel_tracking_sigma_exp,
            self.commands,
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity,
            self.joint_vel,
            self.last_joint_vel,
            self.last_base_lin_vel,
            self.applied_torque,
            self.actions,
            self.previous_actions,
            self.feet_air_time_reward_val,
            self.foot_height_val,
            self.foot_landing_vel_val,
            self.feet_air_penalty_val,
            self.feet_air_penalty_static_val,
            self.joint_vel_l2_val,
            self.dof_pos_l2_walk_val,
            self.dof_pos_l2_stance_val,
            self.hip_dev_l1_val,
            self.grf_balance_stance_val,
            self.joint_limit_val,
            self.first_step_val,
            self.grf_target_val,
            self.max_contact_force_val,
            self.pos_deviation_val,
            self.yaw_deviation_val,
            self.gait_phase_sym_val,
            self.energy_val,
            self.feet_slide_val,
            self.air_time_variance_val,
            self.joint_pos_limits_val,
            self.joint_deviation_val,
            self.feet_air_time_thresh_val,
            self.root_pos_w[:, 2] - self.scene.env_origins[:, 2],
            undesired_contacts,
            self.reset_terminated,
            self.step_dt,
        )
        # Episodic sum feeding the command level curriculum (see _reset_idx).
        self.track_lin_vel_episode_sum += track_lin_vel_per_env
        self.extras.setdefault("log", {})
        self.extras["log"].update(reward_log)
        if self.cfg.command_level_curriculum:
            # The curriculum's own state, so a run can be read back off TensorBoard: how wide the
            # sampled command box currently is. Flat at the limit means the curriculum has
            # finished widening and the slow band is no longer being covered.
            self.extras["log"]["curriculum/command_x_max"] = torch.tensor(
                self.cmd_x_range[1], device=self.device
            )
            self.extras["log"]["curriculum/command_y_max"] = torch.tensor(
                self.cmd_y_range[1], device=self.device
            )
        # Peak per-foot contact force in body weights, for comparing the substep-peak measurement
        # against the MuJoCo eval's grf_peak_stance_N before retuning rew_scale_max_contact_force.
        self.extras["log"]["diag/grf_peak_bw_mean"] = self.grf_peak_bw_val.mean()
        self.extras["log"]["diag/grf_peak_bw_max"] = self.grf_peak_bw_val.max()
        # Mean vertical foot speed at touchdown, over every foot that landed anywhere in the batch
        # this step. Reads 0 on the rare step where nothing landed, so read it as a running average
        # in TensorBoard, not step by step.
        n_first_steps = self.first_step_latency_count.sum()
        self.extras["log"]["diag/first_step_latency_s"] = (
            self.first_step_latency_sum.sum() / n_first_steps.clamp(min=1.0)
        )
        n_landings = self.foot_landing_count.sum()
        self.extras["log"]["diag/foot_landing_speed_mps"] = (
            self.foot_landing_speed_sum.sum() / n_landings.clamp(min=1.0)
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Determines if the episode is over.
        1. Died: Base hit the ground.
        2. Timeout: Episode duration exceeded limit.
        """
        # _get_dones runs before _get_rewards, and both read projected_gravity/root_pos_w. Refresh
        # here too, otherwise terminations would be judged on the previous step's pose while the
        # rewards (including rew_alive, which is driven by reset_terminated) are judged on the
        # current one. _refresh_state is a pure read from sim buffers, so the repeat call in
        # _get_rewards is idempotent -- keeping it there leaves that method self-contained rather
        # than silently depending on _get_dones having run first.
        self._refresh_state()

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Check if base is too tilted (not upright)
        upright_check = (
            self.projected_gravity[:, 2] > -self.cfg.base_angle_termination_thresh
        )

        # Fall detection: if the robot's body is lower than 15cm, it likely fell.
        base_height = self.root_pos_w[:, 2] - self.scene.env_origins[:, 2]

        # Suppress termination during the standby/landing phase so the robot is not
        # killed mid-bounce when it drops from spawn height.
        past_standby = (
            self.episode_length_buf * self.step_dt
        ) > self.standby_duration

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

            # The per-view reset above bypasses super()._reset_idx(), so the base-class buffers
            # have to be cleared by hand here. The custom buffers below are cleared for BOTH
            # branches further down -- they are not specific to the multi-robot path.
            self.episode_length_buf[env_ids] = 0
            self.reset_buf[env_ids] = 0
        else:
            super()._reset_idx(env_ids)
            # Standard Mass/Friction/State randomization
            self._randomize_view_state(env_ids, self.robot)

        # Custom per-episode buffers. These used to live inside the heterogeneous branch only,
        # which meant that in homogeneous mode (phase1/phase2, robot_cfg: GO2) none of them were
        # ever cleared -- super()._reset_idx() only knows about episode_length_buf. Every new
        # episode therefore started with: last_joint_vel/last_base_lin_vel from the *crashed*
        # robot (a spurious dof_acc_l2/base_acc_l2 spike on step 1), previous_actions from the old
        # episode (spurious action_rate_l2), feet_air_time still accumulating for a foot that was
        # airborne at termination, and obs_history_buf feeding the policy 10 frames of the
        # previous, falling episode. Phase 1 is exactly where the base gait is learned.
        # ── Command-range level curriculum (unitree_rl_lab mdp.lin_vel_cmd_levels) ──────
        # Widen both ends of command_x_range / command_y_range by command_level_delta toward
        # command_level_limit_* whenever the mean PER-SECOND episodic track_lin_vel_xy reward of
        # the envs resetting on this step clears command_level_reward_frac of that term's weight.
        # The common_step_counter gate is upstream's: it fires at most once per episode length,
        # i.e. on the step the timed-out cohort comes back.
        #
        # It only ever WIDENS, which is the point for the paper. Unitree starts at +-0.1, where
        # essentially every command is slow, and finishes at +-1.0 -- the uniform cube, where
        # ~1.4% of draws fall below 0.3 m/s. The slow band is covered only while the policy is
        # too poor to walk, and abandoned as soon as it can.
        if self.cfg.command_level_curriculum:
            if self.common_step_counter % self.max_episode_length == 0:
                mean_track = (
                    self.track_lin_vel_episode_sum[env_ids].mean() / self.max_episode_length_s
                )
                threshold = (
                    self.cfg.rew_scale_track_lin_vel_xy_exp * self.cfg.command_level_reward_frac
                )
                if float(mean_track) > threshold:
                    d = float(self.cfg.command_level_delta)
                    lo_x, hi_x = self.cfg.command_level_limit_x
                    lo_y, hi_y = self.cfg.command_level_limit_y
                    self.cmd_x_range = [
                        max(self.cmd_x_range[0] - d, float(lo_x)),
                        min(self.cmd_x_range[1] + d, float(hi_x)),
                    ]
                    self.cmd_y_range = [
                        max(self.cmd_y_range[0] - d, float(lo_y)),
                        min(self.cmd_y_range[1] + d, float(hi_y)),
                    ]
            self.track_lin_vel_episode_sum[env_ids] = 0.0

        self.feet_air_time[env_ids] = 0.0
        self.feet_contact_time[env_ids] = 0.0
        self.last_feet_air_time[env_ids] = 0.0
        self.last_feet_contact_time[env_ids] = 0.0
        self.feet_height_max[env_ids] = 0.0
        self.last_feet_contact[env_ids] = False
        self.last_feet_vel_z[env_ids] = 0.0
        self.last_joint_vel[env_ids] = 0.0
        self.last_base_lin_vel[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.last_strike_time[env_ids] = 0.0
        # Clear first-step state so a new episode's opening command counts as a fresh
        # transition rather than inheriting the previous episode's armed/elapsed state.
        self.first_step_pending[env_ids] = 0.0
        self.first_step_elapsed[env_ids] = 0.0
        self.first_step_lift_time[env_ids] = 0.0
        self.first_step_lifted[env_ids] = 0.0
        self.first_step_val[env_ids] = 0.0
        self.was_moving[env_ids] = 0.0
        self.static_elapsed[env_ids] = 0.0
        lo, hi = self.cfg.standby_duration_range_s
        if hi > lo:
            self.standby_duration[env_ids] = (
                torch.rand(len(env_ids), device=self.device) * (hi - lo) + lo
            )
        else:
            self.standby_duration[env_ids] = lo
        self.stride_duration[env_ids] = 1.0
        # Per-episode sensor bias: resampled here so it is constant WITHIN an episode
        # (that is the whole point -- see _build_obs_sensor_model) and independent across them.
        if self.obs_bias_bound is not None:
            self.obs_bias[env_ids] = (
                (torch.rand((len(env_ids), self.cfg.obs_dim_single), device=self.device) * 2.0 - 1.0)
                * self.obs_bias_bound
            )
        self.gait_phase_sym_val[env_ids] = 0.0
        if self.cfg.obs_history_len > 0:
            self.obs_history_buf[env_ids] = 0.0

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
        _com_x_rng = self.cfg.com_displacement_range
        _com_y_rng = self.cfg.com_displacement_range_y
        if any(v != 0.0 for v in _com_x_rng + _com_y_rng):
            coms = view.root_physx_view.get_coms().clone()
            if not hasattr(view, "default_coms"):
                view.default_coms = coms.clone()

            com_noise_x = sample_uniform(
                _com_x_rng[0],
                _com_x_rng[1],
                (len(env_ids_cpu), 1),
                "cpu",
            )
            # y gets its own range on purpose. It used to reuse the x range, so a fore/aft-skewed
            # setting like [-0.05, 0.1] also put the CoM an average 2.5cm off to one side in every
            # env, and the optimal response to that is a permanently lopsided stance.
            com_noise_y = sample_uniform(
                _com_y_rng[0],
                _com_y_rng[1],
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

        # 0.2 Randomize PD gains (Kp = stiffness, Kd = damping) around their configured defaults.
        #
        # This has to go through the actuator MODEL, not write_joint_stiffness_to_sim(). Isaac Lab
        # notes on that function: "This function isn't setting the values for actuator models"
        # (articulation.py) -- it only writes the PhysX drive gains. And for EXPLICIT actuators
        # (DCMotorCfg on A1/GO2, ActuatorNetMLP on Go1 -- every robot here) Isaac Lab deliberately
        # zeroes the PhysX drive at startup, because the model computes torque in Python and applies
        # it as an effort target. So the old code did not randomize the intended gains at all: it
        # re-enabled a PhysX PD drive that is supposed to stay off, stacked on top of the actuator's
        # torque. Worse, it wrote the sampled value ABSOLUTELY instead of adding it to the default
        # (unlike the joint-friction randomization above, which correctly does base + noise), so with
        # joint_stiffness_range [-5, 5] roughly half the envs got a NEGATIVE stiffness -- a term that
        # pushes away from the target and injects energy. That switched on at phase5 and cost 32-44%
        # of total reward at step 100k in every chained run measured.
        #
        # Inert for ActuatorNetMLP (Go1): that network maps position/velocity error to torque
        # directly and never reads stiffness/damping, so there is no PD gain to randomize on it.
        kp_range = self.cfg.joint_stiffness_range
        kd_range = self.cfg.joint_pd_damping_range
        if kp_range[0] != 0.0 or kp_range[1] != 0.0 or kd_range[0] != 0.0 or kd_range[1] != 0.0:
            # Cache the model's configured gains once, so repeated resets perturb around the
            # original value instead of compounding on the previous episode's random draw.
            # Mirrors the view.default_coms pattern used by the COM randomization above.
            if not hasattr(view, "default_actuator_gains"):
                view.default_actuator_gains = {
                    name: (act.stiffness.clone(), act.damping.clone())
                    for name, act in view.actuators.items()
                }

            for name, actuator in view.actuators.items():
                base_kp, base_kd = view.default_actuator_gains[name]
                n_act_joints = base_kp.shape[1]

                if kp_range[0] != 0.0 or kp_range[1] != 0.0:
                    kp_noise = sample_uniform(
                        kp_range[0], kp_range[1], (len(ids), n_act_joints), self.device
                    )
                    actuator.stiffness[ids] = torch.clamp(base_kp[ids] + kp_noise, min=0.0)
                if kd_range[0] != 0.0 or kd_range[1] != 0.0:
                    kd_noise = sample_uniform(
                        kd_range[0], kd_range[1], (len(ids), n_act_joints), self.device
                    )
                    actuator.damping[ids] = torch.clamp(base_kd[ids] + kd_noise, min=0.0)

                # Implicit actuators let PhysX run the PD, so they additionally need the new gains
                # pushed into the sim. Explicit ones must NOT -- see the note above.
                if actuator.is_implicit_model:
                    view.write_joint_stiffness_to_sim(
                        actuator.stiffness[ids], joint_ids=actuator.joint_indices, env_ids=ids
                    )
                    view.write_joint_damping_to_sim(
                        actuator.damping[ids], joint_ids=actuator.joint_indices, env_ids=ids
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
    rew_scale_dof_pos_l2_walk: float,
    rew_scale_dof_pos_l2_stance: float,
    rew_scale_hip_dev_l1: float,
    rew_scale_dof_torques_l2: float,
    rew_scale_dof_acc_l2: float,
    rew_scale_action_rate_l2: float,
    rew_scale_feet_air_time: float,
    rew_scale_flat_orientation_l2: float,
    rew_scale_foot_height: float,
    rew_scale_foot_landing_vel: float,
    rew_scale_feet_air_penalty: float,
    rew_scale_feet_air_penalty_static: float,
    rew_scale_joint_vel_l2: float,
    rew_scale_base_height_l2: float,
    rew_scale_grf_balance_stance: float,
    rew_scale_joint_limits: float,
    rew_scale_first_step: float,
    rew_scale_grf_target: float,
    rew_scale_max_contact_force: float,
    rew_scale_base_acc_l2: float,
    rew_scale_pos_deviation_l1: float,
    rew_scale_yaw_deviation_l1: float,
    rew_scale_gait_phase_sym: float,
    rew_scale_energy: float,
    rew_scale_feet_slide: float,
    rew_scale_air_time_variance: float,
    rew_scale_joint_pos_limits: float,
    rew_scale_joint_deviation: float,
    rew_scale_feet_air_time_thresh: float,
    target_base_height: float,
    static_velocity_threshold: float,
    command_lin_vel_std: float,
    command_ang_vel_std: float,
    vel_tracking_sigma_exp: float,
    commands: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    joint_vel: torch.Tensor,
    last_joint_vel: torch.Tensor,
    last_base_lin_vel: torch.Tensor,
    joint_torques: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    feet_air_time_reward_val: torch.Tensor,
    foot_height_val: torch.Tensor,
    foot_landing_vel_val: torch.Tensor,
    feet_air_penalty_val: torch.Tensor,
    feet_air_penalty_static_val: torch.Tensor,
    joint_vel_l2_val: torch.Tensor,
    dof_pos_l2_walk_val: torch.Tensor,
    dof_pos_l2_stance_val: torch.Tensor,
    hip_dev_l1_val: torch.Tensor,
    grf_balance_stance_val: torch.Tensor,
    joint_limit_val: torch.Tensor,
    first_step_val: torch.Tensor,
    grf_target_val: torch.Tensor,
    max_contact_force_val: torch.Tensor,
    pos_deviation_val: torch.Tensor,
    yaw_deviation_val: torch.Tensor,
    gait_phase_sym_val: torch.Tensor,
    energy_val: torch.Tensor,
    feet_slide_val: torch.Tensor,
    air_time_variance_val: torch.Tensor,
    joint_pos_limits_val: torch.Tensor,
    joint_deviation_val: torch.Tensor,
    feet_air_time_thresh_val: torch.Tensor,
    base_height_val: torch.Tensor,
    undesired_contacts: torch.Tensor,
    reset_terminated: torch.Tensor,
    step_dt: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
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

    # 10. DOF Position L2 Penalty, split by static/moving -- see dof_pos_l2_walk_val /
    # dof_pos_l2_stance_val comment in _get_observations for why.
    rew_dof_pos_l2_walk = rew_scale_dof_pos_l2_walk * dof_pos_l2_walk_val
    rew_dof_pos_l2_stance = rew_scale_dof_pos_l2_stance * dof_pos_l2_stance_val
    rew_hip_dev_l1 = rew_scale_hip_dev_l1 * hip_dev_l1_val

    # 11. Flat Orientation Penalty (Penalize Pitch/Roll)
    rew_flat_orientation_l2 = rew_scale_flat_orientation_l2 * torch.sum(
        torch.square(projected_gravity[:, :2]), dim=1
    )

    # 12. Foot Height, swing-apex scored at touchdown. Single term carrying both directions:
    # foot_height_val is (match - bias) per landing, so with a POSITIVE scale a landing at
    # target_foot_height earns and one away from it pays. See _compute_reward_terms.
    rew_foot_height = rew_scale_foot_height * foot_height_val

    # 12b. Landing Impact Penalty (touchdown vertical-speed mismatch -- scale must be NEGATIVE)
    rew_foot_landing_vel = rew_scale_foot_landing_vel * foot_landing_vel_val

    # 13. Base Height Penalty
    rew_base_height_l2 = rew_scale_base_height_l2 * torch.square(base_height_val - target_base_height)

    # 14. Integrated Position Deviation L1 Penalty
    rew_pos_deviation = rew_scale_pos_deviation_l1 * pos_deviation_val
    rew_yaw_deviation = rew_scale_yaw_deviation_l1 * yaw_deviation_val

    # 15. Gait Phase Symmetry (cosine-based, all 6 leg pairs, configured offsets)
    rew_gait_phase_sym = rew_scale_gait_phase_sym * gait_phase_sym_val

    rew_feet_air_penalty = rew_scale_feet_air_penalty * feet_air_penalty_val
    rew_feet_air_penalty_static = rew_scale_feet_air_penalty_static * feet_air_penalty_static_val
    rew_joint_vel_l2 = rew_scale_joint_vel_l2 * joint_vel_l2_val
    rew_grf_balance_stance = rew_scale_grf_balance_stance * grf_balance_stance_val
    # Joint-limit proximity (scale must be NEGATIVE); zero inside the allowed band.
    rew_joint_limits = rew_scale_joint_limits * joint_limit_val
    # First-step promptness (scale POSITIVE); non-zero only on the step that completes
    # the first swing after a standing start.
    rew_first_step = rew_scale_first_step * first_step_val
    rew_grf_target = rew_scale_grf_target * grf_target_val
    rew_max_contact_force = rew_scale_max_contact_force * max_contact_force_val

    # Ported unitree_rl_lab / Isaac Lab terms (arm U1). All the maths is in
    # _compute_reward_terms; here they are only weighted. Every scale defaults to 0.0.
    rew_energy = rew_scale_energy * energy_val
    rew_feet_slide = rew_scale_feet_slide * feet_slide_val
    rew_air_time_variance = rew_scale_air_time_variance * air_time_variance_val
    rew_joint_pos_limits = rew_scale_joint_pos_limits * joint_pos_limits_val
    rew_joint_deviation = rew_scale_joint_deviation * joint_deviation_val
    rew_feet_air_time_thresh = rew_scale_feet_air_time_thresh * feet_air_time_thresh_val

    # Per-term breakdown, mean across all parallel envs -- shows up in TensorBoard under
    # "Info / <key>" (skrl's agent config has environment_info: log wired up already).
    log["reward/alive"] = rew_alive.mean()
    log["reward/undesired_contacts"] = rew_undesired_contacts.mean()
    log["reward/track_lin_vel_xy_exp"] = rew_track_lin_vel_xy_exp.mean()
    log["reward/track_ang_vel_z_exp"] = rew_track_ang_vel_z_exp.mean()
    log["reward/lin_vel_z_l2"] = rew_lin_vel_z_l2.mean()
    log["reward/ang_vel_xy_l2"] = rew_ang_vel_xy_l2.mean()
    log["reward/dof_torques_l2"] = rew_dof_torques_l2.mean()
    log["reward/dof_pos_l2_walk"] = rew_dof_pos_l2_walk.mean()
    log["reward/dof_pos_l2_stance"] = rew_dof_pos_l2_stance.mean()
    log["reward/hip_dev_l1"] = rew_hip_dev_l1.mean()
    log["reward/dof_acc_l2"] = rew_dof_acc_l2.mean()
    log["reward/base_acc_l2"] = rew_base_acc_l2.mean()
    log["reward/action_rate_l2"] = rew_action_rate_l2.mean()
    log["reward/feet_air_time"] = rew_feet_air_time.mean()
    log["reward/flat_orientation_l2"] = rew_flat_orientation_l2.mean()
    log["reward/foot_height"] = rew_foot_height.mean()
    log["reward/foot_landing_vel"] = rew_foot_landing_vel.mean()
    log["reward/base_height_l2"] = rew_base_height_l2.mean()
    log["reward/feet_air_penalty"] = rew_feet_air_penalty.mean()
    log["reward/feet_air_penalty_static"] = rew_feet_air_penalty_static.mean()
    log["reward/joint_vel_l2"] = rew_joint_vel_l2.mean()
    log["reward/grf_balance_stance"] = rew_grf_balance_stance.mean()
    log["reward/joint_limits"] = rew_joint_limits.mean()
    log["reward/first_step"] = rew_first_step.mean()
    log["reward/grf_target"] = rew_grf_target.mean()
    log["reward/max_contact_force"] = rew_max_contact_force.mean()
    log["reward/pos_deviation"] = rew_pos_deviation.mean()
    log["reward/yaw_deviation"] = rew_yaw_deviation.mean()
    log["reward/gait_phase_sym"] = rew_gait_phase_sym.mean()
    log["reward/energy"] = rew_energy.mean()
    log["reward/feet_slide"] = rew_feet_slide.mean()
    log["reward/air_time_variance"] = rew_air_time_variance.mean()
    log["reward/joint_pos_limits"] = rew_joint_pos_limits.mean()
    log["reward/joint_deviation"] = rew_joint_deviation.mean()
    log["reward/feet_air_time_thresh"] = rew_feet_air_time_thresh.mean()
    # Raw swing duration actually achieved, over the landings in this batch. The whole
    # feet_air_time argument turns on this number sitting below the threshold, so log it
    # unweighted and unscaled: mean over feet that landed, 0.0 on a step where none did.
    n_landed = (feet_air_time_thresh_val != 0.0).float().sum()
    log["diag/air_time_charge_mean"] = feet_air_time_thresh_val.sum() / n_landed.clamp(min=1.0)
    # Raw (unscaled) diagnostics -- useful to sanity-check a term is actually receiving live,
    # nonzero physical data before worrying about whether its reward *scale* is well tuned.
    log["diag/joint_acc_sum_sq_mean"] = torch.sum(torch.square(joint_acc), dim=1).mean()
    log["diag/joint_vel_sum_sq_mean"] = joint_vel_l2_val.mean()
    log["diag/base_acc_sum_sq_mean"] = torch.sum(torch.square(base_acc), dim=1).mean()

    total_reward = (
        rew_alive
        + rew_undesired_contacts
        + rew_track_lin_vel_xy_exp
        + rew_track_ang_vel_z_exp
        + rew_lin_vel_z_l2
        + rew_ang_vel_xy_l2
        + rew_dof_torques_l2
        + rew_dof_pos_l2_walk
        + rew_dof_pos_l2_stance
        + rew_hip_dev_l1
        + rew_dof_acc_l2
        + rew_base_acc_l2
        + rew_action_rate_l2
        + rew_feet_air_time
        + rew_flat_orientation_l2
        + rew_foot_height
        + rew_foot_landing_vel
        + rew_base_height_l2
        + rew_feet_air_penalty
        + rew_feet_air_penalty_static
        + rew_joint_vel_l2
        + rew_grf_balance_stance
        + rew_joint_limits
        + rew_first_step
        + rew_grf_target
        + rew_max_contact_force
        + rew_pos_deviation
        + rew_yaw_deviation
        + rew_gait_phase_sym
        + rew_energy
        + rew_feet_slide
        + rew_air_time_variance
        + rew_joint_pos_limits
        + rew_joint_deviation
        + rew_feet_air_time_thresh
    )
    # Third return is the PER-ENV linear-velocity tracking reward, which the command level
    # curriculum accumulates over an episode and thresholds on. The log only carries its mean.
    return total_reward, log, rew_track_lin_vel_xy_exp
