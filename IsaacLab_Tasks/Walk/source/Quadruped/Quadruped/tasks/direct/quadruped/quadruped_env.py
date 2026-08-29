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
            
        self._undesired_contact_body_ids, names = self._contact_sensor.find_bodies(".*_thigh|.*_calf|trunk")
        # Positions WITHIN net_undesired_contact_forces (not sensor body ids) of the thigh and calf
        # bodies, so the paper reward set can charge them at its two different weights (-1.0 and
        # -0.2) instead of collapsing them into one undesired-contacts flag. find_bodies returns
        # ids and names in matching order, so indexing by enumerate position is the right mapping.
        self._paper_thigh_local_ids = torch.tensor(
            [k for k, n in enumerate(names) if "thigh" in n], dtype=torch.long, device=self.device
        )
        self._paper_calf_local_ids = torch.tensor(
            [k for k, n in enumerate(names) if "calf" in n], dtype=torch.long, device=self.device
        )

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
        # Mirror of feet_air_time for the STANCE side: seconds this foot has been continuously in
        # contact, reset on liftoff. Only the paper reward set (Dowdy & Chagas Vaz, SII 2026) uses
        # it -- its gait term scores air time and contact time against each other pairwise.
        self.feet_contact_time = torch.zeros(self.num_envs, 4, device=self.device)
        # Duration of each foot's last COMPLETED swing, latched at touchdown and held through
        # the following stance. feet_air_time is 0 for a planted foot, so it cannot answer
        # "were the swings real" at an arbitrary instant; this can. Used only by the paper
        # gait term's duration gate -- see _compute_paper_rewards.
        self.feet_last_air_time = torch.zeros(self.num_envs, 4, device=self.device)
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
        self.foot_height_penalty_val = torch.zeros(self.num_envs, device=self.device)
        # Same apex measurement as the penalty, opposite direction: the Gaussian MATCH
        # (rew_scale_foot_height_reward, POSITIVE) instead of the mismatch. Both are filled
        # every step; which one actually does anything is decided by the phase's scales.
        self.foot_height_reward_val = torch.zeros(self.num_envs, device=self.device)
        # Grounded feet in excess of feet_grounded_allowed, while a move command is active.
        # The one gait term that is NONZERO when the robot is standing still -- see
        # _compute_reward_terms.
        self.feet_grounded_val = torch.zeros(self.num_envs, device=self.device)
        self.foot_landing_vel_val = torch.zeros(self.num_envs, device=self.device)
        self.feet_air_penalty_val = torch.zeros(self.num_envs, device=self.device)
        self.feet_air_penalty_static_val = torch.zeros(self.num_envs, device=self.device)
        self.joint_vel_l2_static_val = torch.zeros(self.num_envs, device=self.device)
        self.dof_pos_l2_walk_val = torch.zeros(self.num_envs, device=self.device)
        self.dof_pos_l2_stance_val = torch.zeros(self.num_envs, device=self.device)
        self.grf_balance_val = torch.zeros(self.num_envs, device=self.device)
        self.grf_target_val = torch.zeros(self.num_envs, device=self.device)
        self.max_contact_force_val = torch.zeros(self.num_envs, device=self.device)
        self.grf_peak_bw_val = torch.zeros(self.num_envs, device=self.device)
        # Diagnostics for the landing-impact penalty: |vz| summed over the feet that touched down
        # this step, and how many did. Kept as sum+count so the log can report a true mean per
        # LANDING EVENT rather than per env.
        self.foot_landing_speed_sum = torch.zeros(self.num_envs, device=self.device)
        self.foot_landing_count = torch.zeros(self.num_envs, device=self.device)
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

    def _compute_paper_rewards(self) -> tuple[torch.Tensor, dict]:
        """Reward set transcribed from Dowdy & Chagas Vaz, "Towards Torque-Driven Reinforcement
        Learning for Quadruped Locomotion", IEEE/SICE SII 2026, Table III.

        Kept entirely separate from compute_rewards() rather than folded into it, because the
        paper's terms are NOT the repo's terms with different weights -- most use an L1 norm where
        this repo squares, so transplanting the weights alone would mean something different. A
        phase that turns this set on should zero the repo equivalents (see phase_paper_torque),
        otherwise both charge for the same physics.

        Every scale defaults to 0.0, so a phase that does not mention these keys is unaffected and
        this method returns exactly zero.

        WHY THIS SET MIGHT SUCCEED WHERE THE REPO'S DID NOT -- and what that claim is NOT:
        The Gait term is DENSE (paid every step, not once per landing) and worth 10.0, so a clean
        trot earns +10/step against an alive weight of 0.75. Measured on the reconstruction here:
        standing scores ~0.0003 at 0.1 s of stance and underflows to 0 beyond that, a clean trot
        scores exactly 1.0, and a three-down-one-up shuffle scores 0. Every gait term in
        compute_rewards() tops out near 0.15 and is charged per event, so the prize for finding a
        gait was ~1% of the per-step reward; here it is 13x the alive term.
        This is a payoff difference, NOT a gradient difference. The product underflows to zero for
        a standing robot just as the repo's terms do, so nothing here gently guides the first foot
        off the ground either -- exploration still has to stumble into a near-trot before the term
        pays anything. If that never happens, this set will plateau standing exactly like the last
        five runs did, and the fix is in exploration (entropy_loss_scale, see below), not rewards.

        DEVIATIONS FROM THE PAPER, all forced by what Table III leaves undefined:
          - Gait: the table gives only "prod(l_sync) * prod(l_async)" without defining l. The form
            here is the air-time/contact-time pairing from Chen et al. 2022 (the paper's ref [7],
            cited for torque control), with sync pairs = the trot diagonals. gait_sync_sigma is not
            in the paper at all and is exposed as a tunable.
          - Air Time Variance: the table's expression sums SIGNED differences over all ordered
            pairs, which is identically zero for any input. Read as absolute differences.
          - Thigh/Calf contact: charged per body in contact, from the existing thigh/calf sensor
            bodies. The paper gives C in R^4 without saying whether it counts bodies or is binary.
        """
        cfg = self.cfg
        log: dict = {}
        total = torch.zeros(self.num_envs, device=self.device)

        def add(name: str, scale: float, value: torch.Tensor):
            term = scale * value
            log[f"reward/paper/{name}"] = term.mean()
            return term

        contact_f = self.contact_bool.float()
        # -- Rewards (Table III, upper block) --
        # Alive: sign(alive) -- 1 every step the episode is still running.
        total = total + add("alive", cfg.rew_scale_paper_alive,
                            torch.ones(self.num_envs, device=self.device))

        # Feet Air Time: sum_i (t_air,i - 0.3), credited at touchdown, command-gated. Unbounded
        # and LINEAR in swing duration, unlike this repo's saturating Gaussian. Transcribed
        # unclamped because that is how Table III writes it AND how Isaac Lab's stock
        # mdp.feet_air_time computes it (the clamped variant is feet_air_time_positive_biped).
        #
        # READ THE SIGN BEFORE TRUSTING THIS TERM. Unclamped, at weight 5.0, it PENALISES every
        # swing shorter than paper_air_time_offset: 0.12 s pays -0.90, 0.20 s pays -0.50, 0.25 s
        # pays -0.25, and it only turns positive past 0.30 s. The swing times actually measured
        # from the policies that walk in this repo are cp7 0.12 s, NiceGait7 ~0.20 s, Basic4
        # ~0.25 s -- all of them negative here. So for a Go2 this term does not pay for stepping,
        # it pays for stepping SLOWLY, and until the policy can hold a 0.3 s swing it is cheaper
        # not to step at all. The paper's own results section reports exactly this artifact
        # ("a longer swing of each leg at low angular and linear velocities").
        # The paper's robot is a B1 -- far heavier, naturally slower cadence -- so 0.3 s is
        # plausible there and probably is not here. paper_air_time_offset is exposed for that
        # reason; ~0.15 would put a Go2's natural cadence on the positive side.
        # feet_last_air_time, NOT feet_air_time. _compute_reward_terms clears feet_air_time for
        # every foot in contact, and it runs before this method -- so by the time we get here a
        # foot that just landed reads 0, and this term evaluated to a flat
        # (0 - offset) = -0.15 per landing no matter how long the swing actually was. It could
        # never go positive and carried no gradient toward longer swings at all: the one term
        # that was supposed to pay for real strides was a constant per-landing tax.
        # feet_last_air_time is latched from feet_air_time at touchdown, before that clear, so
        # on the step a foot lands it holds that swing's true duration.
        total = total + add("feet_air_time", cfg.rew_scale_paper_feet_air_time,
                            torch.sum((self.feet_last_air_time - cfg.paper_air_time_offset)
                                      * self.landed_bool.float(), dim=1) * self.moving_mask_val)

        # Linear / angular velocity error: exp(-||err|| / 0.25). NOTE the norm is not squared and
        # the divisor is a fixed constant -- both differ from track_lin_vel_xy_exp in this repo,
        # which squares the error and scales the divisor with commanded speed.
        lin_err = torch.norm(self.base_lin_vel[:, :2] - self.commands[:, :2], dim=1)
        total = total + add("track_lin_vel", cfg.rew_scale_paper_track_lin_vel,
                            torch.exp(-lin_err / cfg.paper_vel_tracking_sigma))
        ang_err = torch.abs(self.base_ang_vel[:, 2] - self.commands[:, 2])
        total = total + add("track_ang_vel", cfg.rew_scale_paper_track_ang_vel,
                            torch.exp(-ang_err / cfg.paper_ang_tracking_sigma))

        # Gait: product over 2 synchronised pairs (the trot diagonals) and 4 antisynchronised
        # pairs. Foot order is (FL, FR, RL, RR), so diagonals are (FL,RR) and (FR,RL).
        #   l_sync(i,j)  = G(t_air_i  - t_air_j)  * G(t_cont_i - t_cont_j)   -- move together
        #   l_async(i,j) = G(t_air_i  - t_cont_j) * G(t_cont_i - t_air_j)    -- oppose each other
        # with G(x) = exp(-x^2 / gait_sync_sigma). The product is in [0, 1] and is 1 only for a
        # clean trot, so at the paper's weight of 10.0 this is the single largest term in the set.
        t_air, t_con = self.feet_air_time, self.feet_contact_time
        def G(x):
            return torch.exp(-torch.square(x) / cfg.paper_gait_sigma)
        gait = torch.ones(self.num_envs, device=self.device)
        for i, j in ((0, 3), (1, 2)):                       # sync: diagonals
            gait = gait * G(t_air[:, i] - t_air[:, j]) * G(t_con[:, i] - t_con[:, j])
        for i, j in ((0, 1), (2, 3), (0, 2), (1, 3)):       # async: everything else
            gait = gait * G(t_air[:, i] - t_con[:, j]) * G(t_con[:, i] - t_air[:, j])
        # DURATION GATE -- not in the paper, and necessary because the product above is
        # DEGENERATE without it. l_sync and l_async compare only DIFFERENCES of times, so any
        # state where all four feet share the same air and contact times scores 1.0 -- including
        # a foot chattering on and off the ground at 0.01 s. Measured on this reconstruction:
        #   real trot, 0.15 s swing ....... 1.0000
        #   chatter, all times 0.02 s ..... 1.0000
        #   chatter, all times 0.01 s ..... 1.0000
        # A 10 Hz per-foot vibration is worth exactly as much as a clean trot, at the largest
        # weight in the set (10.0, dense). The 2026-08-27 run found it: reward/paper/gait sat at
        # 9.24/10 while feet_air_time stayed negative, which pins the swings at ~0.01-0.02 s,
        # i.e. ~10 Hz per foot. The metrics looked like a gait; the robot was buzzing.
        #
        # The missing constraint is that the times be LARGE, not merely equal. Gate on the last
        # completed swing, latched at touchdown so it persists through stance (max/instantaneous
        # air time drops to 0 at every landing and would chop the reward at each transition).
        # One-sided: swings longer than the target are not punished here, only degenerate ones.
        # Chatter at 0.02 s scores 0.13 of the gate; a 0.15 s trot scores 1.0.
        gait_dur_gate = (
            self.feet_last_air_time.mean(dim=1) / max(cfg.paper_gait_min_swing, 1e-6)
        ).clamp(0.0, 1.0)
        gait = gait * gait_dur_gate
        total = total + add("gait", cfg.rew_scale_paper_gait, gait)
        log["diag/paper_gait_raw"] = gait.mean()
        log["diag/paper_gait_dur_gate"] = gait_dur_gate.mean()
        log["diag/paper_mean_swing_s"] = self.feet_last_air_time.mean()

        # -- Penalties (Table III, lower block). All L1 / plain norms, as written. --
        total = total + add("action_smooth", cfg.rew_scale_paper_action_smooth,
                            torch.norm(self.actions - self.previous_actions, dim=1))

        # Air Time Variance: sum over ordered pairs of |t_air,i - t_air,j|. Forces the four swings
        # to take the same length, which is what stops one leg carrying the whole gait.
        air_var = torch.sum(
            torch.abs(t_air.unsqueeze(2) - t_air.unsqueeze(1)), dim=(1, 2)
        )
        total = total + add("air_time_var", cfg.rew_scale_paper_air_time_var, air_var)

        total = total + add("base_motion", cfg.rew_scale_paper_base_motion,
                            torch.abs(self.base_ang_vel[:, 0]) + torch.abs(self.base_ang_vel[:, 1]))
        total = total + add("base_orientation", cfg.rew_scale_paper_base_orientation,
                            torch.norm(self.projected_gravity[:, :2], dim=1))

        # Foot Slippage: tangential speed of a foot that is in contact. Directly charges the
        # dragging/scuffing that a shuffling policy uses to fake velocity tracking.
        total = total + add("foot_slip", cfg.rew_scale_paper_foot_slip,
                            torch.sum(torch.norm(self.feet_vel_w[:, :, :2], dim=2) * contact_f, dim=1))

        total = total + add("dof_pos", cfg.rew_scale_paper_dof_pos,
                            torch.sum(torch.abs(self.joint_pos - self.desired_joint_pos), dim=1))
        total = total + add("dof_torque", cfg.rew_scale_paper_dof_torque,
                            torch.norm(self.applied_torque, dim=1))
        joint_acc = (self.joint_vel - self.last_joint_vel) / self.step_dt
        total = total + add("dof_acc", cfg.rew_scale_paper_dof_acc, torch.norm(joint_acc, dim=1))
        total = total + add("dof_vel", cfg.rew_scale_paper_dof_vel,
                            torch.norm(self.joint_vel, dim=1))

        # Thigh / calf contact, charged per body in contact rather than lumped into one
        # undesired-contacts flag the way compute_rewards does.
        forces = torch.norm(self.net_undesired_contact_forces, dim=-1) > 1.0
        if self._paper_thigh_local_ids.numel():
            total = total + add("thigh_contact", cfg.rew_scale_paper_thigh_contact,
                                forces[:, self._paper_thigh_local_ids].float().sum(dim=1))
        if self._paper_calf_local_ids.numel():
            total = total + add("calf_contact", cfg.rew_scale_paper_calf_contact,
                                forces[:, self._paper_calf_local_ids].float().sum(dim=1))
        return total, log

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

        # Env block. Only a subset can meaningfully change mid-process: these are re-read from
        # self.cfg every step. The rest are consumed once at construction (buffer sizes, scene,
        # spawned robots, terrain) and cannot be changed without restarting -- which is exactly why
        # launcher.py splits the curriculum across processes at the phase2->phase3 boundary.
        # This whole block used to be skipped entirely, so observation_noise_scale silently stayed
        # at the starting phase's value for the rest of the run (0.05 instead of the 0.1 that
        # phases 4-6 ask for), i.e. a sim2real hardening step that never happened.
        _RUNTIME_SETTABLE_ENV = {
            "observation_noise_scale",
            "base_angle_termination_thresh",
            "action_scale",
            "torque_scale",
            "reward_dt_ref",
            "joint_limit_barrier_stiffness",
            # Not read per step like the rest of this set -- they are consumed once, right
            # below, at the moment the phase is entered. Listed here because the loop treats
            # any env key it does not recognise as a typo and raises.
            "reset_exploration_on_entry",
            "reset_exploration_log_std",
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
            # decimation and the actuator PD gains are both baked in at construction, so the
            # control mode cannot be switched mid-run -- it needs a separate train invocation.
            "control_mode": lambda c: getattr(c, "control_mode", None),
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
            for term_name in ("push_a1", "push_quadruped", "push_go2"):
                try:
                    term_cfg = self.event_manager.get_term_cfg(term_name)
                except (ValueError, KeyError):
                    continue  # term not registered (e.g. non-heterogeneous setups)
                term_cfg.params["velocity_range"] = {
                    "x": (rng[0], rng[1]),
                    "y": (rng[0], rng[1]),
                }
                self.event_manager.set_term_cfg(term_name, term_cfg)
            print(f"[Curriculum] push velocity range -> {tuple(rng)} (enabled={enabled})")

        # Exploration reset. The env has no handle on the agent, so train.py registers a
        # callback here before training starts; without it (play.py, eval, any script that
        # builds the env without an optimizer) the request is reported and skipped rather
        # than failing, since nothing is learning in those cases anyway.
        if getattr(self.cfg, "reset_exploration_on_entry", False):
            log_std = float(getattr(self.cfg, "reset_exploration_log_std", -0.7))
            callback = getattr(self, "on_exploration_reset", None)
            if callback is None:
                print(
                    f"[Curriculum] phase '{next_phase['name']}' asks for an exploration reset, "
                    f"but no on_exploration_reset callback is registered -- skipping."
                )
            else:
                callback(next_phase["name"], log_std)

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

    def _joint_limit_barrier(
        self, torques: torch.Tensor, view, joint_ids, q: torch.Tensor
    ) -> torch.Tensor:
        """One-sided restoring torque that keeps joints off their mechanical stops.

        In position mode `torch.clamp(targets, lower, upper)` makes limit violation
        structurally impossible. Torque mode has no target to clamp, so the same guarantee
        has to be re-established in the same place -- the action path -- rather than handed
        to the reward function. Doing it here matters for the experiment as well as for the
        robot: it leaves the reward set byte-identical between the position and torque
        curricula, which is the entire point of running them against each other.

        Two parts, both active only outside the soft limits:
          1. Any commanded torque still pushing the joint further out is dropped.
          2. A spring proportional to the overshoot pulls it back, at a stiffness chosen to
             match the position mode's Kp so the boundary feels the same in both modes.

        No damping term: blocking the driving torque already removes the energy source, so
        the spring has nothing to fight. Add one here if a barrier oscillation ever shows up.
        """
        lower = view.data.soft_joint_pos_limits[0, joint_ids, 0]
        upper = view.data.soft_joint_pos_limits[0, joint_ids, 1]
        over_hi = (q - upper).clamp(min=0.0)
        over_lo = (lower - q).clamp(min=0.0)

        torques = torch.where(over_hi > 0.0, torques.clamp(max=0.0), torques)
        torques = torch.where(over_lo > 0.0, torques.clamp(min=0.0), torques)
        return torques + self.cfg.joint_limit_barrier_stiffness * (over_lo - over_hi)

    def _apply_torque_action(self, delayed_actions: torch.Tensor) -> None:
        """Drive the joints with commanded effort, bypassing the PD loop entirely.

        Deliberately absent, relative to the position path:
          - No nominal-pose offset. There is no well-defined "nominal torque" the way there is
            a nominal stance, so the action is the whole command, not a residual on one.
          - No backlash deadband. That model is expressed in position units and means nothing
            applied to an effort command.

        Joint limits are still enforced, just structurally rather than by clamping a target --
        see _joint_limit_barrier. The actuator model's torque-speed clamp then applies on top,
        so the commanded effort also stays bounded by what the motor could actually deliver at
        its current speed.
        """
        torques = delayed_actions * self.cfg.torque_scale

        if getattr(self, "is_heterogeneous", False):
            for i, view in enumerate(self.robot_views):
                indices = self.robot_view_indices[i]
                if len(indices) == 0:
                    continue
                v_idx = self._view_joint_dof_idx[i]
                view_torques = self._joint_limit_barrier(
                    torques[indices], view, v_idx, view.data.joint_pos[indices][:, v_idx]
                )
                view.set_joint_effort_target(view_torques, joint_ids=v_idx)
        else:
            torques = self._joint_limit_barrier(
                torques,
                self.robot,
                self._joint_dof_idx,
                self.robot.data.joint_pos[:, self._joint_dof_idx],
            )
            self.robot.set_joint_effort_target(torques, joint_ids=self._joint_dof_idx)

    def _apply_action(self) -> None:
        """
        Applies the neural network action to the robot joints.

        Mode "position" (default): absolute position control -- the action is a residual on
        the nominal stance, tracked by the actuator model's PD loop.

        Mode "torque": the action IS the joint effort, with no PD loop anywhere in the path.
        See the CONTROL_MODE note in quadruped_env_cfg.py for why the three changes (rate,
        zeroed actuator gains, no position clamp) only make sense together.
        """
        # Fetch delayed action
        env_indices = torch.arange(self.num_envs, device=self.device)
        delayed_actions = self.action_history[env_indices, self.env_latencies, :]

        if self.cfg.control_mode == "torque":
            self._apply_torque_action(delayed_actions)
            return

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

        # -- Foot height: two terms, same Gaussian, DIFFERENT accounting on purpose --
        #
        #   rew_scale_foot_height_reward (POSITIVE, foot_height_reward_val) -- DENSE, paid every
        #     step a foot is airborne, on that foot's INSTANTANEOUS height. The lift incentive for
        #     early phases, where nothing else gives the robot a reason to pick a foot up. Written
        #     just below the penalty; see the long note there for why it is dense and not
        #     per-landing (short version: a per-landing reward is worth ~0.5% of the per-step
        #     total and loses to standing still).
        #
        #   rew_scale_foot_height_penalty (NEGATIVE, foot_height_penalty_val) -- charged ONCE per
        #     swing at touchdown, on that swing's APEX, as the MISMATCH (1 - match). The same
        #     conversion gait_phase_sym went through, for the same reason. Later phases: it costs
        #     nothing to stand still and only charges for stepping at the wrong height, above
        #     target (jumping) and below it (scuffing) alike, which is what removes the
        #     high-stepping exploit the dense lift reward creates.
        #
        # Intended usage is one phase at a time -- lift while the gait is being found, penalty
        # once it is. They are no longer two sides of one number (dense-instantaneous vs
        # per-landing-apex), so running both at once is a genuine blend, not a constant offset.
        #
        # The penalty, 1 - exp(-(apex-target)^2/sigma), is bounded in [0, 1], so one wild apex costs
        # at most a single unit of scale and cannot swamp the rest of the reward the way an
        # unbounded squared error would.
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
        foot_height_mismatch = 1.0 - foot_height_match
        # Only feet that landed THIS step are charged (`landed`, computed above). Masked by command
        # like every other gait-shaping term, so a robot told to hold still pays nothing even if it
        # does shuffle a foot -- target_foot_height is meaningless when it is not supposed to step.
        self.foot_height_penalty_val = torch.sum(
            foot_height_mismatch * landed.float() * moving_mask.unsqueeze(1), dim=1
        )

        # -- Lift REWARD: DENSE, every step a foot is airborne (NOT per landing) --
        # The two directions deliberately use different accounting, because they have different
        # jobs. A penalty only has to be avoidable, so charging it once per landing is fine and
        # keeps it from bidding against feet_air_time over swing duration. A reward that has to
        # INDUCE a behaviour the policy does not have yet must be big enough to out-earn standing
        # still, and per-landing accounting cannot be:
        #
        #   a landing is an event, ~8 per second for a 2 Hz gait, while `alive` and the tracking
        #   terms are paid EVERY step. At 50 Hz that is 0.16 landings/step, at 200 Hz only 0.04 --
        #   so at scale 0.4 a perfect gait earned at most ~0.016/step against alive=1.0. Measured
        #   in the 2026-08-26 torque run: reward/foot_height_reward sat at 0.008-0.011, about 0.5%
        #   of the per-step total, while stepping cost strictly more torque, dof_acc and
        #   action_rate. Standing still was the better deal and the policy took it.
        #
        # Dense payment is also what cp7 -- the only configuration in this repo that produced a
        # fast gait -- actually used: up to ~0.4/step with two feet up, i.e. ~25x what the
        # per-landing version could pay. It is rate-invariant too, which per-landing is not: a
        # per-step term is exactly what reward_dt_ref's normalisation is built for, so the same
        # scale means the same thing at 50 Hz and 200 Hz.
        #
        # The cost is cp7's known exploit -- the optimum drifts above target because a taller arc
        # spends more time in the high-reward band -- and that is the whole reason phase2 switches
        # to the penalty above. Scored on INSTANTANEOUS height (not the swing apex), since the
        # point is to pay for the foot being up right now, at every step of the swing.
        airborne = (~contact).float() * moving_mask.unsqueeze(1)
        foot_height_dense_match = torch.exp(
            -torch.square(feet_heights - self.cfg.target_foot_height) / self.cfg.foot_height_sigma
        )
        self.foot_height_reward_val = torch.sum(foot_height_dense_match * airborne, dim=1)

        # -- Grounded-feet PENALTY: the term that has to break the standing local optimum --
        # Every other gait term in this file is multiplied by (~contact) or by `landed`, so for a
        # robot with all four feet planted they are ALL identically zero -- and so are their
        # gradients. feet_air_time only pays a foot that is already airborne; the lift reward only
        # pays a foot that is already up. None of them can tell the policy that picking a foot up
        # would be better, because none of them changes until it already has. That is the whole
        # reason five straight runs plateaued standing still with a valid-looking reward set.
        #
        # This term is nonzero exactly IN that state and falls monotonically as feet leave the
        # ground, so it is a real gradient out of it: charge (feet in contact - allowed), floored
        # at zero, every step, while a move command is active.
        #
        # feet_grounded_allowed = 2 is what makes it safe to point at a gait rather than at
        # jumping: a trot has two feet down at all times and pays exactly zero, so once the robot
        # is trotting this term stops applying pressure entirely. Nothing here rewards having
        # FEWER than two feet down, so it cannot drive flight phases the way a bare "reward air
        # time" would. A three-legged crawl pays 1 -- deliberate, since that is the shuffle this
        # is meant to push past.
        #
        # Gated by moving_mask like the rest of the gait shaping: standing still on all four feet
        # is the correct answer to a zero command and must stay free.
        self.feet_grounded_val = (
            (contact.float().sum(dim=1) - self.cfg.feet_grounded_allowed).clamp(min=0.0)
            * moving_mask
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
        if getattr(self, "is_heterogeneous", False):
            all_feet_vel_z = torch.zeros((self.num_envs, 4), device=self.device)
            for i, view in enumerate(self.robot_views):
                indices = self.robot_view_indices[i]
                feet_ids = self.robot_feet_ids[i]  # relative to Articulation (FL, FR, RL, RR)
                all_feet_vel_z[indices] = view.data.body_lin_vel_w[:, feet_ids, 2]
            feet_vel_z = all_feet_vel_z
        else:
            feet_vel_z = self.robot.data.body_lin_vel_w[:, self._feet_ids_articulation, 2]

        # Full world-frame foot velocity (N, 4, 3). The landing penalty above only needs z; the
        # paper set's foot-slippage term needs the tangential (xy) component of a foot that is
        # supposed to be planted.
        if getattr(self, "is_heterogeneous", False):
            feet_vel_w = torch.zeros((self.num_envs, 4, 3), device=self.device)
            for i, view in enumerate(self.robot_views):
                indices = self.robot_view_indices[i]
                feet_vel_w[indices] = view.data.body_lin_vel_w[:, self.robot_feet_ids[i], :]
        else:
            feet_vel_w = self.robot.data.body_lin_vel_w[:, self._feet_ids_articulation, :]
        self.feet_vel_w = feet_vel_w

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
        # Marching in place is exactly "zero base velocity, large joint velocity", so this is the
        # term that targets it directly -- the velocity-tracking and pos_deviation rewards are both
        # fully satisfied by a robot that steps without translating and give no pressure at all.
        self.joint_vel_l2_static_val = (
            torch.sum(torch.square(self.joint_vel), dim=1) * static_mask
        )

        # DOF position deviation, split by static/moving (same ramp as everything else above) so
        # standing posture and walking posture can be regularized independently -- a joint
        # configuration that's a sensible average across a whole gait cycle isn't necessarily what
        # you want a planted, motionless stance to relax into, and tuning one scale was forcing a
        # compromise between the two.
        dof_pos_err = torch.sum(torch.square(self.joint_pos - self.desired_joint_pos), dim=1)
        self.dof_pos_l2_walk_val = dof_pos_err * moving_mask
        self.dof_pos_l2_stance_val = dof_pos_err * static_mask

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
        #
        # Uses the PEAK force across the physics substeps of this control step, not the single
        # instantaneous sample in net_forces_w. DirectRLEnv.step() calls scene.update() inside the
        # decimation loop, so the contact sensor refreshes every sim substep (5ms) while rewards are
        # computed once per control step (20ms). A touchdown impact only lasts 1-2 substeps, so
        # reading net_forces_w alone samples a near-arbitrary phase of the contact cycle and misses
        # most spikes entirely -- which is why this term logged ~1e-5 while the MuJoCo eval showed
        # 2.3x-bodyweight slams. net_forces_w_history keeps the last `history_length` substeps
        # (index 0 = most recent); with history_length == decimation it spans the whole control step.
        # Only the peak-force term uses this: contact detection and the grf_balance/grf_target terms
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

        # Stance-time bookkeeping, mirroring the air-time lines just below: accumulate while in
        # contact, clear on liftoff. Same ordering rule -- the increment happens before the clear,
        # so a foot that lifts this step still had its full stance credited.
        self.feet_contact_time += self.step_dt
        self.feet_contact_time[~contact] = 0.0
        # Handles for _compute_paper_rewards, which runs after this method from _get_rewards.
        self.contact_bool = contact
        self.landed_bool = landed
        self.moving_mask_val = moving_mask

        # Latch the completed swing duration before feet_air_time is cleared below.
        self.feet_last_air_time = torch.where(landed, self.feet_air_time, self.feet_last_air_time)

        # Reset air time and swing peak height for feet in contact. Must stay AFTER the reward
        # computations above, so the final increment of the swing is credited before clearing.
        self.feet_air_time[contact] = 0.0
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
        # while foot_height/feet_air_time ramped smoothly over [threshold, static_command_ramp].
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
            # still must cost nothing here. Then ramped by moving_mask so the term fades out
            # toward zero command instead of switching off at a threshold.
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

    # Table II of Dowdy & Chagas Vaz (SII 2026): (scaler, injected noise) per observation term,
    # applied per-term rather than through this repo's single observation_noise_scale.
    _PAPER_OBS_SPEC = (
        ("lin_acc",      0.7, 0.02),
        ("ang_vel",      0.7, 0.01),
        ("orientation",  0.7, 0.05),
        ("command",      1.0, 0.00),
        ("joint_pos",    1.0, 0.05),
        ("joint_vel",    1.0, 0.10),
        ("prev_torque",  0.5, 0.05),
        ("prev_action",  1.0, 0.00),
        ("feet_contact", 2.0, 0.00),
    )
    PAPER_OBS_DIM = 3 + 3 + 4 + 3 + 12 + 12 + 12 + 12 + 4  # = 65

    def _paper_observations(self) -> torch.Tensor:
        """Observation vector transcribed from Dowdy & Chagas Vaz (SII 2026), Table II.

        Three differences from this repo's 49-dim vector matter, and they are the paper's whole
        argument for why a torque policy can work without state estimation:

          1. BASE LINEAR VELOCITY IS ABSENT, deliberately. This repo feeds base_lin_vel, which on
             hardware has to come from a state estimator. The paper replaces it with linear
             ACCELERATION (what an IMU actually measures) and lets the policy infer velocity from
             the sequence.
          2. PREVIOUS APPLIED TORQUE is fed back in. With domain randomisation over the actuators,
             this is what lets the policy identify its own actuator dynamics online -- the paper's
             stated reason for expecting torque control to beat position control here.
          3. FOOT CONTACT is an explicit binary input. Nothing in this repo's observation tells the
             policy which feet are down, which is a hard thing to ask it to infer while also
             asking it to produce a contact-scheduled gait.

        Orientation is the raw world-frame quaternion, replacing projected gravity -- the paper is
        explicit about that swap. Command is 3 wide here (vx, vy, wz); this repo's 4th slot
        (heading) has no counterpart in Table II and is dropped.

        Per-term scalers and per-term noise both come from Table II, so cfg.observation_noise_scale
        is NOT used in this mode.

        AMBIGUITY: Table II names the term "Linear Acceleration a_base" without saying whether it
        is body-frame, and whether it includes gravity the way a real accelerometer does. Taken
        here as the body-frame finite difference of base_lin_vel -- the same quantity the repo
        already computes for base_acc_l2 -- i.e. gravity-free.
        """
        lin_acc = (self.base_lin_vel - self.last_base_lin_vel) / self.step_dt
        feet_contact = (
            torch.norm(self.net_contact_forces[:, self._feet_ids, :], dim=-1) > 1.0
        ).float()
        terms = (
            lin_acc,
            self.base_ang_vel,
            self.root_quat_w,
            self.commands[:, :3],
            self.joint_pos - self.desired_joint_pos,
            self.joint_vel,
            self.applied_torque,
            self.actions,
            feet_contact,
        )
        out = []
        for value, (_, scaler, noise) in zip(terms, self._PAPER_OBS_SPEC):
            v = value * scaler
            if noise > 0.0:
                v = v + torch.randn_like(v) * noise
            out.append(v)
        return torch.cat(out, dim=-1)

    def _get_observations(self) -> dict:
        """
        Collects data from the simulation to feed into the neural network.
        """
        # Re-refresh: _reset_idx() ran between _get_rewards() and here, so reset envs have been
        # teleported to their spawn state since _compute_reward_terms() last looked.
        self._refresh_state()

        if self.cfg.obs_mode == "paper":
            obs = self._paper_observations()
        else:
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
        # Pull all sim state fresh, then derive every per-step reward quantity from it, so the
        # whole reward vector is evaluated on the same (current) timestep -- see _refresh_state's
        # docstring for why neither can rely on _get_observations having run.
        self._refresh_state()
        self._compute_reward_terms()

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
            self.cfg.rew_scale_dof_pos_l2_walk,
            self.cfg.rew_scale_dof_pos_l2_stance,
            self.cfg.rew_scale_dof_torques_l2,
            self.cfg.rew_scale_dof_acc_l2,
            self.cfg.rew_scale_action_rate_l2,
            self.cfg.rew_scale_feet_air_time,
            self.cfg.rew_scale_flat_orientation_l2,
            self.cfg.rew_scale_foot_height_penalty,
            self.cfg.rew_scale_foot_height_reward,
            self.cfg.rew_scale_feet_grounded,
            self.cfg.rew_scale_foot_landing_vel,
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
            self.joint_vel,
            self.last_joint_vel,
            self.last_base_lin_vel,
            self.applied_torque,
            self.actions,
            self.previous_actions,
            self.feet_air_time_reward_val,
            self.foot_height_penalty_val,
            self.foot_height_reward_val,
            self.feet_grounded_val,
            self.foot_landing_vel_val,
            self.feet_air_penalty_val,
            self.feet_air_penalty_static_val,
            self.joint_vel_l2_static_val,
            self.dof_pos_l2_walk_val,
            self.dof_pos_l2_stance_val,
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
        # Paper reward set (Dowdy & Chagas Vaz, SII 2026 Table III). Added on top rather than
        # inside compute_rewards: its terms use L1 norms where the repo squares, so they are
        # genuinely different terms, not a reweighting. Every scale defaults to 0.0, so this
        # contributes exactly zero unless a phase turns it on.
        paper_reward, paper_log = self._compute_paper_rewards()
        total_reward = total_reward + paper_reward
        self.extras["log"].update(paper_log)
        # Peak per-foot contact force in body weights, for comparing the substep-peak measurement
        # against the MuJoCo eval's grf_peak_stance_N before retuning rew_scale_max_contact_force.
        self.extras["log"]["diag/grf_peak_bw_mean"] = self.grf_peak_bw_val.mean()
        self.extras["log"]["diag/grf_peak_bw_max"] = self.grf_peak_bw_val.max()
        # Mean vertical foot speed at touchdown, over every foot that landed anywhere in the batch
        # this step. Reads 0 on the rare step where nothing landed, so read it as a running average
        # in TensorBoard, not step by step.
        n_landings = self.foot_landing_count.sum()
        self.extras["log"]["diag/foot_landing_speed_mps"] = (
            self.foot_landing_speed_sum.sum() / n_landings.clamp(min=1.0)
        )
        # Normalise per-step reward to the reference control period, so that per-*second*
        # reward is invariant to the control rate. Without this a 200 Hz torque run collects
        # 4x the return per second of a 50 Hz position run purely from stepping more often,
        # and the two action spaces are not being scored on the same thing. Position mode
        # (step_dt == reward_dt_ref == 0.02) scales by exactly 1.0 and is unaffected.
        if self.cfg.reward_dt_ref > 0.0:
            total_reward = total_reward * (self.step_dt / self.cfg.reward_dt_ref)
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
        self.feet_air_time[env_ids] = 0.0
        self.feet_contact_time[env_ids] = 0.0
        self.feet_last_air_time[env_ids] = 0.0
        self.feet_height_max[env_ids] = 0.0
        self.last_feet_contact[env_ids] = False
        self.last_feet_vel_z[env_ids] = 0.0
        self.last_joint_vel[env_ids] = 0.0
        self.last_base_lin_vel[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.last_strike_time[env_ids] = 0.0
        self.stride_duration[env_ids] = 1.0
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
        # Link Mass, "Scale" in Table I of the paper: a multiplier on EVERY body's mass, not the
        # single additive payload on the trunk above. Applied on top of the payload so the two
        # compose; range defaults to (1.0, 1.0), i.e. off, for every pre-existing phase.
        lo, hi = self.cfg.link_mass_scale_range
        if lo != 1.0 or hi != 1.0:
            scale = sample_uniform(lo, hi, (len(env_ids_cpu), masses.shape[1]), "cpu")
            masses[local_ids_cpu] = view.data.default_mass[local_ids_cpu] * scale
            masses[local_ids_cpu, 0] = masses[local_ids_cpu, 0] + mass_noise[:, 0]
        view.root_physx_view.set_masses(masses, local_ids_cpu)

        # Foot Friction, "New" in Table I: resample the contact material outright rather than
        # perturbing it. Applied to every collider on the robot, not only the feet -- shape ids
        # are not exposed per-body here, and the terrain uses friction_combine_mode="multiply",
        # so this still lands on the foot/ground contact that Table I is about.
        f_lo, f_hi = self.cfg.foot_friction_range
        if f_hi > 0.0:
            # BUCKETED, and it has to be. PhysX caps the scene at 64K unique materials, and a
            # freshly sampled friction per (env, shape) blows straight through that: 4096 envs x
            # ~17 shapes = ~70K on the first full reset, which fails with
            #   "PxPhysics::createMaterial: limit of 64K materials reached"
            # followed by a flood of "material pointer 0 is NULL". Isaac Lab's own
            # randomize_rigid_body_material has num_buckets for exactly this reason.
            #
            # So: draw a fixed, small palette of friction values, and give each env one of them.
            # The number of distinct materials is bounded by the bucket count no matter how many
            # envs exist. 64 values across (0.4, 1.1) is a 0.011 quantisation -- far finer than
            # the sim2real uncertainty this is standing in for.
            #
            # Applied ONCE PER VIEW, on the first reset, not on every reset. Bucketing
            # bounds the number of distinct material VALUES; it does not necessarily bound how
            # many material objects PhysX allocates across repeated set_material_properties
            # calls. Assigning once removes that question entirely, and costs almost nothing:
            # with thousands of envs each holding a different bucket, the policy still sees the
            # whole friction distribution in every single batch -- it just does not get a fresh
            # draw per episode. This is also what Isaac Lab's own material randomisation does
            # (its event term defaults to mode="startup").
            view_key = view_idx if view_idx is not None else -1
            if getattr(self, "_friction_done", None) is None:
                self._friction_done = set()
            if view_key not in self._friction_done:
                self._friction_done.add(view_key)
                n_buckets = self.cfg.friction_num_buckets
                buckets = sample_uniform(f_lo, f_hi, (n_buckets,), "cpu")
                materials = view.root_physx_view.get_material_properties().clone()
                # One bucket per ENV, broadcast across that robot's shapes. A robot with
                # different friction under each of its own feet is not what the paper's "Foot
                # Friction" row describes, and per-shape draws are what multiplied the material
                # count by ~17 in the first place.
                n_env_total = materials.shape[0]
                picks = torch.randint(0, n_buckets, (n_env_total,), device="cpu")
                friction = buckets[picks].unsqueeze(1)      # (n_env_total, 1), broadcasts
                materials[:, :, 0] = friction               # static
                materials[:, :, 1] = friction               # dynamic
                all_ids = torch.arange(n_env_total, device="cpu")
                view.root_physx_view.set_material_properties(materials, all_ids)

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

        # Add random noise to initial joint positions and velocities. Ranges are configurable
        # (defaults reproduce the old hardcoded +-0.2 rad / +-0.5 rad/s); the paper phase widens
        # them to Table I's +-0.3 rad and +-2.5 rad/s, which is a far more aggressive reset --
        # the robot regularly starts mid-tumble, which is itself a source of the varied contact
        # states a gait reward needs to see before it can score one.
        pos_noise = sample_uniform(
            self.cfg.reset_joint_pos_range[0], self.cfg.reset_joint_pos_range[1],
            (len(ids), len(v_idx)), joint_pos.device
        )
        vel_noise = sample_uniform(
            self.cfg.reset_joint_vel_range[0], self.cfg.reset_joint_vel_range[1],
            (len(ids), len(v_idx)), joint_vel.device
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

        # Base pose / velocity randomisation on reset (Table I: Body Position, Body Orientation,
        # Body Velocity -- all "Add"). This repo spawned every robot in an identical, perfectly
        # level, perfectly still pose, so the policy only ever saw one initial condition. All
        # three ranges default to 0.0, leaving existing phases spawning exactly as before.
        n = len(ids)
        if self.cfg.reset_body_pos_range[1] > 0.0:
            lo, hi = self.cfg.reset_body_pos_range
            default_root_state[:, :2] += sample_uniform(lo, hi, (n, 2), self.device)
        if self.cfg.reset_body_ori_range[1] > 0.0:
            # Applied as a small additive perturbation on the quaternion, renormalised. At Table
            # I's +-0.02 the small-angle approximation holds comfortably (~2.3 deg).
            lo, hi = self.cfg.reset_body_ori_range
            quat = default_root_state[:, 3:7] + sample_uniform(lo, hi, (n, 4), self.device)
            default_root_state[:, 3:7] = quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-6)
        if self.cfg.reset_body_vel_range[1] > 0.0:
            lo, hi = self.cfg.reset_body_vel_range
            default_root_state[:, 7:10] += sample_uniform(lo, hi, (n, 3), self.device)

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
    rew_scale_dof_torques_l2: float,
    rew_scale_dof_acc_l2: float,
    rew_scale_action_rate_l2: float,
    rew_scale_feet_air_time: float,
    rew_scale_flat_orientation_l2: float,
    rew_scale_foot_height_penalty: float,
    rew_scale_foot_height_reward: float,
    rew_scale_feet_grounded: float,
    rew_scale_foot_landing_vel: float,
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
    joint_vel: torch.Tensor,
    last_joint_vel: torch.Tensor,
    last_base_lin_vel: torch.Tensor,
    joint_torques: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    feet_air_time_reward_val: torch.Tensor,
    foot_height_penalty_val: torch.Tensor,
    foot_height_reward_val: torch.Tensor,
    feet_grounded_val: torch.Tensor,
    foot_landing_vel_val: torch.Tensor,
    feet_air_penalty_val: torch.Tensor,
    feet_air_penalty_static_val: torch.Tensor,
    joint_vel_l2_static_val: torch.Tensor,
    dof_pos_l2_walk_val: torch.Tensor,
    dof_pos_l2_stance_val: torch.Tensor,
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

    # 10. DOF Position L2 Penalty, split by static/moving -- see dof_pos_l2_walk_val /
    # dof_pos_l2_stance_val comment in _get_observations for why.
    rew_dof_pos_l2_walk = rew_scale_dof_pos_l2_walk * dof_pos_l2_walk_val
    rew_dof_pos_l2_stance = rew_scale_dof_pos_l2_stance * dof_pos_l2_stance_val

    # 11. Flat Orientation Penalty (Penalize Pitch/Roll)
    rew_flat_orientation_l2 = rew_scale_flat_orientation_l2 * torch.sum(
        torch.square(projected_gravity[:, :2]), dim=1
    )

    # 12. Foot Height, swing-apex scored at touchdown -- two directions, see the block comment in
    # _compute_reward_terms. Penalty on the mismatch (rew_scale_foot_height_penalty must be NEGATIVE)
    # and/or reward on the match (rew_scale_foot_height_reward must be POSITIVE, for early phases
    # that need the feet pushed up in the first place).
    rew_foot_height_penalty = rew_scale_foot_height_penalty * foot_height_penalty_val
    rew_foot_height_reward = rew_scale_foot_height_reward * foot_height_reward_val

    # 12c. Grounded-feet penalty (scale must be NEGATIVE). Charged on feet in contact beyond
    # feet_grounded_allowed while moving -- the only gait term that is nonzero while standing.
    rew_feet_grounded = rew_scale_feet_grounded * feet_grounded_val

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
    log["reward/dof_pos_l2_walk"] = rew_dof_pos_l2_walk.mean()
    log["reward/dof_pos_l2_stance"] = rew_dof_pos_l2_stance.mean()
    log["reward/dof_acc_l2"] = rew_dof_acc_l2.mean()
    log["reward/base_acc_l2"] = rew_base_acc_l2.mean()
    log["reward/action_rate_l2"] = rew_action_rate_l2.mean()
    log["reward/feet_air_time"] = rew_feet_air_time.mean()
    log["reward/flat_orientation_l2"] = rew_flat_orientation_l2.mean()
    log["reward/foot_height_penalty"] = rew_foot_height_penalty.mean()
    log["reward/foot_height_reward"] = rew_foot_height_reward.mean()
    log["reward/feet_grounded"] = rew_feet_grounded.mean()
    log["diag/n_feet_grounded_excess"] = feet_grounded_val.mean()
    log["reward/foot_landing_vel"] = rew_foot_landing_vel.mean()
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
        + rew_dof_pos_l2_walk
        + rew_dof_pos_l2_stance
        + rew_dof_acc_l2
        + rew_base_acc_l2
        + rew_action_rate_l2
        + rew_feet_air_time
        + rew_flat_orientation_l2
        + rew_foot_height_penalty
        + rew_foot_height_reward
        + rew_feet_grounded
        + rew_foot_landing_vel
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
