##TODO
# Add a input variable to controll the swing height and the base height
# Add a point cloud of the terrain as input
# Add a friction variable to controll the friction of the terrain
# Add a noise variable to controll the noise of the sensors
# Add a way to recouver from a fall

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sim2Sim environment: three Unitree robots (A1, Quadruped, Go2) side-by-side,
all controlled by the same trained policy checkpoint."""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import sample_uniform

from .quadruped_env import compute_rewards
from .quadruped_sim2sim_env_cfg import QuadrupedSim2SimEnvCfg

# Y-axis offsets so the three robots don't overlap inside a single env cell
_Y_OFFSETS = (-1.8, 0.0, 1.8)
_ROBOT_LABELS = ("A1", "Quadruped", "Go2")


class QuadrupedSim2SimEnv(DirectRLEnv):
    """Three-robot Sim2Sim environment.

    Each scene 'env slot' contains three robots (A1, Quadruped, Go2) placed at
    different Y-offsets.  The policy sees ``scene.num_envs * 3`` parallel
    instances (one stripe per robot type), so no changes to the policy or
    play loop are required.

    Internal indexing convention (N = scene.num_envs):
        logical [0 , N)   → A1  , physical env [0, N)
        logical [N , 2N)  → Quadruped , physical env [0, N)
        logical [2N, 3N)  → Go2 , physical env [0, N)
    """

    cfg: QuadrupedSim2SimEnvCfg

    @property
    def num_envs(self) -> int:
        """The logical number of environments (3x the physical envs)."""
        return getattr(self, "_scene_num_envs", self.cfg.scene.num_envs) * 3

    # ── construction ──────────────────────────────────────────────────────

    def __init__(self, cfg: QuadrupedSim2SimEnvCfg, render_mode: str | None = None, **kwargs):
        # DirectRLEnv.__init__ calls _setup_scene, so robots are ready after this
        super().__init__(cfg, render_mode, **kwargs)

        N = self.scene.num_envs
        self._scene_num_envs = N

        # ── per-robot joint / feet ─────────────────────────────────────────
        self._robots = [self._robot_a1, self._robot_quadruped, self._robot_go2]
        self._contact_sensors = [
            self._cs_a1, self._cs_quadruped, self._cs_go2
        ]

        def _find_joints_and_feet(robot):
            dof_idx, _ = robot.find_joints(
                ".*_hip_joint|.*_thigh_joint|.*_calf_joint"
            )
            feet_ids_raw, _ = robot.find_bodies(".*_foot")
            feet_ids = [feet_ids_raw[2], feet_ids_raw[3],
                        feet_ids_raw[0], feet_ids_raw[1]]
            return dof_idx, feet_ids

        self._dof_idx_a1, self._feet_a1 = _find_joints_and_feet(self._robot_a1)
        self._dof_idx_quadruped, self._feet_quadruped = _find_joints_and_feet(self._robot_quadruped)
        self._dof_idx_go2, self._feet_go2 = _find_joints_and_feet(self._robot_go2)

        self._all_dof_idx = [self._dof_idx_a1, self._dof_idx_quadruped, self._dof_idx_go2]
        self._all_feet = [self._feet_a1, self._feet_quadruped, self._feet_go2]

        # ── desired joint positions (default pose per robot) ───────────────
        self._desired_jp = [
            robot.data.default_joint_pos[:, dof].clone()
            for robot, dof in zip(self._robots, self._all_dof_idx)
        ]

        D = self.device

        # ── re-allocate ALL num_envs-sized buffers to 3N ──────────────────
        self.episode_length_buf   = torch.zeros(3 * N, device=D, dtype=torch.long)
        self.reset_terminated     = torch.zeros(3 * N, device=D, dtype=torch.bool)
        self.reset_time_outs      = torch.zeros(3 * N, device=D, dtype=torch.bool)
        self.reset_buf            = torch.zeros(3 * N, device=D, dtype=torch.bool)

        self.commands             = torch.zeros(3 * N, 4, device=D)
        self.command_timer        = torch.zeros(3 * N, device=D)

        self.actions              = torch.zeros(3 * N, self.cfg.action_space, device=D)
        self.previous_actions     = torch.zeros(3 * N, self.cfg.action_space, device=D)

        # last_joint_vel: use Quadruped joint count as reference (all 3 have 12+)
        num_jnts = self._robots[0].num_joints
        self.last_joint_vel       = torch.zeros(3 * N, num_jnts, device=D)

        self.feet_air_time        = torch.zeros(3 * N, 4, device=D)
        self.last_feet_contact    = torch.zeros(3 * N, 4, device=D, dtype=torch.bool)
        self.feet_air_time_reward_val = torch.zeros(3 * N, device=D)

        # net contact forces — kept per-robot (different body counts possible)
        self._net_cf = [
            torch.zeros(N, r.num_bodies, 3, device=D) for r in self._robots
        ]

        # ── initial command resample ───────────────────────────────────────
        self._resample_commands(torch.arange(3 * N, device=D))

    # ── scene setup ───────────────────────────────────────────────────────

    def _setup_scene(self):
        """Spawn three robots at different Y-offsets then clone."""

        def _make_robot(base_cfg, y_off):
            cfg = base_cfg.copy()
            cfg.init_state.pos = (0.0, y_off, self.cfg.spawn_height)
            cfg.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.1, 0.1)
            )
            return Articulation(cfg)

        self._robot_a1  = _make_robot(self.cfg.robot_a1_cfg,  _Y_OFFSETS[0])
        self._robot_quadruped = _make_robot(self.cfg.robot_quadruped_cfg, _Y_OFFSETS[1])
        self._robot_go2 = _make_robot(self.cfg.robot_go2_cfg, _Y_OFFSETS[2])

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot_a1"]  = self._robot_a1
        self.scene.articulations["robot_quadruped"] = self._robot_quadruped
        self.scene.articulations["robot_go2"] = self._robot_go2

        self._cs_a1  = ContactSensor(self.cfg.contact_sensor_a1)
        self._cs_quadruped = ContactSensor(self.cfg.contact_sensor_quadruped)
        self._cs_go2 = ContactSensor(self.cfg.contact_sensor_go2)

        self.scene.sensors["cs_a1"]  = self._cs_a1
        self.scene.sensors["cs_quadruped"] = self._cs_quadruped
        self.scene.sensors["cs_go2"] = self._cs_go2

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── helpers ───────────────────────────────────────────────────────────

    def _resample_commands(self, env_ids):
        """Resample velocity commands; env_ids in [0, 3N)."""
        self.commands[env_ids, 0] = sample_uniform(
            self.cfg.command_x_range[0], self.cfg.command_x_range[1],
            (len(env_ids),), device=self.device,
        )
        self.commands[env_ids, 1] = sample_uniform(
            self.cfg.command_y_range[0], self.cfg.command_y_range[1],
            (len(env_ids),), device=self.device,
        )
        self.commands[env_ids, 2] = sample_uniform(
            self.cfg.command_yaw_range[0], self.cfg.command_yaw_range[1],
            (len(env_ids),), device=self.device,
        )
        self.commands[env_ids, 3] = 0.0
        self.command_timer[env_ids] = 0.0

    # ── RL interface ──────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions = self.actions.clone()
        self.actions = actions.clone()

        # Update timers and resample stale commands
        self.command_timer += self.step_dt
        stale = (self.command_timer >= self.cfg.command_resampling_time).nonzero(
            as_tuple=False
        ).flatten()
        if len(stale) > 0:
            self._resample_commands(stale)

        # Teleoperation: broadcast keyboard command to all 3 robot groups
        import os
        if os.environ.get("QUADRUPED_TELEOP", "0") == "1":
            if not hasattr(self, "_keyboard"):
                import numpy as np
                from isaaclab.devices.keyboard.se2_keyboard import (
                    Se2Keyboard, Se2KeyboardCfg,
                )

                class WasdKeyboard(Se2Keyboard):
                    def __init__(self, cfg):
                        self.speed_multiplier = 1.0
                        super().__init__(cfg)
                        print(
                            "\n[Sim2Sim Teleop] W/S=fwd  A/D=strafe  Q/E=turn"
                            " | +/= speed up  - slow down\n"
                        )

                    def _create_key_bindings(self):
                        super()._create_key_bindings()
                        self._INPUT_KEY_MAPPING.update({
                            "W":  np.asarray([1., 0., 0.]) * self.v_x_sensitivity,
                            "S":  np.asarray([-1., 0., 0.]) * self.v_x_sensitivity,
                            "A":  np.asarray([0., 1., 0.]) * self.v_y_sensitivity,
                            "D":  np.asarray([0., -1., 0.]) * self.v_y_sensitivity,
                            "Q":  np.asarray([0., 0., 1.]) * self.omega_z_sensitivity,
                            "E":  np.asarray([0., 0., -1.]) * self.omega_z_sensitivity,
                        })

                    def _on_keyboard_event(self, event, *args, **kwargs):
                        import carb.input as ci
                        if event.type == ci.KeyboardEventType.KEY_PRESS:
                            if event.input in (ci.KeyboardInput.EQUAL,
                                               ci.KeyboardInput.NUMPAD_ADD):
                                self.speed_multiplier = round(
                                    min(3.0, self.speed_multiplier + 0.1), 1)
                                print(f"[Teleop] Speed: {self.speed_multiplier:.1f}x")
                            elif event.input in (ci.KeyboardInput.MINUS,
                                                 ci.KeyboardInput.NUMPAD_SUBTRACT):
                                self.speed_multiplier = round(
                                    max(0.1, self.speed_multiplier - 0.1), 1)
                                print(f"[Teleop] Speed: {self.speed_multiplier:.1f}x")
                        return super()._on_keyboard_event(event, *args, **kwargs)

                    def advance(self):
                        return super().advance() * self.speed_multiplier

                kb_cfg = Se2KeyboardCfg(
                    v_x_sensitivity=1.0, v_y_sensitivity=1.0, omega_z_sensitivity=1.2
                )
                kb_cfg.class_type = WasdKeyboard
                kb_cfg.sim_device = self.device
                self._keyboard = kb_cfg.class_type(kb_cfg)

            cmd = self._keyboard.advance()
            self.commands[:, 0] = cmd[0]
            self.commands[:, 1] = cmd[1]
            self.commands[:, 2] = cmd[2]
            self.commands[:, 3] = 0.0

    def _apply_action(self) -> None:
        N = self._scene_num_envs
        for i, (robot, desired_jp, dof_idx) in enumerate(
            zip(self._robots, self._desired_jp, self._all_dof_idx)
        ):
            acts = self.actions[i * N:(i + 1) * N]
            targets = acts * self.cfg.action_scale + desired_jp
            lo = robot.data.soft_joint_pos_limits[0, dof_idx, 0]
            hi = robot.data.soft_joint_pos_limits[0, dof_idx, 1]
            targets = torch.clamp(targets, lo, hi)
            robot.set_joint_position_target(targets, joint_ids=dof_idx)
            robot.set_joint_velocity_target(torch.zeros_like(targets), joint_ids=dof_idx)

    def _get_observations(self) -> dict:
        N = self._scene_num_envs
        obs_chunks = []

        for i, (robot, cs, dof_idx, feet_ids, desired_jp) in enumerate(
            zip(self._robots, self._contact_sensors,
                self._all_dof_idx, self._all_feet, self._desired_jp)
        ):
            slice_ = slice(i * N, (i + 1) * N)

            jpos = robot.data.joint_pos[:, dof_idx]
            jvel = robot.data.joint_vel[:, dof_idx]
            lin_vel = robot.data.root_lin_vel_b
            ang_vel = robot.data.root_ang_vel_b
            proj_grav = robot.data.projected_gravity_b

            # Contact / air-time
            self._net_cf[i] = cs.data.net_forces_w
            contact = (
                torch.norm(self._net_cf[i][:, feet_ids, :], dim=-1) > 1.0
            )
            first_contact = contact & ~self.last_feet_contact[slice_]
            self.feet_air_time[slice_] += self.step_dt
            rew_air = torch.sum(
                (self.feet_air_time[slice_] - 0.5).clamp(min=0.0)
                * first_contact.float(), dim=1
            ) * (torch.norm(self.commands[slice_, :2], dim=1) > 0.1)
            self.feet_air_time_reward_val[slice_] = rew_air
            self.feet_air_time[slice_][contact] = 0.0
            self.last_feet_contact[slice_] = contact

            # Store for reward / done use
            self.last_joint_vel[slice_, :jvel.shape[1]] = jvel

            cmds = self.commands[slice_]
            acts = self.actions[slice_]

            obs = torch.cat(
                (lin_vel, ang_vel, proj_grav, cmds,
                 jpos - desired_jp, jvel, acts),
                dim=-1,
            )
            obs += torch.randn_like(obs) * self.cfg.observation_noise_scale
            obs_chunks.append(obs)

        return {"policy": torch.cat(obs_chunks, dim=0)}  # [3N, 49]

    def _get_rewards(self) -> torch.Tensor:
        N = self._scene_num_envs
        chunks = []

        for i, (robot, dof_idx, desired_jp) in enumerate(
            zip(self._robots, self._all_dof_idx, self._desired_jp)
        ):
            sl = slice(i * N, (i + 1) * N)
            jpos = robot.data.joint_pos[:, dof_idx]
            jvel = robot.data.joint_vel[:, dof_idx]
            last_jvel = self.last_joint_vel[sl, :jvel.shape[1]]

            r = compute_rewards(
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
                self.cfg.command_lin_vel_std,
                self.cfg.command_ang_vel_std,
                self.commands[sl],
                robot.data.root_lin_vel_b,
                robot.data.root_ang_vel_b,
                robot.data.projected_gravity_b,
                jpos, desired_jp, jvel, last_jvel,
                robot.data.applied_torque,
                robot.data.joint_acc,
                self.actions[sl],
                self.previous_actions[sl],
                self.feet_air_time_reward_val[sl],
                self.reset_terminated[sl],
                self.step_dt,
            )
            chunks.append(r)

        return torch.cat(chunks, dim=0)  # [3N]

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        died_chunks = []
        for robot in self._robots:
            proj_grav = robot.data.projected_gravity_b
            upright_fail = proj_grav[:, 2] > -self.cfg.base_angle_termination_thresh
            too_low = robot.data.root_pos_w[:, 2] < 0.15
            died_chunks.append(too_low | upright_fail)

        died = torch.cat(died_chunks, dim=0)  # [3N]
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device)

        # Base-class bookkeeping (episode_length_buf, event manager, etc.)
        # We call DirectRLEnv._reset_idx directly using the logical IDs.
        # episode_length_buf is [3N] so indexing with logical IDs is correct.
        self.episode_length_buf[env_ids] = 0

        N = self._scene_num_envs

        # Split logical IDs into per-robot physical IDs
        a1_phys  = env_ids[env_ids < N]
        quadruped_phys = env_ids[(env_ids >= N) & (env_ids < 2 * N)] - N
        go2_phys = env_ids[env_ids >= 2 * N] - 2 * N

        for phys_ids, robot, dof_idx, desired_jp, logical_base in [
            (a1_phys,  self._robot_a1,  self._dof_idx_a1,  self._desired_jp[0], 0),
            (quadruped_phys, self._robot_quadruped, self._dof_idx_quadruped, self._desired_jp[1], N),
            (go2_phys, self._robot_go2, self._dof_idx_go2, self._desired_jp[2], 2 * N),
        ]:
            if len(phys_ids) == 0:
                continue

            cpu_ids = phys_ids.cpu()

            # Mass randomisation
            masses = robot.root_physx_view.get_masses().clone()
            noise = sample_uniform(-1.0, 3.0, (len(cpu_ids), 1), "cpu")
            masses[cpu_ids, 0] = robot.data.default_mass[cpu_ids, 0] + noise[:, 0]
            robot.root_physx_view.set_masses(masses, cpu_ids)

            # Joint friction and damping randomization
            friction_noise = sample_uniform(
                self.cfg.joint_friction_range[0],
                self.cfg.joint_friction_range[1],
                (len(phys_ids), len(dof_idx)),
                self.device,
            )
            damping_noise = sample_uniform(
                self.cfg.joint_damping_range[0],
                self.cfg.joint_damping_range[1],
                (len(phys_ids), len(dof_idx)),
                self.device,
            )
            robot.write_joint_friction_coefficient_to_sim(friction_noise, joint_ids=dof_idx, env_ids=phys_ids)
            robot.write_joint_damping_to_sim(damping_noise, joint_ids=dof_idx, env_ids=phys_ids)

            # Joint states
            jpos = robot.data.default_joint_pos[phys_ids].clone()
            jvel = robot.data.default_joint_vel[phys_ids].clone()
            pos_noise = sample_uniform(-0.2, 0.2, (len(phys_ids), len(dof_idx)), jpos.device)
            vel_noise = sample_uniform(-0.5, 0.5, (len(phys_ids), len(dof_idx)), jvel.device)
            jpos[:, dof_idx] += pos_noise
            jvel[:, dof_idx] += vel_noise

            # Root state — offset to env origins + Y offset for this robot
            root = robot.data.default_root_state[phys_ids].clone()
            root[:, :3] += self.scene.env_origins[phys_ids]
            root[:, 2] = self.scene.env_origins[phys_ids][:, 2] + self.cfg.spawn_height

            robot.write_root_pose_to_sim(root[:, :7], phys_ids)
            robot.write_root_velocity_to_sim(root[:, 7:], phys_ids)
            robot.write_joint_state_to_sim(jpos, jvel, None, phys_ids)

            # Reset action buffer slice
            logical_ids = phys_ids + logical_base
            self.actions[logical_ids] = 0.0
            self._resample_commands(logical_ids)
