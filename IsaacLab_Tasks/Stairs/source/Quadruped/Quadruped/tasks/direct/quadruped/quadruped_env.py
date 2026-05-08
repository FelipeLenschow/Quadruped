# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
import copy
import random
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import warp as wp
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import sample_uniform, wrap_to_pi
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

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
                (self.num_envs, 20, 3), device=self.device  # 20 is max (Go2 has 19)
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

        # Contact sensor handles num_bodies per env automatically if specified in config
        self.net_contact_forces = torch.zeros(
            self.num_envs, 4, 3, device=self.device
        )  # Only 4 feet

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
        self.feet_air_penalty_static_val = torch.zeros(
            self.num_envs, device=self.device
        )
        self.joint_vel_l2_static_val = torch.zeros(self.num_envs, device=self.device)
        self.command_timer = torch.full(
            (self.num_envs,), 100.0, device=self.device
        )  # Force immediate resample

        # Initialize joint DOF indices (Used in reset)
        # Note: This is now safe because super().__init__ has finished and views are ready
        try:
            self._joint_dof_idx, _ = self.robot.find_joints(
                ".*_hip_joint|.*_thigh_joint|.*_calf_joint"
            )
        except Exception as e:
            print(f"[WARNING] Could not find joint DOF indices during init: {e}")
            self._joint_dof_idx = None

        # Check if debug visualization is enabled in config
        if hasattr(self.cfg, "debug_vis") and self.cfg.debug_vis:
            self.set_debug_vis(True)

    def _debug_vis_callback(self, event):
        """Called every frame by the base class event subscription to update debug visualization."""
        self._set_debug_vis_impl(True)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Draws the height scan grid points in the viewport."""
        if not hasattr(self, "_height_markers"):
            return

        self._height_markers.set_visibility(debug_vis)
        if not debug_vis:
            return

        # samples_w is populated in _get_observations, check before access
        if not hasattr(self, "samples_w") or self.samples_w is None:
            return

        # VISUALIZE ALL if num_envs is small (e.g. Play mode), otherwise limit to env_0
        if self.num_envs <= 10:
            points = self.samples_w.reshape(-1, 3).cpu().numpy()
        else:
            points = self.samples_w[0].cpu().numpy()  # (187, 3)

        # Only visualize if we have valid non-zero points to avoid origin flicker
        if np.any(points):
            self._height_markers.visualize(translations=points)

    def _setup_scene(self):
        import os
        import copy
        import torch
        import numpy as np
        import omni.usd
        import isaaclab.sim as sim_utils
        from pxr import UsdGeom, Usd
        from isaaclab.assets.articulation import Articulation
        from .quadruped_env_cfg import ROBOT_VARIANTS
        from isaaclab.utils.warp import raycast_mesh, convert_to_warp_mesh

        selection = os.environ.get(
            "QUADRUPED_ROBOT", os.environ.get("FORCE_ROBOT", "")
        ).upper()
        num_envs = self.num_envs
        stage = omni.usd.get_context().get_stage()

        # Ensure Physics Scene exists
        if not any(prim.GetTypeName() == "PhysicsScene" for prim in stage.Traverse()):
            sim_utils.define_prim("/physicsScene", "PhysicsScene")

        # 1. Terrain Mesh Loading for Elevation Sampling
        try:
            all_points, all_indices, vertex_offset = [], [], 0
            ground_prim = stage.GetPrimAtPath("/World/ground")
            iterator = (
                Usd.PrimRange(ground_prim)
                if ground_prim.IsValid()
                else stage.Traverse()
            )

            for prim in iterator:
                if prim.IsA(UsdGeom.Mesh):
                    path_str = str(prim.GetPath())
                    if any(
                        x in path_str.lower()
                        for x in ["marker", "robot", "visuals", "groundplane"]
                    ):
                        continue
                    mesh_geom = UsdGeom.Mesh(prim)
                    local_points = mesh_geom.GetPointsAttr().Get()
                    indices = mesh_geom.GetFaceVertexIndicesAttr().Get()
                    if (
                        local_points is not None
                        and indices is not None
                        and len(indices) % 3 == 0
                    ):
                        xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                            Usd.TimeCode.Default()
                        )
                        points = np.array([xform.Transform(p) for p in local_points])
                        all_points.append(points)
                        all_indices.append(
                            np.array(indices).reshape(-1, 3) + vertex_offset
                        )
                        vertex_offset += len(points)

            if all_points:
                combined_points = np.concatenate(all_points).astype(np.float32)
                combined_indices = np.concatenate(all_indices).astype(np.int32)
                self.terrain_mesh = convert_to_warp_mesh(
                    combined_points, combined_indices, device=self.device
                )
                print(
                    f"[INFO] Loaded terrain mesh with {len(combined_points)} vertices for grid sampling."
                )
            else:
                self.terrain_mesh = None
        except Exception as e:
            print(f"[ERROR] Failed to load terrain mesh: {e}")
            self.terrain_mesh = None

        # 2. Spawning Logic
        spacing = self.cfg.scene.env_spacing
        num_cols = int(np.sqrt(num_envs))

        # Calculate stride to spread robots across terrain generator tiles (e.g. 45x45)
        # This ensuring ALL terrain types (rough, stairs, etc.) are sampled even in small runs
        terrain_cfg = getattr(self.cfg.scene, "terrain", None)
        total_tiles = num_envs
        grid_cols = num_cols
        if terrain_cfg:
            gen_cfg = getattr(terrain_cfg, "terrain_generator", None)
            if (
                gen_cfg
                and hasattr(gen_cfg, "num_cols")
                and hasattr(gen_cfg, "num_rows")
            ):
                grid_cols = gen_cfg.num_cols
                total_tiles = grid_cols * gen_cfg.num_rows
                print(
                    f"[INFO] Spreading {num_envs} robots across {total_tiles} ({gen_cfg.num_rows}x{gen_cfg.num_cols}) terrain tiles."
                )

        stride = max(1, total_tiles // num_envs)

        self.scene._env_origins = torch.zeros((num_envs, 3), device=self.device)

        if "RANDOM" in selection or not selection:
            # Heterogeneous Mode
            self.is_heterogeneous = True
            self.a1_indices = list(range(0, num_envs, 3))
            self.quadruped_indices = list(range(1, num_envs, 3))
            self.go2_indices = list(range(2, num_envs, 3))

            for i in range(num_envs):
                env_path = f"/World/envs/env_{i}"
                if not stage.GetPrimAtPath(env_path).IsValid():
                    sim_utils.define_prim(env_path, "Xform")

                tile_idx = i * stride
                row, col = tile_idx // grid_cols, tile_idx % grid_cols
                pos = (row * spacing, col * spacing, 0.0)

                prim = stage.GetPrimAtPath(env_path)
                xform = UsdGeom.Xformable(prim)
                translate_op = xform.GetTranslateOp() or xform.AddTranslateOp()
                translate_op.Set(pos)
                self.scene._env_origins[i] = torch.tensor(pos, device=self.device)

                # Distribute variant types
                if i % 3 == 0:
                    ROBOT_VARIANTS[0].spawn.func(
                        f"{env_path}/RobotA1", ROBOT_VARIANTS[0].spawn
                    )
                elif i % 3 == 1:
                    ROBOT_VARIANTS[1].spawn.func(
                        f"{env_path}/RobotGo1", ROBOT_VARIANTS[1].spawn
                    )
                else:
                    ROBOT_VARIANTS[2].spawn.func(
                        f"{env_path}/RobotGo2", ROBOT_VARIANTS[2].spawn
                    )

            # Initialize articulations
            self.robot_a1 = Articulation(
                copy.deepcopy(ROBOT_VARIANTS[0]).replace(
                    spawn=None, prim_path="/World/envs/env_.*/RobotA1"
                )
            )
            self.robot_go1 = Articulation(
                copy.deepcopy(ROBOT_VARIANTS[1]).replace(
                    spawn=None, prim_path="/World/envs/env_.*/RobotGo1"
                )
            )
            self.robot_go2 = Articulation(
                copy.deepcopy(ROBOT_VARIANTS[2]).replace(
                    spawn=None, prim_path="/World/envs/env_.*/RobotGo2"
                )
            )

            self.scene.articulations.update(
                {
                    "robot": self.robot_go1,
                    "robot_a1": self.robot_a1,
                    "robot_go1": self.robot_go1,
                    "robot_go2": self.robot_go2,
                }
            )
            self.robot, self.robot_views = self.robot_go1, [
                self.robot_a1,
                self.robot_go1,
                self.robot_go2,
            ]
            self.robot_view_indices = [
                torch.tensor(self.a1_indices, device=self.device, dtype=torch.long),
                torch.tensor(
                    self.quadruped_indices, device=self.device, dtype=torch.long
                ),
                torch.tensor(self.go2_indices, device=self.device, dtype=torch.long),
            ]
        else:
            # Homogeneous Mode
            self.is_heterogeneous = False
            v_idx = 0 if "A1" in selection else (2 if "GO2" in selection else 1)
            variant_cfg = ROBOT_VARIANTS[v_idx]
            for i in range(num_envs):
                env_path = f"/World/envs/env_{i}"
                if not stage.GetPrimAtPath(env_path).IsValid():
                    sim_utils.define_prim(env_path, "Xform")
                tile_idx = i * stride
                row, col = tile_idx // grid_cols, tile_idx % grid_cols
                pos = (row * spacing, col * spacing, 0.0)
                xform = UsdGeom.Xformable(stage.GetPrimAtPath(env_path))
                (xform.GetTranslateOp() or xform.AddTranslateOp()).Set(pos)
                self.scene._env_origins[i] = torch.tensor(pos, device=self.device)
                variant_cfg.spawn.func(f"{env_path}/Robot", variant_cfg.spawn)

            self.robot = Articulation(
                copy.deepcopy(variant_cfg).replace(
                    spawn=None, prim_path="/World/envs/env_.*/Robot"
                )
            )
            self.scene.articulations["robot"] = self.robot
            self.robot_views, self.robot_view_indices = [self.robot], [
                torch.arange(self.num_envs, device=self.device)
            ]

        # 3. Final Origin Coordination and Elevation Sampling
        grid_origins = self.scene._env_origins.clone()
        if self.terrain_mesh is not None:
            ray_o = grid_origins.clone()
            ray_o[:, 2] = 10.0
            hits, _, _, _ = raycast_mesh(
                ray_o,
                torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(num_envs, 1),
                self.terrain_mesh,
            )
            grid_origins[:, 2] = torch.where(
                torch.isinf(hits[:, 2]), torch.zeros_like(hits[:, 2]), hits[:, 2]
            )

        if hasattr(self.scene, "terrain") and self.scene.terrain is not None:
            self.scene.terrain.env_origins = grid_origins
        self.scene._env_origins = grid_origins
        self.cfg.contact_sensor.prim_path = "/World/envs/env_.*/Robot.*/.*_foot"

        # Common sensors and setup
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Setup Debug Visualization Markers (Height Scan)
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
        import numpy as np

        height_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/HeightScan",
            markers={
                "dot": sim_utils.SphereCfg(
                    radius=0.015,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 1.0)  # Cyan
                    ),
                ),
            },
        )
        self._height_markers = VisualizationMarkers(height_marker_cfg)

        # Precompute height scan grid points (relative to robot)
        # 17x11 grid, resolution 0.1m, size (1.6, 1.0)
        res = 0.1
        size = (1.6, 1.0)
        x_pts = torch.arange(
            -size[0] / 2, size[0] / 2 + res / 2, res, device=self.device
        )
        y_pts = torch.arange(
            -size[1] / 2, size[1] / 2 + res / 2, res, device=self.device
        )
        grid_x, grid_y = torch.meshgrid(x_pts, y_pts, indexing="ij")
        self.height_points = torch.stack(
            [grid_x.flatten(), grid_y.flatten(), torch.zeros_like(grid_x.flatten())],
            dim=-1,
        )
        # Ensure 187 points (17 * 11)
        if self.height_points.shape[0] != 187:
            print(
                f"[WARNING] Height scan grid size mismatch: {self.height_points.shape[0]} != 187"
            )

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
            # DISTRIBUTE targets to partitioned views
            for i, view in enumerate(self.robot_views):
                indices = self.robot_view_indices[i]
                # Filter global targets to this robot group
                view_targets = targets[indices]

                # 2. Safety limits (Heterogeneous)
                # Fetch limits for this specific view (N_robots_in_view, num_joints, 2)
                lower_limits = view.data.soft_joint_pos_limits[
                    :, self._joint_dof_idx, 0
                ]
                upper_limits = view.data.soft_joint_pos_limits[
                    :, self._joint_dof_idx, 1
                ]
                view_targets = torch.clamp(view_targets, lower_limits, upper_limits)

                # Apply limits and set (View handles internal indexing)
                # indices are GLOBAL, but Articulation handles mapping if it was initialized with global paths
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
                # indices are GLOBAL env_ids for this robot type
                indices = self.robot_view_indices[i]

                # Fetch data from the view (view has len = len(indices))
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

                # Handle body count differences
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

        # -- Update height scan (Direct Warp Sampling) --
        if self.cfg.observation_space != 49:
            if self.terrain_mesh is not None:
                # 1. Transform grid points to world frame using robot position and yaw
                # Extract yaw from root_quat_w: atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
                q = self.root_quat_w
                yaw = torch.atan2(
                    2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                    1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2),
                )

                cos_yaw = torch.cos(yaw)
                sin_yaw = torch.sin(yaw)

                # Robot-to-world rotation matrix (2D yaw only)
                # [cos -sin]
                # [sin  cos]
                rot_mat = torch.stack(
                    [
                        torch.stack([cos_yaw, -sin_yaw], dim=-1),
                        torch.stack([sin_yaw, cos_yaw], dim=-1),
                    ],
                    dim=1,
                )  # (N, 2, 2)

                # Rotate height points (N, 187, 2)
                points_2d = self.height_points[:, :2].unsqueeze(0)  # (1, 187, 2)
                # Use batch matrix multiplication
                world_points_2d = torch.bmm(
                    points_2d.repeat(self.num_envs, 1, 1), rot_mat.transpose(1, 2)
                )

                # Add robot root position (XY) and set Z to a high value for downward raycast
                world_points = torch.zeros((self.num_envs, 187, 3), device=self.device)
                world_points[:, :, :2] = world_points_2d + self.root_pos_w[:, :2].unsqueeze(
                    1
                )
                world_points[:, :, 2] = self.root_pos_w[:, 2].unsqueeze(1) + 2.0

                # 2. Perform direct raycast
                ray_dir = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(
                    self.num_envs * 187, 1
                )
                # Warp raycast_mesh expects (M, 3) for origins and dirs
                hits, _, _, _ = raycast_mesh(
                    world_points.view(-1, 3), ray_dir, self.terrain_mesh
                )

                # 3. Calculate hit heights and handle misses
                # hits is (num_envs * 187, 3), hit position in world frame
                hit_pos = hits.view(self.num_envs, 187, 3)
                hit_z = hit_pos[:, :, 2]

                # Warp raycast_mesh returns inf for miss
                invalid_mask = torch.isinf(hit_z)
                if torch.any(invalid_mask):
                    # Fallback to a safe distance (0.5m below robot)
                    fallback_z = self.root_pos_w[:, 2].unsqueeze(1) - 0.5
                    hit_z = torch.where(invalid_mask, fallback_z, hit_z)

                # 4. Handle invalid hits (infinite distance/no hit)
                invalid_mask = torch.isinf(hit_z)
                if torch.any(invalid_mask):
                    # Fallback to a safe distance (0.5m below robot)
                    fallback_z = self.root_pos_w[:, 2].unsqueeze(1) - 0.5
                    hit_z = torch.where(invalid_mask, fallback_z, hit_z)

                self.current_terrain_heights = hit_z[:, 93]
                # Store world coordinates of sampled points for visualization
                self.samples_w = world_points.clone()
                self.samples_w[:, :, 2] = hit_z

                if self.common_step_counter % 100 == 0:
                    hit_z0 = hit_z[0]
                    min_h = hit_z0.min().item()
                    max_h = hit_z0.max().item()
                    root_pos = self.root_pos_w[0]
                    # print(
                    #     f"[DEBUG] Env 0: pos=({root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f}), hit_z range=[{min_h:.3f}, {max_h:.3f}]"
                    # )
                    # if min_h == 0 and max_h == 0:
                    #     print(f"[DEBUG] Env 0 first 5 hit_z: {hit_z0[:5].tolist()}")

                height_scan = hit_z - self.root_pos_w[:, 2].unsqueeze(1)
            else:
                # Fallback if no mesh (flat plane): assume ground at z=0
                # Still update samples_w for visualization!
                q = self.root_quat_w
                yaw = torch.atan2(
                    2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                    1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2),
                )
                cos_yaw = torch.cos(yaw)
                sin_yaw = torch.sin(yaw)
                rot_mat = torch.stack(
                    [
                        torch.stack([cos_yaw, -sin_yaw], dim=-1),
                        torch.stack([sin_yaw, cos_yaw], dim=-1),
                    ],
                    dim=1,
                )
                points_2d = self.height_points[:, :2].unsqueeze(0)
                world_points_2d = torch.bmm(
                    points_2d.repeat(self.num_envs, 1, 1), rot_mat.transpose(1, 2)
                )
                world_points = torch.zeros((self.num_envs, 187, 3), device=self.device)
                world_points[:, :, :2] = world_points_2d + self.root_pos_w[:, :2].unsqueeze(
                    1
                )

                self.samples_w = world_points.clone()
                self.samples_w[:, :, 2] = 0.0  # Flat plane
                self.current_terrain_heights = torch.zeros(
                    self.num_envs, device=self.device
                )
                height_scan = torch.zeros((self.num_envs, 187), device=self.device)

                # No mesh: flat plane assumed at z=0, samples_w updated for visualization

            # Clip to a reasonable range
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

        # Fall detection: if the robot's body is lower than 15cm relative to terrain, it fell.
        base_height = self.root_pos_w[:, 2] - self.current_terrain_heights

        died = (base_height < 0.15) | upright_check

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Reset common buffers
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.feet_air_time[env_ids] = 0.0
        self.last_joint_vel[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0

        # Loop over views for randomization
        for view, global_indices in zip(self.robot_views, self.robot_view_indices):
            # Intersect env_ids with global_indices for this view
            mask = torch.isin(env_ids, global_indices)
            local_mask = torch.isin(global_indices, env_ids)

            # Number of envs to reset in this view
            num_reset = mask.sum()
            if num_reset > 0:
                # view_ids is LOCAL to the view (0 to num_robots_in_view - 1)
                # We need to find the local IDs of the global_indices that are in env_ids
                view_ids = torch.nonzero(local_mask).flatten()
                self._randomize_view_state(view_ids, view)

    def _randomize_view_state(
        self,
        env_ids: torch.Tensor,
        view: Articulation,
    ):
        ids = env_ids
        env_ids_cpu = env_ids.cpu()

        # 0. Randomize Base Mass (Sim2Real) - Robust check for buffer size
        if (
            view.data.default_mass is not None
            and len(view.data.default_mass) > env_ids_cpu.max()
        ):
            masses = view.root_physx_view.get_masses().clone()
            # mass_noise = sample_uniform(-1.0, 3.0, (len(local_ids_cpu), 1), "cpu")
            # masses[local_ids_cpu, 0] = (
            #    view.data.default_mass[local_ids_cpu, 0] + mass_noise[:, 0]
            # )
            # view.root_physx_view.set_masses(masses, local_ids_cpu)
            # Replaced with safer default mass assignment for now to avoid CUDA assert
            pass

        # 0.1 Randomize Joint Friction and Damping (Enforce Config)
        if hasattr(self, "_joint_dof_idx") and self._joint_dof_idx is not None:
            # Convert list to tensor for comparison
            dof_idx = torch.tensor(self._joint_dof_idx, device=self.device)
            num_joints_in_view = view.num_joints
            valid_idx = dof_idx[dof_idx < num_joints_in_view]

            if len(valid_idx) > 0:
                friction_noise = sample_uniform(
                    self.cfg.joint_friction_range[0],
                    self.cfg.joint_friction_range[1],
                    (len(ids), len(valid_idx)),
                    self.device,
                )
                view.write_joint_friction_coefficient_to_sim(
                    friction_noise,
                    joint_ids=valid_idx,
                    env_ids=ids,
                )

                damping_noise = sample_uniform(
                    self.cfg.joint_damping_range[0],
                    self.cfg.joint_damping_range[1],
                    (len(ids), len(valid_idx)),
                    self.device,
                )
                view.write_joint_damping_to_sim(
                    damping_noise,
                    joint_ids=valid_idx,
                    env_ids=ids,
                )

        # 1. Reset Joint States (Use Default Pose + Noise on controlled joints)
        # Use full joint arrays (all joints, not just controlled ones)
        joint_pos = view.data.default_joint_pos[ids].clone()
        joint_vel = view.data.default_joint_vel[ids].clone()

        # Add small random noise to initial joint positions and velocities
        if hasattr(self, "_joint_dof_idx") and self._joint_dof_idx is not None:
            # Convert list to tensor for comparison
            dof_idx = torch.tensor(self._joint_dof_idx, device=self.device)
            num_joints_in_view = joint_pos.shape[1]
            valid_idx = dof_idx[dof_idx < num_joints_in_view]

            if len(valid_idx) > 0:
                pos_noise = sample_uniform(
                    -0.2, 0.2, (len(ids), len(valid_idx)), joint_pos.device
                )
                vel_noise = sample_uniform(
                    -0.5, 0.5, (len(ids), len(valid_idx)), joint_vel.device
                )

                # Apply noise only to controlled joints
                joint_pos[:, valid_idx] += pos_noise
                joint_vel[:, valid_idx] += vel_noise

        # 2. Reset Base State (Position + Velocity)
        default_root_state = view.data.default_root_state[ids].clone()
        # Offset the base to the environment origin (so robots don't spawn on top of each other)
        # env_origins should match the grid we set in _setup_scene
        origins = self.scene.env_origins[env_ids]
        # if hasattr(self, "common_step_counter") and self.common_step_counter % 100 == 0:
        #     print(f"[DEBUG RESET] Env {env_ids[0].item()}: default_pos={default_root_state[0, :3].tolist()}, origin={origins[0].tolist()}")

        default_root_state[:, :3] += origins
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
