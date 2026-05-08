# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from isaaclab_assets.robots.unitree import (
    UNITREE_A1_CFG,
    UNITREE_GO1_CFG as UNITREE_QUADRUPED_CFG,
    UNITREE_GO2_CFG,
)
from isaaclab_assets.robots.anymal import ANYMAL_B_CFG, ANYMAL_C_CFG, ANYMAL_D_CFG
from isaaclab_assets.robots.spot import SPOT_CFG
from isaaclab.sim.spawners.wrappers.wrappers_cfg import MultiUsdFileCfg
from isaaclab.actuators import DCMotorCfg

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns as sensor_patterns
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from .quadruped_mdp import push_robot_heterogeneous
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


_ter = os.environ.get("QUADRUPED_TERRAIN", "rough")

TC_FLAT = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
)
TC_ALL = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains=ROUGH_TERRAINS_CFG.sub_terrains,
    ),
    max_init_terrain_level=5,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
)
TC_ROUGH = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "random_rough": ROUGH_TERRAINS_CFG.sub_terrains["random_rough"],
        },
    ),
    max_init_terrain_level=5,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
)


# Robot variants for morphological randomization
ROBOT_VARIANTS: list[ArticulationCfg] = [
    UNITREE_A1_CFG.copy(),
    UNITREE_QUADRUPED_CFG.copy(),
    UNITREE_GO2_CFG.copy(),
]
# Set placeholder prim_paths for validation/consistency
for variant in ROBOT_VARIANTS:
    variant.prim_path = "/World/envs/env_.*/Robot"


@configclass
class QuadrupedEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    action_scale = 0.25
    # - spaces definition
    action_space = 12
    observation_space = int(os.environ.get("QUADRUPED_OBS_DIM", 236))  # Dynamic: 49 (state) or 236 (height scan)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=decimation)

    # robot(s)
    spawn_height = 0.50

    # Use A1 as the base for the articulation view data structure
    # This is because A1 has the minimum set of rigid bodies (no head/handle),
    # which ensures the view matches a consistent set of bodies across all variants.
    robot: ArticulationCfg = UNITREE_A1_CFG.copy()
    robot.prim_path = "/World/envs/env_.*/Robot"
    robot.actuators = UNITREE_QUADRUPED_CFG.actuators.copy()

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2000, env_spacing=2.5, replicate_physics=False
    )
    scene.terrain = (
        TC_ROUGH if _ter == "rough" else (TC_FLAT if _ter == "flat" else TC_ALL)
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_foot",
        history_length=3,
        track_air_time=False,
    )

    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=sensor_patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        debug_vis=False,
    )

    # sim2real noise
    observation_noise_scale = 0.05

    # actuator randomization
    joint_friction_range = (0.03, 0.3)
    joint_damping_range = (0.01, 0.1)

    @configclass
    class EventCfg:
        """Configuration for events."""

        push_a1 = EventTerm(
            func=push_robot_heterogeneous,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "asset_cfg": SceneEntityCfg("robot_a1"),
                "velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)},
            },
        )
        push_quadruped = EventTerm(
            func=push_robot_heterogeneous,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "asset_cfg": SceneEntityCfg("robot_quadruped"),
                "velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)},
            },
        )
        push_go2 = EventTerm(
            func=push_robot_heterogeneous,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "asset_cfg": SceneEntityCfg("robot_go2"),
                "velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)},
            },
        )

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        # scale down the terrains because the robot is small
        if (
            hasattr(self.scene, "terrain")
            and self.scene.terrain.terrain_generator is not None
        ):
            self.scene.terrain.terrain_generator.sub_terrains[
                "random_rough"
            ].noise_range = (0.01, 0.06)
            self.scene.terrain.terrain_generator.sub_terrains[
                "random_rough"
            ].noise_step = 0.01

    # rewards
    rew_scale_alive = 1.0  # Encourage stability
    # New Locomotion Rewards
    rew_scale_track_lin_vel_xy_exp = 1.5
    rew_scale_track_ang_vel_z_exp = 0.75
    rew_scale_lin_vel_z_l2 = -2.0
    rew_scale_ang_vel_xy_l2 = -0.05
    rew_scale_dof_pos_l2 = -0.2
    rew_scale_dof_torques_l2 = -0.0002
    rew_scale_dof_acc_l2 = -2.5e-7
    rew_scale_action_rate_l2 = -0.01
    rew_scale_feet_air_time = 0.01
    rew_scale_flat_orientation_l2 = -5.0
    rew_scale_foot_height_exp = 0.2
    rew_scale_feet_air_penalty = -0.05  # General penalty for foot in air
    rew_scale_feet_air_penalty_static = -5.0  # Extra penalty when standing still
    rew_scale_joint_vel_l2_static = -0.1  # Penalty for moving joints when stationary
    target_foot_height = 0.1

    # Command parameters
    command_lin_vel_std = 0.5
    command_ang_vel_std = 0.5

    # - Command ranges (x, y, yaw)
    command_x_range = (-1.0, 1.0)
    command_y_range = (-1.0, 1.0)
    command_yaw_range = (-1.0, 1.0)
    command_resampling_time = 10.0  # seconds

    # termination
    base_angle_termination_thresh = (
        0.7  # Cosine of angle between base z-axis and world z-axis
    )
