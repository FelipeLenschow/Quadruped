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
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns as sensor_patterns
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from .quadruped_mdp import push_robot_heterogeneous
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
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
        border_width=0.0,
        num_rows=45,
        num_cols=45,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.5,
                step_height_range=(0.1, 0.2),
                step_width=0.3,
                platform_width=0.1,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.5,
                step_height_range=(0.1, 0.2),
                step_width=0.3,
                platform_width=0.1,
            ),
        },
    ),
    max_init_terrain_level=0,  # Start at easiest level
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
        num_rows=45,
        num_cols=45,
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
ROBOT_VARIANTS[0].prim_path = "/World/envs/env_.*/RobotA1"
ROBOT_VARIANTS[1].prim_path = "/World/envs/env_.*/RobotGo1"
ROBOT_VARIANTS[2].prim_path = "/World/envs/env_.*/RobotGo2"


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
    debug_vis = True  # Enable by default for perception feedback

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        device="cuda:0",
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_contact_count=2**22,
            gpu_max_rigid_patch_count=2**18,
            gpu_found_lost_pairs_capacity=2**22,
            gpu_found_lost_aggregate_pairs_capacity=2**22,
            gpu_total_aggregate_pairs_capacity=2**22,
            gpu_max_soft_body_contacts=2**11,
            gpu_max_particle_contacts=2**11,
            gpu_heap_capacity=2**27,  # 256MB - More reliable allocation
            gpu_temp_buffer_capacity=2**25,  # 64MB
            gpu_max_num_partitions=7,
        ),
    )

    # robot(s)
    spawn_height = 0.40

    # Use A1 as the base for the articulation view data structure
    # This is because A1 has the minimum set of rigid bodies (no head/handle),
    # which ensures the view matches a consistent set of bodies across all variants.
    robot: ArticulationCfg = UNITREE_A1_CFG.copy()
    robot.prim_path = "/World/envs/env_.*/Robot"
    robot.actuators = UNITREE_QUADRUPED_CFG.actuators.copy()

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=2.5, replicate_physics=False
    )
    scene.terrain = (
        TC_ROUGH if _ter == "rough" else (TC_FLAT if _ter == "flat" else TC_ALL)
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot.*/.*_foot",  # General regex to cover RobotA1, RobotGo1, etc.
        history_length=3,
        track_air_time=False,
    )

    # observation space stays 236: 49 (state) + 187 (height scan: 17x11)

    # sim2real noise
    observation_noise_scale = 0.05

    # actuator randomization
    joint_friction_range = (0.03, 0.3)
    joint_damping_range = (0.01, 0.1)

    @configclass
    class EventCfg:
        """Configuration for events."""

        # Events (MDP terms)
        # Target the unified 'robot' entity which always exists regardless of robot selection
        # DISABLED: push_robot term is temporarily disabled for heterogeneous modes
        # to avoid CUDA index assert caused by multi-view partitioning.
        # push_robot = EventTerm(...)

    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        # scale down the terrains because the robot is small
        if (
            hasattr(self.scene, "terrain")
            and self.scene.terrain.terrain_generator is not None
        ):
            subs = self.scene.terrain.terrain_generator.sub_terrains
            if "random_rough" in subs:
                subs["random_rough"].noise_range = (0.01, 0.06)
                subs["random_rough"].noise_step = 0.01

    # rewards
    rew_scale_alive = 0.0  # Force tracking points (disabled Starfish farm)
    # New Locomotion Rewards
    rew_scale_track_lin_vel_xy_exp = 5.0
    rew_scale_track_ang_vel_z_exp = 0.75
    rew_scale_lin_vel_z_l2 = -2.0
    rew_scale_ang_vel_xy_l2 = -0.05
    rew_scale_dof_pos_l2 = (
        -0.05
    )  # Reduced penalty to allow higher leg lifting on stairs
    rew_scale_dof_torques_l2 = -0.0002
    rew_scale_dof_acc_l2 = -2.5e-7
    rew_scale_action_rate_l2 = -0.01
    rew_scale_feet_air_time = 0.01
    rew_scale_flat_orientation_l2 = -2.0
    rew_scale_foot_height_exp = 0.0
    rew_scale_feet_air_penalty = -0.05  # General penalty for foot in air
    rew_scale_feet_air_penalty_static = -1.0  # Extra penalty when standing still
    rew_scale_joint_vel_l2_static = -0.01  # Penalty for moving joints when stationary
    target_foot_height = 0.1

    # Command parameters
    command_lin_vel_std = 1.0
    command_ang_vel_std = 0.5

    # - Command ranges (x, y, yaw)
    command_x_range = (-1.0, 1.0)
    command_y_range = (-1.0, 1.0)
    command_yaw_range = (-1.0, 1.0)
    command_resampling_time = 10.0  # seconds

    # termination
    base_angle_termination_thresh = (
        0.5  # Cosine of angle between base z-axis and world z-axis (~60 degrees)
    )
