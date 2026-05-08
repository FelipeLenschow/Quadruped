import torch
import os

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from .quadruped_mdp import push_robot_heterogeneous
from .quadruped_curriculum import apply_curriculum, LYING_JOINT_POS
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, TiledCameraCfg
from isaaclab.sim import SimulationCfg


from isaaclab_assets.robots.unitree import UNITREE_A1_CFG, UNITREE_GO1_CFG, UNITREE_GO2_CFG

ROBOT_VARIANTS = {
    "UNITREE_A1_CFG": UNITREE_A1_CFG,
    "UNITREE_GO1_CFG": UNITREE_GO1_CFG,
    "UNITREE_GO2_CFG": UNITREE_GO2_CFG,
}

@configclass
class QuadrupedSceneCfg(InteractiveSceneCfg):
    """Configuration for the quadruped scene."""
    # ground terrain
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=0,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=10,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            sub_terrains={
                "random_rough": ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].replace(
                    proportion=0.4, noise_range=(0.01, 0.06), noise_step=0.01,
                ),
                "boxes": ROUGH_TERRAINS_CFG.sub_terrains["boxes"].replace(
                    proportion=0.2, grid_height_range=(0.025, 0.1),
                ),
                "hf_pyramid_slope": ROUGH_TERRAINS_CFG.sub_terrains["hf_pyramid_slope"].replace(
                    proportion=0.2, slope_range=(0.0, 0.3),
                ),
                "hf_pyramid_slope_inv": ROUGH_TERRAINS_CFG.sub_terrains["hf_pyramid_slope_inv"].replace(
                    proportion=0.2, slope_range=(0.0, 0.3),
                ),
            },
        ),
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = ROBOT_VARIANTS[os.environ.get("QUADRUPED_ROBOT", "UNITREE_GO2_CFG")].replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    # Strengthen physics solver for stability
    robot.spawn.articulation_props.solver_position_iteration_count = 8
    robot.spawn.articulation_props.solver_velocity_iteration_count = 4

    # sensors
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", # Track all bodies for contact penalties
        history_length=3,
        track_air_time=False,
    )

@configclass
class QuadrupedEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_space = 12
    action_scale = 0.25
    observation_space = 54 # Proprioceptive + Commands (Blind) + Contacts + Height
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.6,
            dynamic_friction=0.6,
            restitution=0.0,
        ),
    )

    # scene
    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # events
    events: EventTerm = {
        "push_go2": EventTerm(
            func=push_robot_heterogeneous,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}, # Overriden by curriculum
            },
        ),
    }


    # randomization
    random_pos_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
    random_roll_range = (0, 0)
    random_pitch_range = (0, 0)
    random_yaw_range = (-3.14, 3.14)
    random_height_range = (0, 0)
    spawn_height = 0.35
    observation_noise_scale = 0.00
    random_force_range = (0, 0)  # Newton
    random_force_duration_range = (0, 0)  # Seconds
    random_push_interval_range = (0, 0)  # Seconds
    
    # Missing ranges for randomization
    joint_friction_range = (0.0, 0.0)
    joint_damping_range = (0.0, 0.0)
    random_joint_pos_range = (0.0, 0.0)
    random_nn_delay_range = (0, 0)
    zero_command_prob = 0.25

    # Curriculum & Startup Settings
    startup_mode: str = "standing" # options: lying, standing, passive_drop, active_drop
    randomize_orientation: bool = False
    start_delay_s: float = 0.0
    lying_joint_pos: list[float] = LYING_JOINT_POS
    training_timesteps: int = 100000

    # randomization
    base_angle_termination_thresh = 0.7  # Cosine of angle between base z-axis and world z-axis (terminate if titled past ~45 deg)
    terminate_on_base_contact: bool = False

    def __post_init__(self):
        super().__post_init__()
        # Apply Curriculum Phase
        phase_idx = int(os.environ.get("TRAINING_PHASE", "10"))
        apply_curriculum(self, phase_idx)

    # ── Rewards (Managed by curriculum.py) ──────────────────────────
    # Values here are placeholders; they are overriden in __post_init__
    rew_scale_alive = 1.0
    rew_scale_track_lin_vel_xy_exp = 1.0
    rew_scale_track_ang_vel_z_exp = 1.0
    rew_scale_lin_vel_z_l2 = 0.0
    rew_scale_ang_vel_xy_l2 = 0.0
    rew_scale_dof_pos_l2 = 0.0
    rew_scale_dof_torques_l2 = 0.0
    rew_scale_dof_acc_l2 = 0.0
    rew_scale_action_rate_l2 = 0.0
    rew_scale_feet_air_time = 0.0
    rew_scale_flat_orientation_l2 = 0.0
    rew_scale_foot_height_exp = 0.0
    rew_scale_feet_air_penalty = 0.0
    rew_scale_feet_air_penalty_static = 0.0
    rew_scale_joint_vel_l2_static = 0.0
    rew_scale_base_height_exp = 0.0
    rew_scale_base_contact_penalty = 0.0
    target_base_height = 0.35
    target_foot_height = 0.1

    # Command parameters
    command_lin_vel_std = 0.5
    command_ang_vel_std = 0.5

    # - Command ranges (x, y, yaw)
    command_x_range = (-1.0, 1.0)
    command_y_range = (-1.0, 1.0)
    command_yaw_range = (-1.0, 1.0)
    command_resampling_time = 10.0  # seconds
