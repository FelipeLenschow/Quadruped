# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Quadruped Locomotion Environment Configuration
# ================================================
# Based on IsaacLab Go2 velocity-tracking reference:
#   - IsaacLab/source/isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py
#   - IsaacLab/source/isaaclab_tasks/.../locomotion/velocity/config/go2/rough_env_cfg.py
#   - IsaacLab/source/isaaclab_tasks/.../locomotion/velocity/config/go2/flat_env_cfg.py
#
# Robot: Unitree A1 body + DCMotorCfg (PD controller, Kp=25, Kd=0.5)
# Actuator choice: DCMotorCfg instead of ActuatorNetMLP (Go1) because:
#   - Allows PD gain randomization via write_joint_stiffness/damping_to_sim
#   - Matches our real-robot PD control loop (Kp/Kd are meaningful)
#   - ActuatorNetMLP bypasses PhysX PD entirely (stiffness writes are ignored)

import os
import yaml

# Load training phase configuration
_raw_phase_name = os.environ.get("QUADRUPED_TRAINING_PHASE", "phase1")
_is_sequence = "_to_" in _raw_phase_name or _raw_phase_name.endswith("_onward")
if "_to_" in _raw_phase_name:
    _phase_name, _end_phase = _raw_phase_name.split("_to_")
else:
    _phase_name = _raw_phase_name.replace("_onward", "") if _is_sequence else _raw_phase_name
    _end_phase = None

_yaml_path = os.path.join(os.path.dirname(__file__), "training_phases.yaml")
with open(_yaml_path, "r") as f:
    _all_phases = yaml.safe_load(f)

if _phase_name not in _all_phases.get("phases", {}):
    raise ValueError(f"Training phase '{_phase_name}' not found in training_phases.yaml")

def resolve_phase(all_phases, phase_name):
    import collections.abc
    import copy
    
    def deep_update(d, u):
        if not isinstance(d, collections.abc.Mapping):
            d = {}
        if not isinstance(u, collections.abc.Mapping):
            return d
        for k, v in u.items():
            if v is None:
                continue
            if isinstance(v, collections.abc.Mapping):
                target = d.get(k, {})
                if not isinstance(target, collections.abc.Mapping):
                    target = {}
                d[k] = deep_update(target, v)
            else:
                d[k] = v
        return d

    phase_node = all_phases["phases"].get(phase_name, {})
    parent_name = phase_node.get("inherits", "default")
    
    if parent_name and parent_name != phase_name:
        if parent_name == "default":
            parent_cfg = all_phases.get("default", {})
        else:
            parent_cfg = resolve_phase(all_phases, parent_name)
    else:
        parent_cfg = all_phases.get("default", {}) # Ultimate fallback

    return deep_update(copy.deepcopy(parent_cfg), phase_node)

_phase_cfg = resolve_phase(_all_phases, _phase_name)

_curriculum_phases = []
if _is_sequence:
    phase_keys = list(_all_phases["phases"].keys())
    if _phase_name in phase_keys:
        start_idx = phase_keys.index(_phase_name)
        end_idx = phase_keys.index(_end_phase) if _end_phase and _end_phase in phase_keys else len(phase_keys) - 1
        for k in phase_keys[start_idx+1:end_idx+1]:
            cfg = resolve_phase(_all_phases, k)
            _curriculum_phases.append({
                "name": k,
                "cfg": cfg,
                "max_timesteps": cfg["env"].get("max_timesteps", 500000)
            })

_vel_range = _phase_cfg["events"]["push_velocity_range"] if _phase_cfg["events"]["enable_pushes"] else None

from isaaclab_assets.robots.unitree import (
    UNITREE_A1_CFG,
    UNITREE_GO1_CFG as UNITREE_QUADRUPED_CFG,
    UNITREE_GO2_CFG,
)
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns as sensor_patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp

from .quadruped_mdp import push_robot_heterogeneous


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TERRAIN PRESETS                                                            ║
# ║  Select via env var: QUADRUPED_TERRAIN=flat|rough|all (default: rough)      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_TERRAIN_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    static_friction=1.0,
    dynamic_friction=1.0,
)
_TERRAIN_VISUAL_MATERIAL = sim_utils.MdlFileCfg(
    mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
    project_uvw=True,
    texture_scale=(0.25, 0.25),
)


TC_FLAT = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    collision_group=-1,
    physics_material=_TERRAIN_PHYSICS_MATERIAL,
    visual_material=_TERRAIN_VISUAL_MATERIAL,
    debug_vis=False,
)

TC_ROUGH = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=0.0,
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
    physics_material=_TERRAIN_PHYSICS_MATERIAL,
    visual_material=_TERRAIN_VISUAL_MATERIAL,
    debug_vis=False,
)

TC_ALL = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=0.0,
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
    physics_material=_TERRAIN_PHYSICS_MATERIAL,
    visual_material=_TERRAIN_VISUAL_MATERIAL,
    debug_vis=False,
)
# Terrain selection will happen after YAML parsing


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ROBOT VARIANTS (for heterogeneous multi-robot training)                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

ROBOT_VARIANTS: list[ArticulationCfg] = [
    UNITREE_A1_CFG.copy(),         # index 0
    UNITREE_QUADRUPED_CFG.copy(),  # index 1 (Go1)
    UNITREE_GO2_CFG.copy(),        # index 2
]
for variant in ROBOT_VARIANTS:
    variant.prim_path = "/World/envs/env_.*/Robot"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ENVIRONMENT CONFIGURATION                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@configclass
class QuadrupedEnvCfg(DirectRLEnvCfg):

    _yaml_rob = _phase_cfg["env"].get("robot_cfg", "")
    _rob = _yaml_rob.upper() if _yaml_rob else os.environ.get("QUADRUPED_ROBOT", os.environ.get("FORCE_ROBOT", "RANDOM")).upper()
    robot_choice: str = _rob

    # ── Simulation ────────────────────────────────────────────────────────────
    decimation = 4
    episode_length_s = 20.0
    sim: SimulationCfg = SimulationCfg(
        dt=0.005, 
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            gpu_max_rigid_contact_count=2**24,       # ~16.7M contacts
            gpu_max_rigid_patch_count=2**23,         # ~8.3M patches
            gpu_found_lost_pairs_capacity=2**24,
            gpu_found_lost_aggregate_pairs_capacity=2**24,
            gpu_max_soft_body_contacts=1048576,
            gpu_max_particle_contacts=1048576,
            gpu_heap_capacity=33554432 * 2,
            gpu_temp_buffer_capacity=16777216 * 2,
            gpu_max_num_partitions=8
        )
    )

    # ── Observation / Action spaces ───────────────────────────────────────────
    observation_space = int(os.environ.get("QUADRUPED_OBS_DIM", 51))
    # obs = [lin_vel(3) + ang_vel(3) + gravity(3) + cmd(4) + jpos(12) + jvel(12) + actions(12) + gait_clock(2)] = 51
    action_space = 12
    state_space = 0
    action_scale = _phase_cfg["env"]["action_scale"]
    
    base_max_timesteps = _phase_cfg["env"].get("max_timesteps", 500000)
    _total_timesteps = base_max_timesteps
    for p in _curriculum_phases:
        _total_timesteps += p["cfg"]["env"].get("max_timesteps", 500000)
    max_timesteps = _total_timesteps

    curriculum_phases = _curriculum_phases

    # ── Robot ─────────────────────────────────────────────────────────────────
    # A1 body (minimum rigid bodies → compatible view across all variants)
    # with DCMotorCfg actuators: Kp=25 Nm/rad, Kd=0.5 Nm·s/rad, τ_max=33.5 Nm
    robot: ArticulationCfg = UNITREE_A1_CFG.copy()
    robot.prim_path = "/World/envs/env_.*/Robot"
    robot.actuators = UNITREE_QUADRUPED_CFG.actuators.copy()
    spawn_height = 0.50

    # ── Scene ─────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=_phase_cfg["env"]["num_envs"], 
        env_spacing=2.5, 
        replicate_physics=(_rob != "RANDOM"),
    )
    _ter = os.environ.get("QUADRUPED_TERRAIN", _phase_cfg["env"].get("terrain", "rough"))
    scene.terrain = (
        TC_ROUGH if _ter == "rough" else (TC_FLAT if _ter == "flat" else TC_ALL)
    )

    # ── Sensors ───────────────────────────────────────────────────────────────
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/(.*_foot|.*_calf|.*_thigh)",
        history_length=3,
        track_air_time=False,
    )


    # ── Observation noise (sim2real) ──────────────────────────────────────────
    observation_noise_scale = _phase_cfg["env"]["observation_noise_scale"]

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  DOMAIN RANDOMIZATION                                                 ║
    # ╚════════════════════════════════════════════════════════════════════════╝

    # Base mass and Center of Mass (CoM) Randomization
    payload_mass_range = tuple(_phase_cfg["domain_randomization"].get("payload_mass_range", [-1.0, 3.0]))
    com_displacement_range = tuple(_phase_cfg["domain_randomization"].get("com_displacement_range", [0.0, 0.0]))

    # Joint friction — viscous drag
    joint_friction_range = tuple(_phase_cfg["domain_randomization"]["joint_friction_range"])

    # PD gains
    joint_stiffness_range = tuple(_phase_cfg["domain_randomization"]["joint_stiffness_range"])
    joint_pd_damping_range = tuple(_phase_cfg["domain_randomization"]["joint_pd_damping_range"])

    # Action latency
    action_latency_range_steps = tuple(_phase_cfg["domain_randomization"]["action_latency_range_steps"])

    # Motor backlash
    motor_backlash_range = tuple(_phase_cfg["domain_randomization"]["motor_backlash_range"])

    # ── Events (pushes, external forces) ──────────────────────────────────────
    @configclass
    class EventCfg:
        """Push events configured dynamically via YAML."""
        push_a1 = EventTerm(
            func=push_robot_heterogeneous,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "asset_cfg": SceneEntityCfg("robot_a1"),
                "velocity_range": {"x": (_vel_range[0], _vel_range[1]), "y": (_vel_range[0], _vel_range[1])},
            },
        ) if _vel_range else None
        
        push_quadruped = EventTerm(
            func=push_robot_heterogeneous,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "asset_cfg": SceneEntityCfg("robot_quadruped"),
                "velocity_range": {"x": (_vel_range[0], _vel_range[1]), "y": (_vel_range[0], _vel_range[1])},
            },
        ) if _vel_range else None
        
        push_go2 = EventTerm(
            func=push_robot_heterogeneous,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "asset_cfg": SceneEntityCfg("robot_go2"),
                "velocity_range": {"x": (_vel_range[0], _vel_range[1]), "y": (_vel_range[0], _vel_range[1])},
            },
        ) if _vel_range else None

    events: EventCfg = EventCfg()

    # ── Terrain post-init (scales noise for small robots) ─────────────────────
    def __post_init__(self):
        super().__post_init__()
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

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  REWARDS                                                              ║
    # ║  Based on Go2 reference, tuned for base stability + sim2real          ║
    # ╚════════════════════════════════════════════════════════════════════════╝
    #
    #   Reward                        Go2      Phase2   Phase3    Change
    #   ─────────────────────────────────────────────────────────────────────
    #   alive                         0.0      0.0      0.0
    #   track_lin_vel_xy_exp          1.0      1.5      1.0       ↓ back to Go2 (was chasing speed)
    #   track_ang_vel_z_exp           0.5      0.75     0.5       ↓ back to Go2
    #   feet_air_time                 0.25     0.25     0.125     ↓ less aggressive lifting
    #   foot_height_exp               —        0.5      0.3       ↓ was encouraging jumping
    #   flat_orientation_l2           -2.5     -2.5     -5.0      ↑ stronger body-level penalty
    #   lin_vel_z_l2                  -2.0     -2.0     -4.0      ↑ stop vertical bouncing
    #   ang_vel_xy_l2                 -0.05    -0.05    -0.5      ↑↑ 10× (main fix for rocking)
    #   dof_pos_l2                    0.0      0.0      -0.05     ↑ prevent wild joint configs
    #   dof_torques_l2                -0.0002  -0.0002  -0.0002
    #   dof_acc_l2                    -2.5e-7  -2.5e-7  -2.5e-7
    #   action_rate_l2                -0.01    -0.01    -0.03     ↑ smoother = less jerky

    # Alive and Undesired Contacts
    rew_scale_alive = _phase_cfg["rewards"]["rew_scale_alive"]
    rew_scale_undesired_contacts = _phase_cfg["rewards"]["rew_scale_undesired_contacts"]

    # Velocity tracking
    rew_scale_track_lin_vel_xy_exp = _phase_cfg["rewards"]["rew_scale_track_lin_vel_xy_exp"]
    rew_scale_track_ang_vel_z_exp = _phase_cfg["rewards"]["rew_scale_track_ang_vel_z_exp"]
    rew_scale_pos_deviation_l1 = _phase_cfg["rewards"].get("rew_scale_pos_deviation_l1", 0.0)
    rew_scale_stall = _phase_cfg["rewards"].get("rew_scale_stall", 0.0)
    max_pos_leash = _phase_cfg["rewards"].get("max_pos_leash", 0.4)
    rew_scale_gait_phase = _phase_cfg["rewards"].get("rew_scale_gait_phase", 0.0)
    gait_stride_length = _phase_cfg["rewards"].get("gait_stride_length", 0.12)
    gait_swing_sigma = _phase_cfg["rewards"].get("gait_swing_sigma", 0.1)

    # Gait shaping
    rew_scale_feet_air_time = _phase_cfg["rewards"]["rew_scale_feet_air_time"]
    target_feet_air_time = _phase_cfg["rewards"]["target_feet_air_time"]
    rew_scale_foot_height_exp = _phase_cfg["rewards"]["rew_scale_foot_height_exp"]
    target_foot_height = _phase_cfg["rewards"]["target_foot_height"]
    rew_scale_trot_symmetry = _phase_cfg["rewards"]["rew_scale_trot_symmetry"]
    hip_sym_multiplier = _phase_cfg["rewards"]["hip_sym_multiplier"]
    rew_scale_torque_symmetry = _phase_cfg["rewards"]["rew_scale_torque_symmetry"]
    rew_scale_grf_balance = _phase_cfg["rewards"]["rew_scale_grf_balance"]
    rew_scale_grf_target = _phase_cfg["rewards"].get("rew_scale_grf_target", 0.0)
    rew_scale_max_contact_force = _phase_cfg["rewards"].get("rew_scale_max_contact_force", 0.0)
    max_contact_force_pct = _phase_cfg["rewards"].get("max_contact_force_pct", 0.75)
    rew_scale_max_air_feet = _phase_cfg["rewards"]["rew_scale_max_air_feet"]

    # Stability penalties
    rew_scale_flat_orientation_l2 = _phase_cfg["rewards"]["rew_scale_flat_orientation_l2"]
    rew_scale_lin_vel_z_l2 = _phase_cfg["rewards"]["rew_scale_lin_vel_z_l2"]
    rew_scale_ang_vel_xy_l2 = _phase_cfg["rewards"]["rew_scale_ang_vel_xy_l2"]
    rew_scale_dof_pos_l2 = _phase_cfg["rewards"]["rew_scale_dof_pos_l2"]

    # Smoothness / efficiency
    rew_scale_dof_torques_l2 = _phase_cfg["rewards"]["rew_scale_dof_torques_l2"]
    rew_scale_dof_acc_l2 = _phase_cfg["rewards"]["rew_scale_dof_acc_l2"]
    rew_scale_action_rate_l2 = _phase_cfg["rewards"]["rew_scale_action_rate_l2"]

    # Disabled (kept for interface compatibility)
    rew_scale_feet_air_penalty = _phase_cfg["rewards"]["rew_scale_feet_air_penalty"]
    rew_scale_feet_air_penalty_static = _phase_cfg["rewards"]["rew_scale_feet_air_penalty_static"]
    rew_scale_joint_vel_l2_static = _phase_cfg["rewards"]["rew_scale_joint_vel_l2_static"]
    rew_scale_base_height_l2 = _phase_cfg["rewards"]["rew_scale_base_height_l2"]
    target_base_height = _phase_cfg["rewards"]["target_base_height"]

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  COMMANDS                                                             ║
    # ╚════════════════════════════════════════════════════════════════════════╝

    command_lin_vel_std = _phase_cfg["commands"]["command_lin_vel_std"]
    command_ang_vel_std = _phase_cfg["commands"]["command_ang_vel_std"]

    command_x_range = tuple(_phase_cfg["commands"]["command_x_range"])      # [m/s]
    command_y_range = tuple(_phase_cfg["commands"]["command_y_range"])      # [m/s]
    command_yaw_range = tuple(_phase_cfg["commands"]["command_yaw_range"])    # [rad/s]
    command_resampling_time = _phase_cfg["commands"]["command_resampling_time"] # [s]

    # Gait reward masking: feet_air_time only counted when ‖cmd‖ > this
    static_velocity_threshold = _phase_cfg["commands"]["static_velocity_threshold"]
    stall_velocity_threshold = _phase_cfg["commands"]["stall_velocity_threshold"]

    # Zero-command fraction and single-axis fractions
    zero_command_fraction = _phase_cfg["commands"]["zero_command_fraction"]
    standby_duration_s = _phase_cfg["commands"].get("standby_duration_s", 0.5)
    x_only_command_fraction = _phase_cfg["commands"].get("x_only_command_fraction", 0.0)
    y_only_command_fraction = _phase_cfg["commands"].get("y_only_command_fraction", 0.0)
    yaw_only_command_fraction = _phase_cfg["commands"].get("yaw_only_command_fraction", 0.0)

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  TERMINATION                                                          ║
    # ╚════════════════════════════════════════════════════════════════════════╝

    # Terminate if cos(tilt angle) > this (i.e. too tilted)
    base_angle_termination_thresh = 0.7
