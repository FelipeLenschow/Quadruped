import math
import os

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from clock_scan.assets.robots.unitree import UNITREE_GO2_CFG
from clock_scan.assets.robots.unitree_actuators import GO2_KNEE_RATIO
from clock_scan.tasks.locomotion import mdp


def _env(name: str, default: float) -> float:
    return float(os.environ.get(f"CLOCK_SCAN_{name}", default))


def _range(name: str, lo: float, hi: float) -> tuple[float, float]:
    return (_env(f"{name}_LO", lo), _env(f"{name}_HI", hi))


def _gauss(std: float):
    return Gnoise(std=std) if std else None


# ── Height scan ────────────────────────────────────────────────────────────────────────────
# A 1.6 x 1.0 m grid at 0.1 m (17 x 11 = 187 rays) under the base, turned with its yaw. The actor gets
# it with uniform noise per ray, the critic without.
SCAN_SIZE = (_env("SCAN_LENGTH", 1.6), _env("SCAN_WIDTH", 1.0))
SCAN_RESOLUTION = _env("SCAN_RESOLUTION", 0.1)
SCAN_NOISE = _env("SCAN_NOISE", 0.05)

# ── Terrain ────────────────────────────────────────────────────────────────────────────────
ROUGH_NOISE = _env("ROUGH_NOISE", 0.04)
BOX_HEIGHT = _range("BOX", 0.02, 0.10)
STEP_HEIGHT = _range("STEP", 0.05, 0.20)
SHARE_FLAT = _env("SHARE_FLAT", 0.1)
SHARE_ROUGH = _env("SHARE_ROUGH", 0.15)
SHARE_BOXES = _env("SHARE_BOXES", 0.15)
SHARE_STAIRS_DOWN = _env("SHARE_STAIRS_DOWN", 0.3)
SHARE_STAIRS_UP = _env("SHARE_STAIRS_UP", 0.3)

# ── Commands ───────────────────────────────────────────────────────────────────────────────
MAX_SPEED = _env("MAX_SPEED", 1.0)
MAX_YAW = _env("MAX_YAW", 1.0)
RESAMPLE = _range("RESAMPLE", 4.0, 8.0)
PAUSE_FRACTION = _env("PAUSE_FRACTION", 0.3)
SLOW_FRACTION = _env("SLOW_FRACTION", 0.6)
SLOW_RANGE = _range("SLOW", 0.05, 0.3)
YAW_ONLY = _env("YAW_ONLY", 0.15)
STANDBY = _range("STANDBY", 1.0, 3.0)

# ── Trot clock ─────────────────────────────────────────────────────────────────────────────
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
STRIDE_MIN = _env("STRIDE_MIN", 0.10)
STRIDE_GAIN = _env("STRIDE_GAIN", 0.4)
SWING_TIME = _env("SWING_TIME", 0.3)
MAX_FREQUENCY = _env("MAX_FREQUENCY", 2.0)
SWING_HEIGHT = _env("SWING_HEIGHT", 0.08)

# ── Rewards ────────────────────────────────────────────────────────────────────────────────
LIN_SIGMA_EXP = _env("LIN_SIGMA_EXP", 1.0)
ANG_SIGMA_EXP = _env("ANG_SIGMA_EXP", 2.0)
ANG_STD = _env("ANG_STD", math.sqrt(0.25))
GRACE_S = _env("GRACE_S", 0.6)
GRACE_FLOOR = _env("GRACE_FLOOR", 0.25)
BASE_HEIGHT = _env("BASE_HEIGHT", 0.32)
W_FORCE = _env("W_FORCE", 2.0)
W_VEL = _env("W_VEL", 0.5)
W_SWING = _env("W_SWING", -150.0)
SWING_ABOVE_SCALE = _env("SWING_ABOVE_SCALE", 0.1)
W_LANDING = _env("W_LANDING", -2.0)
W_HEIGHT = _env("W_HEIGHT", -40.0)
W_TERMINATION = _env("W_TERMINATION", -1000.0)
W_STUMBLE = _env("W_STUMBLE", -2.0)
W_CONTACTS = _env("W_CONTACTS", -2.0)
W_HIP_DEV = _env("W_HIP_DEV", -0.2)

# ── Sensor model (actor) ───────────────────────────────────────────────────────────────────
LIN_VEL_NOISE = _env("LIN_VEL_NOISE", 0.125)
ANG_VEL_NOISE = _env("ANG_VEL_NOISE", 0.04)
LIN_VEL_BIAS = _env("LIN_VEL_BIAS", 0.0625)
ANG_VEL_BIAS = _env("ANG_VEL_BIAS", 0.025)
GRAVITY_BIAS = _env("GRAVITY_BIAS", 0.025)
JOINT_POS_BIAS = _env("JOINT_POS_BIAS", 0.019)

# ── Plant randomization ────────────────────────────────────────────────────────────────────
MIN_DELAY = int(_env("MIN_DELAY", 0))
MAX_DELAY = int(_env("MAX_DELAY", 5))
KP = _env("KP", 5.0)
KD = _env("KD", 0.2)
COM_X = _range("COM_X", -0.06, 0.10)
COM_Y = _env("COM_Y", 0.05)
COM_Z = _range("COM_Z", -0.01, 0.08)
ARMATURE = _range("ARMATURE", 0.005, 0.02)
VISCOUS = _range("VISCOUS", 0.0, 1.0)
JOINT_FRICTION = _range("JOINT_FRICTION", 0.03, 0.5)
FOOT_TORSION = _env("FOOT_TORSION", 0.02)
FOOT_TORSION_MIN = _env("FOOT_TORSION_MIN", 0.01)

# ── Starts ─────────────────────────────────────────────────────────────────────────────────
LYING_FRACTION = _env("LYING_FRACTION", 0.2)
DROP_FRACTION = _env("DROP_FRACTION", 0.2)
DROP_HEIGHT = _range("DROP_HEIGHT", 0.1, 0.3)
CONTACT_GRACE = _env("CONTACT_GRACE", STANDBY[1])
LYING_JOINT_POS = {
    ".*L_hip_joint": 0.1,
    ".*R_hip_joint": -0.1,
    ".*_thigh_joint": 1.3,
    ".*_calf_joint": -2.55,
}

_FEET_BODIES = SceneEntityCfg("robot", body_names=FEET, preserve_order=True)
_FEET_CONTACT = SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True)
_SCAN = SceneEntityCfg("height_scanner")


def _robot_cfg() -> ArticulationCfg:
    """Go2 with the calf's torque-speed curve scaled by the knee reduction, actuator delay and a
    torsional contact patch on every collider."""
    from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg

    legs = UNITREE_GO2_CFG.actuators["GO2HV"]

    def per_joint(value, calf_scale):
        return {".*_hip_joint": value, ".*_thigh_joint": value, ".*_calf_joint": value * calf_scale}

    return UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            "GO2HV": legs.replace(
                Y1=per_joint(legs.Y1, GO2_KNEE_RATIO),
                Y2=per_joint(legs.Y2, GO2_KNEE_RATIO),
                X1=per_joint(legs.X1, 1.0 / GO2_KNEE_RATIO),
                X2=per_joint(legs.X2, 1.0 / GO2_KNEE_RATIO),
                min_delay=MIN_DELAY,
                max_delay=MAX_DELAY,
            )
        },
        spawn=UNITREE_GO2_CFG.spawn.replace(
            make_uninstanceable=True,
            collision_props=PhysxCollisionPropertiesCfg(
                torsional_patch_radius=FOOT_TORSION,
                min_torsional_patch_radius=FOOT_TORSION_MIN,
            ),
        ),
    )


ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=SHARE_FLAT),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=SHARE_ROUGH, noise_range=(0.0, ROUGH_NOISE), noise_step=0.005, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=SHARE_BOXES, grid_width=0.45, grid_height_range=BOX_HEIGHT, platform_width=2.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=SHARE_STAIRS_DOWN, step_height_range=STEP_HEIGHT, step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=SHARE_STAIRS_UP, step_height_range=STEP_HEIGHT, step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=1,
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
    robot: ArticulationCfg = _robot_cfg()
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=SCAN_RESOLUTION, size=list(SCAN_SIZE)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": COM_X, "y": (-COM_Y, COM_Y), "z": COM_Z},
        },
    )
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": JOINT_FRICTION,
            "operation": "add",
        },
    )
    randomize_joint_plant = EventTerm(
        func=mdp.randomize_joint_plant,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_range": ARMATURE,
            "viscous_friction_range": VISCOUS,
        },
    )

    # reset, in this order: reset_start_pose overwrites the envs it picks after the nominal reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-1.0, 1.0)},
    )
    randomize_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (-KP, KP),
            "damping_distribution_params": (-KD, KD),
            "operation": "add",
        },
    )
    reset_start_pose = EventTerm(
        func=mdp.reset_start_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lying_fraction": LYING_FRACTION,
            "lying_joint_pos": LYING_JOINT_POS,
            "drop_fraction": DROP_FRACTION,
            "drop_height_range": DROP_HEIGHT,
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=RESAMPLE,
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-MAX_YAW, MAX_YAW)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-MAX_SPEED, MAX_SPEED),
            lin_vel_y=(-min(0.4, MAX_SPEED), min(0.4, MAX_SPEED)),
            ang_vel_z=(-MAX_YAW, MAX_YAW),
        ),
        slow_command_fraction=SLOW_FRACTION,
        slow_command_range=SLOW_RANGE,
        slow_after_full_range=True,
        yaw_only_command_fraction=YAW_ONLY,
        standby_duration_range=STANDBY,
        pause_fraction=PAUSE_FRACTION,
        max_foot_speed=MAX_SPEED,
    )
    # After base_velocity: the clock reads its command.
    clock = mdp.SpeedClockCommandCfg(
        foot_names=tuple(FEET),
        foot_offsets=mdp.GAIT_OFFSETS["trot"],
        stride_min=STRIDE_MIN,
        stride_gain=STRIDE_GAIN,
        swing_time=SWING_TIME,
        max_frequency=MAX_FREQUENCY,
        integrate_phase=True,
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Deployment's 48-wide layout, then the clock (8) and the height scan, in this order."""

        base_lin_vel = ObsTerm(
            func=mdp.biased_obs,
            params={"func": mdp.base_lin_vel, "bias": LIN_VEL_BIAS},
            clip=(-100, 100),
            noise=_gauss(LIN_VEL_NOISE),
        )
        base_ang_vel = ObsTerm(
            func=mdp.biased_obs,
            params={"func": mdp.base_ang_vel, "bias": ANG_VEL_BIAS},
            scale=0.2,
            clip=(-100, 100),
            noise=_gauss(ANG_VEL_NOISE),
        )
        projected_gravity = ObsTerm(
            func=mdp.biased_obs,
            params={"func": mdp.projected_gravity, "bias": GRAVITY_BIAS},
            clip=(-100, 100),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(
            func=mdp.biased_obs,
            params={"func": mdp.joint_pos_rel, "bias": JOINT_POS_BIAS},
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        clock = ObsTerm(func=mdp.speed_clock, params={"command_name": "clock"})
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": _SCAN},
            noise=Unoise(n_min=-SCAN_NOISE, n_max=SCAN_NOISE) if SCAN_NOISE else None,
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        clock = ObsTerm(func=mdp.speed_clock, params={"command_name": "clock"})
        clock_contact = ObsTerm(func=mdp.speed_clock_contact, params={"command_name": "clock"})
        clock_rate = ObsTerm(func=mdp.generated_commands, params={"command_name": "clock"})
        base_height = ObsTerm(func=mdp.base_height_ground, params={"sensor_cfg": _SCAN})
        height_scan = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": _SCAN}, clip=(-1.0, 1.0))

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp_scaled,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "sigma_exp": LIN_SIGMA_EXP,
            "grace_s": GRACE_S,
            "grace_floor": GRACE_FLOOR,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp_scaled,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "std": ANG_STD,
            "sigma_exp": ANG_SIGMA_EXP,
            "grace_s": GRACE_S,
            "grace_floor": GRACE_FLOOR,
        },
    )

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    base_height = RewTerm(
        func=mdp.base_height_moving, weight=W_HEIGHT, params={"target_height": BASE_HEIGHT, "sensor_cfg": _SCAN}
    )
    termination = RewTerm(func=mdp.is_terminated, weight=W_TERMINATION)

    # -- posture
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
            "moving_scale": 0.0,
        },
    )
    hip_dev = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=W_HIP_DEV,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )

    # -- feet
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    clock_contact_force = RewTerm(func=mdp.clock_contact_force, weight=W_FORCE, params={"sensor_cfg": _FEET_CONTACT})
    clock_contact_vel = RewTerm(func=mdp.clock_contact_vel, weight=W_VEL, params={"asset_cfg": _FEET_BODIES})
    clock_swing_height = RewTerm(
        func=mdp.clock_swing_height,
        weight=W_SWING,
        params={
            "asset_cfg": _FEET_BODIES,
            "swing_height": SWING_HEIGHT,
            "sensor_cfg": _FEET_CONTACT,
            "above_scale": SWING_ABOVE_SCALE,
        },
    )
    foot_landing = RewTerm(
        func=mdp.foot_landing_velocity, weight=W_LANDING, params={"asset_cfg": _FEET_BODIES, "sensor_cfg": _FEET_CONTACT}
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble, weight=W_STUMBLE, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)}
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=W_CONTACTS,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact_after,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
            "grace_s": CONTACT_GRACE,
        },
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    physics_blowup = DoneTerm(func=mdp.physics_blowup)


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_tracking)
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)


@configclass
class ClockScanEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        from isaaclab_physx.physics import PhysxCfg

        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
        self.sim.use_newton_actuators = False
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt


# Env 0, the one the video camera follows, always gets the first terrain type, so play keeps one type.
# The slopes are not in training; they are there to see how the policy copes with them.
PLAY_TERRAIN = os.environ.get("CLOCK_SCAN_PLAY_TERRAIN", "boxes")
PLAY_DIFFICULTY = _range("PLAY_DIFFICULTY", 0.6, 1.0)


@configclass
class ClockScanPlayEnvCfg(ClockScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        vel = self.commands.base_velocity
        vel.ranges = vel.limit_ranges
        pinned = os.environ.get("CLOCK_SCAN_PLAY_CMD_X")
        if pinned:
            v = float(pinned)
            vel.ranges.lin_vel_x = (v, v)
            vel.ranges.lin_vel_y = (0.0, 0.0)
            vel.ranges.ang_vel_z = (0.0, 0.0)
            vel.rel_standing_envs = 0.0
            print(f"[ClockScan] every env pinned to lin_vel_x = {v} m/s")

        gen = self.scene.terrain.terrain_generator
        extra = {
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=1.0, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=1.0, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            ),
        }
        terrain = {**gen.sub_terrains, **extra}[PLAY_TERRAIN].replace(proportion=1.0)
        self.scene.terrain.terrain_generator = gen.replace(
            num_rows=2, num_cols=1, sub_terrains={PLAY_TERRAIN: terrain}, difficulty_range=PLAY_DIFFICULTY
        )
        print(f"[ClockScan] play on {PLAY_TERRAIN}, difficulty {PLAY_DIFFICULTY}")
