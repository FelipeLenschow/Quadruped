import math
import os

import numpy as np

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
from clock_scan.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


def _env(name: str, default: float) -> float:
    return float(os.environ.get(f"CLOCK_SCAN_{name}", default))


def _range(name: str, lo: float, hi: float) -> tuple[float, float]:
    return (_env(f"{name}_LO", lo), _env(f"{name}_HI", hi))


def _gauss(std: float):
    return Gnoise(std=std) if std else None


# ── Scans ──────────────────────────────────────────────────────────────────────────────────
# Height map, actor and critic: vertical rays turned with the base yaw, as a height map built from the lidars
# and sampled about the base would be. From just behind the rear feet (x -0.2 m) to 1 m ahead, +-0.5 m to the
# sides; 5 cm apart near the feet (x up to 0.4 m, |y| up to 0.3 m), 10 cm beyond: 19 x 17 = 323 rays. The
# actor's copy has uniform noise per ray and one offset per episode for the whole map.
MAP_BACK = _env("MAP_BACK", -0.2)
MAP_FRONT = _env("MAP_FRONT", 1.0)
MAP_SIDE = _env("MAP_SIDE", 0.5)
MAP_FINE_FRONT = _env("MAP_FINE_FRONT", 0.4)
MAP_FINE_SIDE = _env("MAP_FINE_SIDE", 0.3)
MAP_FINE = _env("MAP_FINE", 0.05)
MAP_COARSE = _env("MAP_COARSE", 0.1)
SCAN_NOISE = _env("SCAN_NOISE", 0.05)
MAP_BIAS = _env("MAP_BIAS", 0.03)


def _axis(lo: float, hi: float, fine_lo: float, fine_hi: float) -> tuple[float, ...]:
    """Points from lo to hi, MAP_FINE apart within [fine_lo, fine_hi] and MAP_COARSE apart outside it."""
    fine = np.arange(fine_lo, fine_hi + 1e-6, MAP_FINE)
    below = np.arange(fine_lo - MAP_COARSE, lo - 1e-6, -MAP_COARSE)[::-1]
    above = np.arange(fine_hi + MAP_COARSE, hi + 1e-6, MAP_COARSE)
    return tuple(round(float(v), 4) for v in np.concatenate([below, fine, above]))


MAP_X = _axis(MAP_BACK, MAP_FRONT, MAP_BACK, MAP_FINE_FRONT)
MAP_Y = _axis(-MAP_SIDE, MAP_SIDE, -MAP_FINE_SIDE, MAP_FINE_SIDE)
# Height grid, critic only: 1.6 x 1.0 m at 0.1 m (187 rays) under the base, turned with its yaw. Base height
# for the reward and the critic is taken from it.
SCAN_SIZE = (_env("SCAN_LENGTH", 1.6), _env("SCAN_WIDTH", 1.0))
SCAN_RESOLUTION = _env("SCAN_RESOLUTION", 0.1)

# ── Terrain ────────────────────────────────────────────────────────────────────────────────
SMOOTH_NOISE = _env("SMOOTH_NOISE", 0.02)
ROUGH_NOISE = _env("ROUGH_NOISE", 0.04)
BOX_HEIGHT = _range("BOX", 0.02, 0.10)
STEP_HEIGHT = _range("STEP", 0.05, 0.20)
# Stair levels: STEP_HEIGHTS_PER_COUNT levels per step count, the step height rising over STEP_HEIGHT inside
# each count and starting low again at the next. A fall lowers the height but never the step count.
STEP_COUNTS = tuple(int(c) for c in os.environ.get("CLOCK_SCAN_STEP_COUNTS", "1,2,3,6").split(","))
STEP_HEIGHTS_PER_COUNT = int(_env("STEP_HEIGHTS_PER_COUNT", 4))
NUM_ROWS = len(STEP_COUNTS) * STEP_HEIGHTS_PER_COUNT
# Stages of mdp.terrain_stages, as the share of envs on each terrain type. 1: Simple's ground. 2: rough and
# boxes added. 3: stairs, STAIRS_SHARE of the envs, the rest kept in stage 2's mix.
STAIRS_SHARE = _env("STAIRS_SHARE", 0.5)
_STAGE2 = {"flat": 0.15, "smooth": 0.15, "random_rough": 0.35, "boxes": 0.35}
TERRAIN_STAGES = [
    {"flat": 0.3, "smooth": 0.7},
    _STAGE2,
    {**{k: v * (1.0 - STAIRS_SHARE) for k, v in _STAGE2.items()},
     "pyramid_stairs": STAIRS_SHARE / 2, "pyramid_stairs_inv": STAIRS_SHARE / 2},
]
STAIR_TYPES = ("pyramid_stairs", "pyramid_stairs_inv")
# Next stage after STAGE_HOLD checks in a row (one per episode length) with the full speed range, tracking
# above STAGE_TRACK x its weight, and (stage 2 to 3) a mean boxes difficulty of at least STAGE_BOXES_DIFFICULTY.
STAGE_TRACK = _env("STAGE_TRACK", 0.65)
STAGE_HOLD = int(_env("STAGE_HOLD", 3))
STAGE_BOXES_DIFFICULTY = _env("STAGE_BOXES_DIFFICULTY", 0.4)
# Once stairs are in: every env on stairs walks forward only for FORWARD_HOLD_ITERS iterations, then the share
# falls linearly to FORWARD_END over FORWARD_RAMP_ITERS. Forward only: 0 <= x <= FORWARD_X_MAX, y x FORWARD_Y_SCALE,
# turning x FORWARD_YAW_SCALE.
FORWARD_HOLD_ITERS = _env("FORWARD_HOLD_ITERS", 1000)
FORWARD_RAMP_ITERS = _env("FORWARD_RAMP_ITERS", 3000)
FORWARD_END = _env("FORWARD_END", 1.0)
FORWARD_X_MAX = _env("FORWARD_X_MAX", 0.6)
FORWARD_Y_SCALE = _env("FORWARD_Y_SCALE", 1.0)
FORWARD_YAW_SCALE = _env("FORWARD_YAW_SCALE", 0.3)
_STEPS_PER_ITER = BasePPORunnerCfg().num_steps_per_env
# Base this far from the tile centre to go up a level (0: off), on all tiles (LEAVE_DISTANCE) or on stair tiles
# (STAIRS_LEAVE_DISTANCE; past the first edge at 1.5 m). An episode commanded twice as far that never got there
# goes down a level, on stairs also from the first row of a step count. terrain_stages logs the share of stair
# episodes that get 2 m out as stairs_left.
LEAVE_DISTANCE = _env("LEAVE_DISTANCE", 0.0)
STAIRS_LEAVE_DISTANCE = _env("STAIRS_LEAVE_DISTANCE", 2.0)
# Fall penalty on stair tiles, as a share of W_TERMINATION.
STAIRS_FALL_SCALE = _env("STAIRS_FALL_SCALE", 0.25)
# Feet hitting a vertical face (feet_stumble) and body contacts on stair tiles, as a share of their weights.
STAIRS_STUMBLE_SCALE = _env("STAIRS_STUMBLE_SCALE", 1.0)
STAIRS_CONTACT_SCALE = _env("STAIRS_CONTACT_SCALE", 1.0)

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
# Body tilt penalty, against the ground plane fitted to the height grid (GROUND_ORIENTATION=1), so the body may pitch
# with the stairs, or against gravity (0, Simple's flat_orientation_l2).
W_ORIENT = _env("W_ORIENT", -2.5)
GROUND_ORIENTATION = _env("GROUND_ORIENTATION", 1.0) > 0.5
# Per m that the lower head (where the lidar sits) is closer than HEAD_CLEARANCE to the ground under it.
HEAD_CLEARANCE = _env("HEAD_CLEARANCE", 0.15)
W_HEAD = _env("W_HEAD", -20.0)
W_TERMINATION = _env("W_TERMINATION", -1000.0)
W_STUMBLE = _env("W_STUMBLE", -5.0)
# A foot counts as hitting a vertical face when its sideways force is more than STUMBLE_RATIO x its vertical force.
STUMBLE_RATIO = _env("STUMBLE_RATIO", 2.0)
W_CONTACTS = _env("W_CONTACTS", -2.0)
W_HIP_DEV = _env("W_HIP_DEV", -0.2)
# Walk's leash without its cap: per m from, and per rad of heading to, a reference moved by the command.
W_LEASH_POS = _env("W_LEASH_POS", -2.0)
W_LEASH_YAW = _env("W_LEASH_YAW", -0.5)
# Leash position penalty on stair tiles, as a share of W_LEASH_POS, so a robot can slow down to climb.
STAIRS_LEASH_SCALE = _env("STAIRS_LEASH_SCALE", 1.0)

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
_MAP = SceneEntityCfg("map_scanner")
_HEAD_RAY = SceneEntityCfg("head_ground")
_HEAD = SceneEntityCfg("robot", body_names="Head_lower")


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
    num_rows=NUM_ROWS,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        "smooth": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.0, SMOOTH_NOISE), noise_step=0.005, border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.0, ROUGH_NOISE), noise_step=0.005, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=BOX_HEIGHT, platform_width=2.0
        ),
        "pyramid_stairs": mdp.CountedStairsTerrainCfg(
            proportion=0.25, step_counts=STEP_COUNTS, step_height_range=STEP_HEIGHT, step_width=0.3, platform_width=3.0
        ),
        "pyramid_stairs_inv": mdp.CountedStairsTerrainCfg(
            proportion=0.25, step_counts=STEP_COUNTS, step_height_range=STEP_HEIGHT, step_width=0.3, platform_width=3.0,
            inverted=True,
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
    map_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=mdp.FootprintPatternCfg(x_points=MAP_X, y_points=MAP_Y),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # One ray straight down from each foot, for the swing height above the ground under it.
    foot_ground_FL = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FL_foot", offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)), ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=[0.0, 0.0]), mesh_prim_paths=["/World/ground"],
    )
    foot_ground_FR = foot_ground_FL.replace(prim_path="{ENV_REGEX_NS}/Robot/FR_foot")
    foot_ground_RL = foot_ground_FL.replace(prim_path="{ENV_REGEX_NS}/Robot/RL_foot")
    foot_ground_RR = foot_ground_FL.replace(prim_path="{ENV_REGEX_NS}/Robot/RR_foot")
    # One ray straight down from the lower head, for the critic's head clearance and its penalty.
    head_ground = foot_ground_FL.replace(prim_path="{ENV_REGEX_NS}/Robot/Head_lower")
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
        forward_only_x_max=FORWARD_X_MAX,
        forward_only_y_scale=FORWARD_Y_SCALE,
        forward_only_yaw_scale=FORWARD_YAW_SCALE,
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
        """Deployment's 48-wide layout, then the clock (8) and the height map, in this order."""

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
        height_map = ObsTerm(
            func=mdp.biased_height_scan,
            params={"sensor_cfg": _MAP, "bias": MAP_BIAS},
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
        height_map = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": _MAP}, clip=(-1.0, 1.0))
        leash_error = ObsTerm(func=mdp.leash_error)
        head_clearance = ObsTerm(func=mdp.head_clearance, params={"sensor_cfg": _HEAD_RAY, "asset_cfg": _HEAD})

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
    orientation = (
        RewTerm(func=mdp.ground_orientation_l2, weight=W_ORIENT, params={"sensor_cfg": _SCAN})
        if GROUND_ORIENTATION
        else RewTerm(func=mdp.flat_orientation_l2, weight=W_ORIENT)
    )
    head_clearance = RewTerm(
        func=mdp.head_clearance_l1,
        weight=W_HEAD,
        params={"sensor_cfg": _HEAD_RAY, "asset_cfg": _HEAD, "min_clearance": HEAD_CLEARANCE},
    )
    base_height = RewTerm(
        func=mdp.base_height_moving, weight=W_HEIGHT, params={"target_height": BASE_HEIGHT, "sensor_cfg": _SCAN}
    )
    leash_pos = RewTerm(
        func=mdp.terrain_scaled,
        weight=W_LEASH_POS,
        params={"term": mdp.leash_position_l1, "term_params": {}, "scales": {name: STAIRS_LEASH_SCALE for name in STAIR_TYPES}},
    )
    leash_yaw = RewTerm(func=mdp.leash_yaw_l1, weight=W_LEASH_YAW)
    termination = RewTerm(
        func=mdp.terminated_by_terrain,
        weight=W_TERMINATION,
        params={"scales": {name: STAIRS_FALL_SCALE for name in STAIR_TYPES}},
    )

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
            "above_scale": SWING_ABOVE_SCALE,
            "ground_sensors": [f"foot_ground_{foot[:2]}" for foot in FEET],
        },
    )
    foot_landing = RewTerm(
        func=mdp.foot_landing_velocity, weight=W_LANDING, params={"asset_cfg": _FEET_BODIES, "sensor_cfg": _FEET_CONTACT}
    )
    feet_stumble = RewTerm(
        func=mdp.terrain_scaled,
        weight=W_STUMBLE,
        params={
            "term": mdp.feet_stumble,
            "term_params": {"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET), "ratio": STUMBLE_RATIO},
            "scales": {name: STAIRS_STUMBLE_SCALE for name in STAIR_TYPES},
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.terrain_scaled,
        weight=W_CONTACTS,
        params={
            "term": mdp.undesired_contacts,
            "term_params": {
                "threshold": 1,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
            },
            "scales": {name: STAIRS_CONTACT_SCALE for name in STAIR_TYPES},
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
    terrain_levels = CurrTerm(
        func=mdp.terrain_stages,
        params={
            "stages": TERRAIN_STAGES,
            "level_types": ("boxes",),
            "level_gates": (0.0, STAGE_BOXES_DIFFICULTY * NUM_ROWS),
            "stair_types": STAIR_TYPES,
            "stair_rows_per_count": STEP_HEIGHTS_PER_COUNT,
            "stair_counts": STEP_COUNTS,
            "stair_height_range": STEP_HEIGHT,
            "track_ratio": STAGE_TRACK,
            "hold": STAGE_HOLD,
            "forward_hold_steps": int(FORWARD_HOLD_ITERS * _STEPS_PER_ITER),
            "forward_ramp_steps": int(FORWARD_RAMP_ITERS * _STEPS_PER_ITER),
            "forward_end_share": FORWARD_END,
            "leave_distance": LEAVE_DISTANCE,
            "stair_leave_distance": STAIRS_LEAVE_DISTANCE,
        },
    )
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
        self.scene.map_scanner.update_period = self.decimation * self.sim.dt
        for foot in FEET:
            getattr(self.scene, f"foot_ground_{foot[:2]}").update_period = self.decimation * self.sim.dt


# Env 0, the one the video camera follows, always gets the first terrain type, so play keeps one type.
# The slopes are not in training; they are there to see how the policy copes with them.
PLAY_TERRAIN = os.environ.get("CLOCK_SCAN_PLAY_TERRAIN", "boxes")
PLAY_DIFFICULTY = _range("PLAY_DIFFICULTY", 0.6, 1.0)
# Tiles in x. Rows run along x, so a robot pinned to walk +x crosses from its tile into the next row's.
PLAY_ROWS = int(_env("PLAY_ROWS", 2))


@configclass
class ClockScanPlayEnvCfg(ClockScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.curriculum.terrain_levels = None
        vel = self.commands.base_velocity
        vel.ranges = vel.limit_ranges
        pinned = os.environ.get("CLOCK_SCAN_PLAY_CMD_X")
        if pinned:
            v = float(pinned)
            vel.ranges.lin_vel_x = (v, v)
            vel.ranges.lin_vel_y = (0.0, 0.0)
            vel.ranges.ang_vel_z = (0.0, 0.0)
            vel.rel_standing_envs = 0.0
            vel.slow_command_fraction = 0.0
            vel.pause_fraction = 0.0
            vel.yaw_only_command_fraction = 0.0
            vel.debug_vis = False
            self.events.reset_start_pose = None
            self.events.push_robot = None
            yaw = _env("PLAY_YAW", 0.0)
            self.events.reset_base.params["pose_range"]["yaw"] = (yaw, yaw)
            print(f"[ClockScan] every env pinned to lin_vel_x = {v} m/s, start yaw {yaw} rad, standing starts, no pushes")

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
            num_rows=PLAY_ROWS, num_cols=1, sub_terrains={PLAY_TERRAIN: terrain}, difficulty_range=PLAY_DIFFICULTY
        )
        print(f"[ClockScan] play on {PLAY_TERRAIN}, difficulty {PLAY_DIFFICULTY}")
