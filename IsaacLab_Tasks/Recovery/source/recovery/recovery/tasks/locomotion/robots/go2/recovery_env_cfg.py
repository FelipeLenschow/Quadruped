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
try:
    from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
except ImportError:
    from isaaclab.utils.noise import UniformNoiseCfg as Unoise
try:
    from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
except ImportError:
    from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

from recovery.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from recovery.tasks.locomotion import mdp


def _env(name: str, default: float) -> float:
    return float(os.environ.get(f"RECOVERY_{name}", default))


# ── Start states ───────────────────────────────────────────────────────────────────────────
# Weights of the five starts in mdp.reset_recovery_start at the top level; they need not sum to 1.
# Mostly the starts that still fail; a little standing and dropping so those are not forgotten.
START_FRACTIONS = {
    "stand": _env("STAND", 0.05),
    "drop": _env("DROP", 0.05),
    "lying": _env("LYING", 0.3),
    "flipped": _env("FLIPPED", 0.3),
    "tumble": _env("TUMBLE", 0.3),
}
# Weights at level 0: no throws and nothing upside down yet.
EASY_FRACTIONS = {
    "stand": _env("EASY_STAND", 0.3),
    "drop": _env("EASY_DROP", 0.3),
    "lying": _env("EASY_LYING", 0.4),
    "flipped": _env("EASY_FLIPPED", 0.0),
    "tumble": _env("EASY_TUMBLE", 0.0),
}
DROP_HEIGHT = (_env("DROP_HEIGHT_LO", 0.1), _env("DROP_HEIGHT_HI", 0.5))
DROP_TILT = _env("DROP_TILT", 0.6)
TUMBLE_HEIGHT = (_env("TUMBLE_HEIGHT_LO", 0.6), _env("TUMBLE_HEIGHT_HI", 1.0))
TUMBLE_LIN_VEL = _env("TUMBLE_LIN_VEL", 1.0)
TUMBLE_ANG_VEL = _env("TUMBLE_ANG_VEL", 4.0)
# Belly-down poses inside the 0.9 soft joint limits (the calf soft limit is about -2.63).
LYING_POSES = [
    {".*L_hip_joint": 0.1, ".*R_hip_joint": -0.1, ".*_thigh_joint": 1.3, ".*_calf_joint": -2.55},
    {".*L_hip_joint": 0.1, ".*R_hip_joint": -0.1, ".*_thigh_joint": 3.2, ".*_calf_joint": -2.55},
    {".*_hip_joint": 0.0, "F[L,R]_thigh_joint": -1.3, "R[L,R]_thigh_joint": 1.6, ".*_calf_joint": -0.95},
    {".*L_hip_joint": 0.6, ".*R_hip_joint": -0.6, ".*_thigh_joint": 1.0, ".*_calf_joint": -1.0},
]

# ── Disturbances ───────────────────────────────────────────────────────────────────────────
# A shove every few seconds.
PUSH_INTERVAL = (_env("PUSH_INTERVAL_LO", 3.0), _env("PUSH_INTERVAL_HI", 6.0))
PUSH_VEL = _env("PUSH_VEL", 0.6)

# ── Rewards ────────────────────────────────────────────────────────────────────────────────
# Standing height of the Go2 in its default pose.
BASE_HEIGHT = _env("BASE_HEIGHT", 0.32)
# The "slow" part: speeds above these caps cost the square of the excess.
# Reco1 (4 rad/s, 2 rad/s, 0.6 m/s, no rise cap) got up from its back in ~1 s, joints at the cap.
# Reco2 (2 rad/s, 1 rad/s) stopped trying from a third of belly starts and some side/back poses.
JOINT_VEL_CAP = _env("JOINT_VEL_CAP", 3.0)
BASE_ANG_VEL_CAP = _env("BASE_ANG_VEL_CAP", 1.5)
BASE_LIN_VEL_CAP = _env("BASE_LIN_VEL_CAP", 0.4)
BASE_RISE_CAP = _env("BASE_RISE_CAP", 0.25)
# Contact force above this on any body counts as an impact.
IMPACT_FORCE = _env("IMPACT_FORCE", 200.0)
# Everything but the feet: a rise in contact force of more than HIT_MARGIN newtons in one step is a hit.
HIT_MARGIN = _env("HIT_MARGIN", 30.0)
W_HITS = _env("W_HITS", -0.02)
_NOT_FEET = SceneEntityCfg("contact_forces", body_names=["base", "Head_.*", ".*_hip", ".*_thigh", ".*_calf"])

EPISODE_S = _env("EPISODE_S", 20.0)

# ── Curriculum ─────────────────────────────────────────────────────────────────────────────
# Per-env level in [0, 1]: +LEVEL_UP when standing as the episode ends, -LEVEL_DOWN when not, so it
# settles where ~2/3 of episodes end standing. It moves the start weights from EASY_FRACTIONS to
# START_FRACTIONS and scales drop height and tilt, and throw height and speed.
# RECOVERY_CURRICULUM=0 trains at level 1 from the start.
CURRICULUM = os.environ.get("RECOVERY_CURRICULUM", "1") == "1"
LEVEL_INIT = _env("LEVEL_INIT", 0.0)
LEVEL_UP = _env("LEVEL_UP", 0.05)
LEVEL_DOWN = _env("LEVEL_DOWN", 0.1)
PLAY_LEVEL = _env("PLAY_LEVEL", 1.0)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_gen.TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            difficulty_range=(0.0, 1.0),
            use_cache=False,
            curriculum=False,
            # Flat: on the 0-4 cm random rough heightfield some feet caught on its edges and stayed put.
            sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
        ),
        max_init_terrain_level=None,
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
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # No offset: the sensor offset turns with the whole base, so one above it ends up below the
    # ground when the robot is on its back. The rays themselves always point down.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.2, 0.2]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # One entry per physics step of a control step (decimation 4), so a short impact is not missed.
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=4, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
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
    start = EventTerm(
        func=mdp.reset_recovery_start,
        mode="reset",
        params={
            "fractions": START_FRACTIONS,
            "easy_fractions": EASY_FRACTIONS,
            "lying_poses": LYING_POSES,
            "drop_height_range": DROP_HEIGHT,
            "drop_tilt": DROP_TILT,
            "tumble_height_range": TUMBLE_HEIGHT,
            "tumble_lin_vel": TUMBLE_LIN_VEL,
            "tumble_ang_vel": TUMBLE_ANG_VEL,
        },
    )
    push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=PUSH_INTERVAL,
        params={"velocity_range": {"x": (-PUSH_VEL, PUSH_VEL), "y": (-PUSH_VEL, PUSH_VEL)}},
    )


@configclass
class CommandsCfg:
    # Always zero. Kept so the actor has unitree's 45-wide layout and runs through policy_runner as is.
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e6, 1e6),
        rel_standing_envs=1.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
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
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        base_height = ObsTerm(func=mdp.base_height_ground, params={"sensor_cfg": SceneEntityCfg("height_scanner")})
        body_contact = ObsTerm(
            func=mdp.body_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "Head_.*", ".*_thigh", ".*_calf", ".*_foot"])},
        )

    critic: CriticCfg = CriticCfg()


_SCAN = SceneEntityCfg("height_scanner")
_STAND = {"target_height": BASE_HEIGHT, "sensor_cfg": _SCAN}


@configclass
class RewardsCfg:
    # -- getting up: belly down, then base up, then the standing pose, then quiet
    upright = RewTerm(func=mdp.upright, weight=1.0)
    base_height = RewTerm(func=mdp.base_height_up, weight=2.0, params=_STAND)
    stand_pose = RewTerm(func=mdp.stand_pose, weight=0.5, params={"std": 0.5, **_STAND})
    stand_still = RewTerm(func=mdp.stand_still, weight=0.5, params={"std": 0.3, **_STAND})
    feet_on_ground = RewTerm(
        func=mdp.feet_on_ground,
        weight=0.5,
        params={"contact_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), **_STAND},
    )

    # -- slow
    joint_vel_over = RewTerm(func=mdp.joint_vel_over, weight=-0.2, params={"limit": JOINT_VEL_CAP})
    base_ang_vel_over = RewTerm(func=mdp.base_ang_vel_over, weight=-1.0, params={"limit": BASE_ANG_VEL_CAP})
    base_lin_vel_over = RewTerm(func=mdp.base_lin_vel_over, weight=-0.5, params={"limit": BASE_LIN_VEL_CAP})
    base_rise_over = RewTerm(func=mdp.base_rise_over, weight=-5.0, params={"limit": BASE_RISE_CAP})

    # -- gentle
    impacts = RewTerm(
        func=mdp.contact_forces,
        weight=-0.01,
        params={"threshold": IMPACT_FORCE, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")},
    )
    body_hits = RewTerm(func=mdp.body_hits, weight=W_HITS, params={"sensor_cfg": _NOT_FEET, "margin": HIT_MARGIN})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.05)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    energy = RewTerm(func=mdp.energy, weight=-5e-5)


@configclass
class CurriculumCfg:
    level = CurrTerm(
        func=mdp.recovery_level,
        params={"initial": LEVEL_INIT, "up": LEVEL_UP, "down": LEVEL_DOWN, "target_height": BASE_HEIGHT, "sensor_cfg": _SCAN},
    )


@configclass
class TerminationsCfg:
    # Nothing ends an episode early: every pose has to be got out of.
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    physics_blowup = DoneTerm(func=mdp.physics_blowup)


# ── Sim-to-real ────────────────────────────────────────────────────────────────────────────
# Simple's Deploy arm, unchanged: sensor bias, actuator delay, PD gains, CoM, joint plant, foot torsion.
ANG_VEL_NOISE = _env("ANG_VEL_NOISE", 0.04)
ANG_VEL_BIAS = _env("ANG_VEL_BIAS", 0.025)
GRAVITY_BIAS = _env("GRAVITY_BIAS", 0.025)
JOINT_POS_BIAS = _env("JOINT_POS_BIAS", 0.019)
# Physics steps of 5 ms.
MIN_DELAY = int(_env("MIN_DELAY", 0))
MAX_DELAY = int(_env("MAX_DELAY", 5))
KP_RANGE = (-_env("KP", 5.0), _env("KP", 5.0))
KD_RANGE = (-_env("KD", 0.2), _env("KD", 0.2))
COM_X = _env("COM_X", 0.05)
COM_Y = _env("COM_Y", 0.03)
ARMATURE = (_env("ARMATURE_LO", 0.005), _env("ARMATURE_HI", 0.02))
VISCOUS = (_env("VISCOUS_LO", 0.0), _env("VISCOUS_HI", 1.0))
JOINT_FRICTION = (_env("JOINT_FRICTION_LO", 0.03), _env("JOINT_FRICTION_HI", 0.5))
FOOT_TORSION = _env("FOOT_TORSION", 0.02)
FOOT_TORSION_MIN = _env("FOOT_TORSION_MIN", 0.01)


def _bias_term(term, bias: float) -> None:
    term.params = {"func": term.func, "bias": bias, **term.params}
    term.func = mdp.biased_obs


def _apply_sensor_model(cfg) -> None:
    policy = cfg.observations.policy
    _bias_term(policy.base_ang_vel, ANG_VEL_BIAS)
    _bias_term(policy.projected_gravity, GRAVITY_BIAS)
    _bias_term(policy.joint_pos_rel, JOINT_POS_BIAS)
    policy.base_ang_vel.noise = Gnoise(std=ANG_VEL_NOISE) if ANG_VEL_NOISE else None


def _apply_calf_reduction(cfg) -> None:
    """The calf sits behind a 1.92:1 knee reduction: 45 N*m and 15.65 rad/s at the joint, not the hip's 23.4 and 30."""
    from recovery.assets.robots.unitree_actuators import GO2_KNEE_RATIO

    legs = cfg.scene.robot.actuators["GO2HV"]

    def per_joint(value, calf_scale):
        return {".*_hip_joint": value, ".*_thigh_joint": value, ".*_calf_joint": value * calf_scale}

    cfg.scene.robot = cfg.scene.robot.replace(
        actuators={
            "GO2HV": legs.replace(
                Y1=per_joint(legs.Y1, GO2_KNEE_RATIO),
                Y2=per_joint(legs.Y2, GO2_KNEE_RATIO),
                X1=per_joint(legs.X1, 1.0 / GO2_KNEE_RATIO),
                X2=per_joint(legs.X2, 1.0 / GO2_KNEE_RATIO),
                min_delay=MIN_DELAY,
                max_delay=MAX_DELAY,
            )
        }
    )


def _apply_plant_randomization(cfg) -> None:
    e = cfg.events
    e.randomize_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": KP_RANGE,
            "damping_distribution_params": KD_RANGE,
            "operation": "add",
        },
    )
    e.randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-COM_X, COM_X), "y": (-COM_Y, COM_Y)},
        },
    )
    e.randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": JOINT_FRICTION,
            "operation": "add",
        },
    )
    e.randomize_joint_plant = EventTerm(
        func=mdp.randomize_joint_plant,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_range": ARMATURE,
            "viscous_friction_range": VISCOUS,
        },
    )


BELLY_PADS = os.environ.get("RECOVERY_BELLY_PADS", "1") == "1"


def _apply_belly_pads(cfg) -> None:
    """Add the Go2's belly pads and put the chin collider where the chin is (see go2_belly_pads.py)."""
    if not BELLY_PADS:
        return
    from recovery.assets.robots.go2_belly_pads import spawn_go2_with_belly_pads

    cfg.scene.robot = cfg.scene.robot.replace(spawn=cfg.scene.robot.spawn.replace(func=spawn_go2_with_belly_pads))


def _apply_foot_torsion(cfg) -> None:
    """Torsional contact patch on every collider; make_uninstanceable because the Go2 USD's collision meshes are read-only instance proxies."""
    if FOOT_TORSION <= 0.0 and FOOT_TORSION_MIN <= 0.0:
        return
    from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg

    cfg.scene.robot = cfg.scene.robot.replace(
        spawn=cfg.scene.robot.spawn.replace(
            make_uninstanceable=True,
            collision_props=PhysxCollisionPropertiesCfg(
                torsional_patch_radius=FOOT_TORSION,
                min_torsional_patch_radius=FOOT_TORSION_MIN,
            ),
        )
    )


@configclass
class RobotRecoveryEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = EPISODE_S
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        if hasattr(self.sim, "physx"):
            self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        else:
            from isaaclab_physx.physics import PhysxCfg

            self.sim.physics = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
            self.sim.use_newton_actuators = False
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        _apply_calf_reduction(self)
        _apply_sensor_model(self)
        _apply_plant_randomization(self)
        _apply_foot_torsion(self)
        _apply_belly_pads(self)
        if not CURRICULUM:
            self.curriculum.level = None
        if type(self).__name__.startswith("RobotRecovery"):
            print(f"[Recovery] starts {EASY_FRACTIONS} at level 0 -> {START_FRACTIONS} at 1, episode {EPISODE_S} s")
        print(
            f"[Recovery] curriculum {'on' if CURRICULUM else 'off (level 1)'}: start {LEVEL_INIT},"
            f" +{LEVEL_UP} when standing at the end, -{LEVEL_DOWN} when not"
        )
        print(
            f"[Recovery] push +-{PUSH_VEL} m/s every {PUSH_INTERVAL} s"
        )
        print(
            f"[Recovery] caps: joints {JOINT_VEL_CAP} rad/s, base {BASE_ANG_VEL_CAP} rad/s and {BASE_LIN_VEL_CAP} m/s,"
            f" rising {BASE_RISE_CAP} m/s,"
            f" impacts over {IMPACT_FORCE} N, non-foot hits over {HIT_MARGIN} N/step x {W_HITS} per N"
        )


def _apply_play(cfg) -> None:
    """RECOVERY_PLAY_START=<mode> starts every env the same way; RECOVERY_PLAY_CYCLE=1 goes through every
    start the task uses, in order, one per episode. RECOVERY_PLAY_PUSHES=0 turns off the pushes. Level held
    at RECOVERY_PLAY_LEVEL (default 1). RECOVERY_PLAY_GRID=<n> puts n robots in a grid
    RECOVERY_PLAY_SPACING apart on one tile, with the viewer looking at all of them.
    RECOVERY_PLAY_EPISODE_S sets the episode length, over whatever the task uses."""
    cfg.scene.num_envs = 32
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 4
    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05), roughness=0.9)
    if cfg.curriculum.level is None:
        cfg.curriculum.level = CurrTerm(func=mdp.recovery_level, params={"target_height": BASE_HEIGHT, "sensor_cfg": _SCAN})
    cfg.curriculum.level.params.update(initial=PLAY_LEVEL, up=0.0, down=0.0)
    if os.environ.get("RECOVERY_PLAY_EPISODE_S"):
        cfg.episode_length_s = float(os.environ["RECOVERY_PLAY_EPISODE_S"])
        print(f"[Recovery] play: {cfg.episode_length_s} s episodes")
    grid = int(os.environ.get("RECOVERY_PLAY_GRID", 0))
    if grid > 0:
        spacing = _env("PLAY_SPACING", 2.0)
        span = spacing * math.ceil(math.sqrt(grid))
        cfg.scene.num_envs = grid
        cfg.scene.env_spacing = spacing
        cfg.scene.terrain.use_terrain_origins = False
        cfg.scene.terrain.env_spacing = spacing
        gen = cfg.scene.terrain.terrain_generator
        gen.num_rows = gen.num_cols = 1
        gen.size = (max(8.0, span + 2.0),) * 2
        cfg.viewer.eye = (span, -span, 0.75 * span)
        cfg.viewer.lookat = (0.0, 0.0, 0.0)
        print(f"[Recovery] play: {grid} robots {spacing} m apart on one tile")
    if os.environ.get("RECOVERY_PLAY_PUSHES", "1") == "0":
        cfg.events.push = None
        if hasattr(cfg.events, "push_start"):
            cfg.events.push_start = None
        print("[Recovery] play: no pushes")
    only = os.environ.get("RECOVERY_PLAY_START")
    if os.environ.get("RECOVERY_PLAY_CYCLE", "0") == "1":
        cfg.events.start.params["cycle"] = True
        print("[Recovery] play: each env goes through every start in turn")
    elif only:
        cfg.events.start.params["fractions"] = {only: 1.0}
        print(f"[Recovery] play: every env starts '{only}'")


@configclass
class RobotRecoveryPlayEnvCfg(RobotRecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play(self)


# ── Lie down ───────────────────────────────────────────────────────────────────────────────
# A safety controller: from whatever state it takes over in (standing, walking, falling, fallen), get
# belly down into LIE_POSE, slowly, and rest the weight on the belly so the motors can then be switched
# off. Measured, holding it for 3 s and then zeroing every gain: this pose rests 51 N of the 167 N on the
# belly at 7.6 cm and does not move when numb. unitree's (hips +-0.35, thighs 1.36, calves -2.6) does the
# same, but the weight pushes its calves past the soft limit; LYING_POSES[0] (hips +-0.1, thighs 1.3)
# stands on its folded legs at 9.7 cm and sinks 1.7 cm when numb.
LIE_POSE = {".*L_hip_joint": 0.5, ".*R_hip_joint": -0.5, ".*_thigh_joint": 1.2, ".*_calf_joint": -2.5}
# With the belly pads in (go2_belly_pads.py) the robot rests on them ~5 deg nose up at 8.6 cm with the chin
# clear. config.yaml's lie_flat: hips spread 0.45 front, 0.65 rear keep the feet 1-2 cm above the ground when on
# the pads; with 0.1/0.5 the feet reach 1-2.4 cm below them and a stiff policy stands on its feet.
if BELLY_PADS:
    LIE_POSE = {"FL_hip_joint": 0.45, "FR_hip_joint": -0.45, "RL_hip_joint": 0.65, "RR_hip_joint": -0.65, ".*_thigh_joint": 1.25, ".*_calf_joint": -2.6}
LIE_HEIGHT = _env("LIE_HEIGHT", 0.086 if BELLY_PADS else 0.076)
LIE_MAX_HEIGHT = _env("LIE_MAX_HEIGHT", 0.10)
LIE_SINK_CAP = _env("LIE_SINK_CAP", 0.3)
# Belly contact force above this is a hit. Lying still on the pads puts 120-140 N on them (51 N on the
# belly without pads); the head has its own terms below.
LIE_BELLY_FORCE = _env("LIE_BELLY_FORCE", 200.0 if BELLY_PADS else 100.0)
LIE_W_BELLY = _env("LIE_W_BELLY", -0.05)
# On top of body_hits: a non-foot jump past LIE_HARD_HIT costs ((jump - LIE_HARD_HIT) / 100 N)^2, so a
# 3000 N slam costs ~3x what the linear term alone charges and a few 400 N touches almost nothing more.
LIE_HARD_HIT = _env("LIE_HARD_HIT", 300.0)
LIE_W_HARD_HIT = _env("LIE_W_HARD_HIT", -0.25)
# Feet are meant to land, so only a jump past this counts (a stiff-legged drop).
LIE_FOOT_HIT = _env("LIE_FOOT_HIT", 800.0)
LIE_W_FOOT_HIT = _env("LIE_W_FOOT_HIT", -0.02)
# Lying on the pads never needs the head on the ground, so any head (chin, lidar cage) contact costs, and a
# jump in it past 50 N costs ((jump - 50) / 50 N)^2 on top: a handstand caught on the face.
LIE_W_HEAD = _env("LIE_W_HEAD", -0.1)
LIE_W_FACE_HIT = _env("LIE_W_FACE_HIT", -0.25)
_HEAD = SceneEntityCfg("contact_forces", body_names="Head_.*")
LIE_EPISODE_S = _env("LIE_EPISODE_S", 10.0)
# A push on the first frame too, on top of the start pose; the interval pushes only come 3-6 s in.
LIE_PUSH_START = _env("LIE_PUSH_START", 1.0) > 0
# The model_4999 run (2026-10-10_00-02-00) had none of the last three and hit hard from a handstand
# (1600-3000 N on the back end, at every tilt from 70 deg).
LIE_FRACTIONS = {
    "stand": _env("LIE_STAND", 0.15),
    "moving": _env("LIE_MOVING", 0.15),
    "drop": _env("LIE_DROP", 0.05),
    "lying": _env("LIE_LYING", 0.05),
    "flipped": _env("LIE_FLIPPED", 0.1),
    "tumble": _env("LIE_TUMBLE", 0.15),
    "handstand": _env("LIE_HANDSTAND", 0.1),
    "hindstand": _env("LIE_HINDSTAND", 0.1),
    "tipping": _env("LIE_TIPPING", 0.15),
}
LIE_EASY_FRACTIONS = {"stand": 0.5, "moving": 0.3, "lying": 0.2}
_LIE = {"lie_joint_pos": LIE_POSE, "max_height": LIE_MAX_HEIGHT, "sensor_cfg": _SCAN}


@configclass
class LieDownRewardsCfg:
    # -- getting down: belly toward the ground, base low, then the pose, then relaxed and still
    upright = RewTerm(func=mdp.upright, weight=1.0)
    base_low = RewTerm(
        func=mdp.base_low, weight=1.0, params={"lie_height": LIE_HEIGHT, "stand_height": BASE_HEIGHT, "sensor_cfg": _SCAN}
    )
    lie_pose_coarse = RewTerm(func=mdp.lie_pose, weight=1.0, params={"lie_joint_pos": LIE_POSE, "std": 1.5})
    lie_pose_fine = RewTerm(func=mdp.lie_pose, weight=1.0, params={"lie_joint_pos": LIE_POSE, "std": 0.3})
    relaxed = RewTerm(func=mdp.relaxed, weight=1.0, params={"scale": 2.0, **_LIE})
    lie_still = RewTerm(func=mdp.lie_still, weight=0.5, params={"std": 0.2, **_LIE})

    # -- slow
    # Not while toppling: catching a fall needs the legs fast.
    joint_vel_over = RewTerm(func=mdp.joint_vel_over_calm, weight=-0.2, params={"limit": JOINT_VEL_CAP})
    base_ang_vel_over = RewTerm(func=mdp.base_ang_vel_over, weight=-1.0, params={"limit": BASE_ANG_VEL_CAP})
    base_lin_vel_over = RewTerm(func=mdp.base_lin_vel_over, weight=-0.5, params={"limit": BASE_LIN_VEL_CAP})
    base_sink_over = RewTerm(func=mdp.base_sink_over, weight=-5.0, params={"limit": LIE_SINK_CAP})

    # -- gentle
    belly_impact = RewTerm(
        func=mdp.contact_forces,
        weight=LIE_W_BELLY,
        params={"threshold": LIE_BELLY_FORCE, "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base")},
    )
    impacts = RewTerm(
        func=mdp.contact_forces,
        weight=-0.01,
        params={"threshold": IMPACT_FORCE, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")},
    )
    body_hits = RewTerm(func=mdp.body_hits, weight=W_HITS, params={"sensor_cfg": _NOT_FEET, "margin": HIT_MARGIN})
    hard_hits = RewTerm(
        func=mdp.body_hits,
        weight=LIE_W_HARD_HIT,
        params={"sensor_cfg": _NOT_FEET, "margin": LIE_HARD_HIT, "square_scale": 100.0},
    )
    foot_hits = RewTerm(
        func=mdp.body_hits,
        weight=LIE_W_FOOT_HIT,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "margin": LIE_FOOT_HIT},
    )
    head_contact = RewTerm(func=mdp.contact_forces, weight=LIE_W_HEAD, params={"threshold": 1.0, "sensor_cfg": _HEAD})
    face_hits = RewTerm(
        func=mdp.body_hits, weight=LIE_W_FACE_HIT, params={"sensor_cfg": _HEAD, "margin": 50.0, "square_scale": 50.0}
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits_unless_lying, weight=-5.0, params=_LIE)
    torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.05)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    energy = RewTerm(func=mdp.energy, weight=-5e-5)


@configclass
class RobotLieDownEnvCfg(RobotRecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = LIE_EPISODE_S
        self.rewards = LieDownRewardsCfg()
        self.events.start.params.update(fractions=LIE_FRACTIONS, easy_fractions=LIE_EASY_FRACTIONS)
        if LIE_PUSH_START and self.events.push is not None:
            self.events.push_start = EventTerm(func=self.events.push.func, mode="reset", params=dict(self.events.push.params))
        if self.curriculum.level is not None:
            self.curriculum.level.params.update(min_s=0.6 * LIE_EPISODE_S, **_LIE)
        print(f"[LieDown] starts {LIE_EASY_FRACTIONS} at level 0 -> {LIE_FRACTIONS} at 1, episode {LIE_EPISODE_S} s")
        print(f"[LieDown] pose {LIE_POSE}, base {LIE_HEIGHT} m (done below {LIE_MAX_HEIGHT} m), sinking over {LIE_SINK_CAP} m/s penalized,"
              f" belly force over {LIE_BELLY_FORCE} N x {LIE_W_BELLY} per N")
        print(f"[LieDown] non-foot jumps over {LIE_HARD_HIT} N squared per 100 N x {LIE_W_HARD_HIT}, foot jumps over {LIE_FOOT_HIT} N x {LIE_W_FOOT_HIT};"
              " joint speed cap off while toppling")
        print(f"[LieDown] push on the first frame {LIE_PUSH_START}")
        print(f"[LieDown] belly pads {BELLY_PADS}; head contact x {LIE_W_HEAD} per N, face jumps over 50 N squared per 50 N x {LIE_W_FACE_HIT}")


@configclass
class RobotLieDownPlayEnvCfg(RobotLieDownEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play(self)
