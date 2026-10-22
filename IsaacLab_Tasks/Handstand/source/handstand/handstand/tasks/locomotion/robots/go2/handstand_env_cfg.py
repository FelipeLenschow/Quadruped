import math
import os

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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

from handstand.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from handstand.tasks.locomotion import mdp


def _env(name: str, default: float) -> float:
    return float(os.environ.get(f"HANDSTAND_{name}", default))


# ── Episodes ───────────────────────────────────────────────────────────────────────────────
# The trick command starts off, then alternates on and off, so each episode has a few goes at doing
# the trick from standing and getting back to standing from it.
EPISODE_S = _env("EPISODE_S", 20.0)
FIRST_OFF = (_env("FIRST_OFF_LO", 0.5), _env("FIRST_OFF_HI", 2.0))
ON_TIME = (_env("ON_LO", 5.0), _env("ON_HI", 8.0))
OFF_TIME = (_env("OFF_LO", 3.0), _env("OFF_HI", 5.0))

# ── Disturbances ───────────────────────────────────────────────────────────────────────────
PUSH_INTERVAL = (_env("PUSH_INTERVAL_LO", 4.0), _env("PUSH_INTERVAL_HI", 8.0))
PUSH_VEL = _env("PUSH_VEL", 0.2)

# ── Shared rewards ─────────────────────────────────────────────────────────────────────────
BASE_HEIGHT = _env("BASE_HEIGHT", 0.32)
# Going up on two legs needs faster joints and base turning than the tricks' 4 rad/s, 1 rad/s and 0.3 m/s.
JOINT_VEL_CAP = _env("JOINT_VEL_CAP", 6.0)
BASE_ANG_VEL_CAP = _env("BASE_ANG_VEL_CAP", 2.0)
BASE_LIN_VEL_CAP = _env("BASE_LIN_VEL_CAP", 0.5)
IMPACT_FORCE = _env("IMPACT_FORCE", 200.0)
# The deploy checks (config.yaml): tilt past 30 deg, and joints in the outer 15% of their hardware range.
# Training keeps a few degrees and percent inside them.
TILT_LIMIT = math.radians(_env("TILT_LIMIT_DEG", 22.0))
ROM_MARGIN = _env("ROM_MARGIN", 0.17)
HARD_LIMITS = {
    ".*_hip_joint": (-1.047, 1.047),
    "F[L,R]_thigh_joint": (-1.571, 3.491),
    "R[L,R]_thigh_joint": (-0.524, 4.538),
    ".*_calf_joint": (-2.723, -0.5),
}

FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
_FEET_CONTACT = SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True)
_FEET_BODIES = SceneEntityCfg("robot", body_names=FEET, preserve_order=True)
_NOT_FEET = SceneEntityCfg("contact_forces", body_names=["base", "Head_.*", ".*_hip", ".*_thigh", ".*_calf"])
_SCAN = SceneEntityCfg("height_scanner")
_CMD = "trick"

# ── Handstand: switch 0 up on the front legs (nose down), switch 1 up on the rear legs (nose up) ──────
# Support pair of each switch, and the pitch toward it (rad) once up.
LEGS = ["F", "R"]
HAND_PITCH = _env("HAND_PITCH", 1.35)
HIND_PITCH = _env("HIND_PITCH", 1.35)
# A share of episodes starts already up, as LieDown's handstand and hindstand starts, at rest; half of those
# with that switch on (hold it), half off (get down from it).
ON_LEGS_FRACTION = _env("ON_LEGS_FRACTION", 0.3)
ON_LEGS_TILT = (_env("ON_LEGS_TILT_LO", 1.0), _env("ON_LEGS_TILT_HI", 1.5))
START_ON_PROB = _env("START_ON_PROB", 0.5)
# Fallen: sideways past this, or tilted past this from level in any direction.
SIDE_LIMIT = _env("SIDE_LIMIT", 1.0)
FALL_TILT = _env("FALL_TILT", 1.9)


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
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.2, 0.2]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
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
    start = EventTerm(func=mdp.reset_trick_start, mode="reset")
    push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=PUSH_INTERVAL,
        params={"velocity_range": {"x": (-PUSH_VEL, PUSH_VEL), "y": (-PUSH_VEL, PUSH_VEL)}},
    )


@configclass
class CommandsCfg:
    trick = mdp.TrickCommandCfg(
        asset_name="robot", first_off_range=FIRST_OFF, on_time_range=ON_TIME, off_time_range=OFF_TIME
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    # unitree's 45-wide actor layout, with the trick command [on, 0, 0] in the velocity command's place, so
    # policy_runner runs it as is.
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        trick_command = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": _CMD})
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
        trick_command = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": _CMD})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        base_height = ObsTerm(func=mdp.base_height_ground, params={"sensor_cfg": _SCAN})
        feet_height = ObsTerm(func=mdp.feet_height, params={"sensor_cfg": _SCAN, "asset_cfg": _FEET_BODIES})
        trick_yaw = ObsTerm(func=mdp.trick_yaw, params={"command_name": _CMD})
        body_contact = ObsTerm(
            func=mdp.body_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "Head_.*", ".*_thigh", ".*_calf", ".*_foot"])},
        )

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # -- standing while off
    stand_pose = RewTerm(func=mdp.stand_pose_off, weight=1.0, params={"std": 0.4, "command_name": _CMD})
    # Wide enough to pull the legs back from any trick pose; with the 0.4 one alone Pee and Wiggle stayed off it.
    stand_pose_coarse = RewTerm(func=mdp.stand_pose_off, weight=1.0, params={"std": 1.0, "command_name": _CMD})
    base_height = RewTerm(
        func=mdp.base_height_off, weight=0.5, params={"target": BASE_HEIGHT, "std": 0.04, "command_name": _CMD, "sensor_cfg": _SCAN}
    )
    level = RewTerm(func=mdp.level_off, weight=-5.0, params={"command_name": _CMD})
    still = RewTerm(func=mdp.still, weight=0.5, params={"std": 0.3, "command_name": _CMD})

    # -- feet stay where they are
    feet_air = RewTerm(func=mdp.support_feet_air, weight=-1.0, params={"sensor_cfg": _FEET_CONTACT, "command_name": _CMD})
    feet_slip = RewTerm(func=mdp.feet_slip, weight=-2.0, params={"sensor_cfg": _FEET_CONTACT, "asset_cfg": _FEET_BODIES})

    # -- inside the deploy checks
    tilt_over = RewTerm(func=mdp.tilt_over, weight=-20.0, params={"limit": TILT_LIMIT})
    safe_rom = RewTerm(func=mdp.safe_rom, weight=-10.0, params={"hard_limits": HARD_LIMITS, "margin": ROM_MARGIN})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- slow and gentle
    joint_vel_over = RewTerm(func=mdp.joint_vel_over, weight=-0.2, params={"limit": JOINT_VEL_CAP})
    base_ang_vel_over = RewTerm(func=mdp.base_ang_vel_over, weight=-1.0, params={"limit": BASE_ANG_VEL_CAP})
    base_lin_vel_over = RewTerm(func=mdp.base_lin_vel_over, weight=-1.0, params={"limit": BASE_LIN_VEL_CAP})
    impacts = RewTerm(
        func=mdp.contact_forces,
        weight=-0.01,
        params={"threshold": IMPACT_FORCE, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")},
    )
    body_hits = RewTerm(func=mdp.body_hits, weight=-0.02, params={"sensor_cfg": _NOT_FEET, "margin": 30.0})
    torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.05)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    energy = RewTerm(func=mdp.energy, weight=-5e-5)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    physics_blowup = DoneTerm(func=mdp.physics_blowup)
    body_down = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "Head_.*", ".*_thigh", ".*_calf"]),
        },
    )
    tipped = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


# ── Sim-to-real ────────────────────────────────────────────────────────────────────────────
# Simple's Deploy arm, as in Recovery: sensor bias, actuator delay, PD gains, CoM, joint plant, foot torsion.
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
    from handstand.assets.robots.unitree_actuators import GO2_KNEE_RATIO

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
class TrickBaseEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

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
        print(f"[Handstand] episode {EPISODE_S} s: off {FIRST_OFF} s, then on {ON_TIME} s and off {OFF_TIME} s in turn")
        print(f"[Handstand] push +-{PUSH_VEL} m/s every {PUSH_INTERVAL} s")
        print(f"[Handstand] inside tilt {math.degrees(TILT_LIMIT):.0f} deg and the outer {ROM_MARGIN:.0%} of each joint's range")


@configclass
class HandstandEnvCfg(TrickBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.trick.switches = len(LEGS)
        self.commands.trick.start_on_prob = START_ON_PROB
        self.events.start.params.update(on_legs_fraction=ON_LEGS_FRACTION, on_legs_tilt=ON_LEGS_TILT)
        self.terminations.tipped = DoneTerm(func=mdp.tipped_on_legs, params={"side_limit": SIDE_LIMIT, "tilt_limit": FALL_TILT})
        legs = {"legs": LEGS, "command_name": _CMD}
        target = {"F": HAND_PITCH, "R": HIND_PITCH}
        r = self.rewards
        r.feet_air.params["lifted"] = [[FEET.index(f) for f in FEET if f[0] != leg] for leg in LEGS]
        r.still.params["when_on"] = False
        r.tilt_over.params["command_name"] = _CMD
        r.unload = RewTerm(func=mdp.on_legs_unload, weight=1.0, params={"load": 75.0, **legs})
        r.pitch = RewTerm(func=mdp.on_legs_pitch, weight=2.0, params={"target": target, **legs})
        r.pitch_exact = RewTerm(func=mdp.on_legs_pitch_exact, weight=2.0, params={"target": target, "std": 0.15, **legs})
        r.balance = RewTerm(func=mdp.on_legs_balance, weight=1.0, params={"std": 0.05, **legs})
        r.hold = RewTerm(func=mdp.on_legs_still, weight=1.0, params={"target": target, "std": 0.5, **legs})
        r.sideways = RewTerm(func=mdp.sideways_on, weight=-5.0, params={"command_name": _CMD})
        print(f"[Handstand] up on the front legs at {HAND_PITCH} rad, on the rear legs at {HIND_PITCH} rad;"
              f" {ON_LEGS_FRACTION:.0%} of starts already up, tilted {ON_LEGS_TILT} rad")


def _apply_play(cfg) -> None:
    """Every robot on the same schedule: HANDSTAND_PLAY_OFF s off, then HANDSTAND_PLAY_ON s on, in turn.
    HANDSTAND_PLAY_PUSHES=0 turns off the pushes. HANDSTAND_PLAY_GRID=<n> puts n robots in a grid
    HANDSTAND_PLAY_SPACING apart on one tile, with the viewer looking at all of them.
    HANDSTAND_PLAY_EPISODE_S sets the episode length."""
    cfg.scene.num_envs = 32
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 4
    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05), roughness=0.9)
    off, on = _env("PLAY_OFF", 2.0), _env("PLAY_ON", 5.0)
    trick = cfg.commands.trick
    trick.first_off_range, trick.on_time_range, trick.off_time_range = (off, off), (on, on), (off, off)
    trick.cycle_switches = True
    cfg.events.start.params = {"joint_noise": 0.0, "lin_vel": 0.0, "ang_vel": 0.0, "yaw_range": 0.0}
    print(f"[Handstand] play: {off} s off, {on} s on, in turn")
    if os.environ.get("HANDSTAND_PLAY_EPISODE_S"):
        cfg.episode_length_s = float(os.environ["HANDSTAND_PLAY_EPISODE_S"])
        print(f"[Handstand] play: {cfg.episode_length_s} s episodes")
    grid = int(os.environ.get("HANDSTAND_PLAY_GRID", 0))
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
        print(f"[Handstand] play: {grid} robots {spacing} m apart on one tile")
    if os.environ.get("HANDSTAND_PLAY_PUSHES", "1") == "0":
        cfg.events.push = None
        print("[Handstand] play: no pushes")


@configclass
class HandstandPlayEnvCfg(HandstandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play(self)
