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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
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
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

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

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

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
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
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
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-1, 1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            # self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        # def __post_init__(self):
        #     self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
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

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)

    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )

    # -- feet
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    # feet_contact_forces = RewTerm(
    #     func=mdp.contact_forces,
    #     weight=-0.02,
    #     params={
    #         "threshold": 100.0,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #     },
    # )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # The height scanner has NO readers: the only term that would consume it is the
        # commented-out height_scan ObsTerm in CriticCfg above. It still casts
        # 187 rays x num_envs every control step (~18M per 24-step iteration at 4096 envs)
        # against the generated terrain mesh, and the result is discarded.
        #
        # MEASURED: removing it changes iteration time by nothing. It was the first suspect for
        # why this env runs ~2.5x slower per iteration than the Direct-env port (4586 vs 1822 ms
        # at 4096 envs), and it is not the answer -- 766k rays/step is apparently cheap next to
        # everything else here. Do not re-test it. The remaining candidates are the generated
        # terrain MESH (vs an analytic ground plane), the contact sensor covering every body with
        # track_air_time, and manager-based per-term dispatch; the last is inherent to this repo.
        #
        # Kept as a toggle because the compute is still wasted, and default OFF so the scene
        # stays identical to the one U0 was trained in. A RayCaster is read-only -- no physics
        # state, no articulation, no RNG -- so dropping it cannot change what a policy learns.
        if os.environ.get("PAPER_NO_HEIGHT_SCANNER", "0") == "1":
            self.scene.height_scanner = None
            print("[PaperArm] height scanner removed (unused; nothing reads height_scan).")

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


# ── Paper arms ────────────────────────────────────────────────────────────────────────────
# Three configs that differ from RobotEnvCfg by exactly one thing each, to separate the two
# candidate causes of the low-speed dead zone:
#
#   RobotEnvCfg          baseline. Unitree's config as shipped.            (arm U0)
#   RobotSigmaEnvCfg     + command-scaled tracking kernel.                 (arm S)  reward SHAPE
#   RobotCoverageEnvCfg  + slow-command quota in the sampler.              (arm C)  command COVERAGE
#   RobotSigmaDensityEnvCfg  + both, to check they are not redundant.      (arm SC)
#
# Nothing else moves: same rewards, weights, curriculum, events, terminations, PPO config and
# sample budget. Overridable from the environment so a sweep needs no edit here; the resolved
# value is printed at startup and lands in the run's params/env.yaml either way.
SIGMA_EXP = float(os.environ.get("PAPER_SIGMA_EXP", 1.0))
SLOW_FRACTION = float(os.environ.get("PAPER_SLOW_FRACTION", 0.6))
SLOW_RANGE = (
    float(os.environ.get("PAPER_SLOW_LO", 0.05)),
    float(os.environ.get("PAPER_SLOW_HI", 0.3)),
)


def _apply_scaled_tracking(cfg) -> None:
    """Swap both tracking rewards for the command-scaled kernel.

    The ATTRIBUTE names stay `track_lin_vel_xy` / `track_ang_vel_z`: mdp.lin_vel_cmd_levels
    looks the linear one up by name to decide when to widen the command box, and renaming it
    would silently disable the curriculum. Only `func` and `params` change.
    """
    cfg.rewards.track_lin_vel_xy.func = mdp.track_lin_vel_xy_exp_scaled
    cfg.rewards.track_lin_vel_xy.params = {
        "command_name": "base_velocity",
        "std": math.sqrt(0.25),
        "sigma_exp": SIGMA_EXP,
    }
    cfg.rewards.track_ang_vel_z.func = mdp.track_ang_vel_z_exp_scaled
    cfg.rewards.track_ang_vel_z.params = {
        "command_name": "base_velocity",
        "std": math.sqrt(0.25),
        "sigma_exp": SIGMA_EXP,
    }
    print(f"[PaperArm] command-scaled tracking kernel, sigma_exp = {SIGMA_EXP}")


def _apply_slow_coverage(cfg) -> None:
    """Give the sampler an explicit slow-command quota."""
    cfg.commands.base_velocity.slow_command_fraction = SLOW_FRACTION
    cfg.commands.base_velocity.slow_command_range = SLOW_RANGE
    print(f"[PaperArm] slow-command coverage, fraction = {SLOW_FRACTION}, range = {SLOW_RANGE}")
    _drop_level_curriculum(cfg)


# Noise on the velocity estimate, as a Unoise half-width in m/s. 0.0 (the default) keeps
# "+ base_lin_vel" a strictly one-variable change from the arm below it. On hardware this channel
# is a state estimate and easily the worst input the policy gets, so a deployable version wants
# something like PAPER_LIN_VEL_NOISE=0.08 -- which is a second experiment, not a free addition.
LIN_VEL_NOISE = float(os.environ.get("PAPER_LIN_VEL_NOISE", 0.0))


@configclass
class PolicyWithLinVelCfg(ObsGroup):
    """unitree's policy observation group with base_lin_vel prepended.

    Declared in full rather than subclassing PolicyCfg and adding a field: ObservationManager
    reads terms from the group's __dict__ in declaration order, and a subclass APPENDS, which
    would put base_lin_vel last. Order matters downstream -- Controller/policy_runner.py builds
    the deployment observation positionally, and this ordering is exactly its 48-wide layout
    ([lin_vel, ang_vel, grav, cmd, jpos, jvel, act]), so a policy trained here runs on the robot
    unchanged. Everything else is identical to PolicyCfg above.
    """

    base_lin_vel = ObsTerm(
        func=mdp.base_lin_vel,
        clip=(-100, 100),
        noise=Unoise(n_min=-LIN_VEL_NOISE, n_max=LIN_VEL_NOISE) if LIN_VEL_NOISE else None,
    )
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"})
    joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5))
    last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


def _apply_lin_vel_obs(cfg) -> None:
    """Give the ACTOR the base linear velocity the critic already sees.

    unitree's actor has no velocity input: it is told what speed to go and never told what speed
    it is going. MEASURED, same checkpoint, same sweep points -- MuJoCo reaches only 0.71-0.76 of
    the speed Isaac does from 0.5 m/s up, and ~0.25 below 0.2 m/s. The fixed arm loses the SAME
    fraction as the baseline (0.71 and 0.75 at 0.5 / 0.75 m/s for both), so the gap is not
    specific to how a policy was trained -- which leaves two candidates: an open-loop policy
    unable to correct a dynamics mismatch, or a genuine physics difference between the two
    simulators. This arm separates them: if it is the former, mujoco/isaac goes to ~1.0; if the
    latter, it improves and plateaus, and the residual is real physics to measure rather than
    train around.

    The critic already has base_lin_vel, which does not help -- only the actor acts.
    """
    cfg.observations.policy = PolicyWithLinVelCfg()
    noise = f", Unoise +-{LIN_VEL_NOISE} m/s" if LIN_VEL_NOISE else ", no noise"
    print(f"[PaperArm] actor observation now includes base_lin_vel (48 dims{noise})")


def _drop_level_curriculum(cfg) -> None:
    """Start at the full command box and remove lin_vel_cmd_levels.

    MEASURED, on the first Both run: the curriculum sat at +-0.1 for 1200 of 3000 iterations
    before widening, against 750 for the unmodified baseline. It widens only when mean
    track_lin_vel_xy clears 0.8 x weight, and BOTH fixes push that mean down -- the scaled kernel
    tightens the tolerance, and the slow quota fills the batch with the commands that are hardest
    to score on. So the arm spent half its budget training on +-0.1 alone, and was under-trained
    at wide commands relative to the baseline it exists to be compared against.

    Removing it is not a shortcut. The curriculum's job is to hand the policy easy (slow)
    commands early; slow_command_fraction supplies exactly those, for the whole run, by
    construction. Keeping both leaves two mechanisms fighting over the same budget.

    Terrain levels are untouched -- only the command-range curriculum goes.
    """
    ranges = cfg.commands.base_velocity.ranges
    limits = cfg.commands.base_velocity.limit_ranges
    ranges.lin_vel_x = limits.lin_vel_x
    ranges.lin_vel_y = limits.lin_vel_y
    ranges.ang_vel_z = limits.ang_vel_z
    cfg.curriculum.lin_vel_cmd_levels = None
    print(f"[PaperArm] level curriculum removed; command box fixed at x {ranges.lin_vel_x} y {ranges.lin_vel_y}")


# Default ON for arms without the slow-command quota (arms with it always drop the curriculum).
# Set PAPER_KEEP_CURRICULUM=0 to drop it there too -- see _drop_level_curriculum.
KEEP_CURRICULUM = os.environ.get("PAPER_KEEP_CURRICULUM", "1") == "1"


def _maybe_drop_curriculum(cfg) -> None:
    if not KEEP_CURRICULUM:
        _drop_level_curriculum(cfg)


@configclass
class RobotSigmaEnvCfg(RobotEnvCfg):
    """Arm S: baseline + the command-scaled tracking kernel, and nothing else."""

    def __post_init__(self):
        super().__post_init__()
        _apply_scaled_tracking(self)
        _maybe_drop_curriculum(self)


@configclass
class RobotCoverageEnvCfg(RobotEnvCfg):
    """Arm C: baseline + slow-command coverage, and nothing else."""

    def __post_init__(self):
        super().__post_init__()
        _apply_slow_coverage(self)


@configclass
class RobotSigmaDensityEnvCfg(RobotEnvCfg):
    """Arm SC: both fixes, to test whether either is redundant given the other."""

    def __post_init__(self):
        super().__post_init__()
        _apply_scaled_tracking(self)
        _apply_slow_coverage(self)


@configclass
class RobotVelEnvCfg(RobotEnvCfg):
    """Arm V: baseline + base_lin_vel in the actor, nothing else. Isolates velocity feedback."""

    def __post_init__(self):
        super().__post_init__()
        _apply_lin_vel_obs(self)


@configclass
class RobotSigmaDensityVelEnvCfg(RobotEnvCfg):
    """Arm SCV: both fixes plus velocity feedback -- the candidate for hardware."""

    def __post_init__(self):
        super().__post_init__()
        _apply_scaled_tracking(self)
        _apply_slow_coverage(self)
        _apply_lin_vel_obs(self)


@configclass
class RobotSigmaVelEnvCfg(RobotSigmaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_lin_vel_obs(self)


FOOT_TARGET_HEIGHT = float(os.environ.get("PAPER_FOOT_TARGET_HEIGHT", 0.07))
FOOT_CLEARANCE_STD = float(os.environ.get("PAPER_FOOT_CLEARANCE_STD", 0.005))
FOOT_CLEARANCE_WEIGHT = float(os.environ.get("PAPER_FOOT_CLEARANCE_WEIGHT", 0.5))


def _apply_foot_clearance(cfg) -> None:
    cfg.rewards.feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=FOOT_CLEARANCE_WEIGHT,
        params={
            "std": FOOT_CLEARANCE_STD,
            "tanh_mult": 2.0,
            "target_height": FOOT_TARGET_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    print(
        f"[PaperArm] foot clearance, target = {FOOT_TARGET_HEIGHT} m, std = {FOOT_CLEARANCE_STD},"
        f" weight = {FOOT_CLEARANCE_WEIGHT}"
    )


@configclass
class RobotVelFootEnvCfg(RobotVelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_foot_clearance(self)


@configclass
class RobotSigmaFootEnvCfg(RobotSigmaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_foot_clearance(self)


@configclass
class RobotSigmaVelFootEnvCfg(RobotSigmaFootEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_lin_vel_obs(self)


@configclass
class RobotSigmaDensityVelFootEnvCfg(RobotSigmaDensityVelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_foot_clearance(self)


ROUGH_NOISE = float(os.environ.get("PAPER_ROUGH_NOISE", 0.02))


def _apply_rough_terrain(cfg) -> None:
    cfg.scene.terrain.terrain_generator = cfg.scene.terrain.terrain_generator.replace(
        sub_terrains={
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=1.0, noise_range=(0.0, ROUGH_NOISE), noise_step=0.005, border_width=0.25
            ),
        }
    )
    print(f"[PaperArm] rough terrain, uniform noise 0-{ROUGH_NOISE} m")


@configclass
class RobotSigmaDensityVelRoughEnvCfg(RobotSigmaDensityVelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_rough_terrain(self)


@configclass
class RobotSigmaVelFootRoughEnvCfg(RobotSigmaVelFootEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_rough_terrain(self)


def _apply_play_overrides(cfg) -> None:
    """Shrink the scene for viewing and open the command range up to the trained limits."""
    cfg.scene.num_envs = 32
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 1
    cfg.commands.base_velocity.ranges = cfg.commands.base_velocity.limit_ranges

    # Play normally samples the FULL +-1.0 command range, where only ~1.4% of draws fall
    # below 0.3 m/s -- so a low-speed dead zone is essentially invisible on screen, and
    # what you watch is not what a slow-command sweep measures. PLAY_CMD_X pins every
    # env to one forward speed (yaw and lateral zeroed, no standing envs) so the slow
    # band can actually be observed here rather than only in the MuJoCo sweep.
    pinned = os.environ.get("PLAY_CMD_X")
    if pinned:
        v = float(pinned)
        cfg.commands.base_velocity.ranges.lin_vel_x = (v, v)
        cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        cfg.commands.base_velocity.rel_standing_envs = 0.0
        print(f"[PlayCfg] Every env pinned to lin_vel_x = {v} m/s.")


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)


@configclass
class RobotSigmaPlayEnvCfg(RobotSigmaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)


@configclass
class RobotCoveragePlayEnvCfg(RobotCoverageEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)


@configclass
class RobotSigmaDensityPlayEnvCfg(RobotSigmaDensityEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)


@configclass
class RobotVelPlayEnvCfg(RobotVelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)


@configclass
class RobotSigmaDensityVelPlayEnvCfg(RobotSigmaDensityVelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)


@configclass
class RobotVelFootPlayEnvCfg(RobotVelFootEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)


@configclass
class RobotSigmaFootPlayEnvCfg(RobotSigmaFootEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)


@configclass
class RobotSigmaVelFootPlayEnvCfg(RobotSigmaVelFootEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)


@configclass
class RobotSigmaDensityVelRoughPlayEnvCfg(RobotSigmaDensityVelRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)


@configclass
class RobotSigmaVelFootRoughPlayEnvCfg(RobotSigmaVelFootRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)


@configclass
class RobotSigmaDensityVelFootPlayEnvCfg(RobotSigmaDensityVelFootEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)


@configclass
class RobotSigmaVelPlayEnvCfg(RobotSigmaVelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)
