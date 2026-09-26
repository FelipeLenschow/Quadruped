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

from simple.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from simple.tasks.locomotion import mdp

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
        if hasattr(self.sim, "physx"):
            self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        else:
            from isaaclab_physx.physics import PhysxCfg

            self.sim.physics = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
            self.sim.use_newton_actuators = False

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


# ── Sigma-Vel-Foot-Rough ───────────────────────────────────────────────────────────────────
# RobotEnvCfg (unitree's config as shipped) plus: command-scaled tracking kernel, base_lin_vel in
# the actor, a foot clearance reward, and rough terrain. Overridable from the environment; the
# resolved values are printed at startup and land in the run's params/env.yaml.
SIGMA_EXP = float(os.environ.get("PAPER_SIGMA_EXP", 1.0))


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
    track_lin_vel_xy clears 0.7 x weight, and BOTH fixes push that mean down -- the scaled kernel
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


# Set PAPER_KEEP_CURRICULUM=0 to drop the command-level curriculum -- see _drop_level_curriculum.
KEEP_CURRICULUM = os.environ.get("PAPER_KEEP_CURRICULUM", "1") == "1"


def _maybe_drop_curriculum(cfg) -> None:
    if not KEEP_CURRICULUM:
        _drop_level_curriculum(cfg)


# Walk's foot_height term with Final7's values. See mdp.foot_apex_reward.
FOOT_TARGET_HEIGHT = float(os.environ.get("PAPER_FOOT_TARGET_HEIGHT", 0.08))
FOOT_HEIGHT_SIGMA = float(os.environ.get("PAPER_FOOT_HEIGHT_SIGMA", 0.002))
FOOT_HEIGHT_BIAS = float(os.environ.get("PAPER_FOOT_HEIGHT_BIAS", 0.5))
FOOT_HEIGHT_WEIGHT = float(os.environ.get("PAPER_FOOT_HEIGHT_WEIGHT", 1.0))


def _apply_foot_clearance(cfg) -> None:
    cfg.rewards.foot_height = RewTerm(
        func=mdp.foot_apex_reward,
        weight=FOOT_HEIGHT_WEIGHT,
        params={
            "target_height": FOOT_TARGET_HEIGHT,
            "sigma": FOOT_HEIGHT_SIGMA,
            "bias": FOOT_HEIGHT_BIAS,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    print(
        f"[PaperArm] foot height at touchdown, target = {FOOT_TARGET_HEIGHT} m, sigma = {FOOT_HEIGHT_SIGMA},"
        f" bias = {FOOT_HEIGHT_BIAS}, weight = {FOOT_HEIGHT_WEIGHT}"
    )


# Walk's speed-dependent swing-time term, replacing unitree's feet_air_time (threshold 0.5 s,
# which never pays). See mdp.feet_air_time_target.
AIR_TIME_SLOW = float(os.environ.get("PAPER_AIR_TIME_SLOW", 0.25))
AIR_TIME_FAST = float(os.environ.get("PAPER_AIR_TIME_FAST", 0.35))
AIR_TIME_SPEED_LO = float(os.environ.get("PAPER_AIR_TIME_SPEED_LO", 0.1))
AIR_TIME_SPEED_HI = float(os.environ.get("PAPER_AIR_TIME_SPEED_HI", 0.35))
AIR_TIME_SIGMA = float(os.environ.get("PAPER_AIR_TIME_SIGMA", 0.05))
AIR_TIME_WEIGHT = float(os.environ.get("PAPER_AIR_TIME_WEIGHT", 0.2))


def _apply_air_time_target(cfg) -> None:
    cfg.rewards.feet_air_time = None
    cfg.rewards.feet_air_time_dyn = RewTerm(
        func=mdp.feet_air_time_target,
        weight=AIR_TIME_WEIGHT,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "target_slow": AIR_TIME_SLOW,
            "target_fast": AIR_TIME_FAST,
            "speed_lo": AIR_TIME_SPEED_LO,
            "speed_hi": AIR_TIME_SPEED_HI,
            "sigma": AIR_TIME_SIGMA,
        },
    )
    print(
        f"[PaperArm] swing time target {AIR_TIME_SLOW} s at {AIR_TIME_SPEED_LO} m/s -> {AIR_TIME_FAST} s at"
        f" {AIR_TIME_SPEED_HI} m/s, sigma = {AIR_TIME_SIGMA}, weight = {AIR_TIME_WEIGHT}"
    )


# Final7's slow-command share: 60% of resamples keep their direction but take a speed in
# 0.05-0.3 m/s. Uniform sampling puts only ~12% there, so the foot terms barely see slow steps.
# It switches on only once the level curriculum has reached the full x range, so a slow-heavy
# batch cannot hold mean tracking under the curriculum's 0.7 x weight bar.
SLOW_FRACTION = float(os.environ.get("PAPER_SLOW_FRACTION", 0.6))
SLOW_RANGE = (
    float(os.environ.get("PAPER_SLOW_LO", 0.05)),
    float(os.environ.get("PAPER_SLOW_HI", 0.3)),
)


def _apply_slow_coverage(cfg) -> None:
    cfg.commands.base_velocity.slow_command_fraction = SLOW_FRACTION
    cfg.commands.base_velocity.slow_command_range = SLOW_RANGE
    cfg.commands.base_velocity.slow_after_full_range = True
    print(
        f"[PaperArm] slow-command coverage after full range, fraction = {SLOW_FRACTION}, range = {SLOW_RANGE}"
    )


# Final7's posture terms: no whole-body pose penalty while walking (a high step needs knee and
# hip bend, which joint_pos charged at -0.7 on every moving step), the stand-still branch kept,
# and a hip-only L1 penalty so the legs do not splay.
POSE_MOVING_SCALE = float(os.environ.get("PAPER_POSE_MOVING_SCALE", 0.0))
HIP_DEV_WEIGHT = float(os.environ.get("PAPER_HIP_DEV_WEIGHT", -0.2))


def _apply_walking_posture(cfg) -> None:
    cfg.rewards.joint_pos.params["moving_scale"] = POSE_MOVING_SCALE
    cfg.rewards.hip_dev = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=HIP_DEV_WEIGHT,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    print(f"[PaperArm] pose penalty while moving x{POSE_MOVING_SCALE}, hip deviation weight = {HIP_DEV_WEIGHT}")


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
class RobotSigmaVelFootRoughEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_scaled_tracking(self)
        _maybe_drop_curriculum(self)
        _apply_slow_coverage(self)
        _apply_foot_clearance(self)
        _apply_air_time_target(self)
        _apply_walking_posture(self)
        _apply_lin_vel_obs(self)
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
class RobotSigmaVelFootRoughPlayEnvCfg(RobotSigmaVelFootRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)


# ── Deploy ─────────────────────────────────────────────────────────────────────────────────
# Sigma-Vel-Foot-Rough trained on a base_lin_vel the simulator hands over exactly. FeetHeight,
# trained that way, leans on it hard -- on the robot its thigh targets move 1-2.4 rad per m/s of
# estimated forward velocity, against 0.1-0.2 rad for Final7 -- and on hardware that input is a
# Kalman estimate driven by leg odometry, so the policy closes a loop through its own leg motion
# and oscillates at standstill. This arm adds what Final7 had and this one did not: a sensor
# model with drift, actuator latency, randomized PD gains, and inertia that is not nominal.
#
# The defaults are Final7's values as it actually ran, which means its yaml numbers times its
# observation_noise_scale of 1.25. Its per-channel WHITE noise is not repeated here except on
# base_lin_vel: unitree's stock policy group already carries more than Final7 did on every other
# channel (ang_vel +-0.2 against sigma 0.0375, joint_vel +-1.5 against sigma 0.375).

# Gaussian sigma, m/s -- Final7 used 0.1 x 1.25. A Unoise half-width of the same number would be
# 1.7x weaker (sigma = a / sqrt(3)), which is why this one channel is Gaussian.
DEPLOY_LIN_VEL_NOISE = float(os.environ.get("PAPER_DEPLOY_LIN_VEL_NOISE", 0.125))
# Per-episode constant offsets, uniform +-value. Estimator drift, gyro bias, IMU mounting tilt,
# and the Unitree zero-point calibration, which is per joint and the one hardware error a
# position-controlled policy cannot see at all.
DEPLOY_LIN_VEL_BIAS = float(os.environ.get("PAPER_DEPLOY_LIN_VEL_BIAS", 0.0625))
DEPLOY_ANG_VEL_BIAS = float(os.environ.get("PAPER_DEPLOY_ANG_VEL_BIAS", 0.025))
# Gaussian sigma on the gyro, rad/s. unitree ships +-0.2 uniform (sigma 0.115), which is ~10x a
# real Go2 IMU and 3x what Final7 used. It costs nothing while the policy only has to hit yaw
# rates near 1 rad/s, but the deficit is at 0.1-0.3 rad/s, and a policy cannot close a loop on a
# rate it cannot see. Measured: this does not change what an already-trained policy outputs
# (0.2 rad/s command, corruption off vs on: 0.040 vs 0.043) -- it is about what can be learned.
DEPLOY_ANG_VEL_NOISE = float(os.environ.get("PAPER_DEPLOY_ANG_VEL_NOISE", 0.04))
DEPLOY_GRAVITY_BIAS = float(os.environ.get("PAPER_DEPLOY_GRAVITY_BIAS", 0.025))
DEPLOY_JOINT_POS_BIAS = float(os.environ.get("PAPER_DEPLOY_JOINT_POS_BIAS", 0.019))
# Actuator command latency, in PHYSICS steps (sim.dt = 5 ms, so 5 = 25 ms). The robot's control
# loop measures ~29 ms per step, and the policy that shook had trained with none.
DEPLOY_MIN_DELAY = int(os.environ.get("PAPER_DEPLOY_MIN_DELAY", 0))
DEPLOY_MAX_DELAY = int(os.environ.get("PAPER_DEPLOY_MAX_DELAY", 5))
# Added to the nominal gains at every reset. Final7: +-5 on Kp 25, +-0.2 on Kd 0.5.
DEPLOY_KP_RANGE = (-float(os.environ.get("PAPER_DEPLOY_KP", 5.0)), float(os.environ.get("PAPER_DEPLOY_KP", 5.0)))
DEPLOY_KD_RANGE = (-float(os.environ.get("PAPER_DEPLOY_KD", 0.2)), float(os.environ.get("PAPER_DEPLOY_KD", 0.2)))
# Base centre of mass, m from nominal: fore/aft may legitimately skew, lateral stays centred.
DEPLOY_COM_X = float(os.environ.get("PAPER_DEPLOY_COM_X", 0.05))
DEPLOY_COM_Y = float(os.environ.get("PAPER_DEPLOY_COM_Y", 0.03))
# The Go2 USD has zero armature and zero viscous damping on every joint. The menagerie MuJoCo
# model carries armature 0.01 and damping 2.0, but eval_mujoco.py zeroes the damping (and sets
# frictionloss to 0.01) before it runs, so the eval sweep does not see it. Neither number has
# been identified on the robot; these ranges are there for robustness and include zero. A
# swing-leg torque/velocity fit against the deployment logs would pin them.
DEPLOY_ARMATURE = (
    float(os.environ.get("PAPER_DEPLOY_ARMATURE_LO", 0.005)),
    float(os.environ.get("PAPER_DEPLOY_ARMATURE_HI", 0.02)),
)
DEPLOY_VISCOUS = (
    float(os.environ.get("PAPER_DEPLOY_VISCOUS_LO", 0.0)),
    float(os.environ.get("PAPER_DEPLOY_VISCOUS_HI", 1.0)),
)
# Added to each joint's friction coefficient. Never negative, so this is a one-sided range.
DEPLOY_JOINT_FRICTION = (
    float(os.environ.get("PAPER_DEPLOY_JOINT_FRICTION_LO", 0.03)),
    float(os.environ.get("PAPER_DEPLOY_JOINT_FRICTION_HI", 0.5)),
)


def _bias_term(term, bias: float) -> None:
    """Wrap an observation term in mdp.biased_obs, keeping its noise, scale and clip."""
    term.params = {"func": term.func, "bias": bias, **term.params}
    term.func = mdp.biased_obs


def _apply_sensor_model(cfg) -> None:
    """Turn the actor's exact measurements into estimates: drift on four channels, noise on one.

    Mutating the terms in place keeps their slots in the group's declaration order, so the
    48-wide deployment layout is unchanged and Controller/policy_runner.py still feeds this
    policy without knowing any of this happened.
    """
    policy = cfg.observations.policy
    _bias_term(policy.base_lin_vel, DEPLOY_LIN_VEL_BIAS)
    _bias_term(policy.base_ang_vel, DEPLOY_ANG_VEL_BIAS)
    _bias_term(policy.projected_gravity, DEPLOY_GRAVITY_BIAS)
    _bias_term(policy.joint_pos_rel, DEPLOY_JOINT_POS_BIAS)
    policy.base_lin_vel.noise = Gnoise(std=DEPLOY_LIN_VEL_NOISE) if DEPLOY_LIN_VEL_NOISE else None
    policy.base_ang_vel.noise = Gnoise(std=DEPLOY_ANG_VEL_NOISE) if DEPLOY_ANG_VEL_NOISE else None
    print(
        f"[PaperArm] base_lin_vel is an estimate: Gaussian sigma {DEPLOY_LIN_VEL_NOISE} m/s,"
        f" per-episode bias +-{DEPLOY_LIN_VEL_BIAS} m/s"
    )
    print(f"[PaperArm] gyro noise: Gaussian sigma {DEPLOY_ANG_VEL_NOISE} rad/s")
    print(
        f"[PaperArm] per-episode bias: ang_vel +-{DEPLOY_ANG_VEL_BIAS} rad/s,"
        f" gravity +-{DEPLOY_GRAVITY_BIAS}, joint_pos +-{DEPLOY_JOINT_POS_BIAS} rad per joint"
    )


def _apply_actuator_delay(cfg) -> None:
    """Lag the joint command by 0-N physics steps, resampled per episode.

    UnitreeActuatorCfg already derives from DelayedPDActuatorCfg, so this is only its delay
    bounds. Replaced rather than mutated: the actuator cfgs are shared with the module-level
    UNITREE_GO2_CFG, and mutating them would leak into any other cfg built in this process.
    """
    cfg.scene.robot = cfg.scene.robot.replace(
        actuators={
            name: actuator.replace(min_delay=DEPLOY_MIN_DELAY, max_delay=DEPLOY_MAX_DELAY)
            for name, actuator in cfg.scene.robot.actuators.items()
        }
    )
    print(f"[PaperArm] actuator delay {DEPLOY_MIN_DELAY}-{DEPLOY_MAX_DELAY} physics steps")


def _apply_gain_randomization(cfg) -> None:
    cfg.events.randomize_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": DEPLOY_KP_RANGE,
            "damping_distribution_params": DEPLOY_KD_RANGE,
            "operation": "add",
        },
    )
    print(f"[PaperArm] PD gains randomized per reset: Kp {DEPLOY_KP_RANGE}, Kd {DEPLOY_KD_RANGE}")


# Foot spin friction, as a PhysX contact-patch radius in metres. A point contact resists no
# rotation about its own normal at all, so in sim the feet are free to spin in place and the
# policy learns a yaw that the real rubber foot, with a centimetre-wide patch under load, does
# not give it back. torsional_patch_radius scales with penetration; min_torsional_patch_radius
# is the floor that applies regardless, and on near-rigid feet it is the one that bites.
DEPLOY_FOOT_TORSION = float(os.environ.get("PAPER_DEPLOY_FOOT_TORSION", 0.02))
DEPLOY_FOOT_TORSION_MIN = float(os.environ.get("PAPER_DEPLOY_FOOT_TORSION_MIN", 0.01))


# Tolerance exponent for the YAW tracking kernel only: denominator = std^2 * |wz_cmd|^exp.
# The shipped exponent of 1 makes a small yaw command cheap to ignore -- at a 0.2 rad/s command
# the arm that turns at 0.06 rad/s (30% of what was asked) still collects 68% of the yaw reward.
# At exponent 2 the kernel is scale-invariant, the same 30% collects 14%, and the tolerance at
# 1 rad/s is untouched (0.94 either way), which is where tracking is already physics-limited.
DEPLOY_ANG_SIGMA_EXP = float(os.environ.get("PAPER_DEPLOY_ANG_SIGMA_EXP", 2.0))
DEPLOY_ANG_STD = float(os.environ.get("PAPER_DEPLOY_ANG_STD", math.sqrt(0.25)))


def _apply_yaw_tracking(cfg) -> None:
    cfg.rewards.track_ang_vel_z.params = {
        "command_name": "base_velocity",
        "std": DEPLOY_ANG_STD,
        "sigma_exp": DEPLOY_ANG_SIGMA_EXP,
    }
    print(f"[PaperArm] yaw tracking kernel: std {DEPLOY_ANG_STD}, sigma_exp {DEPLOY_ANG_SIGMA_EXP}")


# Share of commands reduced to one axis. The MuJoCo sweep and a joystick both ask for pure
# yaw; a uniform command box essentially never does, and the arm trained with foot torsion
# shows the consequence -- 0.24 rad/s of a 0.3 rad/s yaw command while walking at 0.5 m/s,
# 0.18 rad/s of the same command in place.
DEPLOY_X_ONLY = float(os.environ.get("PAPER_DEPLOY_X_ONLY", 0.0))
DEPLOY_Y_ONLY = float(os.environ.get("PAPER_DEPLOY_Y_ONLY", 0.0))
DEPLOY_YAW_ONLY = float(os.environ.get("PAPER_DEPLOY_YAW_ONLY", 0.15))


def _apply_axis_only_coverage(cfg) -> None:
    cfg.commands.base_velocity.x_only_command_fraction = DEPLOY_X_ONLY
    cfg.commands.base_velocity.y_only_command_fraction = DEPLOY_Y_ONLY
    cfg.commands.base_velocity.yaw_only_command_fraction = DEPLOY_YAW_ONLY
    print(
        f"[PaperArm] single-axis command share: x {DEPLOY_X_ONLY}, y {DEPLOY_Y_ONLY},"
        f" yaw {DEPLOY_YAW_ONLY}"
    )


# Episode starts. A share of episodes begin lying on folded legs, a share dropped from above
# standing height with some tilt, and every episode holds a zero command for a few seconds before
# the sampled one is revealed. Without this the policy only ever meets a standstill-to-command
# transition when a 10%-standing env happens to resample, and never has to get up at all.
DEPLOY_LYING_FRACTION = float(os.environ.get("PAPER_DEPLOY_LYING_FRACTION", 0.2))
DEPLOY_DROP_FRACTION = float(os.environ.get("PAPER_DEPLOY_DROP_FRACTION", 0.2))
DEPLOY_DROP_HEIGHT = (
    float(os.environ.get("PAPER_DEPLOY_DROP_HEIGHT_LO", 0.1)),
    float(os.environ.get("PAPER_DEPLOY_DROP_HEIGHT_HI", 0.3)),
)
DEPLOY_STANDBY = (
    float(os.environ.get("PAPER_DEPLOY_STANDBY_LO", 1.0)),
    float(os.environ.get("PAPER_DEPLOY_STANDBY_HI", 3.0)),
)
# Base contact does not terminate for this long; defaults to the longest standby, so a robot that
# starts on its belly or lands a drop on it has the whole standby to get up.
DEPLOY_CONTACT_GRACE = float(os.environ.get("PAPER_DEPLOY_CONTACT_GRACE", DEPLOY_STANDBY[1]))
# Go2 lying on folded legs, inside the 0.9 soft joint limits (the calf soft limit is about -2.63).
LYING_JOINT_POS = {
    ".*L_hip_joint": 0.1,
    ".*R_hip_joint": -0.1,
    ".*_thigh_joint": 1.3,
    ".*_calf_joint": -2.55,
}


# unitree's Go2 config gives every joint the hip motor's torque-speed curve, but the calf sits
# behind an extra 1.92:1 knee reduction: 45 N*m and 15.65 rad/s at the joint, not 23.4 and 30.
# Isaac Lab 3.0 fixed this in its own Go2 config; this asset is unitree's and never got it.
# Measured on the fresh run: the knee sits at the old limit 0.9% of the time in the first 1.5 s
# (standing up, landing) and 0.13% overall, so this corrects the plant more than it changes gaits.
DEPLOY_CALF_REDUCTION = os.environ.get("PAPER_DEPLOY_CALF_REDUCTION", "1") == "1"


def _apply_calf_reduction(cfg) -> None:
    """Scale the calf's torque-speed curve by the knee ratio, per joint within the one group.

    One group rather than a second calf group: the split measured ~10% slower per iteration
    (1.24 -> 1.36 s at 2048 envs), for the same physics.
    """
    if not DEPLOY_CALF_REDUCTION:
        print("[PaperArm] calf uses the hip motor curve (knee reduction off)")
        return
    from simple.assets.robots.unitree_actuators import GO2_KNEE_RATIO

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
            )
        }
    )
    print(
        f"[PaperArm] calf knee reduction {GO2_KNEE_RATIO:.2f}: Y1/Y2 {legs.Y1 * GO2_KNEE_RATIO:.1f}/"
        f"{legs.Y2 * GO2_KNEE_RATIO:.1f} N*m, X1/X2 {legs.X1 / GO2_KNEE_RATIO:.2f}/{legs.X2 / GO2_KNEE_RATIO:.2f} rad/s"
    )


def _apply_start_states(cfg) -> None:
    cfg.events.reset_start_pose = EventTerm(
        func=mdp.reset_start_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "lying_fraction": DEPLOY_LYING_FRACTION,
            "lying_joint_pos": LYING_JOINT_POS,
            "drop_fraction": DEPLOY_DROP_FRACTION,
            "drop_height_range": DEPLOY_DROP_HEIGHT,
        },
    )
    cfg.commands.base_velocity.standby_duration_range = DEPLOY_STANDBY
    cfg.terminations.base_contact = DoneTerm(
        func=mdp.illegal_contact_after,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
            "grace_s": DEPLOY_CONTACT_GRACE,
        },
    )
    print(
        f"[PaperArm] starts: lying {DEPLOY_LYING_FRACTION}, dropped {DEPLOY_DROP_FRACTION}"
        f" (+{DEPLOY_DROP_HEIGHT} m); standby {DEPLOY_STANDBY} s at zero command;"
        f" base contact allowed for the first {DEPLOY_CONTACT_GRACE} s"
    )


def _apply_foot_torsion(cfg) -> None:
    """Give every robot collider a torsional contact patch.

    Authored on the whole articulation, not just the feet: the patch only produces torque where
    there is a contact, and nothing but the feet is meant to touch the ground. make_uninstanceable
    is what makes it stick -- the Go2 USD ships its collision meshes as instance proxies, which are
    read-only, and the schema writer skips them with a warning that is easy to miss.
    """
    if DEPLOY_FOOT_TORSION <= 0.0 and DEPLOY_FOOT_TORSION_MIN <= 0.0:
        print("[PaperArm] foot torsional friction off")
        return
    from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg

    cfg.scene.robot = cfg.scene.robot.replace(
        spawn=cfg.scene.robot.spawn.replace(
            make_uninstanceable=True,
            collision_props=PhysxCollisionPropertiesCfg(
                torsional_patch_radius=DEPLOY_FOOT_TORSION,
                min_torsional_patch_radius=DEPLOY_FOOT_TORSION_MIN,
            ),
        )
    )
    print(
        f"[PaperArm] foot torsional patch radius {DEPLOY_FOOT_TORSION} m,"
        f" minimum {DEPLOY_FOOT_TORSION_MIN} m"
    )


def _apply_inertia_randomization(cfg) -> None:
    """Off-nominal base CoM and joint friction, drawn once per environment.

    Startup rather than reset, unlike Final7: both write through CPU tensors, which Isaac Lab
    warns against doing every episode, and 4096 environments already give the population the
    same spread. The payload mass this sits next to is randomized at startup for the same reason.
    """
    cfg.events.randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-DEPLOY_COM_X, DEPLOY_COM_X), "y": (-DEPLOY_COM_Y, DEPLOY_COM_Y)},
        },
    )
    cfg.events.randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": DEPLOY_JOINT_FRICTION,
            "operation": "add",
        },
    )
    cfg.events.randomize_joint_plant = EventTerm(
        func=mdp.randomize_joint_plant,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_range": DEPLOY_ARMATURE,
            "viscous_friction_range": DEPLOY_VISCOUS,
        },
    )
    print(
        f"[PaperArm] base CoM +-{DEPLOY_COM_X} m fore/aft, +-{DEPLOY_COM_Y} m lateral;"
        f" joint friction +{DEPLOY_JOINT_FRICTION}, armature {DEPLOY_ARMATURE},"
        f" viscous damping {DEPLOY_VISCOUS}"
    )


@configclass
class RobotSigmaVelFootRoughDeployEnvCfg(RobotSigmaVelFootRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_calf_reduction(self)
        _apply_sensor_model(self)
        _apply_actuator_delay(self)
        _apply_gain_randomization(self)
        _apply_inertia_randomization(self)
        _apply_foot_torsion(self)
        _apply_axis_only_coverage(self)
        _apply_yaw_tracking(self)
        _apply_start_states(self)


@configclass
class RobotSigmaVelFootRoughDeployPlayEnvCfg(RobotSigmaVelFootRoughDeployEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)


# ── Clock ──────────────────────────────────────────────────────────────────────────────────
# Deploy plus a trot clock whose rate follows the command (mdp.SpeedClockCommand): stride
# L = stride_min + stride_gain * v, frequency v / L, fixed swing time. The swing-time and apex
# terms are replaced by Walk These Ways' contact schedule and dense swing-height terms. The actor
# gets sin and cos of each foot's phase, 48 + 8 inputs.
CLOCK_FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
CLOCK_STRIDE_MIN = float(os.environ.get("PAPER_CLOCK_STRIDE_MIN", 0.10))
CLOCK_STRIDE_GAIN = float(os.environ.get("PAPER_CLOCK_STRIDE_GAIN", 0.3))
CLOCK_SWING_TIME = float(os.environ.get("PAPER_CLOCK_SWING_TIME", 0.2))
CLOCK_MAX_FREQUENCY = float(os.environ.get("PAPER_CLOCK_MAX_FREQUENCY", 3.0))
CLOCK_SWING_HEIGHT = float(os.environ.get("PAPER_CLOCK_SWING_HEIGHT", 0.08))
CLOCK_W_FORCE = float(os.environ.get("PAPER_CLOCK_W_FORCE", 2.0))
CLOCK_W_VEL = float(os.environ.get("PAPER_CLOCK_W_VEL", 0.5))
# At -20 the term cost 2% of the tracking reward and feet cleared 2-6 cm of the 8 asked.
CLOCK_W_SWING = float(os.environ.get("PAPER_CLOCK_W_SWING", -150.0))
# Standing height of the Go2 in its default pose; walking without this term rose to 0.37-0.38 m.
CLOCK_BASE_HEIGHT = float(os.environ.get("PAPER_CLOCK_BASE_HEIGHT", 0.32))
CLOCK_W_HEIGHT = float(os.environ.get("PAPER_CLOCK_W_HEIGHT", -40.0))


def _apply_clock(cfg) -> None:
    cfg.commands.clock = mdp.SpeedClockCommandCfg(
        foot_names=tuple(CLOCK_FEET),
        stride_min=CLOCK_STRIDE_MIN,
        stride_gain=CLOCK_STRIDE_GAIN,
        swing_time=CLOCK_SWING_TIME,
        max_frequency=CLOCK_MAX_FREQUENCY,
    )
    cfg.observations.policy.clock = ObsTerm(func=mdp.speed_clock, params={"command_name": "clock"})
    cfg.observations.critic.clock = ObsTerm(func=mdp.speed_clock, params={"command_name": "clock"})
    cfg.observations.critic.clock_contact = ObsTerm(func=mdp.speed_clock_contact, params={"command_name": "clock"})
    cfg.observations.critic.clock_rate = ObsTerm(func=mdp.generated_commands, params={"command_name": "clock"})
    if cfg.scene.height_scanner is None:
        raise ValueError("Clock needs the height scanner for base height; unset PAPER_NO_HEIGHT_SCANNER.")
    scanner = SceneEntityCfg("height_scanner")
    cfg.observations.critic.base_height = ObsTerm(func=mdp.base_height_ground, params={"sensor_cfg": scanner})

    r = cfg.rewards
    r.feet_air_time_dyn = None
    r.air_time_variance = None
    r.foot_height = None
    feet_asset = SceneEntityCfg("robot", body_names=CLOCK_FEET, preserve_order=True)
    feet_sensor = SceneEntityCfg("contact_forces", body_names=CLOCK_FEET, preserve_order=True)
    r.clock_contact_force = RewTerm(func=mdp.clock_contact_force, weight=CLOCK_W_FORCE, params={"sensor_cfg": feet_sensor})
    r.clock_contact_vel = RewTerm(func=mdp.clock_contact_vel, weight=CLOCK_W_VEL, params={"asset_cfg": feet_asset})
    r.clock_swing_height = RewTerm(
        func=mdp.clock_swing_height,
        weight=CLOCK_W_SWING,
        params={"asset_cfg": feet_asset, "swing_height": CLOCK_SWING_HEIGHT},
    )
    r.base_height = RewTerm(
        func=mdp.base_height_moving,
        weight=CLOCK_W_HEIGHT,
        params={"target_height": CLOCK_BASE_HEIGHT, "sensor_cfg": scanner},
    )
    print(f"[Clock] base height {CLOCK_BASE_HEIGHT} m above the scan, weight {CLOCK_W_HEIGHT}, critic only")
    print(
        f"[Clock] stride {CLOCK_STRIDE_MIN} + {CLOCK_STRIDE_GAIN} v m, swing {CLOCK_SWING_TIME} s,"
        f" f <= {CLOCK_MAX_FREQUENCY} Hz, swing height {CLOCK_SWING_HEIGHT} m,"
        f" weights force {CLOCK_W_FORCE} vel {CLOCK_W_VEL} swing {CLOCK_W_SWING}"
    )


@configclass
class RobotClockEnvCfg(RobotSigmaVelFootRoughDeployEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_clock(self)


@configclass
class RobotClockPlayEnvCfg(RobotClockEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)
