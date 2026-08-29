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
# Robot: Unitree A1 body with Go1 actuators (see `robot.actuators` assignment below).
#
# NOTE: this block used to claim DCMotorCfg was used "so PD gains can be randomized via
# write_joint_stiffness/damping_to_sim". Both halves were wrong. The code assigns
# UNITREE_QUADRUPED_CFG.actuators (Go1 = ActuatorNetMLP), not DCMotorCfg; and
# write_joint_stiffness_to_sim does not touch actuator-model gains for EITHER type -- Isaac Lab
# zeroes the PhysX drive for all explicit actuators, so writing there just re-enables a drive that
# is meant to stay off. PD-gain randomization now goes through the actuator model instead; see
# _randomize_view_state in quadruped_env.py.
#   - A1, GO2 variants  -> DCMotorCfg      (Kp=25, Kd=0.5; gains are meaningful)
#   - Go1 variant       -> ActuatorNetMLP  (torque from a learned net; gains unused)

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
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = deep_update(d.get(k, {}), v)
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

# ── Replay overlay: the checkpoint's own env.yaml wins on interface keys ──────────────
# QUADRUPED_TRAINING_PHASE is only set by the `train` action. Play / eval / mujoco select a
# CHECKPOINT, not a phase, so they fall through to the "phase1" default above -- which is a
# POSITION-mode phase. Replaying a torque checkpoint that way applies its joint efforts as
# position residuals (targets = action * action_scale + nominal_stance, handed to a PD loop):
# a 3 N*m output becomes a 0.75 rad target offset, and the robot cannot even stand.
#
# The observation width was the only part of this that failed loudly (skrl's "shape '[-1, 65]'
# is invalid for input of size 49"); control_mode, torque_scale and decimation all diverge in
# SILENCE, which is worse. Patching them one environment variable at a time is what let that
# happen -- so instead, point at the file the run actually recorded and overlay from it.
#
# Only INTERFACE keys are overlaid: the ones that define what the policy sees and how its
# output is applied. Terrain, num_envs, episode length, noise and every reward stay under the
# caller's control, because those are legitimate things to vary when replaying a policy.
_REPLAY_OVERLAY_KEYS = (
    "control_mode",
    "torque_scale",
    "obs_mode",
    "obs_history_len",
    "action_scale",
    "joint_limit_barrier_stiffness",
    "spawn_height",
)
_replay_yaml = os.environ.get("QUADRUPED_ENV_YAML", "")
if _replay_yaml and os.path.exists(_replay_yaml):
    try:
        with open(_replay_yaml, "r") as _f:
            _recorded = yaml.load(_f, Loader=yaml.UnsafeLoader) or {}
    except Exception as _e:                                    # noqa: BLE001
        print(f"[Quadruped] WARNING: could not read {_replay_yaml}: {_e}")
        _recorded = {}
    _applied = {}
    for _k in _REPLAY_OVERLAY_KEYS:
        if _k in _recorded and _recorded[_k] is not None:
            _old = _phase_cfg["env"].get(_k, "<unset>")
            if _old != _recorded[_k]:
                _applied[_k] = (_old, _recorded[_k])
            _phase_cfg["env"][_k] = _recorded[_k]
    if _applied:
        print(f"[Quadruped] Replaying checkpoint config from {_replay_yaml}")
        print(f"[Quadruped]   (phase '{_phase_name}' would have given the values on the left)")
        for _k, (_o, _n) in _applied.items():
            print(f"[Quadruped]   {_k}: {_o!r} -> {_n!r}")

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
                "max_timesteps": cfg["env"]["max_timesteps"]
            })

# Push terms are ALWAYS constructed, and "disabled" is expressed as a zero velocity range rather
# than a missing term. Previously this was `... else None`, which dropped the terms entirely when
# the *starting* phase had enable_pushes: false -- so in a phase1_to_phase2 run (phase1 disables
# pushes) the event terms never existed, and phase2 turning them back on had nothing to turn on.
# Phase 2's entire purpose is push hardening, and it was running with zero pushes.
# _transition_to_next_phase() now rewrites these ranges live at each phase change.
_vel_range = _phase_cfg["events"]["push_velocity_range"] if _phase_cfg["events"]["enable_pushes"] else [0.0, 0.0]

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
# ║  CONTROL MODE (position + PD  vs  direct joint torque)                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# "position" (default, unchanged behaviour): the policy outputs a residual on the nominal
#   stance, `target = a * action_scale + q_default`, tracked by the actuator model's PD at
#   200 Hz while the policy runs at 50 Hz (decimation 4).
#
# "torque": the policy outputs joint effort directly, `tau = a * torque_scale`, with no PD
#   loop at all. Three things have to change together for that to mean anything:
#     1. The policy must run at the rate the PD loop was running at -- 200 Hz, decimation 1.
#        Chen et al. (Humanoids 2023) treat this as load-bearing, not incidental: a torque
#        held open-loop for a 20 ms control step is what makes torque control "unstable".
#     2. The actuator model's own PD gains must be zeroed, or the DCMotor keeps adding
#        stiffness * (q_target - q) on top of our effort -- and q_target is whatever was
#        last written (the default pose), i.e. a large unwanted spring.
#     3. Nothing clamps joint positions any more (see _apply_action). Configuration sanity
#        has to come from the dof_pos_l2 reward terms instead -- Chen et al. use L1 pose
#        regularisation on hip and thigh at -1.0 for exactly this reason.
#
# Go1 is excluded: its ActuatorNetMLP maps a position-error history to torque, so there is
# no meaningful way to drive it with a commanded effort.

CONTROL_MODE: str = str(_phase_cfg["env"].get("control_mode", "position")).lower()
if CONTROL_MODE not in ("position", "torque"):
    raise ValueError(
        f"env.control_mode must be 'position' or 'torque', got {CONTROL_MODE!r}"
    )
TORQUE_CONTROL: bool = CONTROL_MODE == "torque"


def _strip_pd_gains(actuator_cfgs: dict) -> dict:
    """Zero an actuator model's PD gains so only the commanded effort reaches the joint.

    With stiffness and damping at 0, IdealPDActuator.compute reduces to
    `computed_effort = control_action.joint_efforts`, and DCMotor._clip_effort then applies
    the torque-speed envelope on top. That clamp is deliberately kept: it is a more faithful
    limit than the fixed ±33 N·m clamp used in the torque-control literature, and it is what
    keeps the commanded torque physically realisable at the joint's current speed.

    ActuatorNetMLP entries (Go1) are left untouched -- that model maps a position-error
    history to torque and has no gains to zero. Torque mode refuses to select Go1 outright,
    see the robot_choice guard below; skipping here just avoids raising while building the
    variant list for a robot the run will never spawn.
    """
    return {
        k: (v if type(v).__name__.startswith("ActuatorNetMLP")
            else v.replace(stiffness=0.0, damping=0.0))
        for k, v in actuator_cfgs.items()
    }


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
    # These are what _setup_scene actually builds the articulation from (it deep-copies the
    # matching ROBOT_VARIANTS entry), so this -- not QuadrupedEnvCfg.robot -- is where the
    # gains have to be zeroed for torque mode to take effect.
    if TORQUE_CONTROL:
        variant.actuators = _strip_pd_gains(variant.actuators)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ENVIRONMENT CONFIGURATION                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@configclass
class QuadrupedEnvCfg(DirectRLEnvCfg):

    _yaml_rob = _phase_cfg["env"]["robot_cfg"]
    _rob = _yaml_rob.upper() if _yaml_rob else os.environ.get("QUADRUPED_ROBOT", os.environ.get("FORCE_ROBOT", "RANDOM")).upper()
    robot_choice: str = _rob
    if TORQUE_CONTROL and (_rob in ("RANDOM", "") or "GO1" in _rob or "QUADRUPED" in _rob):
        raise ValueError(
            f"control_mode: torque cannot be used with robot_cfg={_rob!r}. Go1 drives its "
            f"joints through an ActuatorNetMLP, which derives torque from a joint-position-"
            f"error history -- there is nothing for a commanded effort to drive -- and RANDOM "
            f"mixes Go1 in with the other two. Set env.robot_cfg to GO2 or A1."
        )

    # ── Control mode ──────────────────────────────────────────────────────────
    control_mode: str = CONTROL_MODE
    # N·m per unit of policy output, applied before the actuator's own torque-speed clamp.
    # Scale this to the actuator envelope, not to action_scale. Isaac Lab's DCMotorCfg gives
    # Go2 effort_limit 23.5 N·m and A1 33.5 N·m; Chen et al. use 10.0 on A1, so ~7 is the
    # Go2 equivalent. Ignored in position mode.
    torque_scale: float = float(_phase_cfg["env"].get("torque_scale", 7.0))
    # Torque mode only. Restoring stiffness (N·m per rad of overshoot) for the soft joint
    # limit barrier that replaces position mode's target clamp -- see _joint_limit_barrier.
    # Defaults to the position mode's Kp so the limit feels the same in both modes.
    joint_limit_barrier_stiffness: float = float(
        _phase_cfg["env"].get("joint_limit_barrier_stiffness", 25.0)
    )

    # ── Exploration reset at a curriculum boundary ────────────────────────────
    # When the curriculum enters a phase with this set, the policy's log std is re-inflated
    # and the optimizer's learning rate is restored to its configured value -- see
    # _transition_to_next_phase in quadruped_env.py and the callback train.py registers.
    #
    # This exists because of phase0_torque. A phase whose optimum is "hold perfectly still"
    # is a phase whose optimum is a DETERMINISTIC policy, so PPO anneals std toward zero,
    # and KLAdaptiveLR follows it down (a tight policy means a large KL per unit change).
    # The measured cost: 150k steps of phase0_torque left std at 0.037 and the lr at 1.3e-5,
    # and phase1_torque then had neither the exploration to find a gait nor the step size to
    # learn one. Resetting on entry keeps what phase0 bought -- the mean action, a standing
    # posture -- while undoing the side effect it had on the search distribution.
    #
    # Off by default, so the position curriculum is unaffected and its baselines stand.
    reset_exploration_on_entry: bool = bool(
        _phase_cfg["env"].get("reset_exploration_on_entry", False)
    )
    # log std to re-inflate to. -0.7 -> std 0.5, half the initial_log_std 0.0 (std 1.0) that
    # a from-scratch run starts at: enough noise to search again, not so much that the
    # standing prior is drowned out by 7 N*m of white torque.
    reset_exploration_log_std: float = float(
        _phase_cfg["env"].get("reset_exploration_log_std", -0.7)
    )

    # ── Simulation ────────────────────────────────────────────────────────────
    # Torque control replaces the PD loop, so the policy has to run at the PD loop's rate:
    # sim dt 0.005 with decimation 1 -> 200 Hz, versus 50 Hz for position control.
    decimation = 1 if TORQUE_CONTROL else 4
    episode_length_s = _phase_cfg["env"]["episode_length_s"]
    obs_history_len = _phase_cfg["env"]["obs_history_len"]

    # Per-step reward is normalised to this control period so that per-*second* reward is
    # unchanged when the control rate changes -- otherwise a 200 Hz run collects 4x the
    # return per second of wall-clock and the torque/position comparison is not matched.
    # 0.02 is the 50 Hz position-mode period, so position runs are scaled by exactly 1.0
    # and behave bit-identically to before. Set to 0.0 to disable the normalisation.
    # NOTE: this does not fix the discount horizon -- gamma=0.99 spans 4x less time at
    # 200 Hz. Consider raising gamma in the skrl agent cfg for torque runs.
    reward_dt_ref: float = float(_phase_cfg["env"].get("reward_dt_ref", 0.02))
    sim: SimulationCfg = SimulationCfg(
        dt=0.005, 
        # Physics steps per render. Pinned to 4 rather than `decimation` so the viewer still
        # runs at ~50 Hz in torque mode instead of trying to draw every 200 Hz physics step.
        render_interval=4,
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
    # "repo"  -> the 49-dim vector this project has always used.
    # "paper" -> Table II of Dowdy & Chagas Vaz (SII 2026), 65-dim: no base linear velocity,
    #            plus linear acceleration, previous applied torque and foot contact. See
    #            QuadrupedEnv._paper_observations. History is not supported in paper mode --
    #            the paper stacks nothing.
    # QUADRUPED_OBS_MODE (exported by launcher.py from the checkpoint's env.yaml) wins over the
    # phase yaml. It has to: play/eval pick a checkpoint, not a training phase, so without it a
    # paper-trained policy gets replayed against whatever layout the DEFAULT phase happens to
    # specify. QUADRUPED_OBS_DIM already overrode the declared width that way, which produced a
    # config declaring 65 while _get_observations built 49 -- surfacing much later as skrl's
    # "shape '[-1, 65]' is invalid for input of size 49" at the first reset.
    obs_mode = os.environ.get(
        "QUADRUPED_OBS_MODE", _phase_cfg["env"].get("obs_mode", "repo")
    ).lower()
    _obs_base_dim = 65 if obs_mode == "paper" else 49
    observation_space = int(
        os.environ.get("QUADRUPED_OBS_DIM", _obs_base_dim * (1 + obs_history_len))
    )
    # Fail here, with the two numbers and their sources, rather than deep inside skrl's reshape.
    if observation_space % _obs_base_dim != 0:
        raise ValueError(
            f"observation_space={observation_space} is not a multiple of the "
            f"{_obs_base_dim}-wide '{obs_mode}' observation layout.\n"
            f"  QUADRUPED_OBS_DIM  = {os.environ.get('QUADRUPED_OBS_DIM', '<unset>')}\n"
            f"  QUADRUPED_OBS_MODE = {os.environ.get('QUADRUPED_OBS_MODE', '<unset>')}\n"
            f"  phase env.obs_mode = {_phase_cfg['env'].get('obs_mode', 'repo')}\n"
            f"A 65-dim checkpoint needs obs_mode 'paper'; a 49-dim one needs 'repo'."
        )
    # repo obs = [lin_vel(3) + ang_vel(3) + gravity(3) + cmd(4) + jpos(12) + jvel(12) + act(12)] = 49
    # paper obs = [lin_acc(3) + ang_vel(3) + quat(4) + cmd(3) + jpos(12) + jvel(12)
    #              + prev_torque(12) + prev_action(12) + contact(4)] = 65
    action_space = 12
    state_space = 0
    action_scale = _phase_cfg["env"]["action_scale"]
    
    base_max_timesteps = _phase_cfg["env"]["max_timesteps"]
    _total_timesteps = base_max_timesteps
    for p in _curriculum_phases:
        _total_timesteps += p["cfg"]["env"]["max_timesteps"]
    max_timesteps = _total_timesteps

    curriculum_phases = _curriculum_phases

    # ── Robot ─────────────────────────────────────────────────────────────────
    # A1 body (minimum rigid bodies → compatible view across all variants)
    # with DCMotorCfg actuators: Kp=25 Nm/rad, Kd=0.5 Nm·s/rad, τ_max=33.5 Nm
    robot: ArticulationCfg = UNITREE_A1_CFG.copy()
    robot.prim_path = "/World/envs/env_.*/Robot"
    # Go1's ActuatorNetMLP. This template is not what gets spawned -- _setup_scene builds
    # every articulation from ROBOT_VARIANTS -- so it is left as-is even in torque mode, and
    # the torque-mode gain zeroing happens on the variants above instead.
    robot.actuators = UNITREE_QUADRUPED_CFG.actuators.copy()
    # Height the base is teleported to on reset. 0.50 assumes the PD loop holds the default
    # stance through the ~0.2 m drop so the robot lands on its feet. Torque mode has no PD,
    # and at policy init the commanded torques are random, so the legs are effectively limp:
    # dropped from 0.50 the robot lands on its belly, trips base_height < 0.15 and dies on
    # the first step of every episode. Spawn it standing instead.
    spawn_height = float(
        _phase_cfg["env"].get("spawn_height", 0.35 if TORQUE_CONTROL else 0.50)
    )

    # ── Scene ─────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=_phase_cfg["env"]["num_envs"], 
        env_spacing=2.5, 
        replicate_physics=(_rob != "RANDOM"),
    )
    _ter = os.environ.get("QUADRUPED_TERRAIN", _phase_cfg["env"]["terrain"])
    scene.terrain = (
        TC_ROUGH if _ter == "rough" else (TC_FLAT if _ter == "flat" else TC_ALL)
    )

    # ── Sensors ───────────────────────────────────────────────────────────────
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/(.*_foot|.*_calf|.*_thigh)",
        # Must be >= decimation (4) so net_forces_w_history spans a whole control step -- the
        # max_contact_force penalty takes its peak across this window to catch touchdown impacts
        # that the single net_forces_w sample misses. Was 3, which covered only 15ms of the 20ms.
        # Spans one control step: 4 physics substeps at 50 Hz, 1 at 200 Hz. Kept at >=3 so the
        # touchdown-impact peak still has a window to look back over in torque mode.
        history_length=max(3, decimation),
        track_air_time=False,
    )


    # ── Observation noise (sim2real) ──────────────────────────────────────────
    observation_noise_scale = _phase_cfg["env"]["observation_noise_scale"]

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  DOMAIN RANDOMIZATION                                                 ║
    # ╚════════════════════════════════════════════════════════════════════════╝

    # Base mass and Center of Mass (CoM) Randomization
    payload_mass_range = tuple(_phase_cfg["domain_randomization"]["payload_mass_range"])
    com_displacement_range = tuple(_phase_cfg["domain_randomization"]["com_displacement_range"])

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
        )
        
        push_quadruped = EventTerm(
            func=push_robot_heterogeneous,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "asset_cfg": SceneEntityCfg("robot_quadruped"),
                "velocity_range": {"x": (_vel_range[0], _vel_range[1]), "y": (_vel_range[0], _vel_range[1])},
            },
        )
        
        push_go2 = EventTerm(
            func=push_robot_heterogeneous,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={
                "asset_cfg": SceneEntityCfg("robot_go2"),
                "velocity_range": {"x": (_vel_range[0], _vel_range[1]), "y": (_vel_range[0], _vel_range[1])},
            },
        )

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
    #   foot_height (reward form)     —        0.5      0.3       ↓ was encouraging jumping
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

    # Gait shaping
    rew_scale_feet_air_time = _phase_cfg["rewards"]["rew_scale_feet_air_time"]
    target_feet_air_time = _phase_cfg["rewards"]["target_feet_air_time"]
    feet_air_time_sigma = _phase_cfg["rewards"]["feet_air_time_sigma"]
    # Speed-dependent swing target: target_feet_air_time above is used at/above
    # feet_air_time_speed_hi; target_feet_air_time_slow is used at/below feet_air_time_speed_lo;
    # linearly ramped between. Longer swing at low commanded speed means lower cadence, so the same
    # small commanded velocity can be hit with a normal-amplitude stride instead of a tiny, same-
    # tempo twitch -- see the "why does frequency stay ~3Hz at every speed" discussion this followed.
    target_feet_air_time_slow = _phase_cfg["rewards"].get("target_feet_air_time_slow", target_feet_air_time)
    feet_air_time_speed_lo = _phase_cfg["rewards"].get("feet_air_time_speed_lo", 0.1)
    feet_air_time_speed_hi = _phase_cfg["rewards"].get("feet_air_time_speed_hi", 0.35)
    # NEGATIVE scale: this weights a PENALTY on how far the swing apex ended up from
    # target_foot_height, charged once per foot per landing (the historical values in the table
    # above are from when it was a positive lift reward, hence the "was encouraging jumping" note).
    rew_scale_foot_height_penalty = _phase_cfg["rewards"]["rew_scale_foot_height_penalty"]
    # POSITIVE scale on a DENSE lift reward: exp(-(h - target)^2/sigma) per airborne foot per
    # STEP, on instantaneous height (cp7's form). The incentive for early phases where the robot
    # has no reason to pick a foot up yet -- it has to out-earn standing still, which the
    # per-landing version could not (it was worth ~0.5% of the per-step total). Turn it off and
    # the penalty on once the gait exists: as a payout it rewards stepping high and often, which
    # is the exploit the penalty form removes.
    # .get() so a phase yaml predating this term still loads.
    rew_scale_foot_height_reward = _phase_cfg["rewards"].get("rew_scale_foot_height_reward", 0.0)
    # NEGATIVE scale on the number of feet in contact beyond feet_grounded_allowed, charged every
    # step while a move command is active. This is the term that breaks the standing local optimum:
    # every other gait reward is gated on a foot already being airborne, so all of them read zero
    # -- with zero gradient -- for the four-feet-planted robot they are supposed to get moving.
    # allowed=2 makes it inert for a trot (two feet down at all times), so it applies pressure
    # only until a gait exists and then stops.
    rew_scale_feet_grounded = _phase_cfg["rewards"].get("rew_scale_feet_grounded", 0.0)
    feet_grounded_allowed = _phase_cfg["rewards"].get("feet_grounded_allowed", 2.0)
    target_foot_height = _phase_cfg["rewards"]["target_foot_height"]
    foot_height_sigma = _phase_cfg["rewards"]["foot_height_sigma"]
    # Reset-state randomisation. Previously hardcoded at (-0.2, 0.2) rad and (-0.5, 0.5) rad/s in
    # _randomize_view_state; exposed so the paper phase can use Table I's much wider ranges
    # (joint positions (-0.3, 0.3), joint velocities (-2.5, 2.5), and base pose/velocity noise
    # this repo did not randomise at all). Defaults reproduce the old hardcoded behaviour exactly.
    reset_joint_pos_range = tuple(_phase_cfg["domain_randomization"].get("reset_joint_pos_range", [-0.2, 0.2]))
    reset_joint_vel_range = tuple(_phase_cfg["domain_randomization"].get("reset_joint_vel_range", [-0.5, 0.5]))
    reset_body_vel_range = tuple(_phase_cfg["domain_randomization"].get("reset_body_vel_range", [0.0, 0.0]))
    reset_body_pos_range = tuple(_phase_cfg["domain_randomization"].get("reset_body_pos_range", [0.0, 0.0]))
    reset_body_ori_range = tuple(_phase_cfg["domain_randomization"].get("reset_body_ori_range", [0.0, 0.0]))
    # Table I "Link Mass, Scale" -- multiplier on every body's mass. (1.0, 1.0) disables it.
    link_mass_scale_range = tuple(_phase_cfg["domain_randomization"].get("link_mass_scale_range", [1.0, 1.0]))
    # Table I "Foot Friction, New" -- resampled contact friction. (0.0, 0.0) disables it.
    foot_friction_range = tuple(_phase_cfg["domain_randomization"].get("foot_friction_range", [0.0, 0.0]))
    # Size of the fixed friction palette. PhysX allows at most 64K unique materials in a scene,
    # so friction CANNOT be freshly sampled per env per reset -- see _randomize_view_state.
    # Keep num_buckets * (number of physics scenes) well under 64K; 64 is plenty of resolution.
    friction_num_buckets = int(_phase_cfg["domain_randomization"].get("friction_num_buckets", 64))

    # ── Paper reward set: Dowdy & Chagas Vaz, IEEE/SICE SII 2026, Table III ──────────
    # A separate, self-contained reward set, computed in _compute_paper_rewards() rather than in
    # compute_rewards(). These are NOT the terms above with different weights: most use an L1 norm
    # where this repo squares, so the paper's weights only mean what the paper meant applied to the
    # paper's expressions. A phase turning this set on should zero the repo equivalents.
    # All default 0.0, so phases that do not mention them are unaffected.
    rew_scale_paper_alive = _phase_cfg["rewards"].get("rew_scale_paper_alive", 0.0)
    rew_scale_paper_feet_air_time = _phase_cfg["rewards"].get("rew_scale_paper_feet_air_time", 0.0)
    rew_scale_paper_track_lin_vel = _phase_cfg["rewards"].get("rew_scale_paper_track_lin_vel", 0.0)
    rew_scale_paper_track_ang_vel = _phase_cfg["rewards"].get("rew_scale_paper_track_ang_vel", 0.0)
    rew_scale_paper_gait = _phase_cfg["rewards"].get("rew_scale_paper_gait", 0.0)
    rew_scale_paper_action_smooth = _phase_cfg["rewards"].get("rew_scale_paper_action_smooth", 0.0)
    rew_scale_paper_air_time_var = _phase_cfg["rewards"].get("rew_scale_paper_air_time_var", 0.0)
    rew_scale_paper_base_motion = _phase_cfg["rewards"].get("rew_scale_paper_base_motion", 0.0)
    rew_scale_paper_base_orientation = _phase_cfg["rewards"].get("rew_scale_paper_base_orientation", 0.0)
    rew_scale_paper_foot_slip = _phase_cfg["rewards"].get("rew_scale_paper_foot_slip", 0.0)
    rew_scale_paper_dof_pos = _phase_cfg["rewards"].get("rew_scale_paper_dof_pos", 0.0)
    rew_scale_paper_dof_torque = _phase_cfg["rewards"].get("rew_scale_paper_dof_torque", 0.0)
    rew_scale_paper_dof_acc = _phase_cfg["rewards"].get("rew_scale_paper_dof_acc", 0.0)
    rew_scale_paper_dof_vel = _phase_cfg["rewards"].get("rew_scale_paper_dof_vel", 0.0)
    rew_scale_paper_thigh_contact = _phase_cfg["rewards"].get("rew_scale_paper_thigh_contact", 0.0)
    rew_scale_paper_calf_contact = _phase_cfg["rewards"].get("rew_scale_paper_calf_contact", 0.0)
    # Shape parameters. The first two are the paper's (0.3 s air-time offset, 0.25 tracking
    # divisor); paper_gait_sigma is NOT in the paper -- Table III leaves l_sync/l_async undefined,
    # so this is the width of the Gaussian in the reconstruction, in s^2. 0.01 puts the knee at
    # ~0.1 s of phase error between a pair of feet.
    paper_air_time_offset = _phase_cfg["rewards"].get("paper_air_time_offset", 0.3)
    paper_vel_tracking_sigma = _phase_cfg["rewards"].get("paper_vel_tracking_sigma", 0.25)
    # Split from the linear one so the two can scale differently between robots: a smaller
    # quadruped is slower in m/s but FASTER in rad/s, so the linear tolerance needs tightening
    # on a Go2 while the angular one does not. Defaults to the paper's single 0.25.
    paper_ang_tracking_sigma = _phase_cfg["rewards"].get("paper_ang_tracking_sigma", 0.25)
    paper_gait_sigma = _phase_cfg["rewards"].get("paper_gait_sigma", 0.01)
    # Swing duration below which the gait reward is scaled down, in seconds. Not a paper value --
    # it closes a degenerate solution in the reconstructed gait term, where four feet chattering
    # with equal tiny air/contact times score a perfect 1.0. See _compute_paper_rewards.
    paper_gait_min_swing = _phase_cfg["rewards"].get("paper_gait_min_swing", 0.15)

    # Landing impact: NEGATIVE scale on the foot's vertical speed at touchdown (target is zero --
    # set the foot down, don't drop it). Charged once per landing, on the pre-impact velocity.
    # .get() so a phase yaml predating this term still loads.
    rew_scale_foot_landing_vel = _phase_cfg["rewards"].get("rew_scale_foot_landing_vel", 0.0)
    foot_landing_vel_sigma = _phase_cfg["rewards"].get("foot_landing_vel_sigma", 0.6)

    # Ground reaction forces / impact
    rew_scale_grf_balance = _phase_cfg["rewards"]["rew_scale_grf_balance"]
    rew_scale_grf_target = _phase_cfg["rewards"]["rew_scale_grf_target"]
    rew_scale_max_contact_force = _phase_cfg["rewards"]["rew_scale_max_contact_force"]
    max_contact_force_pct = _phase_cfg["rewards"]["max_contact_force_pct"]

    # Stability penalties
    rew_scale_flat_orientation_l2 = _phase_cfg["rewards"]["rew_scale_flat_orientation_l2"]
    rew_scale_lin_vel_z_l2 = _phase_cfg["rewards"]["rew_scale_lin_vel_z_l2"]
    rew_scale_ang_vel_xy_l2 = _phase_cfg["rewards"]["rew_scale_ang_vel_xy_l2"]
    rew_scale_dof_pos_l2_walk = _phase_cfg["rewards"]["rew_scale_dof_pos_l2_walk"]
    rew_scale_dof_pos_l2_stance = _phase_cfg["rewards"]["rew_scale_dof_pos_l2_stance"]
    rew_scale_base_acc_l2 = _phase_cfg["rewards"]["rew_scale_base_acc_l2"]

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
    rew_scale_pos_deviation_l1 = _phase_cfg["rewards"]["rew_scale_pos_deviation_l1"]
    rew_scale_yaw_deviation_l1 = _phase_cfg["rewards"]["rew_scale_yaw_deviation_l1"]

    # Gait phase symmetry reward (all 6 leg pairs)
    rew_scale_gait_phase_sym  = _phase_cfg["rewards"]["rew_scale_gait_phase_sym"]
    gait_phase_offset_front   = _phase_cfg["rewards"]["gait_phase_offset_front"]
    gait_phase_offset_rear    = _phase_cfg["rewards"]["gait_phase_offset_rear"]
    gait_phase_offset_left    = _phase_cfg["rewards"]["gait_phase_offset_left"]
    gait_phase_offset_right   = _phase_cfg["rewards"]["gait_phase_offset_right"]
    gait_phase_offset_diag1   = _phase_cfg["rewards"]["gait_phase_offset_diag1"]
    gait_phase_offset_diag2   = _phase_cfg["rewards"]["gait_phase_offset_diag2"]

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  COMMANDS                                                             ║
    # ╚════════════════════════════════════════════════════════════════════════╝

    command_lin_vel_std = _phase_cfg["commands"]["command_lin_vel_std"]
    command_ang_vel_std = _phase_cfg["commands"]["command_ang_vel_std"]
    vel_tracking_sigma_exp = _phase_cfg["commands"].get("vel_tracking_sigma_exp", 2.0)

    command_x_range = tuple(_phase_cfg["commands"]["command_x_range"])              # [m/s]
    command_y_range = tuple(_phase_cfg["commands"]["command_y_range"])              # [m/s]
    command_yaw_range = tuple(_phase_cfg["commands"]["command_yaw_range"])            # [rad/s]
    command_resampling_time = _phase_cfg["commands"]["command_resampling_time"]              # [s]

    # Gait reward masking: ramps linearly from fully-static at ‖cmd‖ <= static_velocity_threshold
    # to fully-moving at ‖cmd‖ >= static_command_ramp (was a hard switch at the threshold).
    static_velocity_threshold = _phase_cfg["commands"]["static_velocity_threshold"]
    static_command_ramp = _phase_cfg["commands"].get("static_command_ramp", 0.1)
    max_pos_leash = _phase_cfg["commands"]["max_pos_leash"]
    max_yaw_leash = _phase_cfg["commands"]["max_yaw_leash"]

    # Zero-command fraction and single-axis fractions
    zero_command_fraction = _phase_cfg["commands"]["zero_command_fraction"]
    standby_duration_s = _phase_cfg["commands"]["standby_duration_s"]
    x_only_command_fraction = _phase_cfg["commands"]["x_only_command_fraction"]
    y_only_command_fraction = _phase_cfg["commands"]["y_only_command_fraction"]
    yaw_only_command_fraction = _phase_cfg["commands"]["yaw_only_command_fraction"]

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  TERMINATION                                                          ║
    # ╚════════════════════════════════════════════════════════════════════════╝

    # Terminate if cos(tilt angle) > this (i.e. too tilted)
    base_angle_termination_thresh = _phase_cfg["env"]["base_angle_termination_thresh"]
