# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Commanded-vs-actual velocity sweep, measured in Isaac Sim.

Why this exists rather than reusing the MuJoCo sweep: unitree_rl_lab's actor has NO
``base_lin_vel`` input. It runs open loop on velocity and cannot observe -- let alone correct --
any dynamics difference between simulators, so its MuJoCo numbers understate it badly (~0.74 m/s
against a 1.0 command in MuJoCo vs ~1.0 in Isaac). Every arm derived from that config inherits
the problem, so the dead zone has to be measured in the simulator the policy was trained in.
MuJoCo stays useful, but as a TRANSFER check reported alongside, never as the primary number.

Each sweep point gets its own block of environments, all pinned to that command for the whole
run, so the entire curve comes out of one rollout instead of one run per speed. The first
``--settle`` seconds are discarded and the next ``--measure`` seconds are averaged.

Writes ``isaac_eval_report_<checkpoint>.json`` next to the checkpoint, in the same shape as
Mujoco/eval_mujoco.py's report ({"metadata": ..., "results": {axis: {speed: metrics}}}) so both
plot on one axis.

    python scripts/rsl_rl/sweep.py --task=Unitree-Go2-Velocity-Both --headless \
        --checkpoint logs/rsl_rl/unitree_go2_velocity_both/Both/model_2999.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Velocity sweep for an RSL-RL checkpoint, in Isaac Sim.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--envs_per_point", type=int, default=8, help="Environments averaged per sweep point.")
parser.add_argument("--settle", type=float, default=3.0, help="Seconds discarded before measuring.")
parser.add_argument("--measure", type=float, default=6.0, help="Seconds averaged per point.")
parser.add_argument("--axes", type=str, default="x,y,yaw", help="Comma-separated axes to sweep.")
parser.add_argument(
    "--speeds",
    type=str,
    # The three points around 0.30 pin down where the cliff actually falls; on the MuJoCo sweeps
    # it was known only to be somewhere in the 0.25-0.35 gap.
    default="0.0,0.05,0.10,0.15,0.20,0.25,0.28,0.30,0.32,0.35,0.50,0.75,1.00",
    help="Comma-separated commanded speeds.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _round(x, n=3):
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    if isinstance(x, list):
        return [round(float(v), n) for v in x]
    return round(float(x), n)


class GaitTracker:
    """Per-environment gait statistics accumulated over the measurement window.

    Foot height is taken relative to that environment's terrain origin, not world z: on a
    generated terrain each tile sits at its own height, so an uncorrected world z reports the
    tile the robot happens to be standing on rather than its clearance.
    """

    def __init__(self, num_envs, sensor_foot_ids, robot_foot_ids, device):
        # TWO index sets, deliberately. The contact sensor's prim_path is "Robot/.*", so its body
        # list is built independently of the articulation's and the two orderings do not match.
        # Using the sensor's indices against robot.data.body_pos_w read whichever body happened
        # to share the slot -- it reported foot clearances of 22 METRES before this was split.
        self.sensor_foot_ids = sensor_foot_ids
        self.robot_foot_ids = robot_foot_ids
        self.device, self.n = device, len(robot_foot_ids)
        assert len(sensor_foot_ids) == len(robot_foot_ids), "foot count mismatch between sensor and articulation"
        z = lambda *s: torch.zeros(*s, device=device)
        self.prev_contact = torch.zeros(num_envs, self.n, dtype=torch.bool, device=device)
        self.air_time = z(num_envs, self.n)
        self.swing_peak = z(num_envs, self.n)
        self.lift_sum = z(num_envs, self.n)
        self.swing_sum = z(num_envs, self.n)
        self.landings = z(num_envs, self.n)
        self.base_h_sum = z(num_envs)
        self.vx_sum, self.vy_sum, self.wz_sum = z(num_envs), z(num_envs), z(num_envs)
        # Sum and sum-of-squares, so base_oscillation is a real standard deviation over the
        # window rather than a mean of per-step magnitudes.
        self.osc_sum, self.osc_sq = z(num_envs, 3), z(num_envs, 3)
        self.grf_sum, self.grf_contacts = z(num_envs, self.n), z(num_envs, self.n)
        self.grf_peak = z(num_envs, self.n)
        self.landing_vel_sum = z(num_envs, self.n)
        self.prev_foot_vz = z(num_envs, self.n)
        self.start_pos = None
        self.start_yaw = None
        self.resets = z(num_envs)
        # Contact-phase bookkeeping, matching Mujoco/eval_mujoco.py so the two sweeps' phase
        # numbers mean the same thing. Feet are ordered [FL, FR, RL, RR].
        self.last_strike = z(num_envs, self.n)
        self.stride = z(num_envs, self.n)
        self.phase_sum = z(num_envs, 3)
        self.phase_n = z(num_envs, 3)
        self.steps = 0

    def update(self, env, dt):
        robot = env.scene["robot"]
        sensor = env.scene.sensors["contact_forces"]
        ground = env.scene.env_origins[:, 2]

        contact = sensor.data.net_forces_w[:, self.sensor_foot_ids, :].norm(dim=-1) > 1.0
        foot_h = robot.data.body_pos_w[:, self.robot_foot_ids, 2] - ground.unsqueeze(1)

        self.air_time += dt
        self.swing_peak = torch.maximum(self.swing_peak, foot_h)

        landed = contact & ~self.prev_contact
        if bool(landed.any()):
            self.lift_sum += torch.where(landed, self.swing_peak, torch.zeros_like(self.swing_peak))
            self.swing_sum += torch.where(landed, self.air_time, torch.zeros_like(self.air_time))
            self.landings += landed.float()

        self.air_time = torch.where(contact, torch.zeros_like(self.air_time), self.air_time)
        self.swing_peak = torch.where(contact, torch.zeros_like(self.swing_peak), self.swing_peak)
        self.prev_contact = contact

        # Ground reaction force on each foot while loaded, and the peak seen over the window.
        fz = sensor.data.net_forces_w[:, self.sensor_foot_ids, 2].abs()
        self.grf_sum += torch.where(contact, fz, torch.zeros_like(fz))
        self.grf_contacts += contact.float()
        self.grf_peak = torch.maximum(self.grf_peak, fz)

        # Touchdown speed is read from the PREVIOUS step: by the time a contact force crosses the
        # threshold the collision is already resolved, so this step's vertical velocity is
        # post-impact (near zero for exactly the hard landings this is meant to catch).
        foot_vz = robot.data.body_lin_vel_w[:, self.robot_foot_ids, 2]
        self.landing_vel_sum += torch.where(landed, self.prev_foot_vz.abs(), torch.zeros_like(foot_vz))
        self.prev_foot_vz = foot_vz

        osc = torch.stack(
            [robot.data.root_lin_vel_b[:, 2], robot.data.root_ang_vel_b[:, 0], robot.data.root_ang_vel_b[:, 1]],
            dim=1,
        )
        self.osc_sum += osc
        self.osc_sq += osc.square()

        if self.start_pos is None:
            self.start_pos = robot.data.root_pos_w[:, :2].clone()
            q = robot.data.root_quat_w
            self.start_yaw = torch.atan2(
                2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
            )
        self.end_pos = robot.data.root_pos_w[:, :2]
        q = robot.data.root_quat_w
        self.end_yaw = torch.atan2(
            2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2)
        )

        self._update_phase(landed, dt)

        self.base_h_sum += robot.data.root_pos_w[:, 2] - ground
        self.vx_sum += robot.data.root_lin_vel_b[:, 0]
        self.vy_sum += robot.data.root_lin_vel_b[:, 1]
        self.wz_sum += robot.data.root_ang_vel_b[:, 2]
        self.steps += 1

    # Which foot's landing is measured against which foot's stride. Same three pairs
    # eval_mujoco.py reports, with feet ordered [FL, FR, RL, RR]:
    #   front  FR landing, against FL's stride  -- should be ~50% (antiphase) in a trot
    #   right  RR landing, against FR's stride  -- ~50%
    #   diag   RR landing, against FL's stride  -- ~0%  (the diagonal pair moves together)
    PHASE_PAIRS = ((1, 0), (3, 1), (3, 0))

    def _update_phase(self, landed, dt):
        """Relative contact phase, accumulated at each touchdown.

        Reproduces eval_mujoco.py's calculation exactly, including the order it does things in:
        a foot's stride is measured against its PREVIOUS strike, its strike time is then updated,
        and only afterwards is the phase read off the reference foot. In every pair below the
        reference index is lower than the landing index, which is the order eval_mujoco's per-foot
        loop happens to visit them in -- so feet landing on the same step agree between the two
        implementations rather than differing by one stride.
        """
        t = (self.steps + 1) * dt

        # Stride first (needs the OLD strike time), then the strike time itself.
        started = self.last_strike > 0
        self.stride = torch.where(landed & started, t - self.last_strike, self.stride)
        self.last_strike = torch.where(landed, torch.full_like(self.last_strike, t), self.last_strike)

        for i, (foot, ref) in enumerate(self.PHASE_PAIRS):
            # A reference stride under 0.1 s is a contact flicker, not a gait cycle.
            valid = landed[:, foot] & (self.stride[:, ref] > 0.1)
            if not bool(valid.any()):
                continue
            ref_stride = self.stride[:, ref].clamp(min=1e-4)
            p = ((t - self.last_strike[:, ref]) % ref_stride) / ref_stride
            # Fold onto [0, 0.5]: 0.9 out of phase and 0.1 out of phase are the same offset.
            p = torch.where(p <= 0.5, p, 1.0 - p)
            self.phase_sum[:, i] += p * valid.float()
            self.phase_n[:, i] += valid.float()

    def metrics(self, rows, duration):
        """Reduce to one dict per sweep point, averaging over that point's environments."""
        s = max(self.steps, 1)
        counts = self.landings[rows].clamp(min=1.0)
        return {
            "actual_speed_x": (self.vx_sum[rows] / s).mean(),
            "actual_speed_y": (self.vy_sum[rows] / s).mean(),
            "actual_yaw_rate": (self.wz_sum[rows] / s).mean(),
            "base_height_cm": (self.base_h_sum[rows] / s).mean() * 100.0,
            # Per foot, so a limping gait is visible rather than averaged away.
            "foot_lift_height_cm": (self.lift_sum[rows] / counts).mean(dim=0) * 100.0,
            "foot_swing_time_s": (self.swing_sum[rows] / counts).mean(),
            "step_frequency_hz": (self.landings[rows] / duration).mean(dim=0),
            "landings_total": self.landings[rows].sum(),
            "grf_stance_N": (self.grf_sum[rows] / self.grf_contacts[rows].clamp(min=1.0)).mean(dim=0),
            "grf_peak_stance_N": self.grf_peak[rows].mean(dim=0),
            "foot_landing_vel_ms": (self.landing_vel_sum[rows] / counts).mean(dim=0),
            "base_oscillation": self._std(rows),
            "position_error": self._drift(rows, duration),
            "safety_resets": int(self.resets[rows].sum()),
            **self._phase(rows),
        }

    def _phase(self, rows):
        """Mean relative phase per pair, in percent. 0.0 when no pair ever qualified -- which is
        itself informative: a robot that never takes two consecutive strides has no gait."""
        out = {}
        for i, key in enumerate(("front", "right", "diag")):
            n = self.phase_n[rows, i]
            per_env = self.phase_sum[rows, i] / n.clamp(min=1.0)
            valid = n > 0
            mean = (per_env[valid].mean() * 100.0) if bool(valid.any()) else torch.zeros((), device=self.device)
            out[f"phase_diff_{key}_percent"] = _round(mean, 2)
        return out

    def _std(self, rows):
        s = max(self.steps, 1)
        var = (self.osc_sq[rows] / s - (self.osc_sum[rows] / s).square()).clamp(min=0.0)
        std = var.sqrt().mean(dim=0)
        return {"std_z_vel": _round(std[0]), "std_roll_vel": _round(std[1]), "std_pitch_vel": _round(std[2])}

    def _drift(self, rows, duration):
        """Displacement actually achieved, in the body frame the window STARTED in, minus what
        the command asked for. Positive x means it went further than commanded."""
        d = self.end_pos[rows] - self.start_pos[rows]
        c, s_ = torch.cos(-self.start_yaw[rows]), torch.sin(-self.start_yaw[rows])
        dx = d[:, 0] * c - d[:, 1] * s_
        dy = d[:, 0] * s_ + d[:, 1] * c
        dyaw = (self.end_yaw[rows] - self.start_yaw[rows] + torch.pi) % (2 * torch.pi) - torch.pi
        return {
            "x_m": _round(dx.mean() - self.cmd[0] * duration),
            "y_m": _round(dy.mean() - self.cmd[1] * duration),
            "yaw_rad": _round(dyaw.mean() - self.cmd[2] * duration),
        }


def main():
    speeds = [float(s) for s in args_cli.speeds.split(",")]
    axes = [a.strip() for a in args_cli.axes.split(",") if a.strip()]
    per_point = args_cli.envs_per_point
    num_envs = len(speeds) * per_point

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    # The play cfg shrinks the terrain to 2x1 tiles for viewing; a sweep needs room for every
    # environment, and a robot that walks off its tile would be measured falling off a ledge.
    env_cfg.scene.terrain.terrain_generator.num_rows = 10
    env_cfg.scene.terrain.terrain_generator.num_cols = 20
    env_cfg.scene.num_envs = num_envs
    # Episodes must not end mid-measurement.
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, (args_cli.settle + args_cli.measure) * 1.5)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[Sweep] checkpoint: {resume_path}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(
        env, cli_args.runner_cfg_for_installed_rsl_rl(agent_cfg), log_dir=None, device=agent_cfg.device
    )
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    dt = base.step_dt
    settle_steps = int(args_cli.settle / dt)
    measure_steps = int(args_cli.measure / dt)
    command_term = base.command_manager.get_term("base_velocity")
    # Both index sets, resolved BY NAME into one canonical order. The contact sensor builds its
    # own body list from prim_path "Robot/.*", which is not the articulation's ordering -- here
    # they come back as [4, 8, 14, 18] and [15, 16, 17, 18]. Zipping those positionally pairs
    # each foot's contact with a different foot's position, which silently mis-attributes per-foot
    # lift and every phase pair. Order is [FL, FR, RL, RR], matching Mujoco/eval_mujoco.py.
    FOOT_ORDER = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    def _feet(entity, what):
        ids, names = entity.find_bodies(".*_foot")
        by_name = dict(zip(names, ids))
        missing = [f for f in FOOT_ORDER if f not in by_name]
        if missing:
            raise RuntimeError(f"{what} has no bodies named {missing}; found {sorted(by_name)}")
        return [by_name[f] for f in FOOT_ORDER]

    sensor_foot_ids = _feet(base.scene.sensors["contact_forces"], "contact sensor")
    robot_foot_ids = _feet(base.scene["robot"], "articulation")
    print(f"[Sweep] feet {FOOT_ORDER} -> sensor {sensor_foot_ids}, articulation {robot_foot_ids}")

    # Row block per sweep point: rows [i*per_point, (i+1)*per_point) all hold speeds[i].
    rows = [torch.arange(i * per_point, (i + 1) * per_point, device=base.device) for i in range(len(speeds))]

    results = {}
    for axis in axes:
        col = {"x": 0, "y": 1, "yaw": 2}[axis]
        pinned = torch.zeros(num_envs, 3, device=base.device)
        for i, sp in enumerate(speeds):
            pinned[rows[i], col] = sp

        # rsl-rl 2.3.x returns (obs, extras) here and other versions return obs alone; play.py
        # branches on the version string, which breaks on any release it has not heard of.
        obs = env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]
        tracker = None
        for step in range(settle_steps + measure_steps):
            # Overwrite the sampled command every step. The command manager resamples on its own
            # timer and zeroes the rel_standing_envs share, both of which would corrupt a sweep.
            command_term.vel_command_b[:] = pinned
            command_term.is_standing_env[:] = False
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
            if step == settle_steps:
                tracker = GaitTracker(num_envs, sensor_foot_ids, robot_foot_ids, base.device)
            if tracker is not None:
                command_term.vel_command_b[:] = pinned
                tracker.update(base, dt)
                # A robot that fell and got teleported back is not tracking anything; count it so
                # a suspiciously good number can be checked against how often it reset.
                tracker.resets += base.termination_manager.terminated.float()

        axis_out = {}
        for i, sp in enumerate(speeds):
            tracker.cmd = pinned[rows[i][0]].tolist()
            m = tracker.metrics(rows[i], args_cli.measure)
            actual = {"x": m["actual_speed_x"], "y": m["actual_speed_y"], "yaw": m["actual_yaw_rate"]}[axis]
            axis_out[f"{sp:g}"] = {
                "commanded_speed": _round(sp),
                "actual_speed": _round(actual),
                # None, not NaN: json.dump writes a bare NaN token, which is legal Python and
                # illegal JSON, and JSON.parse rejects the whole response -- taking every other
                # report in the viewer down with it. There is no fraction to report at a zero
                # command, and null says exactly that.
                "tracked_fraction": (_round(float(actual) / sp) if sp else None),
                "foot_lift_height_cm": _round(m["foot_lift_height_cm"], 2),
                "foot_swing_time_s": _round(m["foot_swing_time_s"]),
                "step_frequency_hz": _round(m["step_frequency_hz"], 2),
                # The posture number the MuJoCo reports never carried: neither config has a
                # base-height reward, while flat_orientation_l2 and lin_vel_z_l2 both favour a
                # crouch, so how far the robot sinks is unspecified and worth recording.
                "base_height_cm": _round(m["base_height_cm"], 1),
                "landings_total": int(m["landings_total"]),
                "grf_stance_N": _round(m["grf_stance_N"], 2),
                "grf_peak_stance_N": _round(m["grf_peak_stance_N"], 2),
                "foot_landing_vel_ms": _round(m["foot_landing_vel_ms"]),
                "base_oscillation": m["base_oscillation"],
                "position_error": m["position_error"],
                "safety_resets": m["safety_resets"],
                "phase_diff_front_percent": m["phase_diff_front_percent"],
                "phase_diff_right_percent": m["phase_diff_right_percent"],
                "phase_diff_diag_percent": m["phase_diff_diag_percent"],
            }
            print(
                f"[{axis}] cmd {sp:5.2f} -> {float(actual):6.3f}  "
                f"({100 * float(actual) / sp if sp else 0:5.1f}%)  "
                f"lift {float(m['foot_lift_height_cm'].mean()):5.2f} cm  "
                f"base {float(m['base_height_cm']):5.1f} cm",
                flush=True,
            )
        results[axis] = axis_out

    report = {
        "metadata": {
            "checkpoint": resume_path,
            "checkpoint_name": os.path.basename(resume_path),
            "task": args_cli.task,
            "simulator": "isaac",
            "envs_per_point": per_point,
            "settle_s": args_cli.settle,
            "measure_s": args_cli.measure,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": results,
    }
    name = os.path.splitext(os.path.basename(resume_path))[0]
    out = os.path.join(os.path.dirname(resume_path), f"isaac_eval_report_{name}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Sweep] wrote {out}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
