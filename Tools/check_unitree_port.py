#!/usr/bin/env python3
"""Check that arm U1's ported reward terms equal the upstream functions they copy.

Arm U1 (see Write/Article/plan.md) reruns unitree_rl_lab's Go2 velocity config inside this
repo's Direct RL env, so that every rung above it is interpretable. That only holds if the
reward terms are actually the same functions. This script checks that numerically: it execs the
upstream definitions straight out of the Isaac Lab and unitree_rl_lab source trees with
torch-only stubs, evaluates them and the ported formulas on the same random inputs, and reports
the maximum absolute difference.

It needs torch and nothing else -- no Isaac Sim, no running simulation -- so it is safe to run
while a training job holds the GPU.

    python Tools/check_unitree_port.py

Exits non-zero if any term differs. The ported formulas below are copies of the ones in
quadruped_env._compute_reward_terms; if you edit one there, edit it here too, or this stops
testing anything.
"""

from __future__ import annotations

import os
import re
import sys

import torch

ISAACLAB = os.path.expanduser("~/IsaacLab")
UNITREE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unitree_rl_lab")

SOURCES = {
    "isaaclab": f"{ISAACLAB}/source/isaaclab/isaaclab/envs/mdp/rewards.py",
    "isaaclab_velocity": (
        f"{ISAACLAB}/source/isaaclab_tasks/isaaclab_tasks/manager_based/"
        "locomotion/velocity/mdp/rewards.py"
    ),
    "unitree": f"{UNITREE}/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py",
}

N, J, F = 64, 12, 4          # envs, joints, feet
TOL = 1e-6


class _Stub:
    """Stands in for SceneEntityCfg and the type-only imports upstream annotates with."""

    def __init__(self, *args, **kwargs):
        self.name = args[0] if args else "robot"
        self.joint_ids = slice(None)
        self.body_ids = list(range(F))


def load_upstream(path: str, names: list[str]) -> dict:
    """Exec named functions out of an upstream module with torch-only stubs."""
    src = open(path).read()
    ns = {"torch": torch, "TYPE_CHECKING": False}
    for stubbed in (
        "Articulation", "RigidObject", "ContactSensor", "SceneEntityCfg",
        "ManagerBasedRLEnv", "ManagerBasedEnv", "quat_apply_inverse",
    ):
        ns[stubbed] = _Stub
    out = {}
    for name in names:
        match = re.search(r"\ndef " + name + r"\(.*?(?=\ndef |\Z)", src, re.S)
        if match is None:
            raise SystemExit(f"could not find def {name} in {path}")
        exec(compile("from __future__ import annotations\n" + match.group(0), name, "exec"), ns)
        out[name] = ns[name]
    return out


def main() -> int:
    for label, path in SOURCES.items():
        if not os.path.isfile(path):
            print(f"missing upstream source ({label}): {path}")
            return 2

    upstream = {}
    upstream.update(load_upstream(SOURCES["isaaclab"], ["joint_pos_limits", "undesired_contacts"]))
    upstream.update(load_upstream(SOURCES["isaaclab_velocity"], ["feet_slide", "feet_air_time"]))
    upstream.update(
        load_upstream(SOURCES["unitree"], ["energy", "joint_position_penalty", "air_time_variance_penalty"])
    )

    torch.manual_seed(0)
    joint_pos = torch.randn(N, J)
    default_joint_pos = torch.randn(N, J) * 0.2
    soft_lo = -torch.rand(N, J) * 2 - 0.5
    soft_hi = torch.rand(N, J) * 2 + 0.5
    joint_vel = torch.randn(N, J) * 3
    applied_torque = torch.randn(N, J) * 10
    root_lin_vel_b = torch.randn(N, 3) * 0.5
    feet_vel_w = torch.randn(N, F, 3)
    last_air_time = torch.rand(N, F)
    last_contact_time = torch.rand(N, F)
    commands = torch.randn(N, 3) * 0.4
    commands[:6] = 0.0                      # exercise joint_position_penalty's exactly-zero branch
    first_contact = torch.rand(N, F) > 0.7
    contact = torch.rand(N, F) > 0.5
    force_history = torch.randn(3, N, 8, 3) * 2     # (T, N, bodies, xyz)

    class _Data:
        pass

    asset = type("Asset", (), {})()
    asset.data = _Data()
    asset.data.joint_pos = joint_pos
    asset.data.default_joint_pos = default_joint_pos
    asset.data.soft_joint_pos_limits = torch.stack([soft_lo, soft_hi], dim=-1)
    asset.data.joint_vel = joint_vel
    asset.data.applied_torque = applied_torque
    asset.data.root_lin_vel_b = root_lin_vel_b
    asset.data.body_lin_vel_w = feet_vel_w

    sensor = type("Sensor", (), {})()
    sensor.data = _Data()
    sensor.cfg = _Data()
    sensor.cfg.track_air_time = True
    sensor.data.last_air_time = last_air_time
    sensor.data.last_contact_time = last_contact_time
    sensor.data.net_forces_w_history = force_history.permute(1, 0, 2, 3)
    sensor.compute_first_contact = lambda dt: first_contact

    class RobotCfg(_Stub):
        pass

    class SensorCfg(_Stub):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.name = "contact_forces"

    class UndesiredCfg(SensorCfg):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.body_ids = list(range(4, 8))

    scene = type("Scene", (dict,), {})({"robot": asset, "contact_forces": sensor})
    scene.sensors = {"contact_forces": sensor}
    env = type("Env", (), {})()
    env.num_envs, env.device, env.step_dt, env.scene = N, "cpu", 0.02, scene
    env.command_manager = type("CM", (), {"get_command": staticmethod(lambda name: commands)})()

    results = []

    def compare(label, mine, theirs):
        diff = (mine - theirs.float()).abs().max().item()
        results.append((label, diff))
        status = "MATCH" if diff < TOL else "*** DIFFERS ***"
        print(f"  {label:28s} maxdiff={diff:.3e}  {status}")

    cmd_norm = torch.norm(commands[:, :3], dim=1)
    command_speed_xy = torch.norm(commands[:, :2], dim=1)

    print("porting fidelity, ported formula vs upstream function:")

    # energy -- unitree_rl_lab mdp.energy
    compare(
        "energy",
        torch.sum(joint_vel.abs() * applied_torque.abs(), dim=1),
        upstream["energy"](env, RobotCfg()),
    )

    # joint_pos_limits -- isaaclab mdp.joint_pos_limits
    out_of_limits = -(joint_pos - soft_lo).clamp(max=0.0)
    out_of_limits = out_of_limits + (joint_pos - soft_hi).clamp(min=0.0)
    compare(
        "joint_pos_limits",
        torch.sum(out_of_limits, dim=1),
        upstream["joint_pos_limits"](env, RobotCfg()),
    )

    # joint_position_penalty -- unitree_rl_lab mdp.joint_position_penalty
    body_speed_xy = torch.norm(root_lin_vel_b[:, :2], dim=1)
    joint_dev_norm = torch.linalg.norm(joint_pos - default_joint_pos, dim=1)
    compare(
        "joint_position_penalty",
        torch.where((cmd_norm > 0.0) | (body_speed_xy > 0.3), joint_dev_norm, 5.0 * joint_dev_norm),
        upstream["joint_position_penalty"](env, RobotCfg(), 5.0, 0.3),
    )

    # air_time_variance -- unitree_rl_lab mdp.air_time_variance_penalty
    compare(
        "air_time_variance",
        torch.var(last_air_time.clamp(max=0.5), dim=1)
        + torch.var(last_contact_time.clamp(max=0.5), dim=1),
        upstream["air_time_variance_penalty"](env, SensorCfg()),
    )

    # feet_slide -- isaaclab mdp.feet_slide. Upstream decides "loaded" from the force history;
    # feed it a history consistent with `contact` so only the velocity term is under test.
    sensor.data.net_forces_w_history = torch.where(
        contact[:, None, :, None], torch.full((N, 3, F, 3), 10.0), torch.zeros(N, 3, F, 3)
    )
    compare(
        "feet_slide",
        torch.sum(feet_vel_w[:, :, :2].norm(dim=-1) * contact.float(), dim=1),
        upstream["feet_slide"](env, SensorCfg(), RobotCfg()),
    )

    # feet_air_time -- isaaclab mdp.feet_air_time, threshold form
    compare(
        "feet_air_time(threshold)",
        torch.sum((last_air_time - 0.5) * first_contact.float(), dim=1)
        * (command_speed_xy > 0.1).float(),
        upstream["feet_air_time"](env, "base_velocity", SensorCfg(), 0.5),
    )

    # undesired_contacts -- isaaclab mdp.undesired_contacts, counting form
    sensor.data.net_forces_w_history = force_history.permute(1, 0, 2, 3)
    peak = sensor.data.net_forces_w_history[:, :, UndesiredCfg().body_ids, :].norm(dim=-1).amax(dim=1)
    compare(
        "undesired_contacts(count)",
        (peak > 1.0).float().sum(dim=1),
        upstream["undesired_contacts"](env, 1.0, UndesiredCfg()),
    )

    bad = [label for label, diff in results if diff >= TOL]
    print()
    if bad:
        print(f"{len(bad)} term(s) differ from upstream: {', '.join(bad)}")
        return 1
    print(f"all {len(results)} ported terms match upstream exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
