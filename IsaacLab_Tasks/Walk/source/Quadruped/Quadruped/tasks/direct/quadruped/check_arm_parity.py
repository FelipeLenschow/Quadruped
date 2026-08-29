#!/usr/bin/env python3
"""Report every config difference between the position and torque training arms.

The torque phases used to `inherit` their position counterparts, which made reward parity
structural: you could not change one arm without changing the other. They are standalone
now, which is easier to read and edit but means the two arms can silently drift apart --
and "the two arms were not trained on the same reward" is precisely the caveat the one
published hardware comparison (Chen et al., Humanoids 2023) has to admit to.

So run this after touching either arm. Differences forced by the control mode are listed
and ignored; anything else is a finding and exits non-zero.

    python check_arm_parity.py

No Isaac Lab import: resolve_phase is lifted out of quadruped_env_cfg.py so this runs in
any interpreter, including on a machine with no simulator installed.
"""

from __future__ import annotations

import collections.abc
import copy
import io
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

# Differences that the control mode forces rather than chooses. Everything else must match.
#   control_mode / torque_scale / joint_limit_barrier_stiffness / spawn_height -- the mode itself
#   max_timesteps -- torque runs at 200 Hz, so 4x the agent steps buys the same simulated time
#   reset_exploration_on_entry -- only phase1_torque sets it, and only because it is entered
#     from phase0_torque, the torque-only bootstrap already declared in UNMATCHED_PHASES. It
#     re-inflates the policy std and the learning rate that a phase rewarding stillness
#     annealed away; position phase1 starts fresh and has nothing to undo. It is part of the
#     phase0 asymmetry, not a second one -- report it in the writeup on the same line.
EXPECTED_ENV_DIFFS = {
    "control_mode",
    "torque_scale",
    "joint_limit_barrier_stiffness",
    "spawn_height",
    "max_timesteps",
    "reset_exploration_on_entry",
}

# phase0_torque has no position counterpart: position control gets a standing prior for free
# (a zero action IS the nominal stance), so it needs no bootstrap phase. That asymmetry is
# real and belongs in the writeup alongside decimation, so it is declared here rather than
# checked -- but if a position phase0 is ever added, move it into ARM_PAIRS.
UNMATCHED_PHASES = ["phase0_torque"]

ARM_PAIRS = [
    ("phase1", "phase1_torque"),
    ("phase2", "phase2_torque"),
]


def load_resolver():
    """Lift resolve_phase() verbatim out of quadruped_env_cfg.py.

    Importing the module would pull in isaaclab -> pxr, which needs a running Isaac Sim.
    Executing just this one pure function keeps the check honest (it is the same code the
    trainer uses) without that dependency.
    """
    src = io.open(os.path.join(HERE, "quadruped_env_cfg.py"), encoding="utf-8").read()
    match = re.search(
        r"^def resolve_phase\(all_phases, phase_name\):.*?^(?=_phase_cfg = )",
        src,
        re.S | re.M,
    )
    if match is None:
        sys.exit("could not find resolve_phase() in quadruped_env_cfg.py")
    namespace = {"collections": collections, "copy": copy}
    exec(match.group(0), namespace)
    return namespace["resolve_phase"]


def main() -> int:
    resolve_phase = load_resolver()
    all_phases = yaml.safe_load(
        io.open(os.path.join(HERE, "training_phases.yaml"), encoding="utf-8")
    )
    defined = all_phases.get("phases", {})

    findings = 0
    for phase in UNMATCHED_PHASES:
        if phase in defined:
            print(f"declared  {phase} has no position counterpart (torque-only bootstrap)")

    for position, torque in ARM_PAIRS:
        missing = [p for p in (position, torque) if p not in defined]
        if missing:
            print(f"MISSING  {', '.join(missing)} not defined in training_phases.yaml")
            findings += 1
            continue

        pos_cfg, tq_cfg = resolve_phase(all_phases, position), resolve_phase(all_phases, torque)
        print(f"\n{position}  vs  {torque}")

        for section in ("env", "domain_randomization", "events", "rewards", "commands"):
            a, b = pos_cfg.get(section, {}), tq_cfg.get(section, {})
            for key in sorted(set(a) | set(b)):
                if a.get(key) == b.get(key):
                    continue
                if section == "env" and key in EXPECTED_ENV_DIFFS:
                    print(f"   expected  {section}.{key}: {a.get(key)} -> {b.get(key)}")
                else:
                    print(f"   MISMATCH  {section}.{key}: {a.get(key)} -> {b.get(key)}")
                    findings += 1

    print()
    if findings:
        print(f"{findings} unexpected difference(s). Either fix the drift, or move the key "
              f"into EXPECTED_ENV_DIFFS and say why in the writeup.")
        return 1
    print("Arms are matched: every difference is forced by the control mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
