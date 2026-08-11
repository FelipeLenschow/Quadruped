# CLAUDE.md

This file orients an agent working in this repo. For the deployment/runtime architecture (drivers, telemetry, controller), see `README.md` and `PROJECT_OVERVIEW.md` — this file covers environment setup and the Isaac Lab training side, which those docs don't get into.

## Environment setup

Two separate Python environments are in play; picking the wrong one is the most common cause of "command not found" or import errors (e.g. `tensorboard`, Isaac Lab modules):

- **Isaac Lab / training / MuJoCo bridges** (Python 3.11): `source ~/env_isaacsim/bin/activate`
  Needed for `launcher.py` train/play actions, `tensorboard`, the MuJoCo driver/twin scripts, `eval_mujoco.py`.
- **ROS 2 Humble tooling / Gazebo / real-robot deploy** (Python 3.10, system): `source venv_robot/bin/activate`, or run inside `Docker/` if the host isn't natively Python 3.10. `launcher.py` auto-detects this (`IS_DOCKER`, `IS_ROBOT`, interpreter version) and disables menu options that don't apply to the current environment.

`Command_list.txt` has the exact activation lines and other commands run day to day — check it before assuming a tool isn't installed.

## `launcher.py` — the single entry point

`python launcher.py` is a CLI menu that drives everything: train, play (Isaac Lab / Isaac Sim / MuJoCo / Gazebo), deploy to the real robot, evaluate a policy, and ROS 2 tools (teleop, PlotJuggler, MCAP record/replay, rqt/tf2). Flow:

1. Detects environment (Docker, ARM64 robot, `env_isaacsim` venv) and hides menu options that don't apply.
2. Lists task modules under `IsaacLab_Tasks/` (see below) and prompts for one.
3. Finds checkpoints under `<module>/logs/**/*.pt` or `<module>/checkpoints/*.pt`.
4. For `train`, reads `<module>/source/**/training_phases.yaml` to list phases and can auto-chain a curriculum sequence (e.g. phase1 → phase6), feeding each segment's best checkpoint into the next `train.py` invocation.
5. Dispatches to a subprocess, passing robot/terrain/phase selection as **environment variables** (`QUADRUPED_TRAINING_PHASE`, `QUADRUPED_ROBOT_CFG`, `QUADRUPED_TERRAIN`, `PYTHONPATH=<module>/source/Quadruped`) rather than CLI flags — task code reads these from `os.environ`.
6. For sim/deploy actions, also starts `Controller/reward_estimator_node.py` in the background.

## `IsaacLab_Tasks/` structure

Each subfolder is a fully independent copy of the Isaac Lab task package (own `source/Quadruped/Quadruped/tasks/direct/quadruped/...`, own `logs/`, own `training_phases.yaml`) — nothing is shared between them, so a fix or reward tweak made in one does not propagate to the others. Confirm which module is actually meant before editing:

- **`Walk/`** — the main, actively developed task. Default assumption unless told otherwise.
- **`Walk_GO2/`** — a Go2-only simplification made when a possible internship lab also had a Go2; that plan changed and work went back to `Walk` (3 robots). Likely stale.
- **`Stairs/`** — experiment adding a terrain/height-scan sensor for stair climbing.
- **`Handstand/`** — handstand task.

## Reward/training config pattern (Walk)

Rewards and curriculum for the main task live in `IsaacLab_Tasks/Walk/source/Quadruped/Quadruped/tasks/direct/quadruped/training_phases.yaml`: a `default` block plus `phases.phaseN` entries that `inherits` from a parent phase and deep-merge over it (see `resolve_phase_launcher` in `launcher.py`). Each phase configures `env`, `domain_randomization`, `events`, `rewards`, and `commands`. The yaml only supplies scale factors/thresholds — the actual reward math is in `quadruped_env.py` (`_get_rewards` and the JIT-compiled reward function it calls).
