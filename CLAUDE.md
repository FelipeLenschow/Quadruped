# CLAUDE.md

This file orients an agent working in this repo. For the deployment/runtime architecture (drivers, telemetry, controller), see `README.md` and `PROJECT_OVERVIEW.md` — this file covers environment setup and the Isaac Lab training side, which those docs don't get into.

## Environment setup

Two separate Python environments are in play; picking the wrong one is the most common cause of "command not found" or import errors (e.g. `tensorboard`, Isaac Lab modules):

- **Isaac Lab / training / MuJoCo bridges** (Python 3.12): `source ~/env_isaacsim/bin/activate`
  Needed for `launcher.py` train/play actions, `tensorboard`, the MuJoCo driver/twin scripts, `eval_mujoco.py`.
  Run `unset PYTHONPATH` first: a sourced ROS 2 Humble setup puts its Python 3.10 `site-packages`
  on `PYTHONPATH`, and those leak into this venv and break resolution.
- **ROS 2 Humble tooling / Gazebo / real-robot deploy** (Python 3.10, system): `source venv_robot/bin/activate`, or run inside `Docker/` if the host isn't natively Python 3.10. `launcher.py` auto-detects this (`IS_DOCKER`, `IS_ROBOT`, interpreter version) and disables menu options that don't apply to the current environment.

### Isaac Sim / Isaac Lab versions

Full migration write-up, including every 2.x->3.0 break and its fix: `isaaclab3_migration_notes.md`.

Rebuilt 2026-09-18. `~/env_isaacsim` is Python 3.12 with Isaac Sim **6.1.0.0**; Isaac Lab is
**v3.0.0-EA**, cloned to `~/IsaacLab30` and pip-installed editable. The older `~/IsaacLab`
(on `main`) is no longer what gets imported, and Isaac Sim 5.1 is gone — there is no fallback.

Two things that surprise people:

- **PhysX is still the default backend.** `SimulationCfg.physx` was renamed to
  `SimulationCfg.physics`, and it defaults to `None`, which means `PhysxCfg()`. `PhysxCfg`
  itself moved out of `isaaclab.sim` into `isaaclab_physx.physics`, so the old
  `sim_utils.PhysxCfg` raises `AttributeError: No isaaclab.sim attribute PhysxCfg`.
  All the `gpu_*` capacity fields survived the move unchanged.
- What *is* newly on by default is `use_newton_actuators = True`: Newton executes
  `DCMotorCfg` / `IdealPDActuatorCfg` actuators (PhysX runs them via a host adapter) instead of
  the old Isaac Lab path. This changes actuator behaviour, so it is the flag to set `False` if
  a run needs to match the `Final*` / `ph*` results — not the physics backend.
- Isaac Lab 3.0 is a **uv workspace**. `pip install -e source/<pkg>` pulls only that package's
  own dependencies, not the root project's ~52 externals, so imports fail one at a time
  (`rich`, `hydra-core`, `lazy_loader` first). Prefer `uv sync`; if using pip, install the
  workspace members in topological order with `--no-deps`, then the root dependency list.

`Command_list.txt` has the exact activation lines and other commands run day to day — check it before assuming a tool isn't installed.

## `launcher.py` — the single entry point

`python launcher.py` is a CLI menu that drives everything: train, play (Isaac Lab / Isaac Sim / MuJoCo / Gazebo), deploy to the real robot, evaluate a policy, and ROS 2 tools (teleop, PlotJuggler, MCAP record/replay, rqt/tf2). Flow:

1. Detects environment (Docker, ARM64 robot, `env_isaacsim` venv) and hides menu options that don't apply.
2. Lists task modules under `IsaacLab_Tasks/` (see below) and prompts for one.
3. Finds checkpoints under `<module>/logs/**/*.pt` or `<module>/checkpoints/*.pt`.
4. For `train`, reads `<module>/source/**/training_phases.yaml` to list phases and can auto-chain a curriculum sequence (e.g. phase1 → phase6), feeding each segment's best checkpoint into the next `train.py` invocation.
5. Dispatches to a subprocess, passing robot/terrain/phase selection as **environment variables** (`QUADRUPED_TRAINING_PHASE`, `QUADRUPED_ROBOT_CFG`, `QUADRUPED_TERRAIN`, `PYTHONPATH=<module>/source/Quadruped`) rather than CLI flags — task code reads these from `os.environ`.
   The ROS-side actions run from the ROS 2 packages in `src/` (see below): `ros2 launch quadruped_bringup <mode>.launch.py key:=value ...` for drivers, eval, twins, gamepad teleop, eval sweep, bag record and rviz/plotjuggler/rqt/foxglove; `ros2 run` for the interactive ones that read stdin (console, keyboard teleop, joint tester, MCAP replay). Before those actions the launcher runs `colcon build --symlink-install --base-paths src` when the install is missing or stale, and sources `install/setup.bash`.
6. For sim/deploy actions, can also start the reward estimator (`reward:=true`; off by default, `START_REWARD_ESTIMATOR` in `launcher.py`).
7. For `train`, optionally starts `Tools/auto_eval.py` in the background (one watcher per curriculum
   segment, stopped before the run folder is renamed). It watches the run's `checkpoints/` and puts
   every Nth checkpoint (default 50k) through `quadruped_drivers.eval_mujoco` (run with `-m` from `src/`, no build needed), so the evaluation half of the
   dashboard fills in while the run is still going. The sweep is spawned through a ROS 2 interpreter
   (sourcing `/opt/ros/<distro>/setup.bash` and stripping the Isaac venv off `PATH`) because
   `eval_mujoco.py` imports `rclpy`, which the Isaac venv does not have.

## ROS 2 packages (`src/`)

Runtime code (everything that is not training) is a colcon workspace; build only `src/`, since `IsaacLab_Tasks/` also contains `setup.py` packages:

    colcon build --symlink-install --base-paths src && source install/setup.bash

In Docker run colcon as `python3 -m colcon ...` from the venv, so node scripts get the venv's Python (torch, mujoco).

- `src/`: the ROS 2 packages (colcon workspace, `colcon build --symlink-install --base-paths src`):
  - `quadruped_core`: library, no nodes: `pipeline.py`, `controller/` (policy runner, safety processor, pose generator), `telemetry/` (TelemetryManager, LKF estimator, kinematics), `config_loader.py`, `paths.py`.
  - `quadruped_drivers`: nodes for the real Go2 (`real_driver`, `test_joints`), MuJoCo (`mujoco_driver`, `eval_mujoco`) and Gazebo (`gazebo_driver`).
  - `quadruped_operator`: console, supervisor, teleops, twins, reward estimator, MCAP tool.
  - `quadruped_description`: MuJoCo menagerie, Gazebo world, Go2/Go1/A1 models, kinematics yaml, bundled policies.
  - `quadruped_bringup`: `launch/*.launch.py` and `config/` (`config.yaml`, `joy_f710.config.yaml`).
  - `unitree_sdk2py`: wraps the `third_party/unitree_sdk2_python` submodule.
- `IsaacSim/isaac_driver.py`: stays outside the packages (Isaac Sim's Python 3.12); imports `quadruped_core` from `src/`.

Scripts still take command-line flags (not ROS parameters); `rclpy.utilities.remove_ros_args` strips the `--ros-args` that `Node` appends. Data is found through `quadruped_core.paths` (ament share dir, falling back to `src/`), and `config.yaml` lives at `src/quadruped_bringup/config/config.yaml`.

## `IsaacLab_Tasks/` structure

Each subfolder is a fully independent copy of the Isaac Lab task package (own `source/Quadruped/Quadruped/tasks/direct/quadruped/...`, own `logs/`, own `training_phases.yaml`) — nothing is shared between them, so a fix or reward tweak made in one does not propagate to the others. Confirm which module is actually meant before editing:

- **`Walk/`** — the main, actively developed task. Default assumption unless told otherwise.
- **`Walk_GO2/`** — a Go2-only simplification made when a possible internship lab also had a Go2; that plan changed and work went back to `Walk` (3 robots). Likely stale.
- **`Stairs/`** — experiment adding a terrain/height-scan sensor for stair climbing.
- **`Handstand/`** — handstand task.
- **`Simple/`** — unitree_rl_lab cut down to the Go2 (package `simple`, manager-based, rsl_rl): the Sigma-Vel-Foot-Rough and Deploy tasks. Trains through its own `scripts/rsl_rl/train.py`.
- **`WalkTheseWays/`** — Walk These Ways for the Go2 (package `walk_these_ways`), cut from Simple to the single `Unitree-Go2-Velocity-WTW` task. Same rsl_rl layout as Simple.

## Reward/training config pattern (Walk)

Rewards and curriculum for the main task live in `IsaacLab_Tasks/Walk/source/Quadruped/Quadruped/tasks/direct/quadruped/training_phases.yaml`: a `default` block plus `phases.phaseN` entries that `inherits` from a parent phase and deep-merge over it (see `resolve_phase_launcher` in `launcher.py`). Each phase configures `env`, `domain_randomization`, `events`, `rewards`, and `commands`. The yaml only supplies scale factors/thresholds — the actual reward math is in `quadruped_env.py` (`_get_rewards` and the JIT-compiled reward function it calls).
