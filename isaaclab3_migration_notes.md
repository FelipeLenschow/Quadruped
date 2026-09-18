# Isaac Lab 2.x → 3.0 migration notes

What broke when this repo moved from Isaac Lab `main` (v0.54.4, Isaac Sim 5.1, Python 3.11) to
Isaac Lab **v3.0.0-EA** (Isaac Sim 6.1, Python 3.12) on 2026-09-18, and what each fix was.

The trigger was [IsaacLab#7479](https://github.com/isaac-sim/IsaacLab/issues/7479): `UNITREE_GO2_CFG`
applied one set of DC-motor limits to all twelve leg joints, ignoring the Go2's ~1.92 knee
reduction. The calf was capped at 23.5 N·m instead of 45.43 and allowed 30 rad/s instead of 15.70.
[PR#7564](https://github.com/isaac-sim/IsaacLab/pull/7564) fixed it in `develop`, `release/3.0.0`
and `v3.0.0-EA` — but **not** in `main`, which is why the upgrade was needed at all. There is no
`v3.0.0` GA tag; `v3.0.0-EA` is the newest 3.0 ref.

## Environment

| | before | after |
|---|---|---|
| Python | 3.11.15 | 3.12.12 (uv-managed) |
| Isaac Sim | 5.1.0.0 | 6.1.0.0 |
| Isaac Lab | `main` @ v0.54.4, `~/IsaacLab` | `v3.0.0-EA` @ `ae37b02`, `~/IsaacLab30` |
| torch | 2.7.0+cu128 | 2.11.0+cu128 |
| skrl | 2.1.0 | 2.1.0 |

`~/env_isaacsim` was rebuilt from scratch, so Isaac Sim 5.1 is gone and there is no fallback.
`~/IsaacLab` still exists but is no longer what gets imported, and its `_isaac_sim` symlink dangles.

### Install gotchas

- **`unset PYTHONPATH` before activating.** A sourced ROS 2 Humble setup puts its Python 3.10
  `site-packages` on `PYTHONPATH`, and those leak into the 3.12 venv. The symptom is pip
  complaining about unrelated ROS packages (`launch-ros ... requires pyyaml`).
- **Isaac Lab 3.0 is a uv workspace.** `pip install -e source/<pkg>` pulls only that package's own
  dependencies, not the root project's ~52 externals, so imports fail one at a time. Prefer
  `uv sync`. With pip, install workspace members in topological order with `--no-deps`
  (`isaaclab`, `isaaclab_physx`, `isaaclab_newton`, `isaaclab_contrib`, `isaaclab_assets`,
  `isaaclab_tasks`, `isaaclab_rl`) and then the root dependency list. `rich`, `hydra-core` and
  `lazy_loader` are the ones that bite first.
- **Keep Warp at Isaac Sim's version** (1.17.0), not EA's pinned 1.16.0.
- Four packages violate Isaac Sim's exact pins (`click`, `psutil`, `llvmlite`, `usd-exchange`) but
  boot fine. Do not "fix" them — `numba` requires the newer `llvmlite`.
- EA overrides `newton[sim]==1.5.2` while Isaac Sim 6.1 installs `1.5.0`. pip does not reconcile
  this. First suspect if Newton misbehaves.

## API breaks, in the order they surface

### 1. `PhysxCfg` moved

```
AttributeError: No isaaclab.sim attribute PhysxCfg
```

`PhysxCfg` left `isaaclab.sim` for `isaaclab_physx.physics`. All `gpu_*` capacity fields survived
the move unchanged.

```python
from isaaclab_physx.physics import PhysxCfg   # was sim_utils.PhysxCfg
```

### 2. `SimulationCfg.physx` renamed

The field is now `physics: PhysicsCfg | None`, defaulting to `None`, which means `PhysxCfg()`.
**PhysX is still the default backend** — a common misreading of the 3.0 notes.

### 3. `--headless` is no longer a launcher arg

```
train.py: error: unrecognized arguments: --headless
```

`AppLauncher.add_app_launcher_args` no longer registers it; the visualization args are now
`--visualizer/--viz`, `--livestream`, `--max_visible_envs`, `--xr`. Headless is the default and you
opt *into* rendering.

`headless` still works as an `AppLauncher` key, but the name is **reserved** — adding it to the
parser yourself raises `ValueError: The passed ArgParser object already has the field 'headless'`.
Consume it after parsing instead:

```python
args_cli, hydra_args = parser.parse_known_args()
if "--headless" in hydra_args:
    hydra_args.remove("--headless")
    args_cli.headless = True
if not getattr(args_cli, "headless", False) and args_cli.visualizer is None:
    args_cli.visualizer = "kit"
```

`--enable_cameras` was removed the same way. The `if args_cli.video:` branch still assigns it, so it
does not raise, but video capture may silently do nothing. Untested.

### 4. `InteractiveScene.clone_environments` removed

```
AttributeError: 'InteractiveScene' object has no attribute 'clone_environments'
```

Direct envs drive the cloner themselves. Note the changelog references
`ClonePlan.from_env_0`, which does not exist — the real name is the module-level
`cloner.clone_plan_from_env_0`. `InteractiveScene.env_ns` and `env_regex_ns` are gone too.

```python
src, dest = "/World/envs/env_0", "/World/envs/env_{}"
pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
plan = cloner.clone_plan_from_env_0(
    src, dest, self.scene.num_envs, self.device, pos, global_paths=("/World/ground",)
)
cloner.replicate(plan, stage=self.scene.stage)
if "physx" in self.scene.physics_backend:
    self.scene.filter_collisions(global_prim_paths=["/World/ground"])
```

Terrain still uses grid positions here; the terrain prim goes in `global_paths` and into
`filter_collisions`. See `isaaclab_tasks/contrib/anymal_c_direct/anymal_c_env.py` for the
reference locomotion pattern.

### 5. Replication is queued by the **spawner**

```
RuntimeError: Failed to initialize contact reporter for specified bodies.
        View body count : 12 (64 envs, 0 per env), expected 12 per env
```

`UsdReplicateContext` is only added when the asset cfg has a spawner. The old idiom — spawn
`env_0` manually with `spawn.func(...)`, then set `robot_cfg.spawn = None` — leaves nothing queued,
so `replicate()` silently clones nothing and every view sees one env's worth of bodies. Keep the
spawner on the cfg for the replication path.

### 6. Newton actuators want USD prims

```
ValueError: No NewtonActuator prims found targeting any of: ['FL_hip_joint', ...]
```

`SimulationCfg.use_newton_actuators` defaults to `True` in 3.0, which makes Newton execute
`DCMotorCfg` / `IdealPDActuatorCfg` by authoring `NewtonActuator` USD prims. Setting it to `False`
restores the old Isaac Lab actuator path.

**This flag is load-bearing here and also a behaviour choice.** `False` is what keeps runs
comparable to the PhysX-tuned `Final*` / `ph*` curricula. Leaving it `True` is a real dynamics
change stacked on top of the Go2 calf-limit fix.

### 7. `data.*` returns warp-backed arrays

```
RuntimeError: compute_rewards() Expected a value of type 'Tensor' for argument 'base_lin_vel'
  but instead found type 'ProxyArray'.
```

Every `asset.data.<field>` is now a `ProxyArray` with a `.torch` accessor. This is the big one —
51 call sites in `quadruped_env.py` / `quadruped_mdp.py`. `asset.data.joint_pos` →
`asset.data.joint_pos.torch`.

Related: `root_physx_view.get_masses()` / `get_coms()` return raw warp arrays with **no** `.torch`
attribute (use `wp.to_torch`), and the setters moved to asset level:

```python
view.set_masses_index(masses=masses[ids], env_ids=ids)
view.set_coms_index(coms=coms[ids], env_ids=ids)
```

Device discipline matters. `get_coms()` hands back a CPU tensor while `env_ids` must be a device
int64 array. Mixing them either raises a warp kernel type error or — worse — crashes the process
with `CUDA error 700: an illegal memory access` from inside the backend's device-to-host copy.
Move the payload to `self.device` and keep indices there too.

## Caveats on the ported result

- **The `.torch` sweep was mechanical.** All 51 sites compile and training runs, but that proves
  shapes work, not that every field means what it did in 2.x. Compare a reward curve against `ph3`
  before trusting numbers for the paper.
- **`IsaacLab_Tasks/unitree_rl_lab/` still vendors the old Go2 config** with `effort_limit=23.5`,
  `velocity_limit=30.0`, hardcoded. It does not import from `isaaclab_assets` and was unaffected by
  the upgrade.
- **Only `Walk/` was migrated.** `Stairs/`, `Handstand/` and `Walk_GO2/` are independent copies and
  still contain all seven breaks above.
- `launcher.py`'s Docker/PY3.10 menu options stay disabled. The Isaac env is further from ROS 2
  Humble on 3.12 than it was on 3.11; MuJoCo eval and robot deploy still need `venv_robot` or
  `Docker/`.

## vGPU: "mempool not supported"

Warp prints this at init on the `NVIDIA L40S-48Q`:

```
Support for CUDA memory pools was not detected on devices ['cuda:0'].
This prevents memory allocations in CUDA graphs and may result in poor performance.
```

The whole memory-virtualization family is unavailable on this Q-series vGPU profile:
`MemoryPoolsSupported`, `ManagedMemory`, `ConcurrentManagedAccess` and
`VirtualMemoryManagementSupported` all read 0; only plain `UnifiedAddressing` is on. The hint about
the UVM driver is misleading — managed memory is not offered, not merely disabled. This is gated by
the hypervisor's vGPU manager, so it is not fixable from inside the VM and is not a 3.0 regression;
5.1 had the same limits and simply did not print the warning.

Consequences:

- **Lost CUDA graph replay** is the real cost. Warp batches many small kernels into a captured graph;
  capture needs stream-ordered allocation, so without pools the per-step launch overhead returns.
  That overhead is fixed per step, so low env counts look far worse than high ones. A 64-env smoke
  test measured ~34 it/s and is **not** representative — benchmark at production `num_envs`.
- Newton leans on graph replay harder than the PhysX pipeline, so `use_newton_actuators=False`
  incidentally avoids the worst of it.
- PhysX cannot grow GPU buffers on demand and pre-allocates from the fixed `gpu_*` capacities in
  `PhysxCfg`. Exceeding one is a hard failure, not a quiet realloc.
- No managed memory means no VRAM oversubscription: 48 GiB is a hard ceiling and too large a
  `num_envs` OOMs outright.
