# Training Analysis: Why cp5 Worked and Phase 2 Didn't

## Context

- **cp5** (March 2026): Best checkpoint. Clean gait, good stability.
- **cp7** (March 2026): Weird back leg movements — overtrained past the sweet spot.
- **Current Phase 2**: Front-leaning, quick-stepping, weird gait phase, limping.

The March checkpoints (cp1–7) were trained from a single monolithic config in the `~/Walk` repo. The current system uses `training_phases.yaml` with a curriculum, but the Phase 2 "improvements" made things worse.

---

## Root Cause Analysis

### 1. Foot-Lifting Rewards Were 10× Too Aggressive

| Parameter | cp5 (March) | Current Phase 2 | Problem |
|---|---|---|---|
| `feet_air_time` | **0.01** | 0.1 | 10× increase encourages excessive aerial phases |
| `foot_height_exp` | **0.2** | 0.5 | 2.5× increase forces unnatural high-stepping |
| `target_foot_height` | **0.1** (10cm) | 0.15 (15cm) | 50% higher — extreme for Go2's ~25cm legs |

> cp5's good gait emerged *naturally* from mild shaping. The robot figured out how to trot because it's the most energy-efficient gait at moderate speeds — not because it was forced to lift its feet high.

### 2. Trot Symmetry Penalty Fought The Natural Gait

The `-0.1` trot symmetry penalty didn't exist in cp5. It penalizes:
```
|FL_thigh_action - RR_thigh_action|² + |FR_thigh_action - RL_thigh_action|²
```

**Why this is wrong:** It penalizes action *values* being different between diagonal legs. But in a real trot, diagonal legs are in *opposite phases* — one is swinging while the other is in stance — so their instantaneous actions are *supposed* to be different. This penalty pushes toward marching (all legs identical), not trotting.

### 3. Zero-Command System Created Reward Confusion

cp5 had **none** of this. The current setup adds:
- `zero_command_fraction = 0.25` → 25% of envs permanently at zero velocity
- 1-second standby delay after every command resample
- `static_velocity_threshold` masking that disables foot rewards when standing

This creates a multi-objective problem (walk AND stand AND transition) with conflicting reward signals. cp5 only had to learn to walk — standing came naturally from the static penalties (`feet_air_penalty_static = -5.0`).

### 4. March Had Friction DR + Pushes From Step 0

cp5 always trained with:
- `joint_friction_range = (0.03, 0.3)` — implicit regularization, smoother gaits
- `joint_damping_range = (0.01, 0.1)` — prevents exploiting zero-friction dynamics
- Pushes at ±0.4 m/s — forces wider stability margin, balanced gaits

Current Phase 1+2 had **all of these at zero**. The policy could learn fragile, unbalanced gaits that only work in the idealized simulation.

### 5. Action Pipeline Complexity

March `_apply_action`:
```python
targets = self.actions * self.cfg.action_scale + self.desired_joint_pos
```

Current `_apply_action` adds action history buffer, delayed action indexing, and backlash deadband simulation — all executing every step even when disabled (latency=0, backlash=0).

---

## Full Parameter Comparison

| Parameter | cp5 (March) | Old Phase 1 | Old Phase 2 | Status |
|---|---|---|---|---|
| `rew_scale_alive` | 1.0 | 1.0 | 1.0 | ✅ Same |
| `rew_scale_track_lin_vel_xy_exp` | 1.5 | 1.5 | 1.5 | ✅ Same |
| `rew_scale_track_ang_vel_z_exp` | 0.75 | 0.75 | 0.75 | ✅ Same |
| `rew_scale_feet_air_time` | **0.01** | 0.01 | **0.1** | 🔴 10× too high |
| `rew_scale_foot_height_exp` | **0.2** | 0.2 | **0.5** | 🔴 2.5× too high |
| `target_foot_height` | **0.1** | 0.1 | **0.15** | 🔴 50% too high |
| `rew_scale_flat_orientation_l2` | -5.0 | -5.0 | -5.0 | ✅ Same |
| `rew_scale_lin_vel_z_l2` | -2.0 | -2.0 | -2.0 | ✅ Same |
| `rew_scale_ang_vel_xy_l2` | -0.05 | -0.05 | -0.05 | ✅ Same |
| `rew_scale_dof_pos_l2` | -0.2 | -0.2 | -0.2 | ✅ Same |
| `rew_scale_dof_torques_l2` | -0.0002 | -0.0002 | -0.0002 | ✅ Same |
| `rew_scale_action_rate_l2` | -0.01 | -0.01 | -0.01 | ✅ Same |
| `rew_scale_feet_air_penalty` | -0.05 | -0.05 | -0.05 | ✅ Same |
| `rew_scale_feet_air_penalty_static` | -5.0 | -5.0 | -5.0 | ✅ Same |
| `rew_scale_joint_vel_l2_static` | -0.1 | -0.1 | -0.1 | ✅ Same |
| `rew_scale_trot_symmetry` | **didn't exist** | 0.0 | **-0.1** | 🔴 Counterproductive |
| `rew_scale_torque_symmetry` | **didn't exist** | 0.0 | **-0.001** | 🔴 Counterproductive |
| `robot_cfg` | RANDOM | GO2 | **RANDOM** | ⚠️ Added complexity |
| `num_envs` | 2,000 | 6,000 | 6,000 | — |
| Pushes | **Always on (±0.4)** | OFF | OFF | 🔴 Missing regularization |
| `joint_friction_range` | **(0.03, 0.3)** | (0, 0) | (0, 0) | 🔴 Missing regularization |
| `zero_command_fraction` | **0** | 0.25 | 0.25 | 🔴 Too much standing practice |
| Standby delay | **none** | 1.0s | 1.0s | 🔴 Reward confusion |

---

## Fixes Applied

### training_phases.yaml

**Phase 1 — Reproduce cp5's recipe:**
- Friction DR `[0.03, 0.3]` from step 0 (like cp5)
- Pushes `[-0.4, 0.4]` from step 0 (like cp5)
- All reward weights match cp5 exactly
- `zero_command_fraction` reduced from 0.25 → 0.10
- Trot/torque symmetry penalties = 0.0

**Phase 2 — Objective refinement (gentle additions only):**
- `rew_scale_grf_balance = -0.2` → even force distribution (objective: similar GRF)
- `rew_scale_base_height_l2 = -0.5` → consistent standing height (objective: base stability)
- `foot_height_exp` bumped slightly to 0.3 (not the old 0.5)
- Everything else inherited from Phase 1

**Phase 3–5:** Gradual DR ramp (terrain → multi-robot → sim2real).

### quadruped_env.py

**New GRF balance reward:**
```python
# Coefficient of variation² of contact forces among feet in contact
# 0 = perfectly balanced, >0 = uneven load distribution
feet_forces = torch.norm(net_contact_forces[:, feet_ids, :], dim=-1)
contact_float = contact.float()
n_contact = contact_float.sum(dim=1).clamp(min=1.0)
mean_force = (feet_forces * contact_float).sum(dim=1) / n_contact
force_var = ((feet_forces - mean_force.unsqueeze(1))² * contact_float).sum(dim=1) / n_contact
grf_balance_val = force_var / mean_force².clamp(min=1.0)
```

Works for both 2-foot trot stance (penalizes uneven diagonal pair) and 4-foot standing (penalizes leaning).

**Standby reduced:** 1.0s → 0.5s for smoother transitions.

### quadruped_env_cfg.py

- Added `rew_scale_grf_balance` config field
- Default phase changed from `phase3` → `phase1`

---

## How to Train

```bash
# Phase 1 only (reproduce cp5 quality):
export QUADRUPED_TRAINING_PHASE=phase1

# Full curriculum (phase1 → phase5 sequentially):
export QUADRUPED_TRAINING_PHASE=phase1_onward
```

Recommendation: Train Phase 1 first, verify cp5-level quality, then run `phase1_onward` for the full curriculum.

---

## Mapping to Objectives

| Objective | How it's addressed |
|---|---|
| **Stand still at zero velocity** | `feet_air_penalty_static = -5.0`, `joint_vel_l2_static = -0.1`, `zero_command_fraction = 0.10`, 0.5s standby |
| **Beautiful gait** | Mild foot shaping (cp5 values), no forced symmetry — gait emerges naturally from velocity tracking |
| **Similar GRF in all legs** | New `rew_scale_grf_balance = -0.2` in Phase 2+ penalizes uneven force distribution |
| **Base stability** | `flat_orientation_l2 = -5.0`, `lin_vel_z_l2 = -2.0`, pushes from step 0, `base_height_l2 = -0.5` in Phase 2+ |
