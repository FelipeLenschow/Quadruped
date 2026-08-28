# Rewards Used in Locomotion Papers & Our Quadruped (Grouped & Transposed)

| Reward Category | **Our Quadruped [Phase 6]** | ANYmal Parkour [Locomotion] [cite: 1] | ANYmal Parkour [Navigation] [cite: 1] | Deployable Control [Quadruped] [cite: 2] | Deployable Control [Velocity-Track] [cite: 2] | Robust Perceptive [Locomotion] [cite: 3] | Motion Priors [Low-Level] [cite: 4] | Motion Priors [High-Level] [cite: 4] | Hardware-Agnostic [ANYmal-D] [cite: 5] | Hardware-Agnostic [Spot] [cite: 5] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear / Forward Velocity Tracking** | **3.0** *(exp)* | - | - | 1.0 | 1.0 | 0.75 | 2 | - | 1.0 | 5.0 |
| **Angular / Heading Velocity Tracking** | **2.5** *(exp)* | 5 | - | - | 0.5 | 0.75 | 0.8 | 5 | 0.5 | 5.0 |
| **Position / Direction Tracking** | **-2.0** *(L1 leash)* | 10, 1, -0.5 | 0.15 | - | - | - | - | 15, -2.5 | - | - |
| **Velocity Errors / Body Motion Penalties** | **-2.0** *(vz)*, **-0.05** *(wxy)*, **-4.0** *(stall)* | - | - | - | 2.0, 0.05 | 0.75, 1.0 | - | - | -2.0, -0.05 | -2.0 |
| **Base Orientation / Gravity** | **-3.0** *(proj. gravity xy)* | - | - | 0.5 | 0.5 | - | 0.8 | -0.2, -5.0 | -5.0 | -3.0 |
| **Base Height Tracking** | **-0.5** | - | - | 0.5 | 1.0 | - | - | - | - | - |
| **Joint Torque Penalties** | **-2e-04** *(L2 torques)* | -1e-05 | - | - | 1e-05 | 1e-06 | -2e-05 | -2e-05 | -2.5e-05 | -5e-04 |
| **Torque Limit Penalties** | **-** *(limits & backlash)* | -0.2 | - | - | - | - | -0.2 | -0.2 | - | - |
| **Joint Velocity / Acceleration Penalties** | **-2.5e-07** *(acc)*, **-1e-05** *(static vel)* | -0.001 | - | 0.001 | 2.5e-07 | 0.001 | -7e-05 | -7e-05 | -2.5e-07 | -0.01, -1e-04 |
| **Joint Position Tracking / Limits** | **-0.2** *(L2 default pose)* | -1 | - | - | - | 0.08 | 1.4 | - | - | -0.7 |
| **Action Rate / Smoothness** | **-0.01** *(L2 \Delta action)* | -0.01 | - | 0.01 | 0.05, 0.01 | 0.003 | -0.005 | -0.005 | -0.01 | -1.0 |
| **Collision / Undesired Contacts** | **0.0** *(thigh/calf/trunk >1N)* | -1 | - | - | - | 0.1 | -1 | -1 | -1.0 | -1.0 |
| **Foot Motion (Clearance, Height, Air Time, Slip)**| **0.1** *(height)*, **-0.005** *(air)*, **-5.0** *(static air)* | - | - | - | 3.0 | 0.003, 0.003 | - | - | 0.5 | 0.5, 5.0, -1.0, -0.5 |
| **Feet Contact Forces / Acceleration** | **-0.2** *(GRF bal)*, **-0.15** *(target)*, **-5e-04** *(max)* | -0.002, -1e-05 | - | - | - | - | -1e-04, -0.005 | -1e-04, -0.005 | - | - |
| **Base Acceleration** | **-** | -0.001 | - | - | - | - | - | - | - | - |
| **Termination / Stumble** | **1.0** *(alive bonus)* | -200, -1 | -0.5 | - | - | - | - | -200 | - | - |
| **Task-Specific (Residuals, Gait, Wait)** | **1.0** *(gait clock)*, **-0.5** *(gait L1)* | -1 | - | - | - | - | - | -0.1 | - | 10.0 |

---

# Our Quadruped Locomotion Reward Architecture

Our reward structure is specifically designed for multi-phase curriculum training (`training_phases.yaml`) across flat, rough, and multi-robot domain-randomized environments. It emphasizes velocity-dependent rhythmic gait generation via a phase clock while applying targeted anti-stagnation and physics-normalized Ground Reaction Force (GRF) penalties.

## 1. Curriculum Phase Schedule (`training_phases.yaml`)

We anneal and adjust reward weights across training phases to guide the policy from basic walking on flat ground (`Phase 1`) to robust Sim2Real deployment (`Phase 5-6`):

| Reward Parameter | `default` | `Phase 1` (Walk Base) | `Phase 2` (Hardening) | `Phase 3–5` (Terrain & Sim2Real) | `Phase 6` (Stabilization) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `rew_scale_alive` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` |
| `rew_scale_track_lin_vel_xy_exp` | `3.0` | `3.0` | `3.0` | `3.0` | `3.0` |
| `rew_scale_track_ang_vel_z_exp` | `2.5` | `2.5` | `2.5` | `2.5` | `2.5` |
| `rew_scale_gait_phase` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` |
| `rew_scale_gait_phase_l1` | `-0.5` | `-0.5` | `-0.5` | `-0.5` | `-0.5` |
| `rew_scale_foot_height_penalty` | `0.1` | `0.1` | `0.1` | `0.1` | `0.1` |
| `rew_scale_flat_orientation_l2` | `-6.0` | `-6.0` | `-6.0` | `-6.0` | `-6.0` |
| `rew_scale_lin_vel_z_l2` | `-2.0` | `-2.0` | `-2.0` | `-2.0` | `-2.0` |
| `rew_scale_ang_vel_xy_l2` | `-0.05` | `-0.05` | `-0.05` | `-0.05` | `-0.05` |
| `rew_scale_dof_pos_l2` | `-0.2` | `-0.2` | `-0.2` | `-0.2` | `-0.2` |
| `rew_scale_dof_torques_l2` | `-0.0002` | `-0.0002` | `-0.0002` | `-0.0002` | `-0.0002` |
| `rew_scale_dof_acc_l2` | `-2.5e-7` | `-2.5e-7` | `-2.5e-7` | `-2.5e-7` | `-2.5e-7` |
| `rew_scale_action_rate_l2` | `-0.01` | `-0.01` | `-0.01` | `-0.01` | `-0.01` |
| `rew_scale_base_acc_l2` | `-0.0002` | `-0.0002` | `-0.0002` | `-0.0002` | `-0.0002` |
| `rew_scale_feet_air_penalty` | `-0.005` | `-0.005` | `-0.005` | `-0.005` | `-0.005` |
| `rew_scale_feet_air_penalty_static` | `-5.0` | `-5.0` | `-5.0` | `-5.0` | `-5.0` |
| `rew_scale_joint_vel_l2_static` | `-1.0e-5` | `-1.0e-5` | `-1.0e-5` | `-1.0e-5` | `-1.0e-5` |
| `rew_scale_grf_balance` | `0.0` | **`-0.4`** | **`-0.2`** | `-0.2` | `-0.2` |
| `rew_scale_grf_target` | `0.0` | **`-0.15`** | `-0.15` | `-0.15` | `-0.15` |
| `rew_scale_max_contact_force` | `0.0` | **`-5.0e-4`** | `-5.0e-4` | `-5.0e-4` | `-5.0e-4` |
| `rew_scale_base_height_l2` | `0.0` | `0.0` | **`-0.5`** | `-0.5` | `-0.5` |
| `rew_scale_pos_deviation_l1` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| `rew_scale_stall` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |

---

## 2. Mathematical Formulation (`quadruped_env.py`)

Below is the detailed specification of every reward and penalty term computed in `_get_rewards()` and `_get_observations()`:

### Positive Tracking & Gait Incentives

| Term | Formula | Condition / Mask | Purpose |
| :--- | :--- | :--- | :--- |
| **Alive Bonus** | `rew_alive = 1.0 * (1.0 - reset_terminated)` | Every step while robot is upright (`height > 0.15m`, `gravity_z <= thresh`). | Encourages the agent to survive without triggering termination. |
| **Linear Velocity Tracking** | `exp(-||v_{xy} - v_{xy}^{cmd}||^2 / \sigma_{lin}^2)` where `\sigma_{lin} = 0.3` | All active steps. | Smooth Gaussian tracking of commanded forward and lateral velocities ($x, y$). |
| **Angular Velocity Tracking** | `exp(-(\omega_z - \omega_z^{cmd})^2 / \sigma_{ang}^2)` where `\sigma_{ang} = 0.3` | All active steps. | Smooth Gaussian tracking of yaw turn rate ($\omega_z$). |
| **Gait Phase Clock Reward** | `\sum_{i \in \text{legs}} [ \exp(-d_{phi, i}^2 / \sigma_{swing}^2) \cdot (1 - c_i) ] / N_{air}^{allowed}` | Masked when effective cmd speed $> 0.05 \text{ m/s}$. | Incentivizes lifting feet (`~contact`) during each leg's rhythmic swing window (`\sigma_{swing} = 0.1`). Normalized by `N_{air}^{allowed}` so walk ($1$ leg) and trot ($2$ legs) earn equal per-step rewards. |
| **Foot Swing Height** | `\sum_{i \in \text{legs}} \exp(-(z_{foot, i} - z_{target})^2 / 0.005) \cdot (1 - c_i)` | Masked when cmd speed $> 10^{-6} \text{ m/s}$ (`static_velocity_threshold`). | Encourages feet in swing (`~contact`) to reach and maintain target clearance height (`z_{target} = 0.1\text{m}`). |

### Velocity-Dependent Gait Blending Parameters
The gait frequency and phase offsets scale smoothly with effective command speed ($v_{cmd} = ||v_{xy}|| + 0.25|\omega_z|$):
* **Frequency Law**: $f(v_{cmd}) = 0.6(1 - e^{-5 v_{cmd}}) + 0.4 v_{cmd} \text{ [Hz]}$
* **Walk Offsets (4-Beat)**: `[0.0, 0.5, 0.75, 0.25]` (FL, FR, RL, RR) used at low speeds.
* **Trot Offsets (Diagonal Pairs)**: `[0.0, 0.5, 0.5, 0.0]` (FL, FR, RL, RR) blended in above `trot_speed_threshold = 0.35 m/s` using sigmoid sharpness `10.0`.

---

### Stability & Base Motion Penalties

| Term | Formula | Weight | Description |
| :--- | :--- | :---: | :--- |
| **Flat Orientation** | `-\sum (g_{proj, x}^2 + g_{proj, y}^2)` | `-3.0` | Penalizes base pitch and roll tilt relative to gravity vector. |
| **Linear Velocity Z** | `-v_z^2` | `-2.0` | Penalizes vertical bouncing and bouncing along the Z axis. |
| **Angular Velocity XY** | `-(\omega_x^2 + \omega_y^2)` | `-0.05` | Penalizes unwanted roll and pitch angular rates. |
| **Joint Position Deviation** | `-\sum (q - q_{default})^2` | `-0.2` | Regularizes joint positions toward the default standing pose ($q_{default}$). |
| **Base Height Deviation** | `-(z_{base} - z_{target})^2` (`z_{target}=0.28m`) | `-0.5` *(Phase 2+)* | Prevents the robot from crouching too low or extending legs too high. |
| **Position Leash Deviation** | `-||p_{xy} - p_{xy}^{ref}||_1` | `-1.5` to `-2.0` | Penalizes drifting more than `max_pos_leash = 0.5m` from integrated virtual reference trajectory. |
| **Stall Deficit Penalty** | `-(||v_{xy}^{cmd}|| - ||v_{xy}^{robot}||)` when $v^{cmd} > 0.05$ | `-3.0` to `-4.0` | Penalizes lagging behind command velocity when moving (`stall_thresh = 0.05 m/s`). |

---

### Contact, GRF & Smoothness Penalties

| Term | Formula | Weight | Description |
| :--- | :--- | :---: | :--- |
| **Gait Phase L1 Penalty** | `-\sum \max(d_{phi, i} - \sigma_{swing}, 0) \cdot (1 - c_i)` | `-0.5` | Penalizes keeping a leg airborne outside its designated phase clock window. |
| **GRF Balance ($CV^2$)** | `-\frac{\text{Var}(F_{z, \text{contact}})}{\text{Mean}(F_{z, \text{contact}})^2}` | `-0.4` *(Phase 1)*<br>`-0.2` *(Phase 2+)* | Penalizes relative unevenness in ground reaction forces across contacting feet. Prevents dragging single limbs. |
| **GRF Weight Share ($mg/n$)** | `-\frac{1}{N_{contact}} \sum \frac{(F_{z, i} - mg/N_{contact})^2}{(mg/N_{contact})^2}` | `-0.15` *(Phase 1+)* | Penalizes deviation of foot contact force from equal distribution of robot weight among currently contacting feet. |
| **Max Contact Force** | `-\sum \max(F_{z, i} - 0.75mg, 0)^2 / (mg)^2` | `-5e-4` *(Phase 1+)* | Scale-invariant penalty for impact spikes exceeding $75\%$ of robot total weight ($mg$). |
| **Joint Torques L2** | `-\sum \tau^2` | `-0.0002` | Energy efficiency penalty on applied joint torques. |
| **Joint Acceleration L2** | `-\sum \ddot{q}^2` | `-2.5e-7` | High-frequency noise suppression on motor accelerations. |
| **Base Acceleration L2** | `-\sum \left(\frac{v_t - v_{t-1}}{\Delta t}\right)^2` | `-0.0002` | Penalizes high-frequency shaking, jitter, or jerky changes in base linear velocity. |
| **Action Smoothness** | `-\sum (a_t - a_{t-1})^2` | `-0.01` | Penalizes sharp step-to-step changes in neural network action output. |

---

### Anti-Stagnation & Standing Still Safeguards

When command speed is below `static_velocity_threshold = 1.0e-6 m/s` (or zero), specialized penalties activate to enforce clean standing behavior and prevent standing-still local minima at higher speeds:

1. **Static Feet Air Penalty (`-5.0 * \sum (1 - c_i)`)**: Strongly penalizes lifting any foot when commanded to stand completely still.
2. **Standard Feet Air Penalty (`-0.005 * \sum (1 - c_i)`)**: Slight constant penalty per airborne foot across all speeds, ensuring feet return to ground when swing is not actively rewarded.
3. **Static Joint Velocity Penalty (`-1.0e-5 * \sum \dot{q}^2`)**: Penalizes joint motion when commanded velocity is zero, preventing idle jitter while standing.

---

## 3. Comparison with Literature Patterns

1. **Exponential Tracking vs. Quadratic L2**: Unlike classic methods (`Deployable Control`, `Motion Priors`) which often use linear L1 or quadratic L2 penalties (`-(v - v_{cmd})^2`), our architecture follows `ANYmal Parkour` and modern RL by using bounded exponential kernels (`exp(-e^2/\sigma^2)`). This prevents large velocity errors from producing overwhelming negative gradients during early training while maintaining strong local tracking gradients near the target.
2. **Rhythmic Phase Clock vs. Hardcoded Symmetry**: Early quadruped curricula (`cp5`, legacy walking) relied on explicit HAA (hip abduction) and leg symmetry penalties (`rew_scale_trot_symmetry`). Our policy replaces these heuristics with a **Velocity-Dependent Gait Phase Clock** (`rew_scale_gait_phase`) coupled with **GRF Balance/Target** penalties (`rew_scale_grf_balance`, `rew_scale_grf_target`). This guides natural walk ($4$-beat) and trot ($2$-beat) coordination without artificially locking diagonal joints together.
3. **Physics-Invariant Force Regularization**: By normalizing GRF deviation against exact dynamic robot weight (`mg/N_{contact}` and `0.75 mg`), our contact force penalties work seamlessly across heterogeneous multi-robot curricula (`Go2`, `Go1`, `A1`) where body masses vary significantly due to domain randomization.

