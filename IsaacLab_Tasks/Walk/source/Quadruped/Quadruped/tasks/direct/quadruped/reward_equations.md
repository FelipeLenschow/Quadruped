# Quadruped Walk IsaacLab Task: Reward Equations Explained

This document details every reward and penalty equation implemented in `quadruped_env.py`. The equations are broken down by their mathematical formulation and their physical purpose in shaping the quadruped's locomotion policy.

## 1. Tracking & Task Rewards

These rewards incentivize the robot to follow the commanded velocities and maintain its operational state.

### Alive Bonus

**Equation:** $R_{alive} = \mathbb{1}(\text{not terminated})$
**Explanation:** A constant reward given at every step as long as the episode has not terminated (e.g., base hasn't hit the ground). It encourages the agent to survive and stay upright.

### Linear Velocity Tracking XY (Exponential)

**Equation:** $R_{track\_lin} = \exp\left(-\frac{||v_{xy} - v_{xy}^{cmd}||^2}{\sigma_{lin}^2}\right)$
**Explanation:** A smooth Gaussian kernel that heavily rewards the robot when its base linear velocity ($v_x, v_y$) closely matches the commanded velocity. The exponential prevents massive negative gradients when the error is large.

### Angular Velocity Tracking Z (Exponential)

**Equation:** $R_{track\_ang} = \exp\left(-\frac{(\omega_z - \omega_z^{cmd})^2}{\sigma_{ang}^2}\right)$
**Explanation:** Similar to linear velocity tracking, this Gaussian kernel rewards the robot for accurately tracking the commanded yaw turn rate ($\omega_z$).

### Foot Swing Height Reward

**Equation:** $R_{foot\_height} = \sum_{i} \exp\left(-\frac{(z_{foot, i} - z_{target})^2}{0.005}\right) \times \mathbb{1}(\text{not contact}_i) \times \mathbb{1}(||v_{cmd}|| > v_{static\_thresh})$
**Explanation:** Encourages each swinging (airborne) foot to reach and maintain a specific target clearance height ($z_{target}$). It is masked out when the robot is commanded to stand still.

### Feet Air Time Reward

**Equation:** $R_{air\_time} = \sum_{i} \max(\text{air\_time}_i - t_{target}, 0) \times \mathbb{1}(\text{first\_contact}_i) \times \mathbb{1}(||v_{cmd}|| > v_{static\_thresh})$
**Explanation:** Rewards the robot at the moment a foot strikes the ground, proportional to how long it was in the air beyond a minimum threshold ($t_{target}$). It encourages healthy step durations and avoids rapid, chattering steps.

---

## 2. Gait Phase Clock Rewards (Locomotion Rhythm)

These rewards enforce rhythmic locomotion (walking/trotting) by synchronizing foot lifts with a velocity-dependent phase clock.

*Let $d_i$ be the circular distance from the current phase clock to leg $i$'s designated swing center, and let $W_i = \exp(-d_i^2 / \sigma_{swing}^2)$ be the active swing window.*

### Gait Phase Clock Reward

**Equation:** $R_{gait\_phase} = \left( \frac{\sum_{i} W_i \times \mathbb{1}(\text{not contact}_i)}{N_{air}^{allowed}} \right) \times \mathbb{1}(v^{cmd}_{eff} > 0.05)$
**Explanation:** Rewards a leg for being airborne during its designated swing window. It is normalized by $N_{air}^{allowed}$ so that walking (1 leg swing) and trotting (2 leg swing) yield equal maximum rewards.

### Gait Phase L1 Penalty

**Equation:** $R_{gait\_l1} = -\sum_{i} \max(d_i - \sigma_{swing}, 0) \times \mathbb{1}(\text{not contact}_i) \times \mathbb{1}(v^{cmd}_{eff} > 0.05)$
**Explanation:** Penalizes the robot if a foot is airborne *outside* of its designated swing window. This prevents legs from hovering when they should be in stance phase.

### Gait Missed Lift Penalty

**Equation:** $R_{missed\_lift} = -\sum_{i} W_i \times \mathbb{1}(\text{contact}_i) \times \mathbb{1}(v^{cmd}_{eff} > 0.05)$
**Explanation:** Penalizes the robot if a foot remains in contact with the ground when its phase clock dictates it should be swinging. Without this, the robot might never lift its feet.

---

## 3. Stability & Base Motion Penalties

These penalties ensure the base remains stable, flat, and moves smoothly without unwanted oscillations.

### Flat Orientation Penalty

**Equation:** $R_{flat\_orient} = -(g_{proj, x}^2 + g_{proj, y}^2)$
**Explanation:** Penalizes base pitch and roll tilt relative to the gravity vector. It forces the robot's chassis to stay level to the ground.

### Linear Velocity Z Penalty

**Equation:** $R_{lin\_z} = -v_z^2$
**Explanation:** Penalizes vertical bouncing along the Z axis.

### Angular Velocity XY Penalty

**Equation:** $R_{ang\_xy} = -(\omega_x^2 + \omega_y^2)$
**Explanation:** Penalizes unwanted roll and pitch angular velocities to prevent the body from wobbling dynamically.

### Base Height Penalty

**Equation:** $R_{base\_height} = -(z_{base} - z_{target})^2$
**Explanation:** Prevents the robot from crouching too low or fully extending its legs, keeping the base at an optimal operational height.

### Base Linear Acceleration Penalty

**Equation:** $R_{base\_acc} = -\sum \left(\frac{v_t - v_{t-1}}{\Delta t}\right)^2$
**Explanation:** Penalizes high-frequency shaking, jitter, or jerky changes in the base's linear velocity.

### Stall Penalty

**Equation:** $R_{stall} = -\max(||v^{cmd}_{xy}|| - ||v_{xy}||, 0) \times \mathbb{1}(||v^{cmd}_{xy}|| > 0.05)$
**Explanation:** Penalizes the robot if its actual speed lags behind the commanded speed, preventing it from getting stuck in a standing local minimum when it should be moving.

### Position Deviation (Leash) Penalty

**Equation:** $R_{pos\_dev} = -||p_{xy} - p_{ref, xy}||$
**Explanation:** Penalizes drifting from an integrated virtual reference trajectory. It acts as an invisible leash to prevent arbitrary wandering sideways from the command direction.

---

## 4. Control & Energy Penalties

These penalties regularize the neural network's outputs to produce smooth, energy-efficient, and realistic motor commands.

### DOF Torques L2 Penalty

**Equation:** $R_{torques} = -\sum \tau^2$
**Explanation:** Energy efficiency penalty on applied joint torques. Encourages the robot to use minimal force.

### DOF Acceleration L2 Penalty

**Equation:** $R_{acc} = -\sum \ddot{q}^2$
**Explanation:** Suppresses high-frequency noise and sudden spikes in motor accelerations, prolonging hardware life.

### Action Rate L2 (Smoothness) Penalty

**Equation:** $R_{action\_rate} = -\sum (a_t - a_{t-1})^2$
**Explanation:** Penalizes sharp, step-to-step changes in the raw neural network action outputs, ensuring smooth control signals.

### DOF Position L2 Penalty

**Equation:** $R_{dof\_pos} = -\sum (q - q_{default})^2$
**Explanation:** Regularizes joint positions toward the default standing pose. It prevents weird, highly contorted leg configurations even if they temporarily succeed.

### Trot Symmetry Penalty

**Equation:** $R_{trot\_sym} = -(\text{hip\_sym\_mult} \times \text{hip\_err} + \text{leg\_err})$
*(Where hip error expects FL=-RR and FR=-RL, and leg error expects FL=RR and FR=RL)*
**Explanation:** Explicitly penalizes asymmetric joint actions across diagonal leg pairs. Ensures left and right legs behave as mirrored pairs.

### Torque Symmetry Penalty

**Equation:** $R_{torque\_sym} = -\sum_{j \in \{\text{thigh, calf}\}} (\tau_{FL, j} - \tau_{RR, j})^2 + (\tau_{FR, j} - \tau_{RL, j})^2$
**Explanation:** Similar to action symmetry, this ensures diagonal legs distribute torque symmetrically, leading to a balanced trot.

---

## 5. Contact & Ground Reaction Force (GRF) Penalties

These ensure smooth interactions with the ground, preventing foot dragging and impact spikes.

### Undesired Contacts Penalty

**Equation:** $R_{undesired} = -\mathbb{1}(\max(\text{forces}_{undesired}) > 1.0\text{N})$
**Explanation:** Penalizes ground contact on any body part other than the foot (e.g., thighs, calves, trunk).

### GRF Balance Penalty ($CV^2$)

**Equation:** $R_{grf\_bal} = -\frac{\text{Var}(F_{z, \text{contact}})}{\max(\text{Mean}(F_{z, \text{contact}})^2, 1.0)}$
**Explanation:** Penalizes the relative variance (unevenness) of ground reaction forces across all feet currently on the ground. Prevents limping or heavily favoring one leg.

### GRF Target Penalty ($mg/n$)

**Equation:** $R_{grf\_target} = -\frac{1}{N_{contact}} \sum_{contacting} \frac{(F_{z, i} - mg/N_{contact})^2}{\max((mg/N_{contact})^2, 1.0)}$
**Explanation:** Penalizes deviation of foot contact force from a perfectly equal distribution of the robot's weight ($mg$) among all currently contacting feet.

### Max Contact Force Penalty

**Equation:** $R_{max\_force} = -\frac{\sum_{i} \max(F_{z, i} - 0.75mg, 0)^2}{\max((mg)^2, 1.0)}$
**Explanation:** A scale-invariant penalty for impact spikes. Penalizes any individual foot force exceeding $75\%$ of the robot's total weight.

### Max Air Feet Penalty

**Equation:** $R_{max\_air} = -\max\left(\left(\sum_{i} \mathbb{1}(\text{not contact}_i)\right) - N_{air}^{allowed}, 0\right)$
**Explanation:** Hard penalty for having too many feet in the air simultaneously (e.g., $>1$ for walk, $>2$ for trot), preventing jumping or bouncing behaviors.

---

## 6. Anti-Stagnation & Standing Still Safeguards

These activate specifically when the robot is commanded to stand still ($v_{cmd} < v_{static\_thresh}$).

### Feet Air Penalty

**Equation:** $R_{feet\_air} = -\sum_{i} \mathbb{1}(\text{not contact}_i)$
**Explanation:** A constant slight penalty for every foot in the air at any speed, encouraging feet to return to the ground quickly.

### Static Feet Air Penalty

**Equation:** $R_{feet\_air\_static} = -5.0 \times \sum_{i} \mathbb{1}(\text{not contact}_i) \times \mathbb{1}(v_{cmd} < v_{static\_thresh})$
**Explanation:** A very strong penalty for lifting any foot when commanded to stand completely still.

### Static Joint Velocity Penalty

**Equation:** $R_{joint\_vel\_static} = -\sum \dot{q}^2 \times \mathbb{1}(v_{cmd} < v_{static\_thresh})$
**Explanation:** Penalizes any joint motion when the commanded velocity is zero, eliminating idle jitter and enforcing a frozen, stable stand.

---

## 7. TODOs

- [ ] Make the `0.005` variance in the foot swing height reward proportional to `z_target` (e.g. scaling the tolerance window based on the desired step height).
- [ ] Update the feet air time reward so it is not proportional to total air time; instead, give a flat reward if the foot was airborne for longer than the target time.
- [ ] Combine Gait Phase Clock Reward and Missed Lift Penalty into a single continuous equation using the contact signal to switch between positive and negative (e.g., $W_i \times (\mathbb{1}(\text{not contact}_i) - \mathbb{1}(\text{contact}_i))$).
- [ ] Change the stall penalty from a linearly increasing penalty (`cmd_speed - robot_speed`) to a single flat/heavy penalty that triggers if the robot's speed is below the threshold while commanded to move.
