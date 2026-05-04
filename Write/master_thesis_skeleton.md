# Master's Thesis: High-Performance Quadruped Locomotion via Sim-to-Real Reinforcement Learning

> [!NOTE]
> This skeleton outlines the structure for your Master's thesis based on the Unitree Go2 Locomotion project. It balances both the Software Architecture (the low-latency, hybrid-ROS setup) and the Reinforcement Learning experiments (curriculum learning and recovery). Each section includes bullet points detailing the topics and concepts that should be discussed.

---

## 1. Introduction

*   **1.1 Context and Motivation**
    *   The rise of legged robotics and their versatility in unstructured environments.
    *   The transition from classical control (Model Predictive Control) to Deep Reinforcement Learning (DRL) for agile locomotion.
    *   The Sim-to-Real reality gap: challenges in transferring policies trained in simulation to physical hardware.
    *   The impact of communication latency (e.g., standard ROS 2 overhead) on high-frequency RL control loops.
*   **1.2 Problem Statement**
    *   Standard ROS 2 middleware introduces 5-10ms of jitter, which degrades the performance of policies trained with perfect synchronous simulation steps.
    *   Quadruped robots often suffer from local minima during training (e.g., shuffling, backward crawling, failing to stand up from a fall).
*   **1.3 Objectives**
    *   To develop a **hardware-agnostic, decentralized, low-latency framework** that bypasses ROS 2 communication overhead for the critical control loop.
    *   To design a **robust curriculum learning strategy** in Isaac Lab that trains a policy capable of both walking and recovering from arbitrary falls (self-righting/stand-up).
    *   To validate the system using multiple simulators (MuJoCo, Gazebo, Isaac Sim) ensuring bit-perfect parity before real-world deployment on the Unitree Go2.
*   **1.4 Contributions**
    *   A unified driver architecture with internal inference (the "Hybrid-ROS" approach).
    *   A multi-phase RL curriculum featuring anti-apathy incentives and stabilization rewards.
    *   A successful sim-to-sim and sim-to-real transfer pipeline.
*   **1.5 Thesis Outline**
    *   Brief description of the remaining chapters in the document.

---

## 2. Background and Related Work

*   **2.1 Quadruped Locomotion**
    *   Classical Control Methods: Model Predictive Control (MPC) and Whole-Body Control (WBC).
    *   Deep Reinforcement Learning (DRL) for Locomotion: End-to-end learning vs. hybrid approaches.
*   **2.2 Reinforcement Learning Formulations**
    *   Markov Decision Processes (MDP).
    *   Proximal Policy Optimization (PPO): The primary algorithm used for training stable policies.
    *   Reward shaping and curriculum learning concepts.
*   **2.3 Sim-to-Real Transfer Techniques**
    *   Domain Randomization: Varying mass, friction, motor strength, and sensor noise to create robust policies.
    *   Actuator Modeling: Accurate representation of motor dynamics, latency, and PD control loops.
*   **2.4 Simulation Environments**
    *   **Isaac Lab (NVIDIA):** Used for massively parallel GPU-based training.
    *   **Isaac Sim, MuJoCo, and Gazebo:** High-fidelity simulators used for validation and verification of the trained policies.
*   **2.5 Middleware in Robotics**
    *   ROS 2 (Robot Operating System) and Data Distribution Service (DDS, specifically CycloneDDS).
    *   Trade-offs between modularity (ROS) and performance/latency (Internal loops).

---

## 3. Unified Software Architecture for Sim-to-Real

*   **3.1 System Overview**
    *   The "Hybrid-ROS" architecture: Using ROS 2 strictly for telemetry and monitoring, while keeping the high-frequency control loop (50Hz inference, 200-500Hz actuator loops) entirely internal.
    *   Decentralized Control: How each driver (Sim or Real) locally instantiates the inference engine to achieve sub-millisecond latency.
*   **3.2 Hardware-Agnostic Core Components**
    *   **`TelemetryManager`:** The centralized state standardizer. How it converts disparate sensor data (from Gazebo, MuJoCo, or the real robot) into a universal `StandardState` object (Joint grouping, coordinate frames).
    *   **`CommandProcessor`:** The safety-first action pipeline. Handling hardware-aware action scaling (e.g., scaling actions by 0.25) and saturation limits (clamping to 90% of physical joint limits).
    *   **`PolicyRunner`:** The cross-platform PyTorch/ONNX inference engine.
*   **3.3 Multi-Simulator Drivers**
    *   Developing specific drivers (`mujoco_driver.py`, `gazebo_driver.py`, `isaac_driver.py`) to mimic the firmware of the physical Unitree Go2.
    *   Ensuring bit-perfect parity: If a policy works in the validation simulator, it is mathematically guaranteed to see the same inputs on the real hardware.
*   **3.4 Real-World Deployment Pipeline**
    *   Deploying the exact same `Controller/` module to the Jetson Orin.
    *   Interfacing with the Unitree SDK2 via CycloneDDS.

---

## 4. Reinforcement Learning Environment and Curriculum

*   **4.1 Environment Design (Isaac Lab)**
    *   **Observation Space:** Using blind proprioception (49 dimensions), removing height scanners to force the robot to rely on IMU and joint states for robustness.
    *   **Action Space:** Target joint positions added to the nominal stance.
*   **4.2 Reward Function Shaping**
    *   **Task Rewards:** Tracking linear and angular velocity commands.
    *   **Locomotion Style Rewards:** Foot height lifting rewards, symmetric gait incentives, and `feet_air_time`.
    *   **Penalties:** Base contact penalties, flat orientation tracking (`flat_orientation_l2`), penalizing excessive joint velocities and torques.
*   **4.3 Curriculum Learning Phases**
    *   **Phase 1: Static Standing.** Incentivizing the robot to stand still without shuffling. Balancing rewards to prevent local minima.
    *   **Phase 4: Walking and Following Commands.** Transitioning from standing to robust locomotion.
    *   **Phase 10: Recovery and Self-Righting.** 
        *   Removing state-based termination (allowing the robot to fall upside down).
        *   Implementing Anti-Apathy/Anti-Farming Incentives: Torso-rotation rewards and penalty masking to force the robot to actively attempt a flip instead of farming rewards through useless leg jittering.
*   **4.4 Overcoming Training Challenges**
    *   Addressing the "crawling and backward-walking" local minima by tuning `base_contact_penalty`.
    *   Handling physics-induced instabilities (e.g., drop-shocks from spawning at 0.35m) by randomizing spawn positions and orientations.

---

## 5. Experiments and Results

*   **5.1 Training Performance**
    *   Analyzing learning curves (PPO) in Isaac Lab.
    *   Reward convergence across the different curriculum phases.
*   **5.2 Latency and Performance Analysis**
    *   Comparing control loop times: Traditional ROS 2 architecture (5-10ms jitter) vs. the unified Internal Inference architecture (< 1ms jitter).
*   **5.3 High-Fidelity Validation (Sim2Sim)**
    *   Demonstrating the transfer of the Isaac Lab policy to MuJoCo and Gazebo.
    *   Analyzing the robot's behavior under different friction and actuator noise conditions in these deterministic environments.
*   **5.4 Recovery and Robustness Testing**
    *   Evaluating the stand-up maneuver success rate from various fallen states (side, back).
    *   Testing the "deadman switch" and recovery mechanics.
*   **5.5 Real-World Deployment (Sim2Real)**
    *   Results of deploying the policy to the physical Unitree Go2.
    *   Qualitative analysis of gait symmetry and stability on real terrain.

---

## 6. Discussion

*   **6.1 Architecture Efficacy**
    *   Why the decentralized, "Turbo Mode" inference loop was critical for the success of the sim-to-real transfer.
*   **6.2 The Importance of Curriculum Design**
    *   How balancing the transition from standing to walking, and finally to recovery, prevented the policy from collapsing.
*   **6.3 Limitations**
    *   Current limitations of the blind proprioceptive approach (e.g., struggling with stairs or large obstacles).

---

## 7. Conclusion and Future Work

*   **7.1 Summary of Contributions**
    *   A brief recap of the low-latency architecture and the successful training pipeline.
*   **7.2 Future Directions**
    *   **Exteroception:** Integrating vision and terrain point clouds to allow for dynamic obstacle avoidance.
    *   **Adaptive Friction:** Adding online friction estimation to adapt to slippery surfaces (e.g., ice, wet floors).
    *   **Dynamic Gaits:** Training policies for specific gaits (bounding, trotting, pacing) via user commands.

---

## References
*   [Include academic papers on PPO, Isaac Lab, Unitree robots, Sim-to-Real, Domain Randomization, etc.]

## Appendices
*   **Appendix A:** Network Architectures and Hyperparameters.
*   **Appendix B:** Detailed Reward Function Weights.
*   **Appendix C:** Configuration File (`config.yaml`) Snippets.
