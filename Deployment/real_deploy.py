#!/usr/bin/env python3
"""
Real Deployment Script for Unitree Quadruped (Go1/A1).
This script uses the STABLE PolicyRunner and the ACTUAL Unitree Legged SDK.
"""

import os
import sys
import time
import numpy as np
import torch

# Ensure we can find the PolicyRunner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from policy_runner import PolicyRunner

try:
    import unitree_legged_sdk as sdk
except ImportError:
    print("\n[ERROR] unitree_legged_sdk not found!")
    print("Please ensure the Unitree SDK is in your PYTHONPATH or installed.")
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────────────────────

CHECKPOINT = "checkpoints/best_agent.pt"
ACTUATOR_NET = "unitree_quadruped.pt"
DECIMATION = 20
ACTION_SCALE = 0.25

# Standard Isaac -> Unitree joint mapping
# Isaac Lab: [FL, FR, RL, RR] x [hip, thigh, calf]
# Unitree SDK uses indices: 0-2 (FR), 3-5 (FL), 6-8 (RR), 9-11 (RL)
# We map them to match the Isaac order for the PolicyRunner
ISAAC_TO_REAL = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
REAL_TO_ISAAC = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8] # Inverted mapping

def main():
    # 1. Initialize Policy Runner
    if not os.path.exists(CHECKPOINT):
        print(f"[Error] Checkpoint not found: {CHECKPOINT}")
        return
    
    runner = PolicyRunner(CHECKPOINT)
    act_net = torch.jit.load(ACTUATOR_NET, map_location="cpu").eval()

    # 2. Initialize Unitree SDK
    UDP_ADDR = "192.168.123.10"
    UDP_PORT = 8007
    
    udp = sdk.UDP(sdk.LOWLEVEL, 8080, UDP_ADDR, UDP_PORT)
    cmd = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)

    # 3. Deployment State
    last_actions = np.zeros(12, dtype=np.float32)
    pos_err_hist = np.zeros((3, 12), dtype=np.float32)
    vel_hist = np.zeros((3, 12), dtype=np.float32)
    commands = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32) # [vx, vy, wz, 0]
    
    # Standard home pose to subtract from observations
    desired_qpos_isaac = np.array([0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5], dtype=np.float32)

    print("\n[DEPLOY] Ready. ENTER to start control loop...")
    input()

    start_time = time.time()
    try:
        while True:
            # Sync with real-time (loop at ~50Hz for policy)
            loop_start = time.time()
            
            # A. Get State
            udp.Recv()
            udp.GetRecv(state)
            
            # Transition real state to common object format for PolicyRunner
            # We add base_lin_vel (which on real robot comes from KF/State Estimator)
            # For simplicity, we assume state.velocity exists or uses imu
            # Note: real SDK state.imu.gyroscope/quaternion are already available
            
            # Build observation (Generic builder handles the mapping)
            obs = runner.build_obs(state, commands, last_actions, desired_qpos_isaac, REAL_TO_ISAAC)
            
            # B. Inference
            actions = runner.get_action(obs)
            targets = actions * ACTION_SCALE + desired_qpos_isaac
            
            # C. Actuator Loop (Local 200Hz-1000Hz PD or ActuatorNet)
            # In real deployment, we often push this to a background thread,
            # but here we do it synchronously within the 50Hz step or as a sub-loop.
            for _ in range(5): # Sub-loops for Torques
                udp.Recv() # Update state for PD
                udp.GetRecv(state)
                
                # Real joint values mapped to Isaac
                mj_qpos = np.array([state.motorState[j].q for j in range(12)])
                mj_qvel = np.array([state.motorState[j].dq for j in range(12)])
                jpos_isaac = mj_qpos[REAL_TO_ISAAC]
                jvel_isaac = mj_qvel[REAL_TO_ISAAC]

                pos_err = jpos_isaac - targets
                pos_err_hist = np.roll(pos_err_hist, 1, 0); pos_err_hist[0] = pos_err
                vel_hist = np.roll(vel_hist, 1, 0); vel_hist[0] = jvel_isaac

                net_in = torch.zeros((12, 6))
                net_in[:, :3] = torch.from_numpy(pos_err_hist.T)
                net_in[:, 3:] = torch.from_numpy(vel_hist.T)
                
                with torch.no_grad():
                    torques = act_net(net_in).squeeze().numpy()
                
                # Map torques back to real indices
                for j, real_idx in enumerate(ISAAC_TO_REAL):
                    cmd.motorCmd[real_idx].tau = np.clip(torques[j], -23.7, 23.7)
                    cmd.motorCmd[real_idx].q = 0 # No position control
                    cmd.motorCmd[real_idx].dq = 0
                    cmd.motorCmd[real_idx].Kp = 0
                    cmd.motorCmd[real_idx].Kd = 0
                
                udp.SetSend(cmd)
                udp.Send()
                time.sleep(0.002) # 500Hz actuator loop

            last_actions[:] = actions
            
            # Wait for next policy step (~20ms total)
            while time.time() - loop_start < 0.02:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[DEPLOY] Interrupted. Stopping...")
    finally:
        # Emergency stop
        for i in range(12):
            cmd.motorCmd[i].tau = 0
            cmd.motorCmd[i].Kp = 0
            cmd.motorCmd[i].Kd = 0
        udp.SetSend(cmd)
        udp.Send()

if __name__ == "__main__":
    main()
