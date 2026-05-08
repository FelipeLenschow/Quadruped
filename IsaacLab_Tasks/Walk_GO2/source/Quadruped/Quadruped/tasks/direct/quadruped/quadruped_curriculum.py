import torch
import os

# ── STARTUP MODES ─────────────────────────────────────────────────────────────

STARTUP_LYING = "lying"
STARTUP_STANDING = "standing"
STARTUP_PASSIVE_DROP = "passive_drop"
STARTUP_ACTIVE_DROP = "active_drop"

# ── CONSTANTS ────────────────────────────────────────────────────────────────

# Folded pose for Go2 (Hip, Thigh, Calf) x 4
LYING_JOINT_POS = [0.0, 1.2, -2.4] * 4 

# Squat pose for Go2 (Hip, Thigh, Calf) x 4 - Safer for starting
SQUAT_JOINT_POS = [0.0, 0.9, -1.8] * 4

# ── BASE REWARDS (all zero — each phase declares what it needs) ──────────────

BASE_REWARDS = {
    "rew_scale_alive": 0.0,
    "rew_scale_track_lin_vel_xy_exp": 0.0,
    "rew_scale_track_ang_vel_z_exp": 0.0,
    "rew_scale_lin_vel_z_l2": 0.0,
    "rew_scale_ang_vel_xy_l2": 0.0,
    "rew_scale_dof_pos_l2": 0.0,
    "rew_scale_dof_torques_l2": 0.0,
    "rew_scale_dof_acc_l2": 0.0,
    "rew_scale_action_rate_l2": 0.0,
    "rew_scale_feet_air_time": 0.0,
    "rew_scale_flat_orientation_l2": 0.0,
    "rew_scale_foot_height_exp": 0.0,
    "rew_scale_feet_air_penalty": 0.0,
    "rew_scale_feet_air_penalty_static": 0.0,
    "rew_scale_joint_vel_l2_static": 0.0,
    "rew_scale_base_height_exp": 0.0,
    "rew_scale_base_height_linear": 0.0,
    "rew_scale_base_contact_penalty": 0.0,
    "target_base_height": 0.35,
    "target_foot_height": 0.1,
    "zero_command_prob": 0.0,
}

# ── PHASE DEFINITIONS ─────────────────────────────────────────────────────────

PHASES = {
    1: {
        "name": "Basic Stand",
        "startup": STARTUP_STANDING,
        "spawn_height": 0.35,
        "commands": (0.0, 0.0, 0.0),
        "noise": 0.0,
        "pushes": (0.0, 0.0),
        "timesteps": 50000,
        "episode_length_s": 10.0,
        "terrain": "flat",
        "randomize_orientation": False,
        "overrides": {
            # Standing incentives
            "rew_scale_alive": 1.0,
            "rew_scale_base_height_exp": 10.0,
            "rew_scale_base_height_linear": 8.0,
            # Yaw tracking (cmd=0 → rewards zero yaw velocity)
            "rew_scale_track_ang_vel_z_exp": 0.75,
            # Pose & orientation guidance
            "rew_scale_dof_pos_l2": -0.5,
            "rew_scale_flat_orientation_l2": -2.5,
            "rew_scale_base_contact_penalty": -1.0,
            # Rotation penalties
            "rew_scale_ang_vel_xy_l2": -0.1,
            # Action smoothness
            "rew_scale_action_rate_l2": -0.01,
            # No termination on tilt
            "base_angle_termination_thresh": 0.0,
        }
    },
    2: {
        "name": "Disturbed Stand",
        "startup": STARTUP_LYING,
        "spawn_height": 0.15,
        "commands": (0.0, 0.0, 0.0),
        "noise": 0.01,
        "pushes": (50.0, 100.0),
        "timesteps": 50000,
        "episode_length_s": 20.0,
        "terrain": "flat",
        "randomize_orientation": False,
        "overrides": {
            # Standing incentives
            "rew_scale_alive": 1.0,
            "rew_scale_base_height_exp": 10.0,
            "rew_scale_base_height_linear": 8.0,
            # Yaw tracking (cmd=0 → rewards zero yaw velocity)
            "rew_scale_track_ang_vel_z_exp": 0.75,
            # Pose & orientation guidance
            "rew_scale_dof_pos_l2": -0.5,
            "rew_scale_flat_orientation_l2": -2.5,
            "rew_scale_base_contact_penalty": -1.0,
            # Rotation penalties
            "rew_scale_ang_vel_xy_l2": -0.1,
            # Action smoothness
            "rew_scale_action_rate_l2": -0.1,
            # No termination on tilt
            "base_angle_termination_thresh": 0.0,
        }
    },
    3: {
        "name": "Passive Drop Stability",
        "startup": STARTUP_PASSIVE_DROP,
        "spawn_height": 0.5,
        "commands": (0.0, 0.0, 0.0),
        "noise": 0.01,
        "pushes": (50.0, 100.0),
        "timesteps": 150000,
        "episode_length_s": 20.0,
        "terrain": "rough",
        "randomize_orientation": True,
        "start_delay_s": 0.3,
        "overrides": {
            # Standing incentives (same as Phase 1/2)
            "rew_scale_alive": 1.0,
            "rew_scale_base_height_exp": 10.0,
            "rew_scale_base_height_linear": 8.0,
            # Velocity tracking (cmd=0 → rewards staying still)
            "rew_scale_track_lin_vel_xy_exp": 1.5,
            "rew_scale_track_ang_vel_z_exp": 0.75,
            # Pose & orientation guidance (same as Phase 1/2)
            "rew_scale_dof_pos_l2": -0.5,
            "rew_scale_flat_orientation_l2": -2.5,
            "rew_scale_base_contact_penalty": -1.0,
            # Rotation penalties
            "rew_scale_ang_vel_xy_l2": -0.1,
            # Action smoothness
            "rew_scale_action_rate_l2": -0.1,
            # Phase 3: termination on extreme tilt
            "base_angle_termination_thresh": 0.2,
        }
    },
    4: {
        "name": "Basic Move",
        "startup": STARTUP_STANDING,
        "spawn_height": 0.35,
        "commands": (1.0, 0.5, 0.5),
        "noise": 0.0,
        "pushes": (0.0, 0.0),
        "episode_length_s": 20.0,
        "timesteps": 150000,
        "terrain": "flat",
        "randomize_orientation": False,
        "overrides": {
            # Discovery Settings
            "base_angle_termination_thresh": 0.5, 
            "zero_command_prob": 0.0,             # FORCE every robot to have a move command
            "action_scale": 0.5,                 # Give it more "kick" to move
            "command_lin_vel_std": 1.0,          # Wider reward ramp for easier discovery
            
            # TASK REWARDS (The only signals)
            "rew_scale_track_lin_vel_xy_exp": 5.0, # Massive reward for moving
            "rew_scale_track_ang_vel_z_exp": 0.75,
            
            # ONE ESSENTIAL PENALTY (Don't hit the floor with your torso)
            "rew_scale_base_contact_penalty": -1.0,
            "rew_scale_flat_orientation_l2": -2.5,
        }
    },
    5: {
        "name": "Robust Move",
        "startup": STARTUP_STANDING,
        "spawn_height": 0.35,
        "commands": (1.0, 1.0, 1.0),
        "noise": 0.01,
        "pushes": (0.0, 0.0),
        "timesteps": 300000,
        "episode_length_s": 20.0,
        "terrain": "rough",
        "randomize_orientation": False,
        "overrides": {
            # Locomotion rewards
            "rew_scale_track_lin_vel_xy_exp": 3.0,
            "rew_scale_track_ang_vel_z_exp": 0.75,
            "rew_scale_feet_air_time": 0.25,
            "rew_scale_foot_height_exp": 1.0,
            # Height & orientation
            "rew_scale_base_height_exp": 0.2,
            "rew_scale_flat_orientation_l2": -2.5,
            "rew_scale_base_contact_penalty": -1.0,
            # Regularization
            "rew_scale_dof_pos_l2": -0.2,
            "rew_scale_lin_vel_z_l2": -2.0,
            "rew_scale_ang_vel_xy_l2": -0.05,
            "rew_scale_dof_torques_l2": -0.0002,
            "rew_scale_dof_acc_l2": -2.5e-7,
            "rew_scale_action_rate_l2": -0.01,
            # Static penalties
            "rew_scale_feet_air_penalty_static": -5.0,
            "rew_scale_joint_vel_l2_static": -0.1,
        }
    },
    6: {
        "name": "Combat Move",
        "startup": STARTUP_STANDING,
        "spawn_height": 0.35,
        "commands": (1.0, 1.0, 1.0),
        "noise": 0.01,
        "pushes": (80.0, 120.0),
        "timesteps": 400000,
        "episode_length_s": 20.0,
        "terrain": "rough",
        "randomize_orientation": False,
        "overrides": {
            # Locomotion rewards
            "rew_scale_alive": 1.0,
            "rew_scale_track_lin_vel_xy_exp": 1.5,
            "rew_scale_track_ang_vel_z_exp": 0.75,
            "rew_scale_feet_air_time": 0.25,
            "rew_scale_foot_height_exp": 1.0,
            # Height & orientation
            "rew_scale_base_height_exp": 2.0,
            "rew_scale_flat_orientation_l2": -2.5,
            "rew_scale_base_contact_penalty": -1.0,
            # Regularization
            "rew_scale_dof_pos_l2": -0.2,
            "rew_scale_lin_vel_z_l2": -2.0,
            "rew_scale_ang_vel_xy_l2": -0.05,
            "rew_scale_dof_torques_l2": -0.0002,
            "rew_scale_dof_acc_l2": -2.5e-7,
            "rew_scale_action_rate_l2": -0.01,
            # Static penalties
            "rew_scale_feet_air_penalty_static": -5.0,
            "rew_scale_joint_vel_l2_static": -0.1,
        }
    },
    7: {
        "name": "Final Boss",
        "startup": STARTUP_ACTIVE_DROP,
        "spawn_height": 1.0,
        "commands": (1.0, 1.0, 1.0),
        "noise": 0.03,
        "pushes": (100.0, 150.0),
        "timesteps": 500000,
        "episode_length_s": 15.0,
        "terrain": "rough",
        "randomize_orientation": True,
        "overrides": {
            # Locomotion rewards
            "rew_scale_alive": 2.0,
            "rew_scale_track_lin_vel_xy_exp": 1.5,
            "rew_scale_track_ang_vel_z_exp": 0.75,
            "rew_scale_feet_air_time": 0.25,
            "rew_scale_foot_height_exp": 1.0,
            # Height & orientation
            "rew_scale_base_height_exp": 2.0,
            "rew_scale_flat_orientation_l2": -2.5,
            "rew_scale_base_contact_penalty": -1.0,
            # Regularization
            "rew_scale_dof_pos_l2": -0.2,
            "rew_scale_lin_vel_z_l2": -2.0,
            "rew_scale_ang_vel_xy_l2": -0.05,
            "rew_scale_dof_torques_l2": -0.0002,
            "rew_scale_dof_acc_l2": -2.5e-7,
            "rew_scale_action_rate_l2": -0.01,
            # Static penalties
            "rew_scale_feet_air_penalty_static": -5.0,
            "rew_scale_joint_vel_l2_static": -0.1,
            "zero_command_prob": 0.05,
        }
    },
    8: {
        "name": "Smooth Pattern & Stillness",
        "startup": STARTUP_STANDING,
        "spawn_height": 0.35,
        "commands": (1.0, 1.0, 1.0),
        "noise": 0.03,
        "pushes": (5.0, 15.0),
        "timesteps": 500000,
        "episode_length_s": 15.0,
        "terrain": "flat",
        "randomize_orientation": False,
        "overrides": {
            # Locomotion rewards
            "rew_scale_alive": 2.0,
            "rew_scale_track_lin_vel_xy_exp": 1.5,
            "rew_scale_track_ang_vel_z_exp": 0.75,
            "rew_scale_feet_air_time": 0.5,
            "rew_scale_foot_height_exp": 1.0,
            # Height & orientation
            "rew_scale_base_height_exp": 2.0,
            "rew_scale_flat_orientation_l2": -2.5,
            "rew_scale_base_contact_penalty": -1.0,
            # Regularization
            "rew_scale_dof_pos_l2": -0.2,
            "rew_scale_lin_vel_z_l2": -2.0,
            "rew_scale_ang_vel_xy_l2": -0.05,
            "rew_scale_dof_torques_l2": -0.0002,
            
            # Smooth Pattern Enforcers
            "rew_scale_action_rate_l2": -0.05,
            "rew_scale_dof_acc_l2": -1.0e-6,
            
            # Zero-Velocity Stillness Enforcers
            "zero_command_prob": 0.25,
            "rew_scale_feet_air_penalty_static": -10.0,
            "rew_scale_joint_vel_l2_static": -0.5,
        }
    },
    10: {
        "name": "Official Go2 Walk",
        "startup": STARTUP_STANDING,
        "spawn_height": 0.35,
        "commands": (1.0, 1.0, 1.0),
        "noise": 0.01,
        "pushes": (0.0, 0.0),
        "timesteps": 1000000,
        "episode_length_s": 20.0,
        "terrain": "rough",
        "randomize_orientation": False,
        "overrides": {
            # Official Termination & Scale
            "base_angle_termination_thresh": 0.7, 
            "terminate_on_base_contact": True,
            "action_scale": 0.25,
            "zero_command_prob": 0.02,
            
            # Official Locomotion Rewards
            "rew_scale_track_lin_vel_xy_exp": 1.5,
            "rew_scale_track_ang_vel_z_exp": 0.75,
            "rew_scale_feet_air_time": 0.25,
            
            # Official Regularization
            "rew_scale_lin_vel_z_l2": -2.0,
            "rew_scale_ang_vel_xy_l2": -0.05,
            "rew_scale_dof_torques_l2": -0.0002,
            "rew_scale_dof_acc_l2": -2.5e-7,
            "rew_scale_action_rate_l2": -0.01,
            "rew_scale_flat_orientation_l2": -2.5,
            
            # Disable all curriculum "training wheels"
            "rew_scale_alive": 0.0,
            "rew_scale_base_height_exp": 0.0,
            "rew_scale_base_contact_penalty": 0.0,
            "rew_scale_foot_height_exp": 0.0,
            "rew_scale_dof_pos_l2": 0.0,
        }
    }
}

def apply_curriculum(cfg, phase_idx: int):
    """Applies the curriculum phase configuration and rewards to the environment config."""
    if phase_idx not in PHASES:
        print(f"[Curriculum] Warning: Phase {phase_idx} not found. Using default Phase 1.")
        phase_idx = 1
    
    p = PHASES[phase_idx]
    print(f"\n[Curriculum] Applying Phase {phase_idx}: {p['name']}")
    
    # 1. Base Overrides (System Settings)
    cfg.startup_mode = p["startup"]
    cfg.spawn_height = p["spawn_height"]
    cfg.randomize_orientation = p["randomize_orientation"]
    cfg.start_delay_s = p.get("start_delay_s", 0.0)
    cfg.training_timesteps = p.get("timesteps", 100000)
    cfg.episode_length_s = p.get("episode_length_s", 20.0)
    
    # 2. Commands Overrides
    cmd_x, cmd_y, cmd_yaw = p["commands"]
    cfg.command_x_range = (-cmd_x, cmd_x)
    cfg.command_y_range = (-cmd_y, cmd_y)
    cfg.command_yaw_range = (-cmd_yaw, cmd_yaw)
    
    # 3. Environment/Noise Overrides
    cfg.observation_noise_scale = p["noise"]
    cfg.random_force_range = p["pushes"]
    if p["pushes"][1] > 0:
        cfg.random_push_interval_range = (2.0, 5.0)
        cfg.random_force_duration_range = (0.1, 0.3)
    else:
        cfg.random_push_interval_range = (0.0, 0.0)
    
    # 4. Terrain Overrides
    terrain_type = os.environ.get("QUADRUPED_TERRAIN", p.get("terrain", "flat")).lower()
    if terrain_type == "flat":
        print("[Curriculum] Setting terrain to FLAT (plane)")
        cfg.scene.terrain.terrain_type = "plane"
        # Note: Do NOT set terrain_generator = None — it destroys the config
        # for any future rough-terrain phase. Just switching terrain_type is enough.
    else:
        print("[Curriculum] Setting terrain to ROUGH (generator)")
        cfg.scene.terrain.terrain_type = "generator"

    # 5. Reward Overrides (Systematic Apply)
    # Start with baseline
    for key, val in BASE_REWARDS.items():
        setattr(cfg, key, val)
    
    # Apply phase-specific overrides (including termination thresholds)
    overrides = p.get("overrides", {})
    for key, val in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
        else:
            # For attributes that might not be in the class but we want to set anyway
            setattr(cfg, key, val)
            
    return cfg
