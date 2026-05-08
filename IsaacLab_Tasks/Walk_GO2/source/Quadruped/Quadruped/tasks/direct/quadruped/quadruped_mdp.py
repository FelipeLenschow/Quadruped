import torch
import isaaclab.envs.mdp as mdp
from isaaclab.utils.math import sample_uniform
import isaaclab.utils.math as math_utils

def push_robot_heterogeneous(env, env_ids, asset_cfg, velocity_range):
    """
    Heterogeneous target push event. Filters global env_ids to only those belonging to the specific articulation view.
    """
    # Resolve asset name from object or dict
    if isinstance(asset_cfg, dict):
        asset_name = asset_cfg["name"]
    else:
        asset_name = asset_cfg.name

    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    # If not heterogeneous, use standard logic but implemented here to avoid SceneEntityCfg type issues
    if not getattr(env, "is_heterogeneous", False):
        asset = env.scene[asset_name]
        vel_w = asset.data.root_vel_w[env_ids]
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        vel_w += sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
        asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)
        return

    # Heterogeneous logic
    indices_list = env.robot_view_indices
    
    if "a1" in asset_name:
        view_global_indices = indices_list[0]
    elif "quadruped" in asset_name:
        view_global_indices = indices_list[1]
    elif "go2" in asset_name:
        view_global_indices = indices_list[2]
    else:
        # Fallback
        asset = env.scene[asset_name]
        vel_w = asset.data.root_vel_w[env_ids]
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        vel_w += sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
        asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)
        return

    # Filter global env_ids to those present in this view
    mask = torch.isin(env_ids, view_global_indices)
    valid_global_ids = env_ids[mask]
    
    if len(valid_global_ids) > 0:
        local_ids = torch.isin(view_global_indices, valid_global_ids).nonzero().squeeze(-1)
        asset = env.scene[asset_name]
        
        root_vel = asset.data.root_vel_w[local_ids].clone()
        
        # Sample random velocities using the velocity_range dict
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        
        # Add random samples to current velocity
        root_vel += sample_uniform(ranges[:, 0], ranges[:, 1], root_vel.shape, device=asset.device)
        
        # Apply to simulation using LOCAL indices
        asset.write_root_velocity_to_sim(root_vel, local_ids)
