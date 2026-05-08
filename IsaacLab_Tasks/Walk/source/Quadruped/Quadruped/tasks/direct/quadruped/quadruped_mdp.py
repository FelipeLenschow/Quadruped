import torch
import isaaclab.envs.mdp as mdp
from isaaclab.utils.math import sample_uniform

def push_robot_heterogeneous(env, env_ids, asset_cfg, velocity_range):
    """
    Heterogeneous target push event. Filters global env_ids to only those belonging to the specific articulation view.
    """
    # Resolve asset name from object or dict
    if isinstance(asset_cfg, dict):
        asset_name = asset_cfg["name"]
    else:
        asset_name = asset_cfg.name

    # Fallback for homogeneous mode where env.robot_view_indices isn't set
    if not getattr(env, "is_heterogeneous", False):
        return mdp.push_by_setting_velocity(env, env_ids, asset_cfg, velocity_range)

    # Resolve global indices for the requested asset
    # Indices are: 0 -> A1, 1 -> Quadruped, 2 -> Go2
    indices_list = env.robot_view_indices
    
    if "a1" in asset_name:
        view_global_indices = indices_list[0]
    elif "quadruped" in asset_name:
        view_global_indices = indices_list[1]
    elif "go2" in asset_name:
        view_global_indices = indices_list[2]
    else:
        # Fallback to default behavior if asset name doesn't match
        return mdp.push_by_setting_velocity(env, env_ids, asset_cfg, velocity_range)

    # Filter global env_ids to those present in this view
    mask = torch.isin(env_ids, view_global_indices)
    valid_global_ids = env_ids[mask]
    
    if len(valid_global_ids) > 0:
        # Find LOCAL indices within the Articulation view
        # Example: if view_global_indices is [0, 3, 6] and valid_global_ids is [3, 6], 
        # local_ids should be [1, 2].
        local_ids = torch.isin(view_global_indices, valid_global_ids).nonzero().squeeze(-1)
        
        # Get the asset view
        asset = env.scene[asset_name]
        
        # PUSH logic from mdp.push_by_setting_velocity (using local_ids write)
        root_vel = asset.data.root_vel_w[local_ids].clone()
        # Sample random velocities
        vel_x = sample_uniform(velocity_range["x"][0], velocity_range["x"][1], (len(local_ids),), device=env.device)
        vel_y = sample_uniform(velocity_range["y"][0], velocity_range["y"][1], (len(local_ids),), device=env.device)
        root_vel[:, :2] = torch.stack((vel_x, vel_y), dim=-1)
        
        # Apply to simulation using LOCAL indices
        asset.write_root_velocity_to_sim(root_vel, local_ids)
