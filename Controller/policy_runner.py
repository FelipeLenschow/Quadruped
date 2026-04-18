import os
import torch
import torch.nn as nn
import numpy as np

# Rotation helper
def quat_to_rot_matrix(q):
    """(w, x, y, z) -> [3,3] matrix"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])

class RunningStandardScaler(nn.Module):
    def __init__(self, size, device):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(size))
        self.register_buffer("running_variance", torch.ones(size))
        self.register_buffer("current_count", torch.ones(()))

    def forward(self, x):
        return (x - self.running_mean) / torch.sqrt(self.running_variance + 1e-8)

class PolicyMLP(nn.Module):
    def __init__(self, obs_dim, layers, action_dim):
        super().__init__()
        network_layers = []
        last_dim = obs_dim
        for l in layers:
            network_layers.append(nn.Linear(last_dim, l))
            network_layers.append(nn.ELU())
            last_dim = l
        self.net_container = nn.Sequential(*network_layers)
        self.policy_layer = nn.Linear(last_dim, action_dim)

    def forward(self, x):
        x = self.net_container(x)
        return self.policy_layer(x)

class PolicyRunner:
    def __init__(self, checkpoint_path, obs_dim=None, robot_type="go1", device="cpu"):
        print(f"[PolicyRunner] __init__ called for {checkpoint_path}")
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.robot_type = robot_type
        self.obs_dim = obs_dim or int(os.environ.get("QUADRUPED_OBS_DIM", 49))
        self.is_jit = checkpoint_path.endswith(".jit") or (checkpoint_path.endswith(".pt") and self._check_is_jit(checkpoint_path))
        print(f"[PolicyRunner] is_jit detected: {self.is_jit}")
        
        if self.is_jit:
            print(f"[PolicyRunner] Loading JIT model from {checkpoint_path}")
            self.policy_jit = torch.jit.load(checkpoint_path, map_location=device)
            # Detect obs_dim from JIT model if possible, or fallback
            self.obs_dim = self._detect_jit_obs_dim(self.policy_jit)
            self.action_dim = 12
        else:
            self.obs_dim, self.layers = self._inspect_checkpoint(checkpoint_path)
            self.action_dim = 12
            self.action_scale = 0.25
            
            print(f"[PolicyRunner] Initializing with OBS_DIM={self.obs_dim}, layers={self.layers}")
            
            self.policy = PolicyMLP(self.obs_dim, self.layers, self.action_dim).to(self.device).eval()
            self.scaler = RunningStandardScaler(self.obs_dim, self.device).to(self.device)
            
            self._load_checkpoint(checkpoint_path)
            
        # --- Performance Tracking ---
        self.inf_times = []

    def _check_is_jit(self, path):
        # SKRL usually uses .pt for state dicts. JIT models are different.
        # We only treat as JIT if explicitly told or if .jit extension
        if path.endswith(".jit"): return True
        return False

    def _detect_jit_obs_dim(self, model):
        # Infer obs_dim from the model's forward signature or weight shape if possible
        # For now, we rely on the environment variable or common defaults
        return int(os.environ.get("QUADRUPED_OBS_DIM", 49))

    def _inspect_checkpoint(self, path):
        """Detect obs_dim and layer sizes from checkpoint keys and shapes."""
        obs_dim = 236
        layers = [512, 256, 128] # Default fallback
        try:
            data = torch.load(path, map_location="cpu")
            policy_state = data.get("policy", {})
            
            # Detect OBS_DIM from first layer
            for k, v in policy_state.items():
                if "net" in k and "0.weight" in k:
                    obs_dim = v.shape[1]
                    break
            
            # Detect layers
            layer_sizes = []
            i = 0
            while True:
                key = f"net_container.{i}.weight"
                if key in policy_state:
                    layer_sizes.append(policy_state[key].shape[0])
                    i += 2 # Skip activation
                else:
                    break
            if layer_sizes:
                layers = layer_sizes
                
        except Exception as e:
            print(f"[PolicyRunner] Warning: Inspection failed: {e}")
        return obs_dim, layers

    def _load_checkpoint(self, path):
        print(f"[PolicyRunner] Loading checkpoint weights from {path}")
        data = torch.load(path, map_location=self.device)
        print(f"[PolicyRunner] Checkpoint keys: {list(data.keys())}")
        
        # Load policy
        policy_state = data.get("policy", {})
        # Map keys robustly
        net_keys = {}
        for k, v in policy_state.items():
            if "net" in k or "policy" in k:
                # Remove prefixes like '_model.' if present
                clean_key = k.split("_model.")[-1]
                net_keys[clean_key] = v
        
        self.policy.load_state_dict(net_keys, strict=False)
        
        # Load scaler
        scaler_state = data.get("state_preprocessor") or data.get("running_standard_scaler")
        if scaler_state:
            # Map keys if they have '_model.' prefix
            clean_scaler_state = {}
            for k, v in scaler_state.items():
                clean_key = k.split("_model.")[-1]
                clean_scaler_state[clean_key] = v
            self.scaler.load_state_dict(clean_scaler_state)
            print(f"[PolicyRunner] Loaded obs scaler (mean[0]: {self.scaler.running_mean[0]:.3f})")
        else:
            print("[PolicyRunner] WARNING: No obs scaler found in checkpoint!")

    def build_obs(self, state, commands, last_actions, desired_qpos, mj_to_isaac):
        """
        Generic observation builder that works with LowState (Real or Mock).
        state: object with imu.quaternion, base_lin_vel, imu.gyroscope, motorState[...]
        """
        # Base quaternion (w, x, y, z)
        quat = state.imu.quaternion
        R = quat_to_rot_matrix(quat)

        # Body frame velocities
        lin_vel_b = state.base_lin_vel
        ang_vel_b = state.imu.gyroscope

        # Projected gravity
        gravity_w = np.array([0.0, 0.0, -1.0])
        proj_grav = R.T @ gravity_w

        # Joint states
        num_joints = len(mj_to_isaac)
        mj_qpos = np.array([state.motorState[i].q for i in range(num_joints)])
        mj_qvel = np.array([state.motorState[i].dq for i in range(num_joints)])
        jpos_isaac = mj_qpos[mj_to_isaac]
        jvel_isaac = mj_qvel[mj_to_isaac]

        obs_parts = [
            lin_vel_b,
            ang_vel_b,
            proj_grav,
            commands,
            jpos_isaac - desired_qpos,
            jvel_isaac,
            last_actions,
        ]

        if self.obs_dim != 49:
            # Height scan fallback for simulation (not available on real robot without sensor)
            h_val = 0.0 - state.base_pos[2] if hasattr(state, "base_pos") else -0.3
            hscan = np.full(187, h_val, dtype=np.float32)
            obs_parts.append(hscan)

        # Debug print once
        if not hasattr(self, "_obs_debug_done"):
            print(f"[PolicyRunner] Obs Parts Lengths: {[len(p) for p in obs_parts]} (Sum: {sum(len(p) for p in obs_parts)})")
            self._obs_debug_done = True

        obs = np.concatenate(obs_parts).astype(np.float32)
        return obs

    def get_action(self, obs_np):
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.is_jit:
                action_t = self.policy_jit(obs_t)
            else:
                obs_norm = self.scaler(obs_t)
                action_t = self.policy(obs_norm)
        return action_t.squeeze(0).cpu().numpy()

    def infer(self, state, commands, last_actions, desired_qpos, mapping, verbose=False):
        """High-level inference with timing."""
        t_start = time.perf_counter()
        obs = self.build_obs(state, commands, last_actions, desired_qpos, mapping)
        actions = self.get_action(obs)
        t_end = time.perf_counter()
        
        inf_time = t_end - t_start
        self.inf_times.append(inf_time)
        
        if verbose and len(self.inf_times) >= 100:
            avg = sum(self.inf_times) / len(self.inf_times)
            print(f"[PolicyRunner] Avg Inference: {avg*1000:.2f}ms ({1.0/avg:.1f}Hz)")
            self.inf_times = []
            
        return actions, inf_time
