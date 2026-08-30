import os
import time
import torch
import torch.nn as nn
import numpy as np
# Importable both as `Controller.policy_runner` (the drivers) and as a top-level
# `policy_runner` (Controller/Utils/export_jit.py), so the constant is reached either way.
try:
    from Controller.robot_defaults import DEFAULT_STANCE_QPOS
except ImportError:  # pragma: no cover - depends on caller's sys.path
    from robot_defaults import DEFAULT_STANCE_QPOS


# Rotation helper
def quat_to_rot_matrix(q):
    """(w, x, y, z) -> [3,3] matrix"""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2],
        ]
    )


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
    def __init__(
        self,
        checkpoint_path,
        obs_dim=None,
        robot_type="go1",
        device="cpu",
        verbose=True,
        decimation=4,
    ):
        print(f"[PolicyRunner] __init__ called for {checkpoint_path}")
        self.verbose = verbose
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.robot_type = robot_type
        self.decimation = decimation
        self.counter = 0

        # Default Pose Standards -- shared with the drivers and the safety gate.
        self.desired_qpos = DEFAULT_STANCE_QPOS.copy()

        # Joint mapping: Identity by default (matches our standardized drivers)
        self.mapping = list(range(12))

        self.obs_dim = obs_dim or int(os.environ.get("QUADRUPED_OBS_DIM", 490))

        self._last_infer_time = None
        self._episode_start = True  # Fill all history with the first real obs after a reset

        self.is_jit = checkpoint_path.endswith(".jit") or (
            checkpoint_path.endswith(".pt") and self._check_is_jit(checkpoint_path)
        )
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

            print(
                f"[PolicyRunner] Initializing with OBS_DIM={self.obs_dim}, layers={self.layers}"
            )

            self.policy = (
                PolicyMLP(self.obs_dim, self.layers, self.action_dim)
                .to(self.device)
                .eval()
            )
            self.scaler = RunningStandardScaler(self.obs_dim, self.device).to(
                self.device
            )

            self._load_checkpoint(checkpoint_path)

        # Detect single-step dim based on environment variable (default 49)
        self._obs_dim_single = int(os.environ.get("QUADRUPED_OBS_DIM_SINGLE", 49))
        # Which observation layout this checkpoint expects. See build_obs.
        self.obs_version = int(os.environ.get("QUADRUPED_OBS_VERSION", 2))
        if self.obs_dim % self._obs_dim_single != 0:
            print(f"[PolicyRunner] WARNING: Total obs_dim {self.obs_dim} is not a multiple of single-step dim {self._obs_dim_single}.")

        self._obs_history_len = max(1, self.obs_dim // self._obs_dim_single)
        self._obs_history = np.zeros(
            (self._obs_history_len, self._obs_dim_single), dtype=np.float32
        )  # [0, :] = most recent, [-1, :] = oldest

        # --- Performance Tracking ---
        self.inf_times = []

        # --- Control State ---
        self.last_actions = np.zeros(12, dtype=np.float32)
        self.counter = 0
        self.decimation = 4  # Default for 200Hz -> 50Hz

    def _check_is_jit(self, path):
        # SKRL usually uses .pt for state dicts. JIT models are different.
        # We only treat as JIT if explicitly told or if .jit extension
        if path.endswith(".jit"):
            return True
        return False

    def _detect_jit_obs_dim(self, model):
        # Infer obs_dim from the model's forward signature or weight shape if possible
        # For now, we rely on the environment variable or common defaults
        return int(os.environ.get("QUADRUPED_OBS_DIM", 49))

    @staticmethod
    def _split_policy_state(policy_state):
        """Split a skrl policy state dict into (hidden_sizes, remapped_state).

        skrl writes two different layouts depending on `models: separate:` in the
        agent config, and they disagree about where the OUTPUT head lives:

          separate: False (shared trunk)  net_container.{0,2,4} = hidden
                                          policy_layer          = output head
          separate: True  (own trunk)     net_container.{0,2,4} = hidden
                                          net_container.{6}     = output head

        PolicyMLP always expects the head under `policy_layer`, so the separate
        layout needs its LAST net_container entry moved there. Without this the
        head is left randomly initialised and an extra ELU is applied where the
        output should be -- and because _load_checkpoint uses strict=False, nothing
        complains. The robot simply stands still.
        """
        idxs = sorted(
            int(k.split(".")[1])
            for k in policy_state
            if k.startswith("net_container.") and k.endswith(".weight")
        )
        if not idxs:
            return None, dict(policy_state)

        state = dict(policy_state)
        if "policy_layer.weight" in policy_state:
            hidden = idxs                      # shared layout: head is separate
        else:
            hidden = idxs[:-1]                 # separate layout: last entry is the head
            last = idxs[-1]
            state["policy_layer.weight"] = state.pop(f"net_container.{last}.weight")
            state["policy_layer.bias"] = state.pop(f"net_container.{last}.bias", None)
            if state["policy_layer.bias"] is None:
                del state["policy_layer.bias"]
        return [policy_state[f"net_container.{i}.weight"].shape[0] for i in hidden], state

    def _inspect_checkpoint(self, path):
        """Detect obs_dim and layer sizes from checkpoint keys and shapes."""
        obs_dim = 236
        layers = [512, 256, 128]  # Default fallback
        try:
            data = torch.load(path, map_location="cpu")
            policy_state = data.get("policy", {})

            # Detect OBS_DIM from first layer
            for k, v in policy_state.items():
                if "net" in k and "0.weight" in k:
                    obs_dim = v.shape[1]
                    break

            hidden, _ = self._split_policy_state(policy_state)
            if hidden:
                layers = hidden

        except Exception as e:
            print(f"[PolicyRunner] Warning: Inspection failed: {e}")
        return obs_dim, layers

    def _load_checkpoint(self, path):
        print(f"[PolicyRunner] Loading checkpoint weights from {path}")
        data = torch.load(path, map_location=self.device)
        print(f"[PolicyRunner] Checkpoint keys: {list(data.keys())}")

        # Load policy
        policy_state = data.get("policy", {})
        # Normalise the two skrl layouts onto PolicyMLP's naming (see
        # _split_policy_state) before mapping keys.
        _, policy_state = self._split_policy_state(policy_state)
        net_keys = {}
        for k, v in policy_state.items():
            if "net" in k or "policy" in k:
                # Remove prefixes like '_model.' if present
                clean_key = k.split("_model.")[-1]
                net_keys[clean_key] = v

        result = self.policy.load_state_dict(net_keys, strict=False)
        # strict=False is needed to tolerate extra keys (log_std_parameter,
        # value_layer), but it also silently accepts a policy whose output head
        # never loaded. Missing keys are never acceptable -- that head would stay
        # random and the robot would not move.
        if result.missing_keys:
            raise RuntimeError(
                f"[PolicyRunner] Checkpoint did not supply {result.missing_keys}. "
                f"The policy would run with randomly initialised weights. "
                f"Checkpoint policy keys: {sorted(policy_state)}"
            )

        # Load scaler. skrl renamed this key from "state_preprocessor" to
        # "observation_preprocessor" in 2.1.0, so checkpoints trained before and after the
        # upgrade spell it differently -- accept both. Getting this wrong is not a degradation:
        # the policy is trained on normalized observations, and projected gravity alone has
        # mean -1.0 / std 0.06, so feeding it raw both offsets it by a full unit and shrinks it
        # ~16x. The robot loses its sense of which way is down and collapses on the spot.
        scaler_state = (
            data.get("observation_preprocessor")     # skrl >= 2.1.0
            or data.get("state_preprocessor")        # skrl < 2.1.0
            or data.get("running_standard_scaler")
        )
        if scaler_state:
            # Map keys if they have '_model.' prefix
            clean_scaler_state = {}
            for k, v in scaler_state.items():
                clean_key = k.split("_model.")[-1]
                clean_scaler_state[clean_key] = v
            self.scaler.load_state_dict(clean_scaler_state)
            print(
                f"[PolicyRunner] Loaded obs scaler (mean[0]: {self.scaler.running_mean[0]:.3f})"
            )
        else:
            # Do not let this pass quietly: an unscaled policy does not walk badly, it falls over,
            # and the symptom looks like a bad policy rather than a bad load.
            print(
                "[PolicyRunner] " + "!" * 60 + "\n"
                "[PolicyRunner] WARNING: No obs scaler found in checkpoint -- running UNNORMALIZED.\n"
                f"[PolicyRunner]   checkpoint keys: {sorted(data.keys())}\n"
                "[PolicyRunner]   Expected one of: observation_preprocessor (skrl >= 2.1.0),\n"
                "[PolicyRunner]   state_preprocessor (skrl < 2.1.0), running_standard_scaler.\n"
                "[PolicyRunner]   The robot will almost certainly not stand. Fix the key, do not\n"
                "[PolicyRunner]   retrain.\n"
                "[PolicyRunner] " + "!" * 60
            )

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

        # Accelerometer: body-frame specific force, in g. Both MuJoCo's
        # <accelerometer> and the Unitree IMU report (0, 0, +9.81) at rest and
        # upright, which is the convention Isaac's term was built to match.
        accel_b = np.asarray(state.imu.accelerometer, dtype=np.float64) / 9.81

        # Joint states
        num_joints = len(mj_to_isaac)
        mj_qpos = np.array([state.motorState[i].q for i in range(num_joints)])
        mj_qvel = np.array([state.motorState[i].dq for i in range(num_joints)])
        jpos_isaac = mj_qpos[mj_to_isaac]
        jvel_isaac = mj_qvel[mj_to_isaac]

        # obs_version 2 dropped base_lin_vel (unmeasurable on hardware) and put the
        # accelerometer in its place. Both layouts are 49 wide, so a checkpoint from
        # either era loads without error and only misbehaves at runtime -- hence the
        # explicit switch rather than a shape check. Set QUADRUPED_OBS_VERSION=1 to
        # run a policy trained before the change.
        if self.obs_version >= 2:
            obs_parts = [
                ang_vel_b,
                proj_grav,
                accel_b,
                commands,
                jpos_isaac - desired_qpos,
                jvel_isaac,
                last_actions,
            ]
        else:
            obs_parts = [
                lin_vel_b,
                ang_vel_b,
                proj_grav,
                commands,
                jpos_isaac - desired_qpos,
                jvel_isaac,
                last_actions,
            ]



        # Debug print once
        if not hasattr(self, "_obs_debug_done"):
            print(
                f"[PolicyRunner] Obs Parts Lengths: {[len(p) for p in obs_parts]} (Sum: {sum(len(p) for p in obs_parts)})"
            )
            self._obs_debug_done = True

        obs_single = np.concatenate(obs_parts).astype(np.float32)

        # Roll history: shift oldest out, insert current at front
        self._obs_history = np.roll(self._obs_history, shift=1, axis=0)
        self._obs_history[0, :] = obs_single

        if self._episode_start:
            self._obs_history[:] = obs_single
            self._episode_start = False

        # Fill the whole buffer with the current frame on the first step after a reset.
        #
        # This deliberately does NOT mirror training, where _reset_idx() zeroes obs_history_buf.
        # Feeding real zeros here was tried and is worse: the policy reads them as ten frames of
        # impossible state and commands up to 36 degrees away from the default stance over the
        # first eleven steps, against a steady 29 with replication. In training that transient is
        # absorbed by a robot being respawned; at the pose -> policy handover the robot is already
        # standing still, and a kick that size through the PD loop destabilizes it. Replicating
        # says "the robot has been holding this pose", which is what is actually true here.

        # Return flattened stacked obs
        obs = self._obs_history.flatten()
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

    def should_step(self):
        """Check decimation counter and increment."""
        result = self.counter % self.decimation == 0
        self.counter += 1
        return result

    def infer(self, state, commands, desired_qpos, mapping, dt=0.02, verbose=None):
        """High-level inference with timing and internal history management."""
        # Use class-level verbose if not explicitly overridden
        show_stats = self.verbose if verbose is None else verbose

        t_start = time.perf_counter()
        self._last_infer_time = t_start



        obs = self.build_obs(state, commands, self.last_actions, desired_qpos, mapping)
        actions = self.get_action(obs)
        t_end = time.perf_counter()

        inf_time = t_end - t_start
        self.inf_times.append(inf_time)
        self.last_actions[:] = actions

        if show_stats and len(self.inf_times) >= 100:
            # Stats are now handled by the caller to avoid terminal spam
            # self.inf_times = []
            pass

        return actions, inf_time

    def reset_history(self):
        """Clear all per-episode policy state (call between episodes / eval runs).

        last_actions feeds straight back into the next observation, so leaving it set carries
        the previous episode across the reset -- and carries a NaN across it permanently.
        """
        self._obs_history[:] = 0.0
        self.last_actions[:] = 0.0
        self._episode_start = True

    def step(self, state, commands, dt=0.02, verbose=None):
        """
        Automatic inference step.
        Handles decimation, internal action tracking, and timing.
        Returns the action vector (last produced or newly inferred).
        """
        v = self.verbose if verbose is None else verbose

        if self.counter % self.decimation == 0:
            actions, _ = self.infer(
                state,
                commands,
                self.last_actions,
                self.desired_qpos,
                self.mapping,
                dt=dt,
                verbose=v,
            )
            self.last_actions[:] = actions

        self.counter += 1
        return self.last_actions
