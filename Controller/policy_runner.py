import os
import time
import zipfile
import torch
import torch.nn as nn
import numpy as np
# Importable both as `Controller.policy_runner` (the drivers) and as a top-level
# `policy_runner` (Controller/Utils/export_jit.py), so the constant is reached either way.
try:
    from Controller.robot_defaults import DEFAULT_STANCE_QPOS
except ImportError:  # pragma: no cover - depends on caller's sys.path
    from robot_defaults import DEFAULT_STANCE_QPOS


# Per-term observation scales from unitree_rl_lab's Go2 ObservationsCfg
# (base_ang_vel scale=0.2, joint_vel_rel scale=0.05). Isaac Lab's observation manager
# applies these before the policy's normalizer, so build_obs has to reproduce them for
# the exported TorchScript to see the inputs it was trained on.
UNITREE_ANG_VEL_SCALE = 0.2
UNITREE_JOINT_VEL_SCALE = 0.05


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
        if self.obs_dim % self._obs_dim_single != 0:
            # A checkpoint whose obs_dim is not a multiple of the assumed single-step width
            # but is itself small enough to BE one step is a policy with no observation
            # history and a different command width -- the stock Isaac Lab velocity task
            # (48: 3-wide commands) against our default of 49. Adopt it instead of limping
            # on with a mismatched history buffer, which used to surface as a tensor-size
            # error deep inside the observation scaler rather than here.
            if 45 <= self.obs_dim <= 60:
                print(
                    f"[PolicyRunner] obs_dim {self.obs_dim} is not a multiple of "
                    f"{self._obs_dim_single}; treating it as a single-step, no-history "
                    f"policy and using {self.obs_dim} as the step width."
                )
                self._obs_dim_single = self.obs_dim
            else:
                print(f"[PolicyRunner] WARNING: Total obs_dim {self.obs_dim} is not a multiple of single-step dim {self._obs_dim_single}.")

        # Which observation layout build_obs should emit. 45 is unambiguous: our own
        # layout's fixed blocks already total 45 BEFORE the command block, which is never
        # narrower than 3, so nothing of ours can land there. Override with
        # QUADRUPED_OBS_LAYOUT=unitree|isaac if a future policy breaks that assumption.
        self._obs_layout = os.environ.get(
            "QUADRUPED_OBS_LAYOUT", "unitree" if self._obs_dim_single == 45 else "isaac"
        )
        if self._obs_layout == "unitree":
            print(
                "[PolicyRunner] Using the unitree_rl_lab observation layout: no base_lin_vel, "
                f"ang_vel x{UNITREE_ANG_VEL_SCALE}, joint_vel x{UNITREE_JOINT_VEL_SCALE}."
            )

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
        """True if `path` is a TorchScript archive rather than a state dict.

        Extension alone is not enough: rsl_rl's export_policy_as_jit writes TorchScript to
        `exported/policy.pt`, so the unitree_rl_lab baselines arrive as .pt files that the
        state-dict loader cannot read. Both formats are zip archives since torch 1.6, but
        only TorchScript carries a `code/` directory (the serialized graph) -- a state dict
        holds just data.pkl and its tensor storages. Checking for that entry separates them
        without paying to load the model twice.
        """
        if path.endswith(".jit"):
            return True
        if not path.endswith(".pt"):
            return False
        try:
            with zipfile.ZipFile(path) as z:
                return any("/code/" in name for name in z.namelist())
        except (zipfile.BadZipFile, OSError):
            return False

    def _detect_jit_obs_dim(self, model):
        """Infer the input width of a TorchScript policy from its own parameters.

        The old fallback of 49 was our own layout, so any foreign JIT (the 45-wide Unitree
        baseline) silently got the wrong width and failed downstream on a shape mismatch.
        Two signals, cheapest first: an exported normalizer's running_mean is exactly one
        observation wide, and failing that the first 2-D weight of the MLP is
        [hidden, obs_dim]. QUADRUPED_OBS_DIM still overrides both.
        """
        env_override = os.environ.get("QUADRUPED_OBS_DIM")
        if env_override:
            return int(env_override)
        try:
            sd = model.state_dict()
            for name, tensor in sd.items():
                if "running_mean" in name and tensor.dim() == 1:
                    return int(tensor.shape[0])
            for name, tensor in sd.items():
                if tensor.dim() == 2:
                    return int(tensor.shape[1])
        except Exception as e:
            print(f"[PolicyRunner] Could not infer JIT obs_dim ({e}); falling back to 49.")
        return 49

    def _inspect_checkpoint(self, path):
        """Detect obs_dim and layer sizes from checkpoint keys and shapes."""
        obs_dim = 236
        layers = [512, 256, 128]  # Default fallback
        try:
            data = torch.load(path, map_location="cpu")

            # rsl_rl (unitree_rl_lab, and Isaac Lab's rsl_rl workflow) writes a different
            # archive than skrl: model_state_dict / optimizer_state_dict / iter / infos.
            # None of the lookups below match it, so obs_dim silently kept the 236 fallback
            # and the mismatch only surfaced much later as a tensor-size error inside the
            # observation scaler. Refuse it here, and say what to load instead: rsl_rl's
            # play.py writes a TorchScript policy next to the run, which this loader reads.
            if "model_state_dict" in data and "policy" not in data:
                raise ValueError(
                    f"{os.path.basename(path)} is an rsl_rl checkpoint, which this loader "
                    "cannot read. Run rsl_rl's play.py once and load the TorchScript it "
                    "exports to <run_dir>/exported/policy.pt instead."
                )

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
                    i += 2  # Skip activation
                else:
                    break
            if layer_sizes:
                layers = layer_sizes

        except ValueError:
            raise
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
                "[PolicyRunner]   EXCEPTION: the stock Isaac Lab velocity tasks set\n"
                "[PolicyRunner]   state_preprocessor: null, so their checkpoints legitimately carry no\n"
                "[PolicyRunner]   scaler and are trained on raw observations. For those this warning is\n"
                "[PolicyRunner]   expected and the identity scaler above is correct.\n"
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

        # Joint states
        num_joints = len(mj_to_isaac)
        mj_qpos = np.array([state.motorState[i].q for i in range(num_joints)])
        mj_qvel = np.array([state.motorState[i].dq for i in range(num_joints)])
        jpos_isaac = mj_qpos[mj_to_isaac]
        jvel_isaac = mj_qvel[mj_to_isaac]

        # Command width is whatever is left over once the fixed-size blocks are accounted
        # for: 3 lin_vel + 3 ang_vel + 3 proj_grav + 12 joint_pos + 12 joint_vel + 12
        # actions = 45. Our own policies carry a 4th command element, so they read 49; the
        # stock Isaac Lab velocity task (Isaac-Velocity-Flat-Unitree-Go2-v0) emits only
        # [vx, vy, wz] and reads 48. Slicing here rather than at the caller keeps every
        # command producer in the codebase unchanged -- the extra element is simply not
        # shown to a policy that was never trained on it. The block order is otherwise
        # identical between the two, which is what makes this a one-line difference.
        cmd = np.asarray(commands).ravel()
        if self._obs_layout == "unitree":
            # unitree_rl_lab's Go2 velocity task. Two structural differences from every
            # other policy here, both from its ObservationsCfg:
            #   * base_lin_vel is in the CRITIC group only -- the actor never sees it. It
            #     is an asymmetric actor-critic, so the block is absent, not zeroed.
            #   * base_ang_vel and joint_vel carry per-term `scale=` factors. Those are
            #     applied by Isaac Lab's observation manager BEFORE the policy's own
            #     normalizer, so the exported TorchScript (which bakes in the normalizer)
            #     still expects them pre-scaled here.
            # Getting either wrong produces a plausible-looking but meaningless rollout,
            # which is why the width check below is a hard failure rather than a warning.
            # asarray, not a bare multiply: the real and MuJoCo drivers hand gyroscope in
            # as a plain list, and list * float is a TypeError rather than a scale.
            obs_parts = [
                np.asarray(ang_vel_b, dtype=np.float32) * UNITREE_ANG_VEL_SCALE,
                proj_grav,
                cmd[:3],
                jpos_isaac - desired_qpos,
                np.asarray(jvel_isaac, dtype=np.float32) * UNITREE_JOINT_VEL_SCALE,
                last_actions,
            ]
        else:
            # Command width is whatever is left over once the fixed-size blocks are accounted
            # for: 3 lin_vel + 3 ang_vel + 3 proj_grav + 12 joint_pos + 12 joint_vel + 12
            # actions = 45. Our own policies carry a 4th command element, so they read 49; the
            # stock Isaac Lab velocity task (Isaac-Velocity-Flat-Unitree-Go2-v0) emits only
            # [vx, vy, wz] and reads 48. Slicing here rather than at the caller keeps every
            # command producer in the codebase unchanged -- the extra element is simply not
            # shown to a policy that was never trained on it. The block order is otherwise
            # identical between the two, which is what makes this a one-line difference.
            n_cmd = max(0, self._obs_dim_single - 45)
            obs_parts = [
                lin_vel_b,
                ang_vel_b,
                proj_grav,
                cmd[:n_cmd],
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
        # Fail loudly on a layout/width mismatch. Silently handing a policy the wrong number
        # of inputs used to surface as a torch shape error several frames deeper, or -- worse,
        # when the widths happened to agree but the blocks did not -- as a rollout that looks
        # like a merely bad policy. Either way the sweep numbers would be meaningless.
        if obs_single.shape[0] != self._obs_dim_single:
            raise ValueError(
                f"[PolicyRunner] Built a {obs_single.shape[0]}-wide observation but the policy "
                f"expects {self._obs_dim_single} (layout '{self._obs_layout}'). Block sizes were "
                f"{[len(p) for p in obs_parts]}."
            )

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
