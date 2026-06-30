import os
import sys
import time
import glob
import subprocess
import json
import platform
import yaml
from datetime import datetime


TASKS_DIR = "IsaacLab_Tasks"
LAST_COMMAND_FILE = ".launcher_last_command.json"
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Configs", "config.yaml"))

# Global Environment Detection
IS_DOCKER = os.path.exists("/.dockerenv")
IS_ROBOT = platform.machine().lower() in ["aarch64", "arm64"]

def ckpt_display_name(path):
    """Extract the run folder name from a full checkpoint path."""
    parts = path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p == "checkpoints" and i > 0:
            return parts[i - 1]
    return os.path.join(*parts[-3:-1]) if len(parts) >= 3 else path

def save_last_command(data):
    try:
        with open(LAST_COMMAND_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass

def load_last_command():
    if os.path.exists(LAST_COMMAND_FILE):
        try:
            with open(LAST_COMMAND_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def run_cli_menu():
    is_isaac = "env_isaacsim" in os.environ.get("VIRTUAL_ENV", "") or "env_isaacsim" in sys.executable
    is_robot = IS_ROBOT

    print("\n" + "=" * 50)
    print(" Quadruped Unified Launcher")
    print("=" * 50)
    
    if is_robot:
        print(" [HARDWARE]    Physical Robot (ARM64)")
    else:
        print(" [HARDWARE]    Remote PC/VDI (AMD64)")

    if IS_DOCKER:
        print(" [ENVIRONMENT] Docker Container Detected")
        print("               (IsaacLab options disabled)")
    elif is_isaac:
        print(" [ENVIRONMENT] IsaacSim Native Environment Detected")
    else:
        print(" [ENVIRONMENT] Host Native Environment")
        
    print("=" * 50 + "\n")

    # 1. Action Selection
    last_cmd = load_last_command()
    print("Select Action:")
    if last_cmd:
        print(f"  [0] Repeat Last: {last_cmd.get('action')} on {last_cmd.get('module_name')}")
        print("  " + "-" * 45)
    
    if not IS_DOCKER:
        print("  [1] Train Policy")
        print("  [2] Play IsaacLab")
        print("  " + "-" * 45)
        print("  [3] Play IsaacSim")
    else:
        print("  [X] Train Policy (AVAILABLE ONLY ON ISAACSIM VENV)")
        print("  [X] Play IsaacLab (AVAILABLE ONLY ON ISAACSIM VENV)")
        print("  " + "-" * 45)
        print("  [X] Play IsaacSim (AVAILABLE ONLY ON ISAACSIM VENV)")
        

    # Detect if we are in an environment that CANNOT run ROS 2 Humble natively
    requires_docker = not IS_DOCKER and sys.version_info[:2] != (3, 10)
    
    if not requires_docker:
        print("  [4] Play MuJoCo")
        print("  [5] Play Gazebo")
        print("  [6] Deploy to Robot")
        print("  " + "-" * 45)
        print("  [C] Console")
        print("  [T] Test Joints (Real Robot)")
        print("  " + "-" * 45)
        print("  --- ROS 2 Tools ---")
        print("  [K] Remote Teleop")
        print("  [V] Visualizers")
        print("  [P] PlotJuggler")
        print("  [M] MCAP Log & Replay")
        print("  [R] RQT Graph")
        print("  [F] TF2 Tree")
    else:
        print("  [X] Play MuJoCo (REQUIRES DOCKER OR PY3.10)")
        print("  [X] Play Gazebo (REQUIRES DOCKER OR PY3.10)")
        print("  [X] Deploy to Robot (REQUIRES DOCKER OR PY3.10)")
        print("  " + "-" * 45)
        print("  [X] Console (REQUIRES DOCKER OR PY3.10)")
        print("  [X] Test Joints (Real Robot) (REQUIRES DOCKER OR PY3.10)")
        print("  " + "-" * 45)
        print("  --- ROS 2 Tools ---")
        print("  [X] Remote Teleop (REQUIRES DOCKER OR PY3.10)")
        print("  [X] Visualizers (REQUIRES DOCKER OR PY3.10)")
        print("  [X] PlotJuggler (REQUIRES DOCKER OR PY3.10)")
        print("  [X] MCAP Log & Replay (REQUIRES DOCKER OR PY3.10)")
        print("  [X] RQT Graph (REQUIRES DOCKER OR PY3.10)")
        print("  [X] TF2 Tree (REQUIRES DOCKER OR PY3.10)")

    action_map = {
        "0": "repeat",
        "1": "train",
        "2": "isaac_lab",
        "3": "isaac_sim",
        "4": "mujoco",
        "5": "gazebo",
        "6": "real_deploy",
        "k": "teleop",
        "K": "teleop",
        "v": "visualizers",
        "V": "visualizers",
        "c": "console",
        "C": "console",
        "t": "hardware_tools",
        "T": "hardware_tools",
        "p": "plotjuggler",
        "P": "plotjuggler",
        "m": "mcap",
        "M": "mcap",
        "r": "rqt_graph",
        "R": "rqt_graph",
        "f": "tf2_tree",
        "F": "tf2_tree",
    }
    
    default_action = "0" if last_cmd else "4"
    if requires_docker and default_action == "4":
        default_action = "None" # No valid default if MuJoCo is blocked

    choice = input(f"Enter choice [0-6, K, V, C, T, P, M, R, F] (default {default_action}): ").strip() or default_action
    action = action_map.get(choice.lower(), "None")
    
    if action == "hardware_tools":
        print("\n--- Hardware Tools ---")
        print("  [1] Telemetry Only (Real Driver, No Policy)")
        print("  [2] Joint Tester (Sine Wave Oscillation)")
        hw_choice = input("Enter choice [1-2] (default 1): ").strip() or "1"
        if hw_choice == "2":
            action = "test_joints"
        else:
            action = "real_telemetry"
            
    if action == "teleop":
        tel_choice = input("Select Teleop [1: Keyboard, 2: Gamepad (Joy)] (default 1): ").strip() or "1"
        if tel_choice == "2":
            action = "teleop_joy"
        else:
            action = "teleop_keyboard"
            
    if action == "visualizers":
        vis_choice = input("Select Visualizer [1: MuJoCo Twin, 2: Gazebo Twin, 3: RViz, 4: Foxglove] (default 1): ").strip() or "1"
        if vis_choice == "4":
            action = "foxglove"
        elif vis_choice == "3":
            action = "rviz"
        elif vis_choice == "2":
            action = "gazebo_twin"
        else:
            action = "mujoco_twin"
    
    # 1.1 Handle Repeat
    if action == "repeat" and last_cmd:
        # Load with defaults to avoid KeyError on old config files
        action = last_cmd.get("action", "mujoco")
        
        # Enforce Docker restrictions even on repeated commands
        if IS_DOCKER and action in ["train", "isaac_lab", "isaac_sim"]:
            print(f"\n[WARNING] Last action '{action}' is not available in Docker. Switching to MuJoCo.")
            action = "mujoco"
        
        if not IS_DOCKER and action in ["mujoco", "gazebo", "real_deploy", "real_telemetry", "mujoco_twin", "gazebo_twin", "rviz", "foxglove", "console", "test_joints", "mcap_record", "mcap_replay_rosbag", "mcap_replay_interactive", "teleop_keyboard", "teleop_joy", "rqt_graph", "tf2_tree"]:
            if sys.version_info[:2] != (3, 10):
                print(f"\n[ERROR] Last action '{action}' requires Python 3.10 or Docker. Aborting.")
                sys.exit(1)

        return (
            last_cmd.get("module_name", "None"),
            last_cmd.get("module_path", "."),
            action,
            last_cmd.get("robot_cfg", "UNITREE_GO2_CFG"),
            last_cmd.get("terrain_cfg", "flat"),
            last_cmd.get("num_envs", "1"),
            last_cmd.get("ckpt"),
            last_cmd.get("teleop", False),
            last_cmd.get("headless", IS_DOCKER),
            last_cmd.get("video", False),
            last_cmd.get("run_name", ""),
            last_cmd.get("domain_id", "1"),
            last_cmd.get("use_estimator", False),
            last_cmd.get("record_session", False),
            last_cmd.get("training_phase", ""),
        )

    # 1.2 Validation
    if IS_DOCKER and choice in ["1", "2", "3"]:
        print("\n[ERROR] Training/IsaacSim actions are not available in Docker. Aborting.")
        sys.exit(1)
        
    if requires_docker and choice.lower() in ["4", "5", "6", "k", "v", "c", "t", "p", "m", "r", "f"]:
        print(f"\n[ERROR] Action '{action}' requires ROS 2 Humble (Python 3.10).")
        print("        Please run this task inside DOCKER or switch to a 3.10 environment.")
        sys.exit(1)

    print(f"\n--- Selected Action: {action.upper()} ---\n")
    
    # Rename Terminator window pane based on action and environment
    try:
        if IS_ROBOT:
            env_str = "ROBOT"
        elif IS_DOCKER:
            env_str = "DOCKER"
        elif is_isaac:
            env_str = "ISAAC"
        else:
            env_str = "HOST"
            
        sys.stdout.write(f"\x1b]2;[{env_str}] {action.upper()}\x07")
        sys.stdout.flush()
    except Exception:
        pass

    # 2. Module Selection
    selected_module_name = "None"
    selected_module_path = "."
    
    if action not in ["mujoco_twin", "gazebo_twin", "rviz", "foxglove", "console", "teleop", "teleop_keyboard", "teleop_joy", "test_joints", "real_telemetry", "plotjuggler", "mcap", "rqt_graph", "tf2_tree"]:
        modules = sorted([d for d in os.listdir(TASKS_DIR) if os.path.isdir(os.path.join(TASKS_DIR, d))])
        
        if not modules:
            print(f"[ERROR] No modules found in {TASKS_DIR}!")
            sys.exit(1)

        print("Select Module:")
        for i, m in enumerate(modules):
            print(f"  [{i+1}] {m}")
        
        module_choice = input(f"Enter choice [1-{len(modules)}] (default 1): ").strip() or "1"
        try:
            selected_module_name = modules[int(module_choice) - 1]
        except (ValueError, IndexError):
            selected_module_name = modules[0]

        selected_module_path = os.path.join(TASKS_DIR, selected_module_name)
        print(f"\n--- Operating on {selected_module_name} ---\n")

    # 3. Checkpoint Selection (Agent)
    # Search for any .pt files in logs folder recursively
    search_pattern = os.path.join(selected_module_path, "logs", "**", "*.pt")
    checkpoint_paths = glob.glob(search_pattern, recursive=True)
    
    # Also check a 'checkpoints' folder at the module root just in case
    checkpoint_paths += glob.glob(os.path.join(selected_module_path, "checkpoints", "*.pt"))
    
    # Filter to only show 'best_agent.pt'
    all_ckpts = [p for p in checkpoint_paths if os.path.basename(p) == "best_agent.pt"]
    all_ckpts.sort(reverse=True)
    selected_ckpt = None

    if action not in ["teleop", "teleop_keyboard", "teleop_joy", "mujoco_twin", "gazebo_twin", "rviz", "foxglove", "console", "test_joints", "real_telemetry", "plotjuggler", "mcap", "rqt_graph", "tf2_tree"]:
        print("\nSelect Trained Checkpoint (Agent):")
        if action == "train":
            print("  [0] Train from Scratch (None)")
        
        if not all_ckpts:
            print(f"  [X] No checkpoints found automatically in:")
            print(f"      - {os.path.join(selected_module_path, 'logs/')}")
            print(f"      - {os.path.join(selected_module_path, 'checkpoints/')}")
        
        for i, path in enumerate(all_ckpts):
            print(f"  [{i+1}] {ckpt_display_name(path)} ({os.path.basename(path)})")
        
        print("  [M] Enter Manual Path")
        
        default_val = "0" if action == "train" else ("1" if all_ckpts else "M")
        ckpt_choice = input(f"Enter choice [0-{len(all_ckpts)} or M] (default {default_val}): ").strip() or default_val
        
        if ckpt_choice.lower() == "m":
            selected_ckpt = input("Enter full path to .pt file: ").strip()
        elif ckpt_choice == "0" and action == "train":
            selected_ckpt = None
        else:
            try:
                selected_ckpt = all_ckpts[int(ckpt_choice) - 1]
            except (ValueError, IndexError):
                selected_ckpt = all_ckpts[0] if all_ckpts else None

    if selected_ckpt:
        ckpt_dir = os.path.dirname(selected_ckpt)
        other_ckpts = glob.glob(os.path.join(ckpt_dir, "*.pt"))
        if len(other_ckpts) > 1:
            print(f"\nSelect Specific Checkpoint from this Run:")
            
            def extract_step(filepath):
                base = os.path.basename(filepath)
                if base == "best_agent.pt": return float('inf')
                try:
                    return int(''.join(filter(str.isdigit, base)))
                except ValueError:
                    return -1
                    
            other_ckpts.sort(key=extract_step, reverse=True)
            
            for i, p in enumerate(other_ckpts):
                print(f"  [{i+1}] {os.path.basename(p)}")
                
            sub_choice = input(f"Enter choice [1-{len(other_ckpts)}] (default 1): ").strip() or "1"
            try:
                selected_ckpt = other_ckpts[int(sub_choice) - 1]
            except (ValueError, IndexError):
                selected_ckpt = other_ckpts[0]
                
        print(f"[Launcher] Selected agent: {selected_ckpt}")

    # 4. Environment & Options
    # Load default domain from config.yaml
    default_domain = "1"
    try:
        with open(CONFIG_PATH, 'r') as f:
            cfg_data = yaml.safe_load(f)
            default_domain = str(cfg_data.get("network", {}).get("ros_domain_id", "1"))
    except Exception:
        pass

    domain_id = default_domain
    if action not in ["train", "isaac_lab"]:
        domain_id = input(f"Enter ROS_DOMAIN_ID (default {default_domain}): ").strip() or default_domain
    robot_cfg = "UNITREE_GO2_CFG" # Default for now
    terrain_cfg = "flat"
    num_envs = 1
    headless = IS_ROBOT
    video = False
    teleop = False
    if action in ["mujoco", "gazebo", "real_deploy", "isaac_sim"]:
        teleop = True # Always active internally via /cmd_vel subscription
    run_name = ""
    use_estimator = False
    training_phase = ""

    if action in ["train", "isaac_lab", "isaac_sim", "mujoco"]:
        if action in ["isaac_sim", "mujoco"]:
            robot_choice = input("Select Robot [1: Go2, 2: Go1, 3: A1] (default 1): ").strip() or "1"
            robot_cfg = {"1": "UNITREE_GO2_CFG", "2": "UNITREE_GO1_CFG", "3": "UNITREE_A1_CFG"}.get(robot_choice, "UNITREE_GO2_CFG")
        elif action == "isaac_lab":
            robot_choice = input("Select Robot [1: Go2, 2: Go1, 3: A1, 4: All (Mixed)] (default 1): ").strip() or "1"
            robot_cfg = {"1": "UNITREE_GO2_CFG", "2": "UNITREE_GO1_CFG", "3": "UNITREE_A1_CFG", "4": "RANDOM"}.get(robot_choice, "UNITREE_GO2_CFG")
        else:
            robot_cfg = "RANDOM" # Will be overridden by YAML or fallback to default
        
        if action == "isaac_lab":
            terrain_choice = input("Select Terrain [1: flat, 2: rough] (default 1): ").strip() or "1"
            terrain_cfg = "rough" if terrain_choice == "2" else "flat"
        else:
            terrain_cfg = ""
        
        if action == "train":
            num_envs = input("Number of Envs (Enter to use YAML, or type number): ").strip() or ""
        else:
            num_envs = input("Number of Envs (default 1): ").strip() or "1"
        
        if not IS_ROBOT:
            headless = input("Headless Mode? [y/N]: ").lower().strip() == "y"
        
        if action == "train":
            # Dynamically extract phases from training_phases.yaml if available
            available_phases = ["phase1", "phase2", "phase3"] # fallback
            default_phase_idx = "3"
            
            # Search only in the source directory to avoid picking up backups in logs/
            source_dir = os.path.join(selected_module_path, "source")
            yaml_files = glob.glob(os.path.join(source_dir, "**", "training_phases.yaml"), recursive=True)
            yaml_data = {}
            if yaml_files:
                try:
                    with open(yaml_files[0], 'r') as f:
                        yaml_data = yaml.safe_load(f)
                        if "phases" in yaml_data:
                            extracted_phases = list(yaml_data["phases"].keys())
                            if extracted_phases:
                                available_phases = extracted_phases
                                default_phase_idx = str(len(available_phases))
                except Exception:
                    pass
                    
            phase_options_str = ", ".join([f"{i+1}: {p}" for i, p in enumerate(available_phases)])
            phase_choice = input(f"Select Training Phase [{phase_options_str}] (default {default_phase_idx}): ").strip() or default_phase_idx
            
            try:
                phase_idx = int(phase_choice) - 1
                training_phase = available_phases[phase_idx] if 0 <= phase_idx < len(available_phases) else f"phase{phase_choice}"
            except ValueError:
                training_phase = phase_choice
                
            # Curriculum sequence logic
            furthest_phase = training_phase
            if yaml_data and "phases" in yaml_data and training_phase in available_phases:
                def resolve_phase_launcher(all_phases, phase_name):
                    import collections.abc
                    import copy
                    
                    def deep_update(d, u):
                        for k, v in u.items():
                            if isinstance(v, collections.abc.Mapping):
                                d[k] = deep_update(d.get(k, {}), v)
                            else:
                                d[k] = v
                        return d

                    phase_node = all_phases.get("phases", {}).get(phase_name, {})
                    parent_name = phase_node.get("inherits", "default")
                    
                    if parent_name and parent_name != phase_name:
                        if parent_name == "default":
                            parent_cfg = all_phases.get("default", {})
                        else:
                            parent_cfg = resolve_phase_launcher(all_phases, parent_name)
                    else:
                        parent_cfg = all_phases.get("default", {})

                    return deep_update(copy.deepcopy(parent_cfg), phase_node)
                    
                start_idx = available_phases.index(training_phase)
                start_cfg = resolve_phase_launcher(yaml_data, training_phase)
                start_env = start_cfg.get("env", {})
                start_robot = start_env.get("robot_cfg", "")
                start_terrain = start_env.get("terrain", "rough")
                
                for k in available_phases[start_idx+1:]:
                    curr_cfg = resolve_phase_launcher(yaml_data, k)
                    curr_env = curr_cfg.get("env", {})
                    curr_robot = curr_env.get("robot_cfg", "")
                    curr_terrain = curr_env.get("terrain", "rough")
                    
                    if curr_robot == start_robot and curr_terrain == start_terrain:
                        furthest_phase = k
                    else:
                        break
                        
            if furthest_phase != training_phase:
                ans = input(f"Run as Curriculum Sequence ({training_phase} up to {furthest_phase})? [y/N]: ").lower().strip()
                if ans == "y":
                    training_phase = f"{training_phase}_to_{furthest_phase}"
                
            run_name = input("Enter Run Name (optional): ").strip()
            video = input("Record Video? [y/N]: ").lower().strip() == "y"
            
        if action == "isaac_lab":
            ans = input("Enable Manual Keyboard Control? [y/N]: ").lower().strip()
            if ans == "y":
                teleop = True

    if action in ["mujoco", "gazebo", "real_deploy"]:
        if not IS_ROBOT and action != "real_deploy":
            headless = input("Headless Mode? [y/N]: ").lower().strip() == "y"

    if action in ["mujoco", "gazebo", "isaac_sim"]:
        ans = input("Use State Estimator? [Y/n] (default Y): ").lower().strip()
        use_estimator = ans != "n"

    # Auto-record prompt if launching driver
    record_session = False
    if action in ["mujoco", "gazebo", "isaac_sim", "real_deploy"]:
        # First check config default
        cfg_auto = False
        try:
            with open(CONFIG_PATH, 'r') as f:
                cfg_data = yaml.safe_load(f)
                cfg_auto = cfg_data.get("logging", {}).get("auto_record", False)
        except Exception:
            pass
        
        default_prompt = "Y/n" if cfg_auto else "y/N"
        ans = input(f"Record this session to MCAP? [{default_prompt}]: ").lower().strip()
        if ans == "":
            record_session = cfg_auto
        else:
            record_session = (ans == "y")

    if action == "mcap":
        # MCAP Log & Replay sub-menu
        print("\n--- MCAP Telemetry Manager ---")
        print("  [WARNING] If you started MuJoCo, Gazebo, IsaacSim, or Deploy with auto-record enabled, it is already recording!")
        print("  [1] Record Topics (Ros2 Bag)")
        print("  [2] Replay (Ros2 Bag)")
        print("  [3] Replay (Custom Script)")
        mcap_choice = input("Enter choice [1-3] (default 3): ").strip() or "3"
        
        # Load record_dir from config
        record_dir = "Mcap/Recordings"
        try:
            with open(CONFIG_PATH, 'r') as f:
                cfg_data = yaml.safe_load(f)
                record_dir = cfg_data.get("logging", {}).get("record_dir", "Mcap/Recordings")
        except Exception:
            pass
            
        os.makedirs(record_dir, exist_ok=True)

        if mcap_choice == "1":
            action = "mcap_record"
            filename = input("Enter output directory name (optional, e.g. run1): ").strip()
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"run_{timestamp}"
            run_name = os.path.join(record_dir, filename)
        else:
            if mcap_choice == "2":
                action = "mcap_replay_rosbag"
            else:
                action = "mcap_replay_interactive"
                
            # List files and directories
            items = sorted(os.listdir(record_dir))
            files = []
            for item in items:
                full_path = os.path.join(record_dir, item)
                if item.endswith(".mcap") or os.path.isdir(full_path):
                    files.append(item)

            if not files:
                print(f"[Launcher] No recorded sessions found in '{record_dir}/'.")
                sys.exit(0)
            print("\nSelect Session to Play Back:")
            for i, f in enumerate(files):
                print(f"  [{i+1}] {f}")
            f_choice = input(f"Enter choice [1-{len(files)}] (default 1): ").strip() or "1"
            try:
                selected_file = files[int(f_choice) - 1]
            except (ValueError, IndexError):
                selected_file = files[0]
            run_name = os.path.join(record_dir, selected_file)

    return selected_module_name, selected_module_path, action, robot_cfg, terrain_cfg, num_envs, selected_ckpt, teleop, headless, video, run_name, domain_id, use_estimator, record_session, training_phase

def main():
    (
        module_name,
        module_path,
        action,
        robot_cfg,
        terrain_cfg,
        num_envs,
        ckpt,
        teleop,
        headless,
        video,
        run_name,
        domain_id,
        use_estimator,
        record_session,
        training_phase,
    ) = run_cli_menu()

    # Save for next time
    save_last_command({
        "module_name": module_name,
        "module_path": module_path,
        "action": action,
        "robot_cfg": robot_cfg,
        "terrain_cfg": terrain_cfg,
        "num_envs": num_envs,
        "ckpt": ckpt,
        "teleop": teleop,
        "headless": headless,
        "video": video,
        "run_name": run_name,
        "domain_id": domain_id,
        "use_estimator": use_estimator,
        "record_session": record_session,
        "training_phase": training_phase,
    })

    print("\n" + "=" * 50)
    print(f"Launching {action.upper()} Mode for {module_name}!")
    print(f"Robot:    {robot_cfg}")
    if action == "isaac_lab":
        print(f"Terrain:  {terrain_cfg}")
    print(f"Domain ID: {domain_id}")
    if training_phase:
        print(f"Phase:    {training_phase}")
    if ckpt:
        print(f"Checkpoint: {ckpt_display_name(ckpt)}")
    print(f"Teleop:   {teleop}")
    print("=" * 50 + "\n")

    # Set up environment variables
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(domain_id)
    env["QUADRUPED_ROBOT_CFG"] = robot_cfg
    env["QUADRUPED_ROBOT"] = robot_cfg
    if teleop:
        env["QUADRUPED_TELEOP"] = "1"
    if training_phase:
        env["QUADRUPED_TRAINING_PHASE"] = training_phase
    
    # Search for OBS_DIM in the same folder as the checkpoint
    if ckpt:
        ckpt_dir = os.path.dirname(ckpt)
        params_dir = os.path.abspath(os.path.join(ckpt_dir, "..", "params"))
        agent_cfg = os.path.join(params_dir, "agent.yaml")
        env_cfg_path = os.path.join(params_dir, "env.yaml")
        obs_dim = 0
        
        # Try agent.yaml first
        if os.path.exists(agent_cfg):
            try:
                with open(agent_cfg, 'r') as f:
                    data = yaml.load(f, Loader=yaml.UnsafeLoader)
                    obs_dim = data.get("models", {}).get("policy", {}).get("input_shape", [0])[0]
            except Exception:
                pass
                
        # Try env.yaml if not found in agent.yaml
        if not obs_dim and os.path.exists(env_cfg_path):
            try:
                with open(env_cfg_path, 'r') as f:
                    data = yaml.load(f, Loader=yaml.UnsafeLoader)
                    obs_dim = data.get("observation_space", 0)
            except Exception:
                pass
                
        if obs_dim:
            env["QUADRUPED_OBS_DIM"] = str(obs_dim)
            print(f"[INFO] Detected observation dimension from checkpoint: {obs_dim}")

    # Prepare environment
    if terrain_cfg:
        env["QUADRUPED_TERRAIN"] = terrain_cfg

    # Inject the task module's source directory into PYTHONPATH so that Quadruped.tasks can be imported
    source_path = os.path.abspath(os.path.join(module_path, "source", "Quadruped"))
    if os.path.exists(source_path):
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{source_path}:{current_pythonpath}" if current_pythonpath else source_path

    # Final Command Assembly
    abs_ckpt = os.path.abspath(ckpt) if ckpt else ""
    obs_dim = env.get("QUADRUPED_OBS_DIM", "49")

    def get_robot_key(cfg):
        return {
            "UNITREE_A1_CFG": "a1",
            "UNITREE_GO1_CFG": "go1",
            "UNITREE_GO2_CFG": "go2",
        }.get(cfg or "", "go2")

    robot_key = get_robot_key(robot_cfg)

    if action == "train":
        script_path = os.path.join("scripts", "skrl", "train.py")
        cmd = [sys.executable, script_path, "--task=Template-Quadruped-Direct-v0"]
        if num_envs:
            cmd.append(f"--num_envs={num_envs}")
        if ckpt:
            cmd.append(f"--checkpoint={abs_ckpt}")
        if headless:
            cmd.append("--headless")
        if video:
            cmd.append("--video")
            cmd.append("--video_length=200")
            cmd.append("--video_interval=5000")
        subprocess.run(cmd, env=env, cwd=module_path)

    elif action == "isaac_lab":
        script_path = os.path.join("scripts", "skrl", "play.py")
        cmd = [sys.executable, script_path, "--task=Template-Quadruped-Direct-v0"]
        if num_envs:
            cmd.append(f"--num_envs={num_envs}")
        if ckpt:
            cmd.append(f"--checkpoint={abs_ckpt}")
        if headless:
            cmd.append("--headless")
        subprocess.run(cmd, env=env, cwd=module_path)

    elif action in ("mujoco", "gazebo", "isaac_sim", "real_deploy", "real_telemetry", "mujoco_twin", "gazebo_twin", "rviz", "foxglove", "console", "teleop_keyboard", "teleop_joy", "test_joints", "plotjuggler", "mcap_record", "mcap_replay_rosbag", "mcap_replay_interactive", "rqt_graph", "tf2_tree"):
        # Unified Driver Pipeline
        isaac_python = os.path.expanduser("~/env_isaacsim/bin/python")
        sys_python = sys.executable 


        if action == "isaac_sim":
            bridge_script = os.path.abspath(os.path.join("IsaacSim", "isaac_driver.py"))
            cmd = [
                isaac_python,
                bridge_script,
                f"--robot={robot_key}",
                f"--internal_policy={abs_ckpt}",
                f"--obs_dim={obs_dim}",
            ]
            if use_estimator:
                cmd.append("--use_estimator")
        elif action == "mujoco":
            bridge_script = os.path.abspath(os.path.join("Mujoco", "mujoco_driver.py"))
            cmd = [
                sys_python,
                bridge_script,
                f"--robot={robot_key}",
                f"--internal_policy={abs_ckpt}",
                f"--obs_dim={obs_dim}",
            ]
            # Automatically enable headless in Docker or if headless flag is set
            if headless:
                cmd.append("--headless")
            if use_estimator:
                cmd.append("--use_estimator")
        elif action == "mujoco_twin":
            bridge_script = os.path.abspath(os.path.join("Operator", "mujoco_twin.py"))
            cmd = [
                sys_python,
                bridge_script,
                f"--robot={robot_key}",
            ]
            if use_estimator:
                cmd.append("--use_estimator")
        elif action == "gazebo_twin":
            bridge_script = os.path.abspath(os.path.join("Operator", "gazebo_twin.py"))
            cmd = [
                sys_python,
                bridge_script,
                f"--robot={robot_key}",
            ]
            if use_estimator:
                cmd.append("--use_estimator")
        elif action == "rviz":
            cmd = ["ros2", "run", "rviz2", "rviz2"]
        elif action == "console":
            bridge_script = os.path.abspath(os.path.join("Operator", "console.py"))
            cmd = [
                sys_python,
                bridge_script,
                f"--robot={robot_key}",
            ]
            if use_estimator:
                cmd.append("--use_estimator")
        elif action == "gazebo":
            bridge_script = os.path.abspath(os.path.join("Gazebo", "gazebo_driver.py"))
            cmd = [
                sys_python,
                bridge_script,
                f"--robot={robot_key}",
                f"--internal_policy={abs_ckpt}",
                f"--obs_dim={obs_dim}",
            ]
            if use_estimator:
                cmd.append("--use_estimator")
        elif action in ["real_deploy", "real_telemetry"]:
            bridge_script = os.path.abspath(os.path.join("Unitree", "real_driver.py"))
            cmd = [
                sys_python,
                bridge_script,
                f"--robot={robot_key}",
                f"--obs_dim={obs_dim}",
            ]
            if action == "real_deploy" and abs_ckpt:
                cmd.append(f"--internal_policy={abs_ckpt}")

        elif action == "teleop_keyboard":
            cmd = ["ros2", "run", "teleop_twist_keyboard", "teleop_twist_keyboard"]
            # No robot_key or ckpt needed for this
            
        elif action == "teleop_joy":
            cmd = ["ros2", "launch", "teleop_twist_joy", "teleop-launch.py"]

        elif action == "test_joints":
            bridge_script = os.path.abspath(os.path.join("Unitree", "test_joints.py"))
            cmd = [sys_python, bridge_script]
        
        elif action == "plotjuggler":
            cmd = ["ros2", "run", "plotjuggler", "plotjuggler"]
            
        elif action == "foxglove":
            cmd = ["ros2", "launch", "foxglove_bridge", "foxglove_bridge_launch.xml"]

        elif action == "rqt_graph":
            cmd = ["ros2", "run", "rqt_graph", "rqt_graph"]

        elif action == "tf2_tree":
            cmd = ["ros2", "run", "rqt_tf_tree", "rqt_tf_tree"]

        elif action == "mcap_record":
            cmd = [
                "ros2", "bag", "record",
                "-a",
                "-s", "mcap",
                "-o", os.path.abspath(run_name)
            ]
        elif action == "mcap_replay_rosbag":
            cmd = [
                "ros2", "bag", "play",
                os.path.abspath(run_name)
            ]
        elif action == "mcap_replay_interactive":
            bridge_script = os.path.abspath(os.path.join("Mcap", "mcap_tool.py"))
            cmd = [
                sys_python,
                bridge_script,
                "--replay",
                os.path.abspath(run_name)
            ]

        # Check if we should auto-record this session
        record_proc = None
        if record_session:
            record_dir = "Mcap/Recordings"
            try:
                with open(CONFIG_PATH, 'r') as f:
                    cfg_data = yaml.safe_load(f)
                    record_dir = cfg_data.get("logging", {}).get("record_dir", "Mcap/Recordings")
            except Exception:
                pass
            os.makedirs(record_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mcap_filename = os.path.join(record_dir, f"run_{action}_{timestamp}")
            abs_mcap = os.path.abspath(mcap_filename)
            
            print(f"\n[Launcher] Auto-recording enabled. Saving to: {mcap_filename}")
            
            record_cmd = [
                "ros2", "bag", "record",
                "-a",
                "-s", "mcap",
                "-o", abs_mcap
            ]
            # Start recorder in background
            record_proc = subprocess.Popen(record_cmd, env=env)
            time.sleep(0.5)

        try:
            subprocess.run(cmd, env=env)
        except KeyboardInterrupt:
            pass
        finally:
            if record_proc is not None:
                print("\n[Launcher] Stopping background MCAP recorder...")
                record_proc.terminate()
                try:
                    record_proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    record_proc.kill()
                print("[Launcher] Background recorder terminated cleanly.")

if __name__ == "__main__":
    main()
