import os
import sys
import glob
import subprocess
import json
import platform
import yaml

TASKS_DIR = "IsaacLab_Tasks"
LAST_COMMAND_FILE = ".launcher_last_command.json"

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

    # 0. Load last command
    last_cmd = load_last_command()
    if last_cmd:
        print(f"Last command: {last_cmd.get('action')} on {last_cmd.get('module_name')}")
        quick_start = input("Press Enter to repeat last command or any key to select new: ").strip()
        if not quick_start:
            return (
                last_cmd["module_name"],
                last_cmd["module_path"],
                last_cmd["action"],
                last_cmd["robot_cfg"],
                last_cmd["terrain_cfg"],
                last_cmd["num_envs"],
                last_cmd["ckpt"],
                last_cmd["teleop"],
                last_cmd["headless"],
                last_cmd.get("video", False),
                last_cmd.get("run_name", ""),
            )

    # 1. Action Selection
    print("Select Action:")
    if not IS_DOCKER:
        print("  [1] Train Policy")
        print("  [2] Play Policy (IsaacLab)")
        print("  [3] Play Policy (IsaacSim Bridge)")
    else:
        print("  [X] Train Policy (DISABLED IN DOCKER)")
        print("  [X] Play Policy (IsaacLab) (DISABLED IN DOCKER)")
        print("  [X] Play Policy (IsaacSim Bridge) (DISABLED IN DOCKER)")
        
    print("  [4] Play MuJoCo")
    print("  [5] Play Gazebo")
    print("  [6] Deploy to Robot")
    print("  [7] Remote Teleop")
    print("  [8] Play MuJoCo Digital Twin")

    action_map = {
        "1": "train",
        "2": "isaac_lab",
        "3": "isaac_sim",
        "4": "mujoco",
        "5": "gazebo",
        "6": "real_deploy",
        "7": "teleop",
        "8": "mujoco_twin",
    }
    
    choice = input("Enter choice [1-8] (default 4): ").strip() or "4"
    action = action_map.get(choice, "mujoco")
    
    if IS_DOCKER and choice in ["1", "2", "3"]:
        print("\n[ERROR] Training/IsaacSim actions are not available in Docker. Switching to MuJoCo.")
        action = "mujoco"

    print(f"\n--- Selected Action: {action.upper()} ---\n")

    # 2. Module Selection
    selected_module_name = "None"
    selected_module_path = "."
    
    if action not in ["mujoco_twin", "teleop"]:
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
    
    # Filter to prioritize 'best_agent.pt' but keep others
    best_agents = [p for p in checkpoint_paths if os.path.basename(p) == "best_agent.pt"]
    other_agents = [p for p in checkpoint_paths if os.path.basename(p) != "best_agent.pt"]
    
    # Sort and prioritize best_agents
    best_agents.sort(reverse=True)
    other_agents.sort(reverse=True)
    
    all_ckpts = best_agents + other_agents
    selected_ckpt = None

    if action != "teleop" and action != "mujoco_twin":
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
        print(f"[Launcher] Selected agent: {selected_ckpt}")

    # 4. Environment & Options
    # Load default domain from config.yaml
    default_domain = "1"
    try:
        with open("configs/config.yaml", 'r') as f:
            cfg_data = yaml.safe_load(f)
            default_domain = str(cfg_data.get("network", {}).get("ros_domain_id", "1"))
    except Exception:
        pass

    domain_id = input(f"Enter ROS_DOMAIN_ID (default {default_domain}): ").strip() or default_domain
    robot_cfg = "UNITREE_GO2_CFG" # Default for now
    terrain_cfg = "flat"
    num_envs = 1
    headless = IS_DOCKER
    video = False
    teleop = False
    run_name = ""

    if action in ["train", "isaac_lab", "isaac_sim"]:
        robot_choice = input("Select Robot [1: Go2, 2: Go1, 3: A1] (default 1): ").strip() or "1"
        robot_cfg = {"1": "UNITREE_GO2_CFG", "2": "UNITREE_GO1_CFG", "3": "UNITREE_A1_CFG"}.get(robot_choice, "UNITREE_GO2_CFG")
        
        terrain_choice = input("Select Terrain [1: flat, 2: rough] (default 1): ").strip() or "1"
        terrain_cfg = "rough" if terrain_choice == "2" else "flat"
        
        num_envs = input("Number of Envs (default 1): ").strip() or "1"
        
        if not IS_DOCKER:
            headless = input("Headless Mode? [y/N]: ").lower().strip() == "y"
        
        if action == "train":
            run_name = input("Enter Run Name (optional): ").strip()
            video = input("Record Video? [y/N]: ").lower().strip() == "y"

    if action in ["mujoco", "gazebo", "real_deploy"]:
        teleop = input("Enable Remote Teleop? [y/N]: ").lower().strip() == "y"

    return selected_module_name, selected_module_path, action, robot_cfg, terrain_cfg, num_envs, selected_ckpt, teleop, headless, video, run_name, domain_id

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
        "domain_id": domain_id
    })

    print("\n" + "=" * 50)
    print(f"Launching {action.upper()} Mode for {module_name}!")
    print(f"Robot:    {robot_cfg}")
    print(f"Terrain:  {terrain_cfg}")
    print(f"Domain ID: {domain_id}")
    if ckpt:
        print(f"Checkpoint: {ckpt_display_name(ckpt)}")
    print(f"Teleop:   {teleop}")
    print("=" * 50 + "\n")

    # Set up environment variables
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(domain_id)
    env["QUADRUPED_ROBOT_CFG"] = robot_cfg
    
    # Search for OBS_DIM in the same folder as the checkpoint
    if ckpt:
        ckpt_dir = os.path.dirname(ckpt)
        params_dir = os.path.abspath(os.path.join(ckpt_dir, "..", "params"))
        agent_cfg = os.path.join(params_dir, "agent.yaml")
        if os.path.exists(agent_cfg):
            try:
                with open(agent_cfg, 'r') as f:
                    data = yaml.safe_load(f)
                    # Support for different skrl/rl_games config structures
                    obs_dim = data.get("models", {}).get("policy", {}).get("input_shape", [0])[0]
                    if obs_dim:
                        env["QUADRUPED_OBS_DIM"] = str(obs_dim)
                        print(f"[INFO] Detected observation dimension from checkpoint: {obs_dim}")
            except Exception:
                pass

    # Prepare environment
    env["QUADRUPED_TERRAIN"] = terrain_cfg

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

    elif action in ("mujoco", "gazebo", "isaac_sim", "real_deploy", "mujoco_twin"):
        # Unified Driver Pipeline
        isaac_python = "/home/05680435969@env_isaacsim/bin/python"
        # Use the current Python interpreter to ensure we pick up the correct virtualenv/environment
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
            if IS_DOCKER or headless:
                cmd.append("--headless")
        elif action == "mujoco_twin":
            bridge_script = os.path.abspath(os.path.join("DigitalTwin", "supervisor.py"))
            cmd = [
                sys_python,
                bridge_script,
                f"--robot={robot_key}",
                f"--internal_policy={abs_ckpt}",
                f"--obs_dim={obs_dim}",
            ]
        elif action == "gazebo":
            bridge_script = os.path.abspath(os.path.join("Gazebo", "gazebo_driver.py"))
            cmd = [
                sys_python,
                bridge_script,
                f"--robot={robot_key}",
                f"--internal_policy={abs_ckpt}",
                f"--obs_dim={obs_dim}",
            ]
        elif action == "real_deploy":
            bridge_script = os.path.abspath(os.path.join("Unitree", "unitree_driver.py"))
            cmd = [
                sys_python,
                bridge_script,
                f"--robot={robot_key}",
                f"--internal_policy={abs_ckpt}",
                f"--obs_dim={obs_dim}",
            ]

        elif action == "teleop":
            cmd = ["ros2", "run", "teleop_twist_keyboard", "teleop_twist_keyboard"]
            # No robot_key or ckpt needed for this

        if teleop and action != "teleop":
            cmd.append("--teleop")

        subprocess.run(cmd, env=env)

if __name__ == "__main__":
    main()
