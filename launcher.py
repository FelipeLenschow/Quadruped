import os
import sys
import glob
import subprocess
import json


TASKS_DIR = "IsaacLab_Tasks"
LAST_COMMAND_FILE = ".launcher_last_command.json"


def ckpt_display_name(path):
    """Extract the run folder name from a full checkpoint path.
    e.g. '.../quadruped_direct/2026-04-24_10-30-00/checkpoints/best_agent.pt'
         → '2026-04-24_10-30-00'
    """
    parts = path.replace("\\", "/").split("/")
    # Walk backwards to find the folder right before 'checkpoints'
    for i, p in enumerate(parts):
        if p == "checkpoints" and i > 0:
            return parts[i - 1]
    # Fallback: last two parent dirs
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
    is_docker = os.path.exists("/.dockerenv")
    is_isaac = "env_isaacsim" in os.environ.get("VIRTUAL_ENV", "") or "env_isaacsim" in sys.executable

    print("\n" + "=" * 50)
    print(" Quadruped Unified Launcher")
    print("=" * 50)
    if is_docker:
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

    # 1. Selection: Dynamic Module Detection
    if not os.path.exists(TASKS_DIR):
        print(f"[ERROR] Tasks directory '{TASKS_DIR}' not found.")
        sys.exit(1)

    available_modules = [
        d
        for d in os.listdir(TASKS_DIR)
        if os.path.isdir(os.path.join(TASKS_DIR, d)) and not d.startswith(".")
    ]
    available_modules.sort()

    if not available_modules:
        print(f"[ERROR] No task modules found in {TASKS_DIR}/")
        sys.exit(1)

    print("Select Module:")
    for i, m in enumerate(available_modules):
        print(f"  [{i+1}] {m}")

    mod_idx = input(f"Enter choice [1-{len(available_modules)}] (default 1): ").strip()
    try:
        idx = int(mod_idx) - 1
        if not (0 <= idx < len(available_modules)):
            idx = 0
    except ValueError:
        idx = 0

    selected_module_name = available_modules[idx]
    selected_module_path = os.path.join(TASKS_DIR, selected_module_name)

    print(f"\n--- Operating on {selected_module_name} ---\n")

    # 1. Action
    print("Select Action:")
    if not is_docker:
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
    
    tp = input(
        "Enter choice [1-7] (default 4): " if is_docker else "Enter choice [1-7] (default 2): "
    ).strip()
    
    if tp == "1" and not is_docker:
        action = "train"
    elif tp == "2" and not is_docker:
        action = "isaac_lab"
    elif tp == "3" and not is_docker:
        action = "isaac_sim"
    elif tp == "5":
        action = "gazebo"
    elif tp == "6":
        action = "real_deploy"
    elif tp == "7":
        action = "teleop"
    else:
        # Default dynamically based on environment if invalid/empty/disabled option chosen
        action = "mujoco" if is_docker else "isaac_lab"

    # 2. Robot selection
    selected_robot_cfg = "UNITREE_GO2_CFG"  # Global default
    if action == "train":
        ROBOT_CHOICES = {
            "1": ("Unitree A1", "UNITREE_A1_CFG"),
            "2": ("Unitree Go1", "UNITREE_GO1_CFG"),
            "3": ("Unitree Go2", "UNITREE_GO2_CFG"),
        }
        print("\nSelect Robot Configuration for Training:")
        for key, (name, _) in ROBOT_CHOICES.items():
            print(f"  [{key}] {name}")
        rob_idx = input("Enter choice [1-3] (default 3 for Go2): ").strip()
        if not rob_idx or rob_idx not in ROBOT_CHOICES:
            rob_idx = "3"
        _, selected_robot_cfg = ROBOT_CHOICES[rob_idx]
    elif action == "isaac_lab":
        # Isaac Play mode (Sim2Sim) currently spawns all three for comparison
        selected_robot_cfg = None
    else:
        # MuJoCo/Gazebo/Real deployments now default to Go2 as per user requested standardization
        selected_robot_cfg = "UNITREE_GO2_CFG"

    # 3. Terrain
    selected_terrain = "flat"
    if action == "isaac_lab" or action == "train":
        TERRAIN_CHOICES = {
            "1": ("Flat Plane", "flat"),
            "2": ("Random Rough", "rough"),
            "3": ("All Terrains (Stairs, Slopes, Boxes)", "all"),
        }
        print("\nSelect Terrain:")
        for key, (name, _) in TERRAIN_CHOICES.items():
            print(f"  [{key}] {name}")
        ter_idx = input("Enter choice [1-3] (default 1): ").strip()
        if not ter_idx or ter_idx not in TERRAIN_CHOICES:
            ter_idx = "1"
        _, selected_terrain = TERRAIN_CHOICES[ter_idx]

    # 4. Num Envs
    num_envs = ""
    if action in ("train", "isaac_lab"):
        num_envs = input(
            "\nEnter number of environments (leave blank to use config default): "
        ).strip()
    elif action == "isaac_lab":
        num_envs = input(
            "\nEnter number of environments [1-50] (default 1): "
        ).strip()
        if not num_envs:
            num_envs = "1"
    else:
        num_envs = "1"

    selected_ckpt = None
    teleop = False
    headless = False
    video = False
    run_name = ""

    # 5. Checkpoint Selection
    checkpoint_paths = glob.glob(
        os.path.join(
            selected_module_path,
            "logs",
            "skrl",
            "quadruped_direct",
            "*",
            "checkpoints",
            "best_agent.pt",
        )
    )
    checkpoint_paths.sort(reverse=True)

    needs_ckpt = action in ("isaac_lab", "isaac_sim", "mujoco", "gazebo", "real_deploy")

    if needs_ckpt and not checkpoint_paths:
        print(
            f"\n[ERROR] No best_agent.pt checkpoints found in {selected_module_path}/logs/skrl/quadruped_direct/"
        )
        sys.exit(1)

    if checkpoint_paths:
        if action == "train":
            print("\nSelect Checkpoint to Resume Training:")
            print("  [0] Train from Scratch (None)")
            for i, path in enumerate(checkpoint_paths):
                print(f"  [{i+1}] {ckpt_display_name(path)}")
            ckpt_idx = input(
                f"Enter choice [0-{len(checkpoint_paths)}] (default 0): "
            ).strip()
            try:
                val = int(ckpt_idx)
                if val == 0:
                    selected_ckpt = None
                else:
                    idx = (val - 1) if 1 <= val <= len(checkpoint_paths) else 0
                    selected_ckpt = checkpoint_paths[idx]
            except ValueError:
                selected_ckpt = None
        else:
            # Deployment/Sim2Sim Mode: User wants to pick the checkpoint
            print("\nSelect Trained Checkpoint:")
            for i, path in enumerate(checkpoint_paths):
                print(f"  [{i+1}] {ckpt_display_name(path)}")
            ckpt_idx = input(
                f"Enter choice [1-{len(checkpoint_paths)}] (default 1): "
            ).strip()
            try:
                val = int(ckpt_idx) - 1
                idx = val if 0 <= val < len(checkpoint_paths) else 0
            except ValueError:
                idx = 0
            selected_ckpt = checkpoint_paths[idx]
            print(f"[Launcher] Selected checkpoint: {ckpt_display_name(selected_ckpt)}")

    if action in ("isaac_sim", "mujoco", "gazebo", "real_deploy"):
        teleop = False  # Using ROS 2 teleop instead of internal WASD
    elif action == "isaac_lab":
        t_input = input("\nEnable WASD Teleoperation? [Y/n]: ").strip().lower()
        teleop = t_input != "n"

    if action in ("isaac_lab", "isaac_sim", "mujoco", "gazebo"):
        headless = False
    elif action == "train":
        h_input = input("\nEnable Headless Mode? [Y/n]: ").strip().lower()
        headless = h_input != "n"

    run_name = ""
    if action == "train":
        v_input = input("\nEnable Video Recording? [Y/n]: ").strip().lower()
        video = v_input != "n"

    return (
        selected_module_name,
        selected_module_path,
        action,
        selected_robot_cfg,
        selected_terrain,
        num_envs,
        selected_ckpt,
        teleop,
        headless,
        video,
        run_name,
    )


if __name__ == "__main__":
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
    ) = run_cli_menu()

    print(f"\n{'='*50}")
    print(f"Launching {action.upper()} Mode for {module_name}!")
    if robot_cfg:
        print(f"Robot:    {robot_cfg}")
    else:
        print(f"Robots:   A1 + Go1 + Go2 (Sim2Sim)")
    print(f"Terrain:  {terrain_cfg}")
    print(f"Envs:     {num_envs if num_envs else 'Config Default'}")
    if action in ("isaac_lab", "isaac_sim", "mujoco", "gazebo"):
        print(f"Checkpoint: {ckpt_display_name(ckpt) if ckpt else 'None'}")
        print(f"Teleop:   {teleop}")
    else:
        print(f"Headless: {headless}")
        if action == "train":
            print(f"Video:    {video}")
            if run_name:
                print(f"Run Name: {run_name}")
    print(f"{'='*50}\n")

    # Save last command
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
    })

    # Prepare environment
    env = os.environ.copy()
    env["QUADRUPED_TELEOP"] = "1" if teleop else "0"
    if robot_cfg:
        env["QUADRUPED_ROBOT"] = robot_cfg

    # Ensure the correct source directory is in PYTHONPATH so 'import Quadruped.tasks' works
    source_dir = os.path.abspath(os.path.join(module_path, "source", "Quadruped"))
    if os.path.exists(source_dir):
        env["PYTHONPATH"] = source_dir + os.pathsep + env.get("PYTHONPATH", "")

    # Environment Sanitization for ROS 2 Bridges (MuJoCo/Gazebo/Isaac)
    # This prevents pollution from Isaac Sim's site-packages (Python 3.11)
    # when we want to use the system ROS 2 (Python 3.10)
    if action in ("mujoco", "gazebo", "isaac_sim"):
        if action == "isaac_sim":
            # Use Isaac Sim's internal ROS 2 libraries (compiled for Python 3.11)
            isaac_ros_path = "/home/05680435969@corp.udesc.br/env_isaacsim/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/rclpy"
            if os.path.exists(isaac_ros_path):
                env["PYTHONPATH"] = (
                    isaac_ros_path + os.pathsep + env.get("PYTHONPATH", "")
                )
                # Also add the lib folder for shared objects
                env["LD_LIBRARY_PATH"] = (
                    os.path.join(os.path.dirname(isaac_ros_path), "lib")
                    + os.pathsep
                    + env.get("LD_LIBRARY_PATH", "")
                )

            # Add IsaacLab source paths
            isaaclab_path = "/home/05680435969@corp.udesc.br/IsaacLab"
            isaaclab_sources = [
                os.path.join(isaaclab_path, "source", "isaaclab"),
                os.path.join(isaaclab_path, "source", "isaaclab_assets"),
                os.path.join(isaaclab_path, "source", "isaaclab_tasks"),
                os.path.join(isaaclab_path, "source", "isaaclab_rl"),
            ]
            for src in isaaclab_sources:
                if os.path.exists(src):
                    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        else:
            # Ensure ROS 2 standard paths are present (Humble uses dist-packages)
            ros_python_path = "/opt/ros/humble/local/lib/python3.10/dist-packages"
            if ros_python_path not in env.get("PYTHONPATH", ""):
                env["PYTHONPATH"] = (
                    ros_python_path + os.pathsep + env.get("PYTHONPATH", "")
                )

            # Add fallback just in case
            fallback_path = "/opt/ros/humble/lib/python3.10/site-packages"
            if fallback_path not in env.get("PYTHONPATH", ""):
                env["PYTHONPATH"] = (
                    env.get("PYTHONPATH", "") + os.pathsep + fallback_path
                )

            # Ensure ROS 2 shared libraries are in LD_LIBRARY_PATH
            ros_lib_path = "/opt/ros/humble/lib:/opt/ros/humble/local/lib"
            if "/opt/ros/humble/lib" not in env.get("LD_LIBRARY_PATH", ""):
                env["LD_LIBRARY_PATH"] = (
                    ros_lib_path + os.pathsep + env.get("LD_LIBRARY_PATH", "")
                )

        # Unset virtualenv variables that might confuse the system python
        if action != "isaac_sim":  # Keep env for Isaac
            env.pop("VIRTUAL_ENV", None)
            env.pop("PYTHONHOME", None)

        # VDI Fixes for Gazebo/ROS 2
        env["FORCE_SOFTWARE_RENDER"] = "1"
        env["GZ_PARTITION"] = "quadruped_sim"
        # GZ_HEADLESS=1 can be used to disable GUI for performance
        # env["GZ_HEADLESS"] = "1"

    # Dynamic observation space detection
    if ckpt and os.path.exists(ckpt):
        try:
            import torch

            data = torch.load(ckpt, map_location="cpu")
            # skrl stores weights in 'policy.net.0.weight' or 'policy.net_container.0.weight'
            # We check the input dimension (last element of shape)
            policy_state = data.get("policy", {})
            obs_dim = 236  # Default
            for k, v in policy_state.items():
                if "net" in k and hasattr(v, "shape") and len(v.shape) == 2:
                    obs_dim = v.shape[1]
                    break
            print(f"[INFO] Detected observation dimension from checkpoint: {obs_dim}")
            env["QUADRUPED_OBS_DIM"] = str(obs_dim)
        except Exception as e:
            print(f"[WARNING] Could not inspect checkpoint dimensions: {e}")

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

    elif action in ("mujoco", "gazebo", "isaac_sim", "real_deploy"):
        # Unified Driver Pipeline
        isaac_python = "/home/05680435969@corp.udesc.br/env_isaacsim/bin/python"
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
            bridge_script = os.path.abspath(os.path.join("Unitree", "real_driver.py"))
            cmd = [
                sys_python,
                bridge_script,
                f"--robot={robot_key}",
                f"--internal_policy={abs_ckpt}",
                f"--obs_dim={obs_dim}",
            ]

        print(f"[Launcher] Starting Driver: {' '.join(cmd)}")
        bridge_env = env.copy()
        if action != "isaac_sim":
            bridge_env.pop("VIRTUAL_ENV", None)
            bridge_env.pop("PYTHONHOME", None)

        proc = subprocess.Popen(cmd, env=bridge_env, cwd=module_path)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        sys.exit(0)

    elif action == "teleop":
        print(f"\n{'='*50}")
        print(f"Launching REMOTE TELEOP!")
        print(f"Controls: I/K=fwd/back, J/L=left/right, U/O=turn")
        print(f"{'='*50}\n")
        cmd = ["ros2", "run", "teleop_twist_keyboard", "teleop_twist_keyboard"]
        subprocess.run(cmd, env=env, cwd=module_path)

    else:  # isaac play
        script_path = os.path.join("scripts", "skrl", "play.py")
        cmd = [sys.executable, script_path, "--task=Template-Quadruped-Direct-v0"]
        if num_envs:
            cmd.append(f"--num_envs={num_envs}")
        if ckpt:
            cmd.append(f"--checkpoint={abs_ckpt}")
        subprocess.run(cmd, env=env, cwd=module_path)
