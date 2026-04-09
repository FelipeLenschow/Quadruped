import os
import sys
import glob
import subprocess


TASKS_DIR = "IsaacLab_Tasks"


def run_cli_menu():
    print("\n" + "=" * 50)
    print(" Quadruped Unified Launcher")
    print("=" * 50 + "\n")

    # 0. Selection: Dynamic Module Detection
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
    tp = input(
        "Select Action:\n  [1] Train\n"
        "  [2] Play IsaacSim\n"
        "  [3] Play MuJoCo\n"
        "Enter choice [1-3] (default 2): "
    ).strip()
    if tp == "1":
        action = "train"
    elif tp == "3":
        action = "mujoco"
    else:
        action = "isaac"

    # 2. Robot selection
    selected_robot_cfg = None
    if action != "isaac":
        ROBOT_CHOICES = {
            "1": ("All (A1, Go1, and Go2)", "RANDOM"),
            "2": ("Unitree A1", "UNITREE_A1_CFG"),
            "3": ("Unitree Go1", "UNITREE_GO1_CFG"),
            "4": ("Unitree Go2", "UNITREE_GO2_CFG"),
        }
        print("\nSelect Robot Configuration:")
        for key, (name, _) in ROBOT_CHOICES.items():
            print(f"  [{key}] {name}")
        rob_idx = input("Enter choice [1-4] (default 1): ").strip()
        if not rob_idx or rob_idx not in ROBOT_CHOICES:
            rob_idx = "1"
        _, selected_robot_cfg = ROBOT_CHOICES[rob_idx]

    # 3. Terrain
    selected_terrain = "flat"
    if action == "isaac" or action == "train":
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
    num_envs = input(
        "\nEnter number of environments (leave blank to use config default): "
    ).strip()

    selected_ckpt = None
    teleop = False
    headless = False

    # 5. Checkpoint Selection
    checkpoint_paths = glob.glob(
        os.path.join(
            selected_module_path,
            "logs",
            "skrl",
            "quadruped_direct",
            "cp*",
            "checkpoints",
            "best_agent.pt",
        )
    )
    checkpoint_paths.sort(reverse=True)

    needs_ckpt = action in ("isaac", "mujoco")

    if needs_ckpt and not checkpoint_paths:
        print(
            f"\n[ERROR] No best_agent.pt checkpoints found in {selected_module_path}/logs/skrl/quadruped_direct/"
        )
        sys.exit(1)

    if checkpoint_paths:
        if needs_ckpt:
            print("\nSelect Trained Checkpoint:")
            for i, path in enumerate(checkpoint_paths):
                print(f"  [{i+1}] {path}")
            ckpt_idx = input(
                f"Enter choice [1-{len(checkpoint_paths)}] (default 1): "
            ).strip()
            try:
                val = int(ckpt_idx) - 1
                idx = val if 0 <= val < len(checkpoint_paths) else 0
            except ValueError:
                idx = 0
            selected_ckpt = checkpoint_paths[idx]
        else:  # train
            print("\nSelect Checkpoint to Resume Training:")
            print("  [0] Train from Scratch (None)")
            for i, path in enumerate(checkpoint_paths):
                print(f"  [{i+1}] {path}")
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

    if action in ("isaac", "mujoco"):
        t_input = input("\nEnable WASD Teleoperation? [Y/n]: ").strip().lower()
        teleop = t_input != "n"
    else:
        h_input = input("\nEnable Headless Mode? [Y/n]: ").strip().lower()
        headless = h_input != "n"

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
    )


if __name__ == "__main__":
    module_name, module_path, action, robot_cfg, terrain_cfg, num_envs, ckpt, teleop, headless = (
        run_cli_menu()
    )

    print(f"\n{'='*50}")
    print(f"Launching {action.upper()} Mode for {module_name}!")
    if robot_cfg:
        print(f"Robot:    {robot_cfg}")
    else:
        print(f"Robots:   A1 + Go1 + Go2 (Sim2Sim)")
    print(f"Terrain:  {terrain_cfg}")
    print(f"Envs:     {num_envs if num_envs else 'Config Default'}")
    if action in ("isaac", "mujoco"):
        print(f"Checkpoint: {ckpt}")
        print(f"Teleop:   {teleop}")
    else:
        print(f"Headless: {headless}")
    print(f"{'='*50}\n")

    # Prepare environment
    env = os.environ.copy()
    env["QUADRUPED_TELEOP"] = "1" if teleop else "0"
    if robot_cfg:
        env["QUADRUPED_ROBOT"] = robot_cfg

    # Ensure the correct source directory is in PYTHONPATH so 'import Quadruped.tasks' works
    source_dir = os.path.abspath(os.path.join(module_path, "source", "Quadruped"))
    if os.path.exists(source_dir):
        env["PYTHONPATH"] = source_dir + os.pathsep + env.get("PYTHONPATH", "")

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

    if action == "train":
        script_path = os.path.join("scripts", "skrl", "train.py")
        task = "Template-Quadruped-Direct-v0"
        cmd = [
            sys.executable,
            script_path,
            f"--task={task}",
        ]
        if num_envs:
            cmd.append(f"--num_envs={num_envs}")
        if ckpt:
            abs_ckpt = os.path.abspath(ckpt)
            cmd.append(f"--checkpoint={abs_ckpt}")
        if headless:
            cmd.append("--headless")
    elif action == "mujoco":
        robot_key = {
            "UNITREE_A1_CFG": "a1",
            "UNITREE_GO1_CFG": "go1",
            "UNITREE_GO2_CFG": "go2",
        }.get(robot_cfg or "", "go1")
        # MuJoCo script is in the Mujoco/ directory
        abs_mujoco_script = os.path.abspath(os.path.join("Mujoco", "mujoco_sim2sim.py"))
        abs_ckpt = os.path.abspath(ckpt) if ckpt else ""
        cmd = [
            sys.executable,
            abs_mujoco_script,
            f"--checkpoint={abs_ckpt}",
            f"--robot={robot_key}",
        ]
    else:  # play
        script_path = os.path.join("scripts", "skrl", "play.py")
        task = "Template-Quadruped-Direct-v0"
        cmd = [
            sys.executable,
            script_path,
            f"--task={task}",
        ]
        if num_envs:
            cmd.append(f"--num_envs={num_envs}")
        if ckpt:
            abs_ckpt = os.path.abspath(ckpt)
            cmd.append(f"--checkpoint={abs_ckpt}")

    print(f"[INFO] Executing in {module_path}: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, cwd=module_path)
