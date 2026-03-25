import os
import sys
import glob
import subprocess


def run_cli_menu():
    print("\n" + "=" * 50)
    print(" Quadruped Unified Launcher")
    print("=" * 50 + "\n")

    # 0. Selection: Walk or Handstand
    print("Select Module:")
    print("  [1] Walk")
    print("  [2] Handstand")
    mod_idx = input("Enter choice [1-2] (default 1): ").strip()
    if mod_idx == "2":
        selected_module = "Handstand"
    else:
        selected_module = "Walk"

    print(f"\n--- Operating on {selected_module} ---\n")

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
    if action == "train":
        default_envs = "2000"
    elif action == "isaac":
        default_envs = "6"
    elif action == "mujoco":
        default_envs = "1"
    else:
        default_envs = "100"
    num_envs = input(
        f"\nEnter number of environments (default {default_envs}): "
    ).strip()
    if not num_envs:
        num_envs = default_envs

    selected_ckpt = None
    teleop = False
    headless = False

    # 5. Checkpoint Selection
    checkpoint_paths = glob.glob(
        os.path.join(
            selected_module,
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
            f"\n[ERROR] No best_agent.pt checkpoints found in {selected_module}/logs/skrl/quadruped_direct/"
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
        selected_module,
        action,
        selected_robot_cfg,
        selected_terrain,
        num_envs,
        selected_ckpt,
        teleop,
        headless,
    )


if __name__ == "__main__":
    module, action, robot_cfg, terrain_cfg, num_envs, ckpt, teleop, headless = (
        run_cli_menu()
    )

    print(f"\n{'='*50}")
    print(f"Launching {action.upper()} Mode for {module}!")
    if robot_cfg:
        print(f"Robot:    {robot_cfg}")
    else:
        print(f"Robots:   A1 + Go1 + Go2 (Sim2Sim)")
    print(f"Terrain:  {terrain_cfg}")
    print(f"Envs:     {num_envs}")
    if action in ("isaac", "mujoco"):
        print(f"Checkpoint: {ckpt}")
        print(f"Teleop:   {teleop}")
    else:
        print(f"Headless: {headless}")
    print(f"{'='*50}\n")

    # Prepare environment
    env = os.environ.copy()
    env["QUADRUPED_TERRAIN"] = terrain_cfg
    env["QUADRUPED_TELEOP"] = "1" if teleop else "0"
    if robot_cfg:
        env["QUADRUPED_ROBOT"] = robot_cfg

    # Ensure the correct source directory is in PYTHONPATH so 'import Quadruped.tasks' works
    source_dir = os.path.abspath(os.path.join(module, "source", "Quadruped"))
    env["PYTHONPATH"] = source_dir + os.pathsep + env.get("PYTHONPATH", "")

    if action == "train":
        script_path = os.path.join("scripts", "skrl", "train.py")
        task = "Template-Quadruped-Direct-v0"
        cmd = [
            sys.executable,
            script_path,
            f"--task={task}",
            f"--num_envs={num_envs}",
        ]
        if ckpt:
            # Checkpoint is already relative to root, needs to be relative to module or absolute
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
            f"--num_envs={num_envs}",
        ]
        if ckpt:
            abs_ckpt = os.path.abspath(ckpt)
            cmd.append(f"--checkpoint={abs_ckpt}")

    print(f"[INFO] Executing in {module}: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, cwd=module)
