import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction, Shutdown

from quadruped_bringup.launch_common import arg, common_args, flag, get, include, record, reward, reward_args
from quadruped_core import paths


def setup(context):
    # Isaac Sim runs on its own Python 3.12, so the driver stays a plain script outside the packages.
    checkpoint = get(context, "checkpoint")
    cmd = [
        get(context, "python"), paths.repo("IsaacSim", "isaac_driver.py"),
        f"--robot={get(context, 'robot')}",
        f"--obs_dim={get(context, 'obs_dim')}",
    ]
    if checkpoint:
        cmd.append(f"--internal_policy={checkpoint}")
    if flag(context, "use_estimator"):
        cmd.append("--use_estimator")
    actions = [ExecuteProcess(cmd=cmd, cwd=paths.REPO, output="screen", emulate_tty=True,
                              sigterm_timeout="10", on_exit=[Shutdown()])]
    if flag(context, "joy"):
        actions.append(include("joy.launch.py"))
    return actions + record(context, "isaac_sim") + reward(context)


def generate_launch_description():
    return LaunchDescription([
        *common_args(),
        arg("python", os.path.expanduser("~/env_isaacsim/bin/python"), "Isaac Sim interpreter"),
        arg("checkpoint"),
        arg("obs_dim", "49"),
        arg("use_estimator", "false"),
        arg("joy", "false", "also start the gamepad teleop"),
        *reward_args(),
        OpaqueFunction(function=setup),
    ])
