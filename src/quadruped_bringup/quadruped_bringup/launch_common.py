import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from quadruped_core import paths

LAUNCH_DIR = os.path.join(get_package_share_directory("quadruped_bringup"), "launch")


def arg(name, default="", description=""):
    return DeclareLaunchArgument(name, default_value=str(default), description=description)


def common_args():
    return [
        arg("robot", "go2"),
        arg("record", "false", "record every topic to MCAP"),
        arg("record_path", "", "bag output folder (default Mcap/Recordings/run_<name>_<time>)"),
    ]


def reward_args():
    return [
        arg("reward", "false", "also run the reward estimator"),
        arg("reward_config", "", "training_phases.yaml for the reward estimator"),
        arg("reward_phase", "", "phase the checkpoint finished under"),
    ]


def get(context, name):
    return LaunchConfiguration(name).perform(context)


def flag(context, name):
    return get(context, name).strip().lower() in ("1", "true", "yes", "on")


def node(package, executable, *args, main=False, **kwargs):
    """main=True ends the whole launch when this node exits."""
    return Node(
        package=package,
        executable=executable,
        arguments=[a for a in args if a],
        cwd=paths.REPO,
        output=kwargs.pop("output", "screen"),
        emulate_tty=True,
        sigterm_timeout="10",
        on_exit=[Shutdown()] if main else None,
        **kwargs,
    )


def include(name, **launch_args):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(LAUNCH_DIR, name)),
        launch_arguments={k: str(v) for k, v in launch_args.items()}.items(),
    )


def record(context, label):
    if not flag(context, "record"):
        return []
    path = get(context, "record_path")
    if not path:
        path = paths.repo("Mcap", "Recordings", f"run_{label}_{datetime.now():%Y%m%d_%H%M%S}")
    return [include("record.launch.py", path=path)]


def reward(context):
    if not flag(context, "reward"):
        return []
    return [include("reward.launch.py", robot=get(context, "robot"),
                    config_path=get(context, "reward_config"), phase=get(context, "reward_phase"))]
