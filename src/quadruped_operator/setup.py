import os

from setuptools import find_packages, setup

package_name = "quadruped_operator"


def tree(*dirs):
    files = []
    for d in dirs:
        for root, _, names in os.walk(d):
            if names:
                files.append((os.path.join("share", package_name, root), [os.path.join(root, n) for n in names]))
    return files


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ] + tree(),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Felipe Lenschow",
    maintainer_email="felipe10lenschow@gmail.com",
    description="Operator tools: console, supervisor, teleops, twins, reward estimator, MCAP tool.",
    license="TODO: License declaration",
    entry_points={"console_scripts": [
        "console = quadruped_operator.console:main",
        "supervisor = quadruped_operator.supervisor:main",
        "gait_teleop = quadruped_operator.gait_teleop:main",
        "keyboard_teleop = quadruped_operator.keyboard_teleop:main",
        "sweep_teleop = quadruped_operator.sweep_teleop:main",
        "mujoco_twin = quadruped_operator.mujoco_twin:main",
        "gazebo_twin = quadruped_operator.gazebo_twin:main",
        "reward_estimator = quadruped_operator.reward_estimator_node:main",
        "mcap_tool = quadruped_operator.mcap_tool:main",
    ]},
)
