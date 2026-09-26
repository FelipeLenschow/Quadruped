import os
from glob import glob

from setuptools import find_packages, setup

package_name = "quadruped_drivers"


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
        ("lib/" + package_name, glob("scripts/*.py")),
    ] + tree(),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Felipe Lenschow",
    maintainer_email="felipe10lenschow@gmail.com",
    description="Robot drivers: real Go2, MuJoCo, Gazebo, MuJoCo evaluation.",
    license="TODO: License declaration",
    entry_points={"console_scripts": [
        "real_driver = quadruped_drivers.real_driver:main",
        "test_joints = quadruped_drivers.test_joints:main",
        "mujoco_driver = quadruped_drivers.mujoco_driver:main",
        "eval_mujoco = quadruped_drivers.eval_mujoco:main",
        "gazebo_driver = quadruped_drivers.gazebo_driver:main",
        "mujoco_sim2sim = quadruped_drivers.mujoco_sim2sim:main",
        "gazebo_sim2sim = quadruped_drivers.gazebo_sim2sim:main",
    ]},
)
