import os

from setuptools import find_packages, setup

package_name = "quadruped_description"


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
    ] + tree("mujoco_menagerie", "gazebo", "robots", "config", "policies", "scripts"),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Felipe Lenschow",
    maintainer_email="felipe10lenschow@gmail.com",
    description="Robot models (MuJoCo, Gazebo, URDF), kinematics and bundled policies.",
    license="TODO: License declaration",
    entry_points={"console_scripts": [
    ]},
)
