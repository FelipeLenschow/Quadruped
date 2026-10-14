from setuptools import find_packages, setup

package_name = "quadruped_common"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Felipe Lenschow",
    maintainer_email="felipe10lenschow@gmail.com",
    description="Python helpers shared by every node: paths, config, realtime core, stance, kinematics (library, no nodes).",
    license="TODO: License declaration",
)
