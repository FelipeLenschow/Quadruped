from setuptools import find_packages, setup

package_name = "quadruped_perception"

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
    description="Perception: lidar body filter, cone detection.",
    license="TODO: License declaration",
    entry_points={"console_scripts": [
        "lidar_filter = quadruped_perception.lidar_filter:main",
        "lidar_level = quadruped_perception.lidar_level:main",
        "lidar_imu_calib = quadruped_perception.lidar_imu_calib:main",
    ]},
)
