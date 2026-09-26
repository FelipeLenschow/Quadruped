from setuptools import find_packages, setup

setup(
    name="unitree_sdk2py",
    version="1.0.1",
    packages=find_packages(include=["unitree_sdk2py", "unitree_sdk2py.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/unitree_sdk2py"]),
        ("share/unitree_sdk2py", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
)
