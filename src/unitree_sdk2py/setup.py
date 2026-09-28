from setuptools import find_namespace_packages, setup

setup(
    name="unitree_sdk2py",
    version="1.0.1",
    packages=find_namespace_packages(include=["unitree_sdk2py", "unitree_sdk2py.*"], exclude=["*.__pycache__"]),
    package_data={"unitree_sdk2py.utils": ["lib/*.so"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/unitree_sdk2py"]),
        ("share/unitree_sdk2py", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
)
