"""Where the repo's data lives, whether the packages run from a colcon install or straight from src/."""
import os

SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REPO = os.environ.get("QUADRUPED_REPO") or os.path.dirname(SRC)


def share(package, *parts):
    try:
        from ament_index_python.packages import get_package_share_directory
        base = get_package_share_directory(package)
    except Exception:
        base = os.path.join(SRC, package)
    return os.path.join(base, *parts)


def config(*parts):
    return share("quadruped_bringup", "config", *parts)


def description(*parts):
    return share("quadruped_description", *parts)


def repo(*parts):
    return os.path.join(REPO, *parts)
