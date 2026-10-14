import os
import yaml

from quadruped_common import paths

def load_config(config_path=None):
    """
    Loads the controller configuration from a YAML file.
    Defaults to quadruped_bringup/config/config.yaml.
    """
    if config_path is None:
        config_path = paths.config("config.yaml")

    if not os.path.exists(config_path):
        print(f"[ConfigLoader] WARNING: Config file not found at {config_path}. Using defaults.")
        return {}

    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as exc:
            print(f"[ConfigLoader] ERROR: Could not parse YAML: {exc}")
            return {}


def load_policies(path=None):
    """quadruped_bringup/config/policies.json: {"safety_policy": name, "policies": {name: {path, ...}}}."""
    import json

    path = path or paths.config("policies.json")
    if not os.path.exists(path):
        print(f"[ConfigLoader] WARNING: {path} not found. No policies to switch to, no safety policy.")
        return {}
    with open(path) as f:
        return json.load(f)
