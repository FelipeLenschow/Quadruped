import os
import yaml

def load_config(config_path=None):
    """
    Loads the controller configuration from a YAML file.
    Defaults to Controller/config/config.yaml.
    """
    if config_path is None:
        # Default path relative to this file (now in configs/)
        base_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_path, "config.yaml")

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
