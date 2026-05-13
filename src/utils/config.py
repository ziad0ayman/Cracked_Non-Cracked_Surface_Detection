import json
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
_config_path = os.path.join(_script_dir, "..", "..", "config", "config.json")


def load_config():
    with open(_config_path, "r") as f:
        return json.load(f)


def get_global():
    return load_config()["global"]


def get_model(name):
    """Return the best-known hyperparameters for model `name`."""
    return load_config()["models"][name]["best"]


def get_search_space(name):
    """Return the Optuna search-space definition for model `name`."""
    return load_config()["models"][name]["search_space"]


def get_dataset_link():
    return load_config()["dataset_link"]


def get_split_ratios():
    return load_config()["split_ratios"]
