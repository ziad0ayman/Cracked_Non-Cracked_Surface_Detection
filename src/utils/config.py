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


def update_best(model_name, params, val_accuracy):
    """Update best params in config.json if val_accuracy beats current best.

    Returns True if config was updated, False otherwise.
    """
    with open(_config_path, "r") as f:
        cfg = json.load(f)

    best = cfg["models"][model_name]["best"]
    current_best = best.get("val_accuracy", 0.0)

    if val_accuracy <= current_best:
        return False

    new_best = {"val_accuracy": round(val_accuracy, 4)}
    new_best.update(params)
    cfg["models"][model_name]["best"] = new_best

    with open(_config_path, "w") as f:
        json.dump(cfg, f, indent=4)
    return True


def get_dataset_link():
    return load_config()["dataset_link"]


def get_split_ratios():
    return load_config()["split_ratios"]
