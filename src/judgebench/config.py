from pathlib import Path

import yaml


def load_config(path="configs/config.yaml"):
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")

    model = config.get("judge", {}).get("model")

    if not model:
        raise ValueError("Judge model is missing from configuration")

    return config
