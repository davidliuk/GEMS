"""Model workflow definitions for ComfyClaw benchmarks.

Each model is defined as a YAML file in this directory. To add a new model,
drop a .yaml file here — it will be auto-discovered by short_name.
"""

from pathlib import Path

import yaml

_DIR = Path(__file__).resolve().parent


def _load_model(yaml_path: Path) -> dict:
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for node_id, node in cfg["workflow"].items():
        inputs = node.get("inputs", {})
        for key, val in inputs.items():
            if isinstance(val, list) and len(val) == 2:
                inputs[key] = [str(val[0]), int(val[1])]

    return cfg


def _discover() -> dict[str, dict]:
    models: dict[str, dict] = {}
    for yaml_path in sorted(_DIR.glob("*.yaml")):
        cfg = _load_model(yaml_path)
        models[cfg["short_name"]] = cfg
    return models


MODELS = _discover()
