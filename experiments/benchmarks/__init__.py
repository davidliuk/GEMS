"""Benchmark prompt loaders for ComfyClaw experiments.

Each benchmark is defined as a YAML file in this directory. To add a new
benchmark, drop a .yaml file here — it will be auto-discovered by short_name.

The YAML `data.format` field controls how prompts are loaded:
  - jsonl:           one JSON object per line
  - json_array:      a single JSON array
  - json_or_jsonl:   auto-detect (starts with '[' → array, else JSONL)
  - jsonl_or_txtdir: JSONL file, or a directory of .txt files
"""

import json
import os
from pathlib import Path

import yaml

_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DIR.parent.parent


def _resolve_default_path(relative: str) -> str:
    return str((_REPO_ROOT / relative).resolve())


def _make_loader(data_cfg: dict):
    """Build a load_prompts(n, data_path) function from the YAML data config."""
    env_var = data_cfg["env_var"]
    default_path = _resolve_default_path(data_cfg["default_path"])
    fmt = data_cfg["format"]
    prompt_field = data_cfg["prompt_field"]
    fallback_field = data_cfg.get("prompt_field_fallback")

    def _load_jsonl(path: str, n: int) -> list[dict]:
        items = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                row = json.loads(line)
                items.append({
                    "prompt": row.get(prompt_field, row.get(fallback_field, "")) if fallback_field else row[prompt_field],
                    "idx": i,
                    "meta": row,
                })
        return items

    def _load_json_array(path: str, n: int) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = []
        for i, row in enumerate(data):
            if i >= n:
                break
            prompt = row.get(prompt_field, "")
            if not prompt and fallback_field:
                prompt = row.get(fallback_field, "")
            items.append({"prompt": prompt, "idx": i, "meta": row})
        return items

    def _load_txtdir(path: str, n: int) -> list[dict]:
        txt_files = sorted(Path(path).glob("*.txt"), key=lambda p: p.stem.zfill(10))
        items = []
        for i, txt_file in enumerate(txt_files):
            if i >= n:
                break
            prompt = txt_file.read_text(encoding="utf-8").strip()
            items.append({"prompt": prompt, "idx": i, "meta": {"file": txt_file.name}})
        return items

    def load_prompts(n: int, data_path: str | None = None) -> list[dict]:
        path = data_path or os.environ.get(env_var, default_path)

        if fmt == "jsonl":
            return _load_jsonl(path, n)
        elif fmt == "json_array":
            return _load_json_array(path, n)
        elif fmt == "jsonl_or_txtdir":
            if os.path.isdir(path):
                return _load_txtdir(path, n)
            return _load_jsonl(path, n)
        elif fmt == "json_or_jsonl":
            with open(path, encoding="utf-8") as f:
                content = f.read().strip()
            if content.startswith("["):
                data = json.loads(content)
                items = []
                for i, row in enumerate(data):
                    if i >= n:
                        break
                    prompt = row.get(prompt_field, "")
                    if not prompt and fallback_field:
                        prompt = row.get(fallback_field, "")
                    items.append({"prompt": prompt, "idx": i, "meta": row})
                return items
            else:
                return _load_jsonl(path, n)
        else:
            raise ValueError(f"Unknown benchmark data format: {fmt!r}")

    return load_prompts


def _discover() -> dict[str, dict]:
    benchmarks: dict[str, dict] = {}
    for yaml_path in sorted(_DIR.glob("*.yaml")):
        with open(yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cfg["load_prompts"] = _make_loader(cfg["data"])
        cfg["data_env_var"] = cfg["data"]["env_var"]
        cfg["default_data_path"] = _resolve_default_path(cfg["data"]["default_path"])
        benchmarks[cfg["short_name"]] = cfg
    return benchmarks


BENCHMARKS = _discover()
