from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


AGENTS = ("spec", "tester", "coder", "validator", "bulk_reader")


class ConfigError(Exception):
    """Raised when configuration cannot be read or is invalid."""


def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(project_dir: Path, home_dir: Path | None = None) -> dict:
    effective = _read_config(files("k0ntrol") / "defaults" / "config.yaml")
    home = home_dir or Path.home()
    for path in (home / ".k0ntrol/config.yaml", project_dir / ".k0-mem/config.yaml"):
        if path.exists():
            effective = deep_merge(effective, _read_config(path))
    return effective


def _read_config(path: Any) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: invalid YAML: {error}") from error
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: top level must be a mapping")
    return dict(data)
