from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


AGENTS = ("spec", "tester", "coder", "validator", "bulk_reader")
BACKENDS = ("cursor",)


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
    validate_config(effective)
    return effective


def validate_config(cfg: dict) -> None:
    errors = _validate_top_level(cfg)
    errors.extend(_validate_agents(cfg.get("agents")))
    if errors:
        raise ConfigError("; ".join(errors))


def _validate_top_level(cfg: dict) -> list[str]:
    errors = []
    if cfg.get("backend") not in BACKENDS:
        errors.append("backend: must be cursor")
    errors.extend(_validate_integer("threshold_lines", cfg.get("threshold_lines"), 1))
    errors.extend(_validate_text("test_cmd", cfg.get("test_cmd")))
    errors.extend(_validate_integer("max_retries", cfg.get("max_retries"), 0))
    return errors


def _validate_integer(name: str, value: object, minimum: int) -> list[str]:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return [f"{name}: must be an integer >= {minimum}"]
    return []


def _validate_text(name: str, value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return [f"{name}: must be a non-empty string"]
    return []


def _validate_agents(value: object) -> list[str]:
    if not isinstance(value, Mapping):
        return ["agents: must be a mapping"]
    errors = []
    for name in AGENTS:
        agent = value.get(name)
        if not isinstance(agent, Mapping):
            errors.append(f"agents.{name}: must be a mapping")
            continue
        if not isinstance(agent.get("model"), str) or not agent["model"].strip():
            errors.append(f"agents.{name}.model: must be a non-empty string")
    return errors


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
