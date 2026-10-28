from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when configuration cannot be read or is invalid."""


AGENTS = ("spec", "tester", "coder", "reviewer", "chatty")
HELPERS = ("bulk_read", "reflect")

_AGENT_KEYS = {"provider", "model", "effort", "params", "model_calls"}
_HELPER_KEYS = {"provider", "model", "effort", "params"}
_EFFORT_LEVELS = {"low", "medium", "high"}

# 0.0.x keys that are gone in 0.1.0, mapped to their replacement.
_RETIRED_KEYS = {
    "backend": "set provider on each agents:/helpers: entry",
    "max_retries": "max_attempts",
    "retrieval_top_k": "the search_memory tool's limit argument",
    "reflect_every": "nothing; reflect runs after each run",
    "validator": "VALIDATOR is a deterministic code gate now; use test_cmd and gates",
    "retrieval_gate": "the search_memory tool",
    "bulk_reader": "helpers.bulk_read",
}

# Keys that must never appear in config: secrets belong in vendor env vars.
_SECRET_MARKERS = ("api_key", "token", "secret", "password")

# Keys that would inject tool definitions through params.
_TOOL_MARKERS = ("tool", "function")


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
    if errors:
        raise ConfigError("; ".join(errors))


def resolved_params(cfg: dict, entry: dict) -> dict:
    """Merge an agent or helper's params over its provider's params."""
    providers = cfg.get("providers") or {}
    provider = providers.get(entry.get("provider")) or {}
    return deep_merge(provider.get("params") or {}, entry.get("params") or {})


def _validate_top_level(cfg: dict) -> list[str]:
    errors = _check_keys(cfg)
    errors.extend(_validate_integer("threshold_lines", cfg.get("threshold_lines"), 1))
    errors.extend(_validate_text("test_cmd", cfg.get("test_cmd")))
    errors.extend(_validate_gates(cfg.get("gates")))
    errors.extend(_validate_optional_text("tracker", cfg.get("tracker")))
    errors.extend(_validate_optional_text("base_branch", cfg.get("base_branch")))
    errors.extend(_validate_integer("max_attempts", cfg.get("max_attempts"), 1))
    errors.extend(_validate_integer("test_timeout", cfg.get("test_timeout"), 1))
    errors.extend(_validate_roster("agents", cfg.get("agents")))
    errors.extend(_validate_roster("helpers", cfg.get("helpers")))
    errors.extend(_validate_providers(cfg.get("providers")))
    return errors


def _validate_providers(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Mapping):
        return ["providers: must be a mapping"]
    errors = []
    for name, entry in value.items():
        errors.extend(_validate_provider(f"providers.{name}", entry))
    return errors


def _validate_provider(path: str, value: object) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{path}: must be a mapping"]
    errors = []
    if "base_url" in value:
        errors.extend(_validate_text(f"{path}.base_url", value["base_url"]))
    errors.extend(_validate_params(f"{path}.params", value.get("params")))
    errors.extend(
        f"{path}.{key}: unknown key" for key in value if key not in {"base_url", "params"}
    )
    return errors


def _validate_roster(kind: str, value: object) -> list[str]:
    names = AGENTS if kind == "agents" else HELPERS
    if not isinstance(value, Mapping):
        return [f"{kind}: must be a mapping"]
    errors = _unknown_roster_names(kind, names, value)
    for name in names:
        errors.extend(_validate_entry(f"{kind}.{name}", value.get(name), kind))
    return errors


def _unknown_roster_names(kind: str, names: tuple, value: Mapping) -> list[str]:
    return [
        f"{kind}.{name}: unknown {kind} entry"
        for name in value
        if name not in names and str(name).lower() not in _RETIRED_KEYS
    ]


def _check_keys(node: object, path: str = "", in_params: bool = False) -> list[str]:
    if isinstance(node, Mapping):
        return _check_mapping_keys(node, path, in_params)
    if isinstance(node, list):
        return _check_list_keys(node, path, in_params)
    return []


def _check_mapping_keys(node: Mapping, path: str, in_params: bool) -> list[str]:
    errors = []
    for key, value in node.items():
        key_path = f"{path}.{key}" if path else str(key)
        errors.extend(_check_key(key, key_path, in_params))
        errors.extend(
            _check_keys(value, key_path, in_params or str(key).lower() == "params")
        )
    return errors


def _check_list_keys(node: list, path: str, in_params: bool) -> list[str]:
    errors = []
    for index, item in enumerate(node):
        errors.extend(_check_keys(item, f"{path}[{index}]", in_params))
    return errors


def _check_key(key: object, path: str, in_params: bool) -> list[str]:
    if not isinstance(key, str):
        return [f"{path}: keys must be strings"]
    lowered = key.lower()
    if lowered in _RETIRED_KEYS:
        return [f"{path}: retired in 0.1.0; {_RETIRED_KEYS[lowered]}"]
    if any(marker in lowered for marker in _SECRET_MARKERS):
        return [f"{path}: secrets are not allowed in config; use the provider's env vars"]
    if in_params and any(marker in lowered for marker in _TOOL_MARKERS):
        return [f"{path}: tool definitions are not allowed in params"]
    return []


def _validate_entry(path: str, value: object, kind: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{path}: must be a mapping"]
    errors = _validate_text(f"{path}.provider", value.get("provider"))
    errors.extend(_validate_text(f"{path}.model", value.get("model")))
    errors.extend(_validate_effort(f"{path}.effort", value.get("effort")))
    errors.extend(_validate_params(f"{path}.params", value.get("params")))
    errors.extend(_unknown_entry_keys(path, value, kind))
    if kind == "agents":
        errors.extend(_validate_integer(f"{path}.model_calls", value.get("model_calls"), 1))
    return errors


def _unknown_entry_keys(path: str, entry: Mapping, kind: str) -> list[str]:
    allowed = _AGENT_KEYS if kind == "agents" else _HELPER_KEYS
    return [f"{path}.{key}: unknown key" for key in entry if key not in allowed]


def _validate_effort(name: str, value: object) -> list[str]:
    if value is None:
        return []
    if value not in _EFFORT_LEVELS:
        return [f"{name}: must be one of low, medium, high"]
    return []


def _validate_params(name: str, value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Mapping):
        return [f"{name}: must be a mapping"]
    return []


def _validate_gates(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return ["gates: must be a list"]
    errors = []
    for index, gate in enumerate(value):
        errors.extend(_validate_gate(index, gate))
    return errors


def _validate_gate(index: int, gate: object) -> list[str]:
    if not isinstance(gate, Mapping):
        return [f"gates[{index}]: must be a mapping"]
    errors = _validate_text(f"gates[{index}].name", gate.get("name"))
    errors.extend(_validate_text(f"gates[{index}].cmd", gate.get("cmd")))
    return errors


def _validate_integer(name: str, value: object, minimum: int) -> list[str]:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return [f"{name}: must be an integer >= {minimum}"]
    return []


def _validate_text(name: str, value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return [f"{name}: must be a non-empty string"]
    return []


def _validate_optional_text(name: str, value: object) -> list[str]:
    if value is None:
        return []
    return _validate_text(name, value)


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
