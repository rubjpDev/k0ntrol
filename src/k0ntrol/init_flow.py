"""The `k0 init` questions: a tracker for the project, a provider for the user.

`init_project` writes the `.k0-mem` layout; this module asks the two setup
questions from the provider decision (#14) and writes the answers:

- `tracker` goes to the project config (`.k0-mem/config.yaml`). GitHub is the
  only tracker in 0.1.0; the others are listed but refused.
- one provider for every agent and helper goes to the user config
  (`~/.k0ntrol/config.yaml`), but only when that file sets none yet — the
  provider choice is personal (keys, hardware), never committed.

Anthropic, Bedrock and OpenRouter take the shipped Claude defaults in their
own model IDs; the other providers ask for one model per default tier
(strong/middle/light), and Ollama and LM Studio offer only installed models
that can call tools. Init ends by printing the extra to install, the env var
to set, `k0 doctor`, and the Langfuse warning.
"""

from collections.abc import Callable
from pathlib import Path

import yaml

from k0ntrol.config import AGENTS, HELPERS, ConfigError, deep_merge
from k0ntrol.memory.layout import PROJECT_CONFIG_HEAD, init_project
from k0ntrol.providers import (
    LMSTUDIO_BASE_URL,
    OLLAMA_BASE_URL,
    REGISTRY,
    tool_calling_models,
)

TRACKER = "github"
UNSHIPPED_TRACKERS = ("gitlab", "linear", "jira")

# The shipped Claude defaults (#14) in each provider's own model IDs.
_CLAUDE_TIER_MODELS = {
    "anthropic": {
        "strong": "claude-opus-5-5",
        "middle": "claude-sonnet-5-5",
        "light": "claude-haiku-5-5",
    },
    "openrouter": {
        "strong": "anthropic/claude-opus-5.5",
        "middle": "anthropic/claude-sonnet-5.5",
        "light": "anthropic/claude-haiku-5.5",
    },
    "bedrock": {
        "strong": "us.anthropic.claude-opus-5-5-v1:0",
        "middle": "us.anthropic.claude-sonnet-5-5-v1:0",
        "light": "us.anthropic.claude-haiku-5-5-v1:0",
    },
}

# Default tiers: which roster entries share one model, and the Claude effort.
_TIER_ROSTER = {
    "strong": ("spec", "reviewer"),
    "middle": ("tester", "coder"),
    "light": ("chatty", "bulk_read", "reflect"),
}
_TIER_EFFORT = {"strong": "high", "middle": "medium"}

_USER_CONFIG_HEAD = "# k0 user config: provider and model choices.\n"


def run_init(
    root: Path,
    *,
    tracker: str | None = None,
    provider: str | None = None,
    read_line: Callable[[str], str] | None = None,
    write: Callable[[str], None] = print,
    home: Path | None = None,
    fetch: Callable | None = None,
) -> int:
    """Run `k0 init`: layout, tracker question, provider question, notices."""
    read_line = read_line or input
    error = _check_flags(tracker, provider, write)
    if error:
        return 1
    for line in init_project(root):
        write(line)
    if not _tracker_step(root, tracker, read_line, write):
        return 1
    home = home or Path.home()
    if not _provider_step(root, home, provider, read_line, write, fetch):
        return 1
    return 0


def _check_flags(tracker: str | None, provider: str | None,
                 write: Callable[[str], None]) -> bool:
    """Refuse a flag value init cannot honour; True means a failure."""
    if tracker is not None and tracker != TRACKER:
        write(f"error: tracker '{tracker}' is not in 0.1.0; only '{TRACKER}' is")
        return True
    if provider is not None and provider not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        write(f"error: unknown provider '{provider}'; expected one of: {known}")
        return True
    return False


# --- tracker ----------------------------------------------------------------


def _tracker_step(root, tracker, read_line, write) -> bool:
    path = root / ".k0-mem" / "config.yaml"
    try:
        existing = _read_mapping(path).get("tracker") if path.exists() else None
    except ConfigError as error:
        write(f"config error: {error}")
        return False
    if existing:
        write(f"tracker '{existing}' already set in {path}; keeping it")
        return True
    try:
        name = _pick_tracker(tracker, read_line, write)
    except EOFError:
        write("error: init needs a terminal; pass --tracker")
        return False
    _write_project_tracker(path, name, write)
    return True


def _pick_tracker(tracker, read_line, write) -> str:
    if tracker is not None:
        return tracker
    write("tracker:")
    write(f"  {TRACKER}")
    for name in UNSHIPPED_TRACKERS:
        write(f"  {name} (not in 0.1.0)")
    while True:
        answer = read_line(f"tracker [{TRACKER}]: ").strip().lower()
        if answer in ("", "1", TRACKER):
            return TRACKER
        write(f"'{answer}' is not in 0.1.0; only '{TRACKER}' is")


def _write_project_tracker(path: Path, tracker: str, write) -> None:
    """Merge `tracker` into `.k0-mem/config.yaml`, keeping the comments."""
    data = {"tracker": tracker, **_read_mapping(path)}
    path.write_text(
        PROJECT_CONFIG_HEAD + yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    write(f"tracker '{tracker}' written to {path}")


def _read_mapping(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: invalid YAML: {error}") from error
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


# --- provider ---------------------------------------------------------------


def _provider_step(root, home, provider, read_line, write, fetch) -> bool:
    path = home / ".k0ntrol" / "config.yaml"
    try:
        user_data = _read_mapping(path) if path.exists() else {}
    except ConfigError as error:
        write(f"config error: {error}")
        return False
    in_effect = _providers_in_effect(user_data)
    if in_effect:
        names = ", ".join(f"'{p}'" for p in in_effect)
        write(f"provider {names} already set in {path}; keeping it")
        _print_notices(in_effect, write)
        return True
    return _provider_write(root, path, user_data, provider,
                           read_line, write, fetch)


def _provider_write(root, path, user_data, provider,
                    read_line, write, fetch) -> bool:
    try:
        name = _pick_provider(provider, read_line, write)
    except EOFError:
        write("error: init needs a terminal; pass --provider")
        return False
    roster = _collect_roster(name, root, read_line, write, fetch)
    if roster is None:
        return False
    _write_user_config(path, user_data, roster)
    write(f"provider '{name}' written to {path}")
    _print_notices([name], write)
    return True


def _providers_in_effect(user_data: dict) -> list[str]:
    providers = {
        entry.get("provider")
        for kind in ("agents", "helpers")
        for entry in (user_data.get(kind) or {}).values()
        if isinstance(entry, dict)
    }
    return sorted(p for p in providers if p)


def _pick_provider(provider, read_line, write) -> str:
    if provider is not None:
        return provider
    names = sorted(REGISTRY)
    write("provider for every agent and helper:")
    for index, name in enumerate(names, 1):
        write(f"  {index}. {name}")
    while True:
        answer = read_line(f"provider [{names[0]}]: ").strip().lower()
        chosen = _menu_answer(answer, names)
        if chosen:
            return chosen
        known = ", ".join(names)
        write(f"unknown provider '{answer}'; expected one of: {known}")


def _menu_answer(answer: str, options: list[str]) -> str | None:
    """A menu answer: an exact option, empty for the first, or a 1-based number."""
    if answer in options:
        return answer
    if answer == "":
        return options[0]
    if answer.isdigit() and 1 <= int(answer) <= len(options):
        return options[int(answer) - 1]
    return None


def _collect_roster(name, root, read_line, write, fetch) -> dict | None:
    if name in _CLAUDE_TIER_MODELS:
        return _roster(name, _CLAUDE_TIER_MODELS[name], claude=True)
    try:
        tiers = _ask_tier_models(name, root, read_line, write, fetch)
    except EOFError:
        write("error: init needs a terminal; pass --provider")
        return None
    if tiers is None:
        return None
    return _roster(name, tiers, claude=False)


def _roster(provider: str, tier_models: dict, *, claude: bool) -> dict:
    """`agents:`/`helpers:` entries: one provider, the tier's model."""
    roster: dict = {"agents": {}, "helpers": {}}
    for name in AGENTS + HELPERS:
        tier = _tier_of(name)
        entry = {"provider": provider, "model": tier_models[tier]}
        if claude and tier in _TIER_EFFORT:
            entry["effort"] = _TIER_EFFORT[tier]
        roster["agents" if name in AGENTS else "helpers"][name] = entry
    return roster


def _tier_of(name: str) -> str:
    for tier, names in _TIER_ROSTER.items():
        if name in names:
            return tier
    raise AssertionError(f"roster name without a tier: {name}")


def _ask_tier_models(provider, root, read_line, write, fetch):
    """One model per tier; local providers offer only tool-capable installs."""
    if provider in ("ollama", "lmstudio"):
        models = _local_tool_models(provider, root, write, fetch)
        if not models:
            return None
        return {
            tier: _ask_menu(tier, models, read_line, write)
            for tier in _TIER_ROSTER
        }
    return {tier: _ask_text(tier, read_line, write) for tier in _TIER_ROSTER}


def _local_tool_models(provider, root, write, fetch) -> list[str] | None:
    base_url = _provider_base_url(root, provider)
    try:
        models = tool_calling_models(provider, base_url, fetch=fetch)
    except Exception as error:
        write(f"error: cannot list {provider} models: {_short(error)}")
        return None
    if not models:
        write(f"error: no tool-capable {provider} models installed")
        return None
    return models


def _provider_base_url(root: Path, provider: str) -> str:
    default = OLLAMA_BASE_URL if provider == "ollama" else LMSTUDIO_BASE_URL
    path = root / ".k0-mem" / "config.yaml"
    data = _read_mapping(path) if path.exists() else {}
    providers = data.get("providers") or {}
    return (providers.get(provider) or {}).get("base_url") or default


def _ask_menu(tier, models, read_line, write) -> str:
    write(f"{tier} tier model ({', '.join(_TIER_ROSTER[tier])}):")
    for index, model in enumerate(models, 1):
        write(f"  {index}. {model}")
    while True:
        answer = read_line("model: ").strip()
        chosen = _menu_answer(answer, models)
        if chosen:
            return chosen
        write("pick a listed model that can call tools")


def _ask_text(tier, read_line, write) -> str:
    roles = ", ".join(_TIER_ROSTER[tier])
    while True:
        answer = read_line(f"{tier} tier model ({roles}): ").strip()
        if answer:
            return answer
        write("a model name is required")


def _write_user_config(path: Path, user_data: dict, roster: dict) -> None:
    merged = deep_merge(user_data, roster)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _USER_CONFIG_HEAD + yaml.safe_dump(merged, sort_keys=False),
        encoding="utf-8",
    )


# --- closing notices ----------------------------------------------------------


def _print_notices(providers: list[str], write) -> None:
    extras = sorted({REGISTRY[p].extra for p in providers if p in REGISTRY})
    for extra in extras:
        write(f"install the provider extra: pip install 'k0ntrol[{extra}]'")
    for name in providers:
        _print_credential_line(name, write)
    write("run `k0 doctor` to check the setup")
    write("warning: when Langfuse tracing is enabled, it receives source code")


def _print_credential_line(provider: str, write) -> None:
    spec = REGISTRY.get(provider)
    if spec is None:
        return
    if not spec.credentials:
        write(f"provider '{provider}' needs no credential env var")
        return
    names = " or ".join(spec.credentials)
    if spec.credentials_required:
        write(f"set the credential env var: {names}")
    else:
        write(f"credential env vars, if the chain does not supply one: {names}")


def _short(error: Exception) -> str:
    return (str(error) or type(error).__name__)[:200]
