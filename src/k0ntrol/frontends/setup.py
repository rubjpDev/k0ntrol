"""Setup screens: the project once per repo, the provider once per machine.

Both are Claude Code-style questions over :class:`Prompt`. The project screen
asks where the code is hosted, which issue tracker fits that host, the base
branch and the test command, and writes ``.k0-mem/config.yaml``. The provider
screen rewrites the one user config, ``~/.k0ntrol/config.yaml``. Both run
again from ``/config``.
"""

import getpass
import re
from collections.abc import Callable
from pathlib import Path

import yaml
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from k0ntrol.config import AGENTS, HELPERS, VCS_TRACKERS, load_config
from k0ntrol.credentials import save_credential
from k0ntrol.frontends.prompt import Prompt
from k0ntrol.frontends.theme import BORDER, BULLET, DIM, OK, WARN
from k0ntrol.init_flow import (
    _CLAUDE_TIER_MODELS,
    _TIER_EFFORT,
    _TIER_ROSTER,
    _provider_base_url,
    _tier_of,
)
from k0ntrol.memory.layout import PROJECT_CONFIG_HEAD, init_project
from k0ntrol.providers import REGISTRY, tool_calling_models
from k0ntrol.workflow.change import default_base_branch, origin_url

_OTHER = "other…"
_USER_CONFIG_HEAD = (
    "# k0 user config: provider and model choices (written by the setup screen).\n"
    "# Never put API keys here; k0 keeps them in ~/.k0ntrol/credentials.\n"
)

# --- project setup ------------------------------------------------------------

_VCS_HOSTS = {"github": "GitHub", "gitlab": "GitLab", "bitbucket": "Bitbucket"}
_TRACKER_NAMES = {
    "github": "GitHub Issues",
    "gitlab": "GitLab Issues",
    "linear": "Linear",
    "jira": "Jira",
}
# Saved as given; only GitHub code + GitHub Issues work with /full in 0.1.0.
_UNSHIPPED = "  coming soon: /full needs GitHub in 0.1.0; chat and /fast still work here"
_HOST_PATTERN = re.compile(r"^(?:\w+://)?(?:[^@/]+@)?([^/:]+)")
_TEST_COMMANDS = (
    ("uv.lock", "uv run pytest -q"),
    ("pyproject.toml", "pytest -q"),
    ("setup.cfg", "pytest -q"),
    ("pytest.ini", "pytest -q"),
    ("package.json", "npm test"),
    ("go.mod", "go test ./..."),
    ("Cargo.toml", "cargo test"),
)


def run_project_setup(root: Path, prompt: Prompt, console: Console) -> Path:
    """Ask code host, tracker, base branch and test command; write the config."""
    _print_title(
        console,
        f"Set up k0 for {root.name}",
        "Saved in .k0-mem/config.yaml; you can change it later.",
    )
    vcs = _ask_vcs(root, prompt)
    tracker = _ask_tracker(vcs, prompt)
    if vcs != "github" or tracker != "github":
        console.print(Text(_UNSHIPPED, style=WARN))
    answers = {
        "vcs": vcs,
        "tracker": tracker,
        "base_branch": _ask_base_branch(root, prompt),
        "test_cmd": _ask_test_cmd(root, prompt),
    }
    init_project(root)
    path = root / ".k0-mem" / "config.yaml"
    path.write_text(
        PROJECT_CONFIG_HEAD + yaml.safe_dump(answers, sort_keys=False), encoding="utf-8"
    )
    _print_ok(console, f"Saved {path.relative_to(root).as_posix()}")
    return path


def suggest_test_command(root: Path) -> str | None:
    return next((cmd for name, cmd in _TEST_COMMANDS if (root / name).exists()), None)


def detect_vcs(root: Path) -> str | None:
    """The code host named by the ``origin`` remote, if it is a known one."""
    match = _HOST_PATTERN.match(origin_url(root))
    host = match.group(1).lower() if match else ""
    if host == "github.com":
        return "github"
    if host == "bitbucket.org":
        return "bitbucket"
    return "gitlab" if "gitlab" in host else None


def _ask_vcs(root: Path, prompt: Prompt) -> str:
    detected = detect_vcs(root)
    hosts = sorted(_VCS_HOSTS, key=lambda host: host != detected)
    options = [(host, _host_label(host, detected)) for host in hosts]
    return prompt.choose("Where is this repo's code hosted?", options)


def _host_label(host: str, detected: str | None) -> str:
    tags = ["detected"] if host == detected else []
    tags += [] if host == "github" else ["coming soon"]
    return f"{_VCS_HOSTS[host]} ({', '.join(tags)})" if tags else _VCS_HOSTS[host]


def _ask_tracker(vcs: str, prompt: Prompt) -> str:
    options = [(name, _tracker_label(name)) for name in VCS_TRACKERS[vcs]]
    return prompt.choose("Which issue tracker does this repo use?", options)


def _tracker_label(name: str) -> str:
    label = _TRACKER_NAMES[name]
    return label if name == "github" else f"{label} (coming soon)"


def _ask_base_branch(root: Path, prompt: Prompt) -> str:
    detected = default_base_branch(root)
    options = [(detected, f"{detected} (detected)"), (_OTHER, "Another branch…")]
    choice = prompt.choose("Which branch do pull requests target?", options)
    return _ask_nonempty(prompt, "base branch") if choice == _OTHER else choice


def _ask_test_cmd(root: Path, prompt: Prompt) -> str:
    suggestion = suggest_test_command(root)
    options = [(suggestion, suggestion)] if suggestion else []
    options.append((_OTHER, "Another command…"))
    choice = prompt.choose("How do you run this repo's tests?", options)
    return _ask_nonempty(prompt, "test command") if choice == _OTHER else choice


def _ask_nonempty(prompt: Prompt, label: str) -> str:
    while True:
        answer = prompt.ask_text(label).strip()
        if answer:
            return answer


# --- provider setup -----------------------------------------------------------

_PROVIDERS = [
    ("ollama", "Ollama (local)"),
    ("bedrock", "Amazon Bedrock"),
    ("lmstudio", "LM Studio (local)"),
    ("anthropic", "Anthropic API (untested)"),
    ("openai", "OpenAI API (untested)"),
    ("openrouter", "OpenRouter (untested)"),
    ("vertex", "Google Vertex (untested)"),
]
_LOCAL_NAMES = {"ollama": "Ollama", "lmstudio": "LM Studio"}
_TIER_QUESTION = {
    "strong": "SPEC and REVIEWER",
    "middle": "TESTER and CODER",
    "light": "CHATTY and helpers",
}
_CLAUDE_NAMES = {"strong": "Opus", "middle": "Sonnet", "light": "Haiku"}
_BEDROCK_TESTED = {
    "strong": "qwen.qwen3-next-80b-a3b",
    "middle": "qwen.qwen3-coder-next",
    "light": "qwen.qwen3-next-80b-a3b",
}
_BEDROCK_PRESETS = [
    ("qwen.qwen3-next-80b-a3b", "Qwen3 Next 80B (tested)"),
    ("qwen.qwen3-coder-next", "Qwen3 Coder Next (tested)"),
    ("us.amazon.nova-pro-v1:0", "Amazon Nova Pro"),
    ("us.amazon.nova-lite-v1:0", "Amazon Nova Lite"),
]
_PLACEHOLDERS = {
    "openai": ["gpt-6.1-terra", "gpt-6-luna"],
    "openai_compatible": ["gpt-6.1-terra", "gpt-6-luna"],
    "vertex": ["gemini-3-pro", "gemini-3-flash"],
}
_EFFORTS = [
    ("high", "High"),
    ("medium", "Medium"),
    ("low", "Low"),
    ("none", "Provider default (no effort)"),
]
_REGIONS = [("us-east-1", "us-east-1"), ("us-west-2", "us-west-2"), ("eu-west-1", "eu-west-1")]
_KEY_CHOICES = [
    ("save", "Save an API key for k0 (stored in ~/.k0ntrol/credentials, readable only by you)"),
    ("env", "I set it myself (environment or credential chain)"),
]
_BEDROCK_KEY = "AWS_BEARER_TOKEN_BEDROCK"
_ROSTER_KINDS = ("providers", "agents", "helpers")


def run_provider_setup(
    home: Path,
    prompt: Prompt,
    console: Console,
    *,
    root: Path,
    list_models: Callable[[str, str], list[str]] | None = None,
    read_secret: Callable[[str], str] = getpass.getpass,
) -> Path:
    """Pick a provider, a model and an effort per tier; write the user config."""
    _print_title(
        console,
        "Choose the models k0 uses",
        "Saved in ~/.k0ntrol/config.yaml; run /config to change it later.",
    )
    provider = prompt.choose("Which provider runs your models?", _PROVIDERS)
    installed = _installed_models(provider, root, console, list_models or _default_lister)
    tiers = {tier: _ask_tier(prompt, provider, tier, installed) for tier in _TIER_ROSTER}
    settings = _ask_provider_settings(prompt, provider)
    _ask_credential(home, prompt, console, provider, read_secret)
    config = _build_config(provider, tiers, settings)
    path = _write_user_config(home, config)
    load_config(root, home_dir=home)
    _print_summary(console, config)
    return path


def _default_lister(provider: str, base_url: str) -> list[str]:
    return tool_calling_models(provider, base_url)


def _installed_models(
    provider: str, root: Path, console: Console, lister: Callable[[str, str], list[str]]
) -> list[str] | None:
    """Installed tool-capable models of a local provider; None for the others."""
    if provider not in _LOCAL_NAMES:
        return None
    try:
        models = list(lister(provider, _provider_base_url(root, provider)))
        reason = "no installed model can call tools"
    except Exception as error:
        models, reason = [], f"cannot reach the server ({_short(error)})"
    if not models:
        name = _LOCAL_NAMES[provider]
        console.print(Text(f"  {name}: {reason}. Start {name} and pull a model "
                           "that can call tools, or type a model name.", style=WARN))
    return models


def _ask_tier(prompt: Prompt, provider: str, tier: str, installed: list[str] | None) -> dict:
    model = _ask_model(prompt, tier, _model_options(provider, tier, installed))
    return {"model": model, "effort": _ask_effort(prompt, tier)}


def _model_options(
    provider: str, tier: str, installed: list[str] | None
) -> list[tuple[str, str]]:
    if installed is not None:
        return [(model, model) for model in installed]
    if provider == "bedrock":
        return _bedrock_options(tier)
    if provider in _CLAUDE_TIER_MODELS:
        return _claude_options(provider, tier)
    return [(m, f"{m} (placeholder)") for m in _PLACEHOLDERS.get(provider, [])]


def _claude_options(provider: str, tier: str) -> list[tuple[str, str]]:
    tiers = [tier, *(other for other in _TIER_ROSTER if other != tier)]
    return [
        (_CLAUDE_TIER_MODELS[provider][name], f"Claude {_CLAUDE_NAMES[name]} 5.5")
        for name in tiers
    ]


def _bedrock_options(tier: str) -> list[tuple[str, str]]:
    presets = _BEDROCK_PRESETS + _claude_options("bedrock", tier)
    tested = [preset for preset in presets if preset[0] == _BEDROCK_TESTED[tier]]
    return tested + [preset for preset in presets if preset not in tested]


def _ask_model(prompt: Prompt, tier: str, options: list[tuple[str, str]]) -> str:
    if not options:
        return _ask_nonempty(prompt, "model id")
    question = f"Model for {_TIER_QUESTION[tier]}"
    choice = prompt.choose(question, [*options, (_OTHER, "Another model…")])
    return _ask_nonempty(prompt, "model id") if choice == _OTHER else choice


def _ask_effort(prompt: Prompt, tier: str) -> str | None:
    default = _TIER_EFFORT.get(tier, "none")
    ordered = sorted(_EFFORTS, key=lambda option: option[0] != default)
    choice = prompt.choose(f"Effort for {_TIER_QUESTION[tier]}", ordered)
    return None if choice == "none" else choice


def _ask_provider_settings(prompt: Prompt, provider: str) -> dict:
    if provider != "bedrock":
        return {}
    choice = prompt.choose("AWS region for Bedrock", [*_REGIONS, (_OTHER, "Another region…")])
    region = _ask_nonempty(prompt, "region") if choice == _OTHER else choice
    return {provider: {"params": {"region_name": region}}}


def _ask_credential(
    home: Path,
    prompt: Prompt,
    console: Console,
    provider: str,
    read_secret: Callable[[str], str],
) -> None:
    names = REGISTRY[provider].credentials
    if not names or prompt.choose("API key", _KEY_CHOICES) != "save":
        return
    name = _BEDROCK_KEY if provider == "bedrock" else names[0]
    value = read_secret(f"{name}: ").strip()
    if not value:
        console.print(Text("  No key entered; nothing saved.", style=WARN))
        return
    save_credential(home, name, value)
    _print_ok(console, f"Saved {name} to ~/.k0ntrol/credentials")


def _build_config(provider: str, tiers: dict, settings: dict) -> dict:
    config: dict = {"providers": settings} if settings else {}
    config["agents"] = {name: _entry(provider, name, tiers) for name in AGENTS}
    config["helpers"] = {name: _entry(provider, name, tiers) for name in HELPERS}
    return config


def _entry(provider: str, name: str, tiers: dict) -> dict:
    tier = _tier_of(name)
    chosen = tiers[tier]
    entry = {"provider": provider, "model": chosen["model"]}
    if chosen["effort"]:
        entry["effort"] = chosen["effort"]
    elif tier in _TIER_EFFORT:
        entry["effort"] = None  # cancel the shipped default effort
    return entry


def _write_user_config(home: Path, config: dict) -> Path:
    path = home / ".k0ntrol" / "config.yaml"
    kept = _kept_settings(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        path.unlink()
    text = yaml.safe_dump({**kept, **config}, sort_keys=False)
    path.write_text(_USER_CONFIG_HEAD + text, encoding="utf-8")
    return path


def _kept_settings(path: Path) -> dict:
    """Other settings already in the user config survive the rewrite."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if key not in _ROSTER_KINDS}


# --- output -------------------------------------------------------------------


def _print_title(console: Console, title: str, hint: str) -> None:
    body = Group(Text(title, style="bold"), Text(hint, style=DIM))
    console.print(Panel(body, box=box.ROUNDED, border_style=BORDER, expand=False))


def _print_ok(console: Console, message: str) -> None:
    console.print(Text(f"{BULLET} {message}", style=OK))


def _print_summary(console: Console, config: dict) -> None:
    table = Table(box=None, show_header=False, padding=(0, 2))
    for name, entry in _summary_entries(config):
        effort = entry.get("effort")
        model = f"{entry['model']} · {effort}" if effort else entry["model"]
        table.add_row(Text(name, style=DIM), model)
    _print_ok(console, "Saved ~/.k0ntrol/config.yaml")
    console.print(table)
    console.print(Text("  check it with /status · k0 doctor --live", style=DIM))


def _summary_entries(config: dict) -> list[tuple[str, dict]]:
    agents = [(name.upper(), config["agents"][name]) for name in AGENTS]
    return agents + [(name, config["helpers"][name]) for name in HELPERS]


def _short(error: Exception) -> str:
    return (str(error) or type(error).__name__)[:120]
