"""The provider registry: one config entry to one chat model.

Each ``agents:``/``helpers:`` entry names a k0 provider from ``REGISTRY``;
``build_chat_model`` maps it onto ``init_chat_model`` with the registry
defaults (per the provider research and ADR 0004): LM Studio is OpenAI on
``localhost:1234``, Vertex is ``google_genai`` with ``vertexai=True``, Bedrock
is ``bedrock_converse``, and Ollama gets a 64k ``num_ctx``. An entry's
``params`` override its provider's (``config.resolved_params``); secrets live
in vendor env vars and credential chains, never in config.

k0 sets no ``temperature`` and no forced ``tool_choice`` anywhere, and passes
``effort`` only when the built model's ``profile`` lists the level.
"""

import json
import os
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from k0ntrol.config import resolved_params

OLLAMA_BASE_URL = "http://localhost:11434"
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
OLLAMA_MIN_NUM_CTX = 65536


class ProviderError(Exception):
    """Raised when a config entry names no provider in the registry."""


@dataclass(frozen=True)
class ProviderSpec:
    """How one k0 provider name maps onto ``init_chat_model``."""

    name: str
    model_provider: str  # init_chat_model's model_provider
    extra: str  # pip extra that installs the partner package
    module: str  # importable module that proves the extra is installed
    credentials: tuple[str, ...] = ()  # env var names; any one set counts
    credentials_required: bool = False  # missing is an error, else a warning
    defaults: Mapping[str, Any] = field(default_factory=dict)
    api_key_env: str | None = None  # env var forwarded as the api_key kwarg


REGISTRY: dict[str, ProviderSpec] = {
    spec.name: spec
    for spec in (
        ProviderSpec("ollama", "ollama", "ollama", "langchain_ollama",
                     defaults={"num_ctx": OLLAMA_MIN_NUM_CTX}),
        ProviderSpec("lmstudio", "openai", "openai", "langchain_openai",
                     credentials=("LM_STUDIO_API_KEY",),
                     defaults={"base_url": LMSTUDIO_BASE_URL,
                               "api_key": "lm-studio",
                               "stream_usage": True},
                     api_key_env="LM_STUDIO_API_KEY"),
        ProviderSpec("openai", "openai", "openai", "langchain_openai",
                     credentials=("OPENAI_API_KEY",),
                     credentials_required=True),
        ProviderSpec("openai_compatible", "openai", "openai", "langchain_openai",
                     credentials=("OPENAI_COMPATIBLE_API_KEY",),
                     api_key_env="OPENAI_COMPATIBLE_API_KEY"),
        ProviderSpec("anthropic", "anthropic", "anthropic", "langchain_anthropic",
                     credentials=("ANTHROPIC_API_KEY",),
                     credentials_required=True),
        ProviderSpec("openrouter", "openrouter", "openrouter", "langchain_openrouter",
                     credentials=("OPENROUTER_API_KEY",),
                     credentials_required=True),
        ProviderSpec("bedrock", "bedrock_converse", "aws", "langchain_aws",
                     credentials=("AWS_ACCESS_KEY_ID", "AWS_BEARER_TOKEN_BEDROCK",
                                  "AWS_PROFILE", "AWS_DEFAULT_PROFILE")),
        ProviderSpec("vertex", "google_genai", "google", "langchain_google_genai",
                     credentials=("GOOGLE_APPLICATION_CREDENTIALS",
                                  "GOOGLE_GENAI_USE_VERTEXAI",
                                  "GOOGLE_CLOUD_PROJECT"),
                     defaults={"vertexai": True}),
    )
}

# Providers checked through a local server or an unverifiable base URL; for
# these ``model.profile.tool_calling`` is skipped in favour of server probes.
_LOCAL_PROVIDERS = frozenset({"ollama", "lmstudio", "openai_compatible"})


def provider_spec(name: str) -> ProviderSpec:
    """The registry entry for ``name`` or a ``ProviderError`` naming it."""
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise ProviderError(f"unknown provider {name!r}; expected one of: {known}") from None


def model_kwargs(cfg: dict, entry: dict, environ: Mapping[str, str] | None = None) -> dict:
    """The init kwargs for one entry: registry defaults, then config params."""
    spec = provider_spec(entry["provider"])
    provider_cfg = (cfg.get("providers") or {}).get(entry["provider"]) or {}
    kwargs = dict(spec.defaults)
    if provider_cfg.get("base_url"):
        kwargs["base_url"] = provider_cfg["base_url"]
    kwargs.update(resolved_params(cfg, entry))
    api_key = _api_key_from_env(spec, environ)
    if api_key is not None:
        kwargs["api_key"] = api_key
    return kwargs


def build_chat_model(cfg: dict, entry: dict, *,
                     init=init_chat_model,
                     environ: Mapping[str, str] | None = None):
    """Build the chat model one ``agents:``/``helpers:`` entry names.

    ``effort`` is passed with the provider's own setting only when the built
    model's ``profile`` lists the level; otherwise it is dropped (doctor warns).
    """
    spec = provider_spec(entry["provider"])
    kwargs = model_kwargs(cfg, entry, environ=environ)
    model = init(entry["model"], model_provider=spec.model_provider, **kwargs)
    effort = entry.get("effort")
    if effort and supports_effort(getattr(model, "profile", None), effort):
        kwargs.update(_effort_kwargs(entry["provider"], effort))
        model = init(entry["model"], model_provider=spec.model_provider, **kwargs)
    return model


def supports_effort(profile: Mapping | None, level: str) -> bool:
    """True when ``model.profile`` lists the configured effort level."""
    levels = (profile or {}).get("reasoning_effort_levels") or ()
    return level in levels


def dropped_effort(entry: dict, profile: Mapping | None) -> str | None:
    """The configured effort level the model ignores, or ``None``."""
    effort = entry.get("effort")
    if effort and not supports_effort(profile, effort):
        return effort
    return None


def _effort_kwargs(provider: str, level: str) -> dict:
    """The per-provider setting table from the provider-config decision (#14)."""
    if provider == "anthropic":
        return {"effort": level}
    if provider == "openrouter":
        return {"reasoning": {"effort": level}}
    if provider == "vertex":
        return {"thinking_level": level}
    if provider == "ollama":
        return {"reasoning": level}
    return {"reasoning_effort": level}


def roster_entries(cfg: dict, names=None) -> dict[str, dict]:
    """``{"agents.coder": entry, ...}`` for the roster a run uses.

    ``names`` is an iterable of ``"<kind>.<name>"`` labels; ``None`` selects
    every configured agent and helper.
    """
    wanted = set(names) if names is not None else None
    entries = {}
    for kind in ("agents", "helpers"):
        entries.update(_roster_kind(cfg, kind, wanted))
    return entries


def _roster_kind(cfg: dict, kind: str, wanted: set | None) -> dict[str, dict]:
    return {
        f"{kind}.{name}": entry
        for name, entry in (cfg.get(kind) or {}).items()
        if wanted is None or f"{kind}.{name}" in wanted
    }


def _api_key_from_env(spec: ProviderSpec, environ: Mapping[str, str] | None) -> str | None:
    if spec.api_key_env is None:
        return None
    env = os.environ if environ is None else environ
    return env.get(spec.api_key_env)


# --------------------------------------------------------------------------
# k0 doctor
#
# Free checks spend no tokens: config validity, extras, credential *names*
# (never values), `gh auth status` and the git remote, local-server
# reachability and tool capability, `model.profile.tool_calling`, Ollama
# `num_ctx` >= 64k and dropped effort. `free_checks` accepts `names` so a run
# can gate start on only the agents and helpers it uses.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """One doctor result; ``status`` is ``ok``, ``warning`` or ``error``."""

    name: str
    status: str
    detail: str


def has_errors(checks: Iterable[Check]) -> bool:
    """True when any check failed; warnings never block."""
    return any(c.status == "error" for c in checks)


def report_lines(checks: Iterable[Check]) -> list[str]:
    """Human-readable doctor output: one line per check plus the notices."""
    lines = [f"{c.status}: {c.name}: {c.detail}" for c in checks]
    lines.append("note: agent-written code runs unsandboxed as your user")
    lines.append("note: when Langfuse tracing is enabled, it receives source code")
    return lines


def free_checks(
    cfg: dict,
    *,
    names: Iterable[str] | None = None,
    root: Path | None = None,
    init: Callable = init_chat_model,
    importable: Callable[[str], bool] | None = None,
    fetch: Callable | None = None,
    run: Callable | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[Check]:
    """Every free doctor check for ``names`` (``"<kind>.<name>"``, all by default)."""
    entries = roster_entries(cfg, names)
    env = os.environ if environ is None else environ
    is_installed = importable or (lambda m: find_spec(m) is not None)
    http = fetch or _fetch_json
    runner = run or _run_cmd
    checks = [Check("config", "ok", "valid")]
    checks.extend(_provider_membership(cfg, entries))
    checks.extend(_extra_checks(entries, is_installed))
    checks.extend(_credential_checks(entries, env))
    checks.extend(_gh_checks(runner, root))
    checks.extend(_remote_checks(runner, root))
    checks.extend(_local_server_checks(entries, cfg, http))
    checks.extend(_model_checks(entries, cfg, init, env))
    return checks


def _provider_membership(cfg: dict, entries: dict[str, dict]) -> list[Check]:
    known = ", ".join(sorted(REGISTRY))
    checks = [
        Check(label, "error",
              f"unknown provider {entry.get('provider')!r}; expected one of: {known}")
        for label, entry in entries.items()
        if entry.get("provider") not in REGISTRY
    ]
    checks.extend(
        Check(f"providers.{name}", "warning",
              f"unknown provider {name!r} in the providers map")
        for name in (cfg.get("providers") or {})
        if name not in REGISTRY
    )
    return checks


def _used_providers(entries: dict[str, dict]) -> list[str]:
    return sorted({e.get("provider") for e in entries.values()} - {None})


def _extra_checks(entries: dict[str, dict], importable: Callable[[str], bool]) -> list[Check]:
    checks = []
    for provider in _used_providers(entries):
        spec = REGISTRY.get(provider)
        if spec is None:
            continue
        if importable(spec.module):
            checks.append(Check(f"provider {provider}", "ok",
                                f"extra '{spec.extra}' installed ({spec.module})"))
        else:
            checks.append(Check(f"provider {provider}", "error",
                                f"missing package {spec.module}; install k0ntrol[{spec.extra}]"))
    return checks


def _credential_checks(entries: dict[str, dict], environ: Mapping[str, str]) -> list[Check]:
    checks = []
    for provider in _used_providers(entries):
        check = _credential_check(provider, environ)
        if check is not None:
            checks.append(check)
    return checks


def _credential_check(provider: str, environ: Mapping[str, str]) -> Check | None:
    spec = REGISTRY.get(provider)
    if spec is None or not spec.credentials:
        return None
    present = [v for v in spec.credentials if environ.get(v)]
    name = f"provider {provider}"
    if present:
        return Check(name, "ok", f"credential {present[0]} set")
    if spec.credentials_required:
        return Check(name, "error",
                     f"missing credential; set {' or '.join(spec.credentials)}")
    names = ", ".join(spec.credentials)
    return Check(name, "warning",
                 f"no credential in env ({names}); "
                 "the provider's credential chain may still supply one")


def _gh_checks(run: Callable, root: Path | None) -> list[Check]:
    code, output = run(["gh", "auth", "status"], root)
    if code is None:
        return [Check("gh", "error", "gh is not on PATH; install the GitHub CLI")]
    if code != 0:
        tail = output.strip().splitlines()[-1] if output.strip() else f"exit {code}"
        return [Check("gh", "error", f"gh auth status failed: {tail}")]
    return [Check("gh", "ok", "authenticated")]


def _remote_checks(run: Callable, root: Path | None) -> list[Check]:
    code, _ = run(["git", "remote", "get-url", "origin"], root)
    if code == 0:
        return [Check("remote", "ok", "remote 'origin' configured")]
    return [Check("remote", "error",
                  "no git remote 'origin'; a run cannot push its change")]


def _local_server_checks(entries: dict[str, dict], cfg: dict, fetch: Callable) -> list[Check]:
    checks = []
    for provider in _used_providers(entries):
        if provider == "ollama":
            models = _used_models(entries, "ollama")
            checks.extend(_ollama_checks(_base_url(cfg, "ollama", OLLAMA_BASE_URL),
                                         models, fetch))
        elif provider == "lmstudio":
            models = _used_models(entries, "lmstudio")
            checks.extend(_lmstudio_checks(_base_url(cfg, "lmstudio", LMSTUDIO_BASE_URL),
                                           models, fetch))
    return checks


def _ollama_checks(base_url: str, models: list[str], fetch: Callable) -> list[Check]:
    try:
        fetch(f"{base_url}/api/version")
    except Exception as error:
        return [Check("provider ollama", "error",
                      f"not reachable at {base_url}: {_short(error)}")]
    checks = [Check("provider ollama", "ok", f"reachable at {base_url}")]
    for model in models:
        checks.append(_ollama_model_check(base_url, model, fetch))
    return checks


def _ollama_model_check(base_url: str, model: str, fetch: Callable) -> Check:
    try:
        info = fetch(f"{base_url}/api/show", {"model": model})
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return Check("provider ollama", "error",
                         f"model {model!r} is not pulled")
        return Check("provider ollama", "error",
                     f"/api/show for {model!r} failed: {_short(error)}")
    except Exception as error:
        return Check("provider ollama", "error",
                     f"/api/show for {model!r} failed: {_short(error)}")
    capabilities = info.get("capabilities")
    if capabilities is None:
        return Check("provider ollama", "warning",
                     f"tool-calling capability of {model!r} is unknown")
    if "tools" in capabilities:
        return Check("provider ollama", "ok", f"{model} can call tools")
    return Check("provider ollama", "error", f"{model} cannot call tools")


def _lmstudio_checks(base_url: str, models: list[str], fetch: Callable) -> list[Check]:
    try:
        data = fetch(f"{base_url}/models")
    except Exception as error:
        return [Check("provider lmstudio", "error",
                      f"not reachable at {base_url}: {_short(error)}")]
    checks = [Check("provider lmstudio", "ok", f"reachable at {base_url}")]
    loaded = {item.get("id") for item in data.get("data", [])}
    for model in models:
        if model in loaded:
            checks.append(Check("provider lmstudio", "ok", f"{model} is loaded"))
        else:
            checks.append(Check("provider lmstudio", "error",
                                f"model {model!r} is not loaded in LM Studio"))
    return checks


def _model_checks(entries: dict[str, dict], cfg: dict,
                  init: Callable, environ: Mapping[str, str]) -> list[Check]:
    checks = []
    for label, entry in entries.items():
        provider = entry.get("provider")
        if provider not in REGISTRY:
            continue
        model = _build_or_report(checks, label, cfg, entry, init, environ)
        if model is not None:
            checks.extend(_profile_checks(label, provider, entry, model))
        checks.extend(_num_ctx_checks(label, provider, cfg, entry))
    return checks


def _build_or_report(checks: list[Check], label: str, cfg: dict,
                     entry: dict, init: Callable, environ: Mapping[str, str]):
    try:
        return build_chat_model(cfg, entry, init=init, environ=environ)
    except (ImportError, ModuleNotFoundError):
        return None  # the extra check already reports the missing package
    except Exception as error:
        checks.append(Check(label, "error", f"cannot build model: {_short(error)}"))
        return None


def _profile_checks(label: str, provider: str, entry: dict, model) -> list[Check]:
    profile = getattr(model, "profile", None)
    checks = []
    calling = _tool_calling_check(label, provider, profile)
    if calling is not None:
        checks.append(calling)
    dropped = dropped_effort(entry, profile)
    if dropped:
        checks.append(Check(label, "warning",
                            f"effort {dropped!r} dropped; the model does not "
                            "list that level"))
    return checks


def _tool_calling_check(label: str, provider: str, profile: Mapping | None) -> Check | None:
    # Ollama's /api/show probe is authoritative; no profile warning for it.
    if provider == "ollama":
        return None
    if provider in _LOCAL_PROVIDERS:
        return Check(label, "warning",
                     "tool calling cannot be verified behind this provider; "
                     "'k0 doctor --live' proves it")
    tool_calling = (profile or {}).get("tool_calling")
    if tool_calling is True:
        return Check(label, "ok", "model profile can call tools")
    if tool_calling is False:
        return Check(label, "error", "model profile reports no tool calling")
    return Check(label, "warning",
                 "tool-calling ability is unknown for this model")


def _num_ctx_checks(label: str, provider: str, cfg: dict, entry: dict) -> list[Check]:
    if provider != "ollama":
        return []
    num_ctx = model_kwargs(cfg, entry).get("num_ctx")
    if isinstance(num_ctx, int) and num_ctx < OLLAMA_MIN_NUM_CTX:
        return [Check(label, "warning",
                      f"num_ctx {num_ctx} is below {OLLAMA_MIN_NUM_CTX}; "
                      "agent prompts will truncate")]
    return []


def _used_models(entries: dict[str, dict], provider: str) -> list[str]:
    return sorted({e["model"] for e in entries.values()
                   if e.get("provider") == provider})


def _base_url(cfg: dict, provider: str, default: str) -> str:
    provider_cfg = (cfg.get("providers") or {}).get(provider) or {}
    return provider_cfg.get("base_url") or default


@tool
def _ping() -> str:
    """Confirm the model can call tools. Always call it when asked."""
    return "pong"


def live_checks(
    cfg: dict,
    *,
    names: Iterable[str] | None = None,
    init: Callable = init_chat_model,
    environ: Mapping[str, str] | None = None,
) -> list[Check]:
    """`--live`: one tiny tool-call round trip per distinct (provider, model).

    Tools are bound but ``tool_choice`` is never forced; a model that answers
    in prose instead of calling ``_ping`` is an error.
    """
    entries = roster_entries(cfg, names)
    seen = {}
    for label, entry in entries.items():
        seen.setdefault((entry.get("provider"), entry.get("model")), (label, entry))
    return [_live_check(label, entry, cfg, init, environ)
            for (label, entry) in seen.values()]


def _live_check(label, entry, cfg, init, environ) -> Check:
    pair = f"{entry.get('provider')}:{entry.get('model')}"
    try:
        model = build_chat_model(cfg, entry, init=init, environ=environ)
        reply = model.bind_tools([_ping]).invoke(
            [HumanMessage(content="Call the `_ping` tool.")])
    except Exception as error:
        return Check(label, "error", f"{pair} live call failed: {_short(error)}")
    if getattr(reply, "tool_calls", None):
        return Check(label, "ok", f"{pair} made a tool call")
    return Check(label, "error", f"{pair} answered without a tool call")


def _short(error: Exception) -> str:
    text = str(error) or type(error).__name__
    return text[:200]


def _run_cmd(argv: list[str], root: Path | None) -> tuple[int | None, str]:
    """Run a check command as the user (unscrubbed env, short timeout)."""
    try:
        proc = subprocess.run(argv, cwd=root, capture_output=True, text=True,
                              timeout=15, check=False)
    except OSError:
        return (None, f"cannot run {argv[0]}")
    except subprocess.TimeoutExpired:
        return (None, f"{argv[0]} timed out")
    return (proc.returncode, proc.stdout + proc.stderr)


def _fetch_json(url: str, payload: Mapping | None = None, timeout: float = 5.0):
    """GET ``url`` (or POST ``payload`` as JSON) and decode the JSON body."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())
