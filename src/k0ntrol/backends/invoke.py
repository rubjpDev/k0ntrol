import contextvars
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from k0ntrol.modespec import ModeSpec


class BackendError(Exception):
    pass


_EVENT_CONTEXT: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "k0ntrol_event_context",
    default={"run": "", "step": ""},
)


def set_event_context(run: str, step: str) -> None:
    _EVENT_CONTEXT.set({"run": run, "step": step})


def current_event_context() -> dict[str, str]:
    return dict(_EVENT_CONTEXT.get())


def _resolve_agent_binary() -> str:
    env_bin = os.environ.get("K0_CURSOR_AGENT")
    if env_bin:
        return env_bin
    found = shutil.which("cursor-agent")
    if found:
        return found
    raise BackendError("cursor-agent: not on PATH")


def _build_prompt(mode: ModeSpec, prompt: str) -> str:
    stripped = mode.instructions.strip()
    if stripped:
        return f"{stripped}\n\n{prompt}"
    return prompt


def _build_command(mode: ModeSpec, prompt: str) -> list[str]:
    agent_bin = _resolve_agent_binary()
    cli_prompt = _build_prompt(mode, prompt)
    return [agent_bin, "-p", "--output-format", "json", "--model", mode.model, cli_prompt]


def _parse_payload(stdout: str) -> dict:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise BackendError(f"unparseable stdout from cursor-agent: {exc}") from exc

    if not isinstance(data, dict):
        raise BackendError("cursor-agent output must be a JSON object")

    if data.get("is_error") is True:
        err_msg = data.get("result") or "cursor-agent returned is_error=True"
        raise BackendError(str(err_msg))

    result = data.get("result")
    if not isinstance(result, str):
        raise BackendError("missing or invalid result in cursor-agent output")

    return data


def _token_count(usage: object) -> int | None:
    if not isinstance(usage, dict):
        return None
    incoming = usage.get("input_tokens", usage.get("inputTokens"))
    outgoing = usage.get("output_tokens", usage.get("outputTokens"))
    if isinstance(incoming, int) and isinstance(outgoing, int):
        return incoming + outgoing
    return None


def _record_event(
    mode: ModeSpec,
    elapsed_ms: int,
    *,
    result: str = "",
    ok: bool,
    err: str = "",
    data: dict | None = None,
) -> None:
    mem_dir = Path.cwd() / ".k0-mem"
    mem_dir.mkdir(parents=True, exist_ok=True)
    context = current_event_context()
    event: dict[str, object] = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run": context["run"],
        "step": context["step"],
        "mode": mode.name,
        "model": mode.model,
        "ms": elapsed_ms,
        "chars": len(result),
        "ok": ok,
        "err": err,
    }
    tokens = _token_count(data.get("usage")) if data is not None else None
    if tokens is not None:
        event["tokens"] = tokens

    events_file = mem_dir / "events.jsonl"
    with events_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, separators=(",", ":")) + "\n")


def _run_subprocess(cmd: list[str]) -> tuple[subprocess.CompletedProcess[str], int]:
    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise BackendError(f"failed to run cursor-agent: {exc}") from exc

    elapsed_ms = max(0, int((time.monotonic() - start) * 1000))

    if proc.returncode != 0:
        err = proc.stderr.strip() or f"cursor-agent exited with code {proc.returncode}"
        raise BackendError(err)

    return proc, elapsed_ms


def invoke(mode: ModeSpec, prompt: str) -> str:
    start = time.monotonic()
    try:
        cmd = _build_command(mode, prompt)
        proc, elapsed_ms = _run_subprocess(cmd)
        data = _parse_payload(proc.stdout)
    except BackendError as error:
        elapsed_ms = max(0, int((time.monotonic() - start) * 1000))
        _record_event(mode, elapsed_ms, ok=False, err=str(error))
        raise
    _record_event(mode, elapsed_ms, result=data["result"], ok=True, data=data)
    return data["result"]
