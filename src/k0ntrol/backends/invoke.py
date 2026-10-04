import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from k0ntrol.modespec import ModeSpec


class BackendError(Exception):
    pass


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


def _record_event(data: dict, elapsed_ms: int) -> None:
    mem_dir = Path.cwd() / ".k0-mem"
    mem_dir.mkdir(parents=True, exist_ok=True)
    result = data["result"]
    event: dict[str, int] = {"ms": elapsed_ms, "chars": len(result)}
    tokens = _token_count(data.get("usage"))
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
    cmd = _build_command(mode, prompt)
    proc, elapsed_ms = _run_subprocess(cmd)
    data = _parse_payload(proc.stdout)
    _record_event(data, elapsed_ms)
    return data["result"]
