import json
import os
import shutil
import subprocess
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


def _parse_payload(stdout: str) -> str:
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

    return result


def _run_subprocess(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise BackendError(f"failed to run cursor-agent: {exc}") from exc

    if proc.returncode != 0:
        err = proc.stderr.strip() or f"cursor-agent exited with code {proc.returncode}"
        raise BackendError(err)

    return proc


def invoke(mode: ModeSpec, prompt: str) -> str:
    cmd = _build_command(mode, prompt)
    proc = _run_subprocess(cmd)
    return _parse_payload(proc.stdout)
