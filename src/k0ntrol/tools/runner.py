"""The one command runner behind ``run_tests``, ``test_cmd`` and ``gates``.

ADR 0006: every command k0 runs for agents or gates gets a scrubbed
environment (provider and tracker vendor variables and secret-shaped names
removed), a ``test_timeout`` (a timeout counts as red), and truncated output
before an agent sees it. Agent-written code still runs unsandboxed.
"""

import os
import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# Chars of stdout+stderr kept, from the tail: pytest failures print last.
OUTPUT_LIMIT = 10_000

_SECRET_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_CREDENTIALS")

# Vendor variables the suffixes miss: cloud credential chains, provider
# endpoints and tracker settings an agent's test run must not inherit.
_VENDOR_ENV_VARS = frozenset({
    "AWS_ACCESS_KEY_ID",
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "GOOGLE_CLOUD_PROJECT",
    "GCLOUD_PROJECT",
    "ANTHROPIC_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENROUTER_BASE_URL",
    "OLLAMA_HOST",
    "LANGFUSE_HOST",
})


def scrubbed_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy ``environ`` (default ``os.environ``) minus secrets and vendor vars."""
    source = environ if environ is not None else os.environ
    return {
        name: value
        for name, value in source.items()
        if name not in _VENDOR_ENV_VARS and not name.endswith(_SECRET_SUFFIXES)
    }


# A cached ``.pyc`` is trusted on (mtime in seconds, size): an edit landing
# in the same second as an earlier run, with the same size, would be
# tested as the old code. Never writing bytecode keeps every run fresh.
_FRESH_IMPORTS_ENV = {"PYTHONDONTWRITEBYTECODE": "1"}


@dataclass(frozen=True)
class CommandResult:
    """What one command run produced. ``timed_out`` counts as red."""

    argv: tuple[str, ...]
    returncode: int | None
    output: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_command(
    cmd: str | list[str],
    *,
    root: Path,
    timeout: float,
    extra_args: list[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> CommandResult:
    """Run one argv list under ``root`` with a scrubbed env and a timeout."""
    argv = _argv(cmd, extra_args)
    try:
        proc = subprocess.run(
            argv,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**scrubbed_env(environ), **_FRESH_IMPORTS_ENV},
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return _timeout_result(argv, error, timeout)
    except OSError as error:
        return CommandResult(tuple(argv), None, f"cannot run {argv[0]}: {error}")
    return CommandResult(
        tuple(argv), proc.returncode, _truncate(proc.stdout + proc.stderr)
    )


def _argv(cmd: str | list[str], extra_args: list[str] | None) -> list[str]:
    argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    return argv + list(extra_args or [])


def _timeout_result(
    argv: list[str], error: subprocess.TimeoutExpired, timeout: float
) -> CommandResult:
    partial = _partial_output(error)
    note = f"timed out after {timeout:g}s"
    return CommandResult(tuple(argv), None, _truncate(partial + note), timed_out=True)


def _partial_output(error: subprocess.TimeoutExpired) -> str:
    chunks = []
    for stream in (error.stdout, error.stderr):
        if isinstance(stream, bytes):
            chunks.append(stream.decode("utf-8", errors="replace"))
        elif stream:
            chunks.append(stream)
    return "".join(chunks)


def _truncate(output: str) -> str:
    if len(output) <= OUTPUT_LIMIT:
        return output
    return f"... output truncated, last {OUTPUT_LIMIT} chars kept ...\n" + output[-OUTPUT_LIMIT:]
