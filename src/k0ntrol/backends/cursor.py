import os
import re
import shutil
import subprocess
from pathlib import Path


def _resolve_agent_binary() -> str | None:
    env_bin = os.environ.get("K0_CURSOR_AGENT")
    if env_bin:
        path = Path(env_bin)
        return str(path) if path.is_file() else None
    found = shutil.which("cursor-agent")
    return found if found and Path(found).is_file() else None


def _check_binary() -> str | None:
    return _resolve_agent_binary()


def _check_login(exe: str) -> str | None:
    proc = subprocess.run([exe, "status"], capture_output=True, text=True, check=False)
    combined = f"{proc.stdout}\n{proc.stderr}"
    if proc.returncode != 0 or re.search(r"(?i)not authenticated|not logged in", combined):
        return "cursor-agent: not logged in"
    return None


def _check_models(exe: str, cfg: dict) -> list[str]:
    proc = subprocess.run([exe, "--list-models"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.strip()
        msg = f"cursor-agent: --list-models failed {err}".strip()
        return [msg]

    stdout = proc.stdout
    problems: list[str] = []
    agents = cfg.get("agents", {})
    for agent_name, agent_cfg in agents.items():
        raw_model = agent_cfg.get("model", "")
        bare_id = raw_model.split("[", 1)[0]
        if bare_id not in stdout:
            problems.append(f"{agent_name}: {bare_id}")
    return problems


def cursor_problems(cfg: dict) -> list[str]:
    exe = _check_binary()
    if not exe:
        return ["cursor-agent: not on PATH"]

    login_problem = _check_login(exe)
    if login_problem:
        return [login_problem]

    return _check_models(exe, cfg)
