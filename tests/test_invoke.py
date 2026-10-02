import json
import stat
import sys
from pathlib import Path

import pytest

from k0ntrol.backends.invoke import BackendError, invoke
from k0ntrol.modespec import ModeSpec


MODE = ModeSpec(
    name="bulk_reader",
    instructions="You are a precise code analyst.",
    model="gpt-5.6-luna[effort=high,fast=false]",
    temperature=0.2,
    one_shot=True,
)


def write_agent(path: Path, body: str) -> None:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_invoke_returns_result_and_sends_instructions_with_model(tmp_path, monkeypatch):
    argv_file = tmp_path / "argv.json"
    agent = tmp_path / "cursor-agent"
    write_agent(
        agent,
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['K0_ARGV']).write_text(json.dumps(sys.argv[1:]))\n"
        "print(json.dumps({'type':'result','subtype':'success','is_error':False,"
        "'result':'hello bullets'}))\n",
    )
    monkeypatch.setenv("K0_CURSOR_AGENT", str(agent))
    monkeypatch.setenv("K0_ARGV", str(argv_file))
    monkeypatch.chdir(tmp_path)
    assert invoke(MODE, "what is x?") == "hello bullets"
    argv = json.loads(argv_file.read_text())
    assert argv == [
        "-p",
        "--output-format",
        "json",
        "--model",
        "gpt-5.6-luna[effort=high,fast=false]",
        "You are a precise code analyst.\n\nwhat is x?",
    ]


def test_nonzero_exit_raises_backend_error_with_stderr(tmp_path, monkeypatch):
    agent = tmp_path / "cursor-agent"
    write_agent(
        agent,
        "import sys\n"
        "sys.stderr.write('Not authenticated\\n')\n"
        "raise SystemExit(1)\n",
    )
    monkeypatch.setenv("K0_CURSOR_AGENT", str(agent))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(BackendError, match="Not authenticated"):
        invoke(MODE, "q")


def test_is_error_payload_raises_even_on_exit_zero(tmp_path, monkeypatch):
    agent = tmp_path / "cursor-agent"
    write_agent(
        agent,
        "import json\n"
        "print(json.dumps({'type':'result','is_error':True,'result':'boom'}))\n",
    )
    monkeypatch.setenv("K0_CURSOR_AGENT", str(agent))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(BackendError, match="boom"):
        invoke(MODE, "q")
