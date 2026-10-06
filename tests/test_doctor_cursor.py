import stat
import sys
from pathlib import Path

import pytest

from k0ntrol.cli import main


def write_agent(path: Path, body: str) -> None:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


def test_doctor_names_a_missing_binary(project, monkeypatch, capsys):
    monkeypatch.setenv("K0_CURSOR_AGENT", str(project / "missing-cursor-agent"))
    assert main(["doctor"]) == 1
    err = capsys.readouterr().err
    assert "cursor-agent: not on PATH" in err
    assert "threshold_lines" not in err


def test_doctor_names_a_failed_login(project, monkeypatch, capsys):
    agent = project / "cursor-agent"
    write_agent(
        agent,
        "import sys\n"
        "if sys.argv[1:] == ['status']:\n"
        "    sys.stderr.write('Not authenticated\\n')\n"
        "    raise SystemExit(1)\n"
        "raise SystemExit(0)\n",
    )
    monkeypatch.setenv("K0_CURSOR_AGENT", str(agent))
    assert main(["doctor"]) == 1
    assert "cursor-agent: not logged in" in capsys.readouterr().err


def test_doctor_names_each_config_model_missing_from_list_models(project, monkeypatch, capsys):
    agent = project / "cursor-agent"
    write_agent(
        agent,
        "import sys\n"
        "if sys.argv[1:] == ['status']:\n"
        "    print('Logged in')\n"
        "    raise SystemExit(0)\n"
        "if sys.argv[1:] == ['--list-models']:\n"
        "    print('gpt-5.6-luna')\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit('unexpected')\n",
    )
    monkeypatch.setenv("K0_CURSOR_AGENT", str(agent))
    assert main(["doctor"]) == 1
    err = capsys.readouterr().err
    assert "spec" in err
    assert "claude-opus-5-5" in err


def test_doctor_passes_when_binary_login_and_models_are_present(project, monkeypatch, capsys):
    agent = project / "cursor-agent"
    write_agent(
        agent,
        "import sys\n"
        "if sys.argv[1:] == ['status']:\n"
        "    print('Logged in')\n"
        "    raise SystemExit(0)\n"
        "if sys.argv[1:] == ['--list-models']:\n"
        "    print('claude-opus-5-5-high')\n"
        "    print('gpt-5.6-luna-max')\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit('unexpected')\n",
    )
    monkeypatch.setenv("K0_CURSOR_AGENT", str(agent))
    assert main(["doctor"]) == 0
    assert "threshold_lines: 350" in capsys.readouterr().out


def test_doctor_rejects_a_model_id_that_is_only_a_prefix(project, monkeypatch, capsys):
    (project / ".k0-mem").mkdir()
    (project / ".k0-mem" / "config.yaml").write_text(
        "agents:\n"
        "  spec: {model: claude-opus-5-5}\n"
        "  tester: {model: gpt-5.6-luna-high}\n"
        "  coder: {model: gpt-5.6-luna-high}\n"
        "  validator: {model: gpt-5.6-luna-high}\n"
        "  bulk_reader: {model: gpt-5.6-luna-high}\n"
    )
    agent = project / "cursor-agent"
    write_agent(
        agent,
        "import sys\n"
        "if sys.argv[1:] == ['status']:\n"
        "    print('Logged in')\n"
        "    raise SystemExit(0)\n"
        "if sys.argv[1:] == ['--list-models']:\n"
        "    print('claude-opus-5-5-high - Claude')\n"
        "    print('gpt-5.6-luna-high - Luna')\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit('unexpected')\n",
    )
    monkeypatch.setenv("K0_CURSOR_AGENT", str(agent))
    assert main(["doctor"]) == 1
    err = capsys.readouterr().err
    assert "spec: claude-opus-5-5" in err
    assert "tester:" not in err
