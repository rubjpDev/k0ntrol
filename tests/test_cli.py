import pytest

from k0ntrol import __version__
from k0ntrol.cli import main


def test_version_flag_prints_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_help_flag_shows_k0_usage(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "usage: k0" in capsys.readouterr().out


def test_no_args_opens_a_session(capsys, monkeypatch):
    answers = iter(["/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "usage: k0" not in captured.out
    assert captured.err == ""


def test_no_args_config_error_does_not_open_a_session(capsys, monkeypatch):
    from k0ntrol.config import ConfigError

    def boom(path):
        raise ConfigError("backend: must be cursor")

    monkeypatch.setattr("k0ntrol.cli.load_config", boom)

    def refuse_input(prompt=""):
        raise AssertionError("session opened")

    monkeypatch.setattr("builtins.input", refuse_input)
    assert main([]) == 1
    assert capsys.readouterr().err == "config error: backend: must be cursor\n"
