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


def test_no_args_prints_help_and_returns_zero(capsys):
    assert main([]) == 0
    assert "usage: k0" in capsys.readouterr().out
