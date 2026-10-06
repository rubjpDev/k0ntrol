from pathlib import Path

import pytest

from k0ntrol.frontends.session import run_session


def config():
    return {
        "backend": "cursor",
        "threshold_lines": 350,
        "agents": {"bulk_reader": {"model": "gpt-secret-model"}},
    }


def forbid(mode, prompt, **kwargs):
    raise AssertionError("model called")


def test_help_stays_on_screen_and_quit_exits(tmp_path):
    lines = iter(["/help", "/fast later", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        invoke=forbid,
        bulk_read=forbid,
    )
    assert code == 0
    text = "\n".join(shown)
    assert "› /help" in text
    assert "/ask" in text
    assert "/fast" in text
    assert "/full" in text
    assert text.index("/ask") < text.index("/fast is not in 0.0.1")
    assert "\x1b[?1049h" not in text


def test_full_and_unknown_and_blank_do_not_invoke(tmp_path):
    lines = iter(["", "/full later", "/nope", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        invoke=forbid,
        bulk_read=forbid,
    )
    assert code == 0
    text = "\n".join(shown)
    assert "/full is not in 0.0.1" in text
    assert "unknown command: /nope" in text


def test_ctrl_c_and_eof_exit_zero(tmp_path):
    def interrupt():
        raise KeyboardInterrupt

    assert (
        run_session(
            interrupt,
            lambda _line: None,
            root=tmp_path,
            config=config(),
            invoke=forbid,
            bulk_read=lambda *args, **kwargs: forbid(None, None),
        )
        == 0
    )

    def end():
        raise EOFError

    assert (
        run_session(
            end,
            lambda _line: None,
            root=tmp_path,
            config=config(),
            invoke=forbid,
            bulk_read=lambda *args, **kwargs: forbid(None, None),
        )
        == 0
    )


def test_ask_prints_the_answer_and_a_missing_file_stays_in_session(tmp_path, monkeypatch):
    target = tmp_path / "app.py"
    target.write_text("x\n")

    def fake_invoke(mode, prompt):
        return "- app (file) line 1: x\n"

    lines = iter(["/ask what is line 1? @app.py", "/ask @missing.py", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        invoke=fake_invoke,
        bulk_read=forbid,
    )
    assert code == 0
    text = "\n".join(shown)
    assert "- app (file) line 1: x" in text
    assert "missing.py" in text


def test_transcript_rereads_what_was_shown(tmp_path):
    lines = iter(["/help", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        invoke=forbid,
        bulk_read=forbid,
    )
    assert code == 0
    files = list((tmp_path / ".k0-mem" / "sessions").glob("*.txt"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    for chunk in shown:
        assert chunk in text


def test_session_prints_the_banner_once(tmp_path):
    from k0ntrol import __version__

    lines = iter(["/quit"])
    shown = []
    run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        invoke=forbid,
        bulk_read=forbid,
        color=False,
    )
    text = "\n".join(shown)
    assert text.count("▄█████▙") == 1
    assert f"k0ntrol v{__version__}" in text
    assert str(tmp_path) in text
    assert "cursor" in text
    assert "gpt-secret-model" not in text
    assert "\x1b" not in text
