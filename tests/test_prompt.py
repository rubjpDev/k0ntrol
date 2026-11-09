import io
from pathlib import Path

import pytest
from rich.console import Console

from k0ntrol.frontends import prompt as prompt_module
from k0ntrol.frontends.prompt import (
    Prompt,
    complete_command,
    decode_key,
    digit_choice,
    next_index,
    parse_choice,
    render_welcome,
    terminal_key_reader,
)

COMMANDS = {"/full": "work item to PR", "/fast": "quick change", "/help": "list"}
OPTIONS = [("ok", "Approve"), ("redo", "Send back with feedback")]


def make_console(color_system=None):
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        record=True,
        width=100,
        force_terminal=False,
        color_system=color_system,
    )
    return console, buffer


def lines_input(*answers):
    pending = list(answers)
    prompts = []

    def read(prompt):
        prompts.append(prompt)
        answer = pending.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    read.prompts = prompts
    return read


def keys(*sequence):
    pending = list(sequence)
    return lambda: pending.pop(0)


def make_prompt(console, **kwargs):
    kwargs.setdefault("tty", False)
    return Prompt(console, commands=COMMANDS, **kwargs)


def test_welcome_prints_banner_and_tips():
    console, _ = make_console()
    render_welcome(console, version="0.1.0", provider="ollama", cwd="/w", color=False)
    text = console.export_text()
    assert "k0ntrol v0.1.0" in text and "ollama" in text and "/w" in text
    assert "Just type to ask about this repo · /help for commands" in text
    assert "1. type a question: k0 answers and remembers the conversation" in text
    assert "2. #<issue> takes a work item to a pull request" in text


def test_complete_command_matches_prefix_only_for_slash():
    assert complete_command("/f", COMMANDS) == ["/fast", "/full"]
    assert complete_command("/", COMMANDS) == ["/fast", "/full", "/help"]
    assert complete_command("f", COMMANDS) == []
    assert complete_command("", COMMANDS) == []


def test_next_index_wraps_and_ignores_other_keys():
    assert next_index(0, "up", 3) == 2
    assert next_index(2, "down", 3) == 0
    assert next_index(0, "k", 3) == 2
    assert next_index(1, "j", 3) == 2
    assert next_index(1, "x", 3) == 1


def test_digit_choice_is_bounded_by_option_count():
    assert digit_choice("1", 2) == 0
    assert digit_choice("2", 2) == 1
    assert digit_choice("3", 2) is None
    assert digit_choice("0", 2) is None
    assert digit_choice("enter", 2) is None


def test_parse_choice_takes_key_or_number():
    assert parse_choice(" OK ", OPTIONS) == "ok"
    assert parse_choice("2", OPTIONS) == "redo"
    assert parse_choice("REDO", OPTIONS) == "redo"
    assert parse_choice("maybe", OPTIONS) is None
    assert parse_choice("", OPTIONS) is None


def test_decode_key():
    assert decode_key("\x1b[A") == "up"
    assert decode_key("\x1bOB") == "down"
    assert decode_key("\x1b[C") == ""
    assert decode_key("\r") == "enter"
    assert decode_key("\n") == "enter"
    assert decode_key("7") == "7"
    with pytest.raises(KeyboardInterrupt):
        decode_key("\x03")


def test_read_key_decodes_the_raw_sequence(monkeypatch):
    monkeypatch.setattr(prompt_module, "_read_sequence", lambda: "\x1b[B")
    assert prompt_module.read_key() == "down"


def test_key_pending_asks_select(monkeypatch):
    monkeypatch.setattr(prompt_module.select, "select", lambda *a: ([3], [], []))
    assert prompt_module._key_pending(3) is True
    monkeypatch.setattr(prompt_module.select, "select", lambda *a: ([], [], []))
    assert prompt_module._key_pending(3) is False


def test_terminal_key_reader_needs_termios(monkeypatch):
    assert terminal_key_reader() is prompt_module.read_key
    monkeypatch.setattr(prompt_module, "termios", None)
    assert terminal_key_reader() is None


def test_read_non_tty_is_plain():
    console, buffer = make_console()
    read = lines_input("hello")
    assert make_prompt(console, read_input=read).read() == "hello"
    assert read.prompts == ["› "]
    assert buffer.getvalue() == ""


def test_read_tty_draws_frame_and_returns_cursor():
    console, buffer = make_console()
    read = lines_input("/full #3")
    assert make_prompt(console, tty=True, read_input=read).read() == "/full #3"
    assert read.prompts == ["❯ "]
    out = buffer.getvalue()
    assert out.count("─" * 100) == 2
    assert "? for shortcuts" in out
    # after Enter the frame collapses to the submitted line
    assert out.index("\x1b[3A") < out.index("\x1b[2A")
    assert out.endswith("❯ /full #3\n")


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt()])
def test_read_tty_propagates_and_restores_cursor(error):
    console, buffer = make_console()
    prompt = make_prompt(console, tty=True, read_input=lines_input(error))
    with pytest.raises(type(error)):
        prompt.read()
    assert buffer.getvalue().endswith("\x1b[3B")


def test_read_tty_colours_the_readline_prompt():
    console, _ = make_console(color_system="truecolor")
    read = lines_input("x")
    make_prompt(console, tty=True, read_input=read).read()
    assert read.prompts == ["\001\x1b[38;2;94;234;212m\002❯ \001\x1b[0m\002"]


def test_tty_is_detected_when_not_given(monkeypatch):
    monkeypatch.setattr(prompt_module, "_is_tty", lambda: False)
    console, _ = make_console()
    read = lines_input("a")
    Prompt(console, commands=COMMANDS, read_input=read).read()
    assert read.prompts == ["› "]


def test_is_tty_needs_both_streams(monkeypatch):
    monkeypatch.setattr(prompt_module.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(prompt_module.sys.stdout, "isatty", lambda: False, raising=False)
    assert prompt_module._is_tty() is False


def test_ask_text_plain_and_tty():
    console, _ = make_console()
    plain = lines_input("why")
    assert make_prompt(console, read_input=plain).ask_text("feedback") == "why"
    assert plain.prompts == ["feedback › "]
    fancy = lines_input("why")
    make_prompt(console, tty=True, read_input=fancy).ask_text("feedback")
    assert fancy.prompts == ["feedback ❯ "]


def test_choose_by_line_accepts_key_or_number_and_reasks():
    console, _ = make_console()
    read = lines_input("nope", " REDO ")
    assert make_prompt(console, read_input=read).choose("Approve?", OPTIONS) == "redo"
    out = console.export_text()
    assert "Approve? [ok/redo]" in out
    assert "Answer ok/redo or 1-2." in out
    number = lines_input("1")
    assert make_prompt(console, read_input=number).choose("Q", OPTIONS) == "ok"


def test_choose_by_line_passes_eof_up():
    console, _ = make_console()
    prompt = make_prompt(console, read_input=lines_input(EOFError()))
    with pytest.raises(EOFError):
        prompt.choose("Q", OPTIONS)


def test_choose_without_key_reader_falls_back_to_line():
    console, _ = make_console()
    prompt = make_prompt(console, tty=True, read_input=lines_input("2"))
    assert prompt.choose("Q", OPTIONS) == "redo"


def test_choose_with_keys_arrows_then_enter():
    console, buffer = make_console()
    prompt = make_prompt(console, tty=True, read_key=keys("down", "enter"))
    assert prompt.choose("Approve?", OPTIONS) == "redo"
    out = buffer.getvalue()
    assert "Approve?" in out
    assert "❯ 1. Approve" in out and "  2. Send back with feedback" in out
    assert "❯ 2. Send back with feedback" in out
    assert "↑/↓ to move · enter to select · 1-9 to pick" in out
    assert "\x1b[3A" in out
    # once chosen, the menu collapses to the answer, like Claude Code
    assert out.rstrip().endswith("❯ Send back with feedback")


def test_choose_with_keys_enter_takes_first_and_digit_picks():
    console, _ = make_console()
    first = make_prompt(console, tty=True, read_key=keys("enter"))
    assert first.choose("Q", OPTIONS) == "ok"
    digit = make_prompt(console, tty=True, read_key=keys("x", "2"))
    assert digit.choose("Q", OPTIONS) == "redo"


def test_choose_with_keys_propagates_ctrl_c():
    console, _ = make_console()

    def interrupt():
        raise KeyboardInterrupt

    prompt = make_prompt(console, tty=True, read_key=interrupt)
    with pytest.raises(KeyboardInterrupt):
        prompt.choose("Q", OPTIONS)


def test_help_table_aligns_commands_and_lists_shortcuts():
    console, _ = make_console()
    make_prompt(console).help_table()
    lines = console.export_text().splitlines()
    assert "  /fast  quick change" in lines
    assert "  /help  list" in lines
    assert "ctrl-c pauses a run" in lines[-1]
    assert "tab completes /commands" in lines[-1]


def test_complete_walks_the_matches():
    console, _ = make_console()
    prompt = make_prompt(console)
    assert prompt.complete("/f", 0) == "/fast"
    assert prompt.complete("/f", 1) == "/full"
    assert prompt.complete("/f", 2) is None


class FakeReadline:
    __doc__ = "GNU readline"

    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, args))
            if self.fail and name.endswith("_history_file"):
                raise FileNotFoundError(args[0])

        return record


class FakeLibedit(FakeReadline):
    __doc__ = "libedit readline"


def test_history_is_loaded_limited_and_saved(monkeypatch, tmp_path):
    fake = FakeReadline()
    monkeypatch.setattr(prompt_module, "readline", fake)
    path = tmp_path / "sub" / "history"
    console, _ = make_console()
    prompt = make_prompt(console, history=path, tty=True, read_input=lines_input("x"))
    prompt.read()
    names = [name for name, _ in fake.calls]
    assert ("set_history_length", (1000,)) in fake.calls
    assert ("read_history_file", (str(path),)) in fake.calls
    assert ("write_history_file", (str(path),)) in fake.calls
    assert ("parse_and_bind", ("tab: complete",)) in fake.calls
    assert names.index("read_history_file") < names.index("write_history_file")
    assert path.parent.is_dir()


def test_libedit_binds_tab_differently(monkeypatch):
    fake = FakeLibedit()
    monkeypatch.setattr(prompt_module, "readline", fake)
    console, _ = make_console()
    make_prompt(console)
    assert ("parse_and_bind", ("bind ^I rl_complete",)) in fake.calls


def test_missing_history_file_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_module, "readline", FakeReadline(fail=True))
    console, _ = make_console()
    prompt = make_prompt(
        console, history=tmp_path / "none", tty=True, read_input=lines_input("x")
    )
    assert prompt.read() == "x"


def test_without_readline_everything_still_works(monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_module, "readline", None)
    console, _ = make_console()
    prompt = make_prompt(
        console, history=tmp_path / "h", tty=True, read_input=lines_input("x")
    )
    assert prompt.read() == "x"
    assert not Path(tmp_path / "h").exists()
