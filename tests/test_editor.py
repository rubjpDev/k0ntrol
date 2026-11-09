import io
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from k0ntrol.frontends import editor
from k0ntrol.frontends.editor import (
    DEFAULT_MODES,
    Cancelled,
    EditorState,
    Env,
    Geometry,
    LineEditor,
    Submitted,
    add_entry,
    apply_key,
    caret_position,
    crop,
    effective_mode,
    erase_frame,
    escape_complete,
    layout,
    load_history,
    normalize_key,
    paint,
    paint_frame,
    prefix_mode_name,
    raw_key_reader,
    read_one,
    render_box,
    sanitize_text,
    save_history,
    utf8_length,
    wrap_segments,
)
from k0ntrol.frontends.theme import ACCENT, BRAND, WARN

MODES = list(DEFAULT_MODES)
COMMANDS = {"/help": "help", "/history": "history", "/quit": "quit", "/fast": "fast"}
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def type_keys(keys: list[str], env: Env | None = None, state: EditorState | None = None):
    state = state or EditorState()
    for key in keys:
        state = apply_key(state, key, env)
        if not isinstance(state, EditorState):
            return state
    return state


def typed(text: str) -> list[str]:
    return list(text)


# --- apply_key: text -------------------------------------------------------


def test_printable_text_inserts_at_the_cursor():
    state = type_keys(typed("helo") + ["left", "l"])
    assert (state.buffer, state.cursor) == ("hello", 4)


def test_spanish_characters_insert_as_one_character_each():
    state = type_keys(typed("¿qué año?"))
    assert state.buffer == "¿qué año?"
    assert state.cursor == len("¿qué año?")


def test_paste_turns_newlines_into_spaces_and_never_submits():
    state = type_keys(["line one\nline two\r\nline three\n"])
    assert isinstance(state, EditorState)
    assert state.buffer == "line one line two line three "


def test_empty_or_unknown_input_is_ignored():
    state = type_keys(typed("ab"))
    assert apply_key(state, "\x1b[99~") == state
    assert apply_key(state, "\x1bOZ") == state
    assert apply_key(state, "\x00") == state


# --- apply_key: editing ----------------------------------------------------


@pytest.mark.parametrize("key", ["backspace", "\x7f", "\x08"])
def test_backspace_deletes_left(key):
    state = type_keys(typed("abc") + [key])
    assert (state.buffer, state.cursor) == ("ab", 2)


def test_backspace_at_start_does_nothing():
    assert type_keys(["backspace"]) == EditorState()


@pytest.mark.parametrize("key", ["delete", "\x1b[3~"])
def test_delete_removes_right(key):
    state = type_keys(typed("abc") + ["home", key])
    assert (state.buffer, state.cursor) == ("bc", 0)


def test_cursor_moves_and_clamps():
    state = type_keys(typed("abc") + ["left", "left", "left", "left"])
    assert state.cursor == 0
    state = type_keys(["right"] * 5, state=state)
    assert state.cursor == 3


@pytest.mark.parametrize("home", ["home", "ctrl_a", "\x01", "\x1b[H", "\x1bOH", "\x1b[1~"])
def test_home_keys(home):
    assert type_keys(typed("abc") + [home]).cursor == 0


@pytest.mark.parametrize("end", ["end", "ctrl_e", "\x05", "\x1b[F", "\x1bOF", "\x1b[4~"])
def test_end_keys(end):
    assert type_keys(typed("abc") + ["home", end]).cursor == 3


def test_arrow_escape_sequences_move_the_cursor():
    state = type_keys(typed("abc") + ["\x1b[D", "\x1b[D", "\x1b[C"])
    assert state.cursor == 2


def test_ctrl_u_clears_to_start():
    state = type_keys(typed("hello world") + ["left"] * 5 + ["ctrl_u"])
    assert (state.buffer, state.cursor) == ("world", 0)


def test_ctrl_w_deletes_the_previous_word():
    state = type_keys(typed("one two  three") + ["ctrl_w"])
    assert (state.buffer, state.cursor) == ("one two  ", 9)
    state = type_keys(["ctrl_w"], state=state)
    assert (state.buffer, state.cursor) == ("one ", 4)


def test_ctrl_w_keeps_the_text_after_the_cursor():
    state = type_keys(typed("alpha beta") + ["left"] * 4 + ["ctrl_w"])
    assert (state.buffer, state.cursor) == ("beta", 0)


def test_enter_submits_the_buffer():
    assert type_keys(typed("go") + ["enter"]) == Submitted("go")
    assert type_keys(typed("go") + ["\r"]) == Submitted("go")


def test_ctrl_c_cancels_as_interrupt():
    assert type_keys(["ctrl_c"]) == Cancelled(eof=False)
    assert type_keys(["\x03"]) == Cancelled(eof=False)


def test_ctrl_d_on_empty_buffer_is_eof_else_deletes_right():
    assert type_keys(["ctrl_d"]) == Cancelled(eof=True)
    state = type_keys(typed("ab") + ["home", "ctrl_d"])
    assert state.buffer == "b"


# --- apply_key: modes, tab, history ---------------------------------------


def test_shift_tab_cycles_modes_and_wraps():
    env = Env(mode_count=3)
    state = type_keys(["shift_tab"], env)
    assert state.mode_index == 1
    state = type_keys(["\x1b[Z", "shift_tab"], env, state)
    assert state.mode_index == 0


def test_tab_completes_a_unique_prefix():
    env = Env(commands=COMMANDS)
    state = type_keys(typed("/qu") + ["tab"], env)
    assert (state.buffer, state.cursor) == ("/quit", 5)


def test_tab_completes_the_common_prefix_of_several():
    env = Env(commands=COMMANDS)
    state = type_keys(typed("/h") + ["tab"], env)
    assert state.buffer == "/h"
    state = type_keys(typed("/hi") + ["tab"], env)
    assert state.buffer == "/history"
    state = type_keys(["tab"], env, EditorState(buffer="/", cursor=1))
    assert state.buffer == "/"


@pytest.mark.parametrize("text", ["hello", "/nope", "/quit now"])
def test_tab_does_nothing_without_a_match(text):
    env = Env(commands=COMMANDS)
    state = type_keys(typed(text) + ["\t"], env)
    assert state.buffer == text


def test_up_and_down_browse_history_and_keep_the_draft():
    env = Env(history=("first", "second"))
    state = type_keys(typed("draft") + ["up"], env)
    assert (state.buffer, state.history_index) == ("second", 1)
    state = type_keys(["up"], env, state)
    assert state.buffer == "first"
    state = type_keys(["up"], env, state)
    assert state.buffer == "first"
    state = type_keys(["down", "down"], env, state)
    assert (state.buffer, state.history_index, state.cursor) == ("draft", None, 5)
    assert type_keys(["down"], env, state) == state


def test_up_without_history_does_nothing():
    state = type_keys(typed("x") + ["up"])
    assert state.buffer == "x"


# --- modes by prefix -------------------------------------------------------


@pytest.mark.parametrize(
    ("buffer", "expected"),
    [
        ("/fast fix it", "fast"),
        ("/fast", "fast"),
        ("/full build", "full"),
        ("#123", "full"),
        ("#123 now", "full"),
        ("/ask why", "chat"),
        ("/fastest", None),
        ("#abc", None),
        ("hello #1", None),
        ("", None),
    ],
)
def test_prefix_mode_name(buffer, expected):
    assert prefix_mode_name(buffer) == expected


def test_effective_mode_prefix_beats_selection():
    assert effective_mode("/fast x", 0, MODES).name == "fast"
    assert effective_mode("/ask x", 2, MODES).name == "chat"
    assert effective_mode("hello", 2, MODES).name == "full"
    assert effective_mode("/fast x", 0, [MODES[0]]).name == "chat"


# --- rendering -------------------------------------------------------------


def plain(lines: list[str]) -> list[str]:
    return [ANSI.sub("", line) for line in lines]


def box(buffer: str = "", cursor: int | None = None, mode_index: int = 0, width: int = 40):
    state = EditorState(
        buffer=buffer, cursor=len(buffer) if cursor is None else cursor, mode_index=mode_index
    )
    return render_box(state, width=width, status="bedrock · 25s", modes=MODES, color=False)


def test_box_is_closed_and_every_row_has_the_same_width():
    lines = box("what does slugify do?")
    assert lines[0] == "╭─ chat " + "─" * 31 + "╮"
    assert lines[1].startswith("│ ❯ what does slugify do?")
    assert lines[1].endswith(" │")
    assert lines[2] == "╰" + "─" * 38 + "╯"
    assert lines[3] == "  bedrock · 25s"
    assert {len(line) for line in lines[:3]} == {40}


def test_empty_input_shows_the_mode_hint_as_placeholder():
    assert "ask about this repo" in box()[1]
    assert box()[1].startswith("│ ❯ ")


def test_mode_name_sits_in_the_top_border():
    assert box(mode_index=1)[0].startswith("╭─ fast ")
    assert box(mode_index=2)[0].startswith("╭─ full ")


def test_prefix_recolours_border_and_name():
    assert box("/fast x")[0].startswith("╭─ fast ")
    assert box("#12")[0].startswith("╭─ full ")
    assert box("/ask x", mode_index=1)[0].startswith("╭─ chat ")


def test_border_is_painted_with_the_mode_colour():
    state = EditorState(buffer="/fast x", cursor=7)
    lines = render_box(state, width=40, status="s", modes=MODES, color=True)
    red, green, blue = (int(WARN[i : i + 2], 16) for i in (1, 3, 5))
    assert f"\x1b[38;2;{red};{green};{blue}m╭─ fast" in lines[0]
    assert "\x1b[38;2;" in lines[1]
    chat = render_box(EditorState(), width=40, status="s", modes=MODES, color=True)
    assert f"{int(BRAND[1:3], 16)};" in chat[0]
    full = render_box(EditorState(mode_index=2), width=40, status="s", modes=MODES)
    assert f"{int(ACCENT[1:3], 16)};" in full[0]


def test_color_off_emits_no_colour_codes():
    assert all("\x1b" not in line for line in box("/fast x"))


def test_long_text_wraps_inside_the_box_and_it_grows():
    text = "x" * 50
    lines = box(text, width=40)
    assert len(lines) == 5  # top, 2 rows, bottom, status
    assert lines[1] == "│ ❯ " + "x" * 34 + " │"
    assert lines[2] == "│   " + "x" * 16 + " " * 18 + " │"
    assert {len(line) for line in lines[:4]} == {40}


def test_exactly_full_row_gets_an_empty_row_for_the_caret():
    lines = box("y" * 34, width=40)
    assert len(lines) == 5
    assert lines[2].strip("│ ") == ""
    state = EditorState(buffer="y" * 34, cursor=34)
    assert caret_position(state, 40) == (2, 4)
    mid = EditorState(buffer="y" * 34, cursor=10)
    assert len(layout(mid, 40)) == 1


def test_wide_characters_count_two_cells():
    segments = wrap_segments("日本語日本語", 6)
    assert segments == [(0, 3), (3, 6)]
    lines = box("日" * 10, width=20)
    assert {editor.cell_len(line) for line in lines[:-1]} == {20}


def test_status_is_cropped_with_an_ellipsis():
    state = EditorState()
    lines = render_box(state, width=30, status="s" * 80, modes=MODES, color=False)
    assert len(lines[-1]) == 30
    assert lines[-1].endswith("…")
    assert crop("short", 10) == "short"
    assert crop("abcdef", 4) == "abc…"
    assert crop("abc", 0) == "…" or crop("abc", 0) == ""


def test_narrow_width_is_clamped():
    lines = box("hi", width=5)
    assert len(lines[0]) == editor.MIN_WIDTH


def test_caret_position_follows_the_cursor_across_rows():
    text = "a" * 50
    assert caret_position(EditorState(buffer=text, cursor=0), 40) == (1, 4)
    assert caret_position(EditorState(buffer=text, cursor=10), 40) == (1, 14)
    assert caret_position(EditorState(buffer=text, cursor=34), 40) == (2, 4)
    assert caret_position(EditorState(buffer=text, cursor=50), 40) == (2, 20)
    assert caret_position(EditorState(), 40) == (1, 4)


def test_paint_helpers():
    assert paint("x", "#102030", False) == "x"
    assert paint("", "#102030", True) == ""
    assert paint("x", "#102030", True) == "\x1b[38;2;16;32;48mx\x1b[0m"
    assert sanitize_text("a\tb\x1b[0m\nc") == "a b[0m c"


# --- frame painting --------------------------------------------------------


def test_first_frame_has_no_move_up_and_parks_at_the_caret():
    out = paint_frame(["top", "row", "bottom", "status"], (1, 4), None)
    assert out.startswith("\x1b[?25l\x1b[2Ktop\n\x1b[2Krow\n")
    assert out.endswith("\x1b[2A\x1b[5G\x1b[?25h")


def test_redraw_moves_to_the_old_top_and_clears_leftover_rows():
    previous = Geometry(rows=5, caret_row=2)
    out = paint_frame(["top", "row", "bottom", "status"], (1, 4), previous)
    assert out.startswith("\x1b[?25l\x1b[2A\r\x1b[2Ktop")
    assert out.count("\x1b[2K") == 5  # one per old row, the extra one blank
    assert out.endswith("\x1b[3A\x1b[5G\x1b[?25h")


def test_redraw_at_the_top_row_skips_the_move_up():
    out = paint_frame(["a"], (0, 0), Geometry(rows=1, caret_row=0))
    assert out.startswith("\x1b[?25l\r")


def test_erase_frame_goes_to_the_top_and_clears_below():
    assert erase_frame(None) == ""
    assert erase_frame(Geometry(rows=4, caret_row=2)) == "\x1b[2A\r\x1b[J"
    assert erase_frame(Geometry(rows=4, caret_row=0)) == "\r\x1b[J"


# --- history files ---------------------------------------------------------


def test_add_entry_skips_blank_and_consecutive_duplicates():
    entries = add_entry([], "one")
    assert add_entry(entries, "one") == ["one"]
    assert add_entry(entries, "   ") == ["one"]
    assert add_entry(entries, "two") == ["one", "two"]
    assert add_entry(["two", "one"], "two") == ["two", "one", "two"]


def test_history_is_capped_at_the_limit():
    entries = [str(i) for i in range(editor.HISTORY_LIMIT)]
    result = add_entry(entries, "last")
    assert len(result) == editor.HISTORY_LIMIT
    assert result[-1] == "last"
    assert result[0] == "1"


def test_history_round_trips_through_a_file(tmp_path: Path):
    path = tmp_path / "nested" / "history"
    save_history(path, ["one", "two"])
    assert path.read_text() == "one\ntwo\n"
    assert load_history(path) == ["one", "two"]
    path.write_text("a\na\n\nb\n")
    assert load_history(path) == ["a", "b"]


def test_missing_or_unwritable_history_is_ignored(tmp_path: Path):
    assert load_history(tmp_path / "missing") == []
    blocker = tmp_path / "file"
    blocker.write_text("x")
    save_history(blocker / "history", ["one"])  # parent is a file: no error


# --- LineEditor ------------------------------------------------------------


def keyboard(keys: list[str]) -> Iterator[str]:
    yield from keys


def make_editor(keys, tmp_path: Path | None = None, **kwargs):
    out = io.StringIO()
    source = keyboard(keys)
    path = tmp_path / "history" if tmp_path else None
    line_editor = LineEditor(
        out,
        read_key=lambda: next(source),
        commands=COMMANDS,
        history_path=path,
        width=lambda: 40,
        color=False,
        **kwargs,
    )
    return line_editor, out


def read(line_editor: LineEditor) -> tuple[str, str]:
    return line_editor.read(status=lambda: "status line")


def test_read_returns_mode_and_text():
    line_editor, out = make_editor([*typed("what is this"), "enter"])
    assert read(line_editor) == ("chat", "what is this")
    assert "╭─ chat" in out.getvalue()


def test_submit_replaces_the_frame_with_one_dim_line():
    line_editor, out = make_editor([*typed("hi"), "enter"])
    read(line_editor)
    written = out.getvalue()
    tail = written.rsplit("\x1b[J", 1)[1]
    assert tail == "\x1b[?25h❯ hi\n"
    assert "\x1b[1A\r\x1b[J" in written  # moved to the top, then wiped


def test_prefix_decides_the_returned_mode():
    line_editor, _ = make_editor([*typed("/fast x"), "enter"])
    assert read(line_editor) == ("fast", "/fast x")
    line_editor, _ = make_editor([*typed("#12"), "enter"])
    assert read(line_editor) == ("full", "#12")


def test_shift_tab_changes_the_mode_and_it_persists_between_reads():
    line_editor, out = make_editor(["shift_tab", *typed("a"), "enter", *typed("b"), "enter"])
    assert line_editor.mode == "chat"
    assert read(line_editor) == ("fast", "a")
    assert line_editor.mode == "fast"
    assert read(line_editor) == ("fast", "b")
    assert "╭─ fast" in out.getvalue()


def test_custom_modes_are_used():
    from k0ntrol.frontends.editor import Mode

    modes = [Mode("one", "#ffffff"), Mode("two", "#000000", "hint")]
    line_editor, _ = make_editor(["shift_tab", "enter"], modes=modes)
    assert read(line_editor) == ("two", "")
    assert line_editor.mode == "two"


def test_status_is_asked_for_on_every_redraw():
    line_editor, out = make_editor([*typed("ab"), "enter"])
    calls = []

    def status() -> str:
        calls.append(1)
        return f"tick {len(calls)}"

    line_editor.read(status=status)
    assert len(calls) == 3
    assert "tick 3" in out.getvalue()


def test_ctrl_c_erases_the_frame_and_raises_keyboard_interrupt():
    line_editor, out = make_editor([*typed("x"), "ctrl_c"])
    with pytest.raises(KeyboardInterrupt):
        read(line_editor)
    assert out.getvalue().endswith("\x1b[1A\r\x1b[J\x1b[?25h")


def test_ctrl_d_on_empty_raises_eof_error():
    line_editor, out = make_editor(["ctrl_d"])
    with pytest.raises(EOFError):
        read(line_editor)
    assert "\x1b[J" in out.getvalue()


def test_reader_raising_keyboard_interrupt_also_erases_the_frame():
    def boom() -> str:
        raise KeyboardInterrupt

    out = io.StringIO()
    line_editor = LineEditor(out, read_key=boom, commands={}, width=lambda: 40, color=False)
    with pytest.raises(KeyboardInterrupt):
        read(line_editor)
    assert "\x1b[J" in out.getvalue()


def test_each_key_redraws_the_frame_from_its_top():
    line_editor, out = make_editor([*typed("ab"), "enter"])
    read(line_editor)
    assert out.getvalue().count("\x1b[?25l") == 3
    assert out.getvalue().count("\x1b[1A\r") >= 2


def test_history_is_saved_and_browsed_with_up(tmp_path: Path):
    first, _ = make_editor([*typed("one"), "enter", *typed("one"), "enter", "enter"], tmp_path)
    assert read(first) == ("chat", "one")
    assert read(first) == ("chat", "one")
    assert read(first) == ("chat", "")
    assert (tmp_path / "history").read_text() == "one\n"
    second, _ = make_editor(["up", "enter"], tmp_path)
    assert second.history == ["one"]
    assert read(second) == ("chat", "one")


def test_given_history_is_merged_with_the_file(tmp_path: Path):
    (tmp_path / "history").write_text("old\n")
    line_editor, _ = make_editor(["up", "up", "enter"], tmp_path, history=["seed"])
    assert line_editor.history == ["seed", "old"]
    assert read(line_editor) == ("chat", "seed")


def test_tab_completes_in_the_editor():
    line_editor, _ = make_editor([*typed("/qu"), "tab", "enter"])
    assert read(line_editor) == ("chat", "/quit")


def test_default_width_comes_from_the_terminal(monkeypatch):
    monkeypatch.setattr(editor.shutil, "get_terminal_size", lambda: type("S", (), {"columns": 33})())
    assert editor._terminal_width() == 33
    source = keyboard(["enter"])
    out = io.StringIO()
    line_editor = LineEditor(out, read_key=lambda: next(source), commands={}, color=False)
    read(line_editor)
    assert "╭─ chat " + "─" * 24 + "╮" in out.getvalue()


def test_read_key_with_a_session_runs_inside_it():
    events = []

    class Reader:
        def __init__(self):
            self.keys = keyboard(["enter"])

        def __call__(self) -> str:
            events.append("key")
            return next(self.keys)

        def session(self):
            from contextlib import contextmanager

            @contextmanager
            def manager():
                events.append("enter")
                yield
                events.append("exit")

            return manager()

    line_editor = LineEditor(io.StringIO(), read_key=Reader(), commands={}, color=False)
    read(line_editor)
    assert events == ["enter", "key", "exit"]


# --- raw key decoding ------------------------------------------------------


def source_of(text: str, pending_after: int | None = None):
    chars = iter(text)
    remaining = [len(text)]

    def read_char() -> str:
        remaining[0] -= 1
        return next(chars)

    def pending() -> bool:
        return remaining[0] > 0 if pending_after is None else remaining[0] > pending_after

    return read_char, pending


@pytest.mark.parametrize(
    ("raw", "name"),
    [
        ("\x1b[A", "up"),
        ("\x1b[B", "down"),
        ("\x1b[C", "right"),
        ("\x1b[D", "left"),
        ("\x1b[H", "home"),
        ("\x1bOF", "end"),
        ("\x1b[3~", "delete"),
        ("\x1b[Z", "shift_tab"),
        ("\x1b[1;5C", ""),
        ("\x1b", ""),
        ("\t", "tab"),
        ("\r", "enter"),
        ("\x7f", "backspace"),
        ("\x03", "ctrl_c"),
        ("\x04", "ctrl_d"),
        ("\x17", "ctrl_w"),
        ("\x15", "ctrl_u"),
        ("á", "á"),
        ("¿", "¿"),
    ],
)
def test_read_one_decodes_keys(raw, name):
    assert read_one(*source_of(raw)) == name


def test_read_one_leaves_unread_input_alone():
    read_char, pending = source_of("\x1b[Aq")
    assert read_one(read_char, pending) == "up"
    assert read_char() == "q"


def test_newline_with_more_input_waiting_is_a_paste_space():
    assert read_one(*source_of("\nrest")) == " "
    assert read_one(*source_of("\n")) == "enter"


def test_bracketed_paste_is_one_text_without_newlines():
    raw = "\x1b[200~line one\nline two\x1b[201~"
    assert read_one(*source_of(raw)) == "line one line two"
    assert apply_key(EditorState(), "line one line two").buffer == "line one line two"


def test_utf8_length_from_the_lead_byte():
    assert [utf8_length(b) for b in (0x61, 0xC3, 0xE2, 0xF0, 0x80)] == [1, 2, 3, 4, 1]
    assert "á".encode()[0] == 0xC3


@pytest.mark.parametrize(
    ("sequence", "complete"),
    [
        ("\x1b", False),
        ("\x1b[", False),
        ("\x1b[1", False),
        ("\x1b[A", True),
        ("\x1b[3~", True),
        ("\x1bO", False),
        ("\x1bOH", True),
        ("\x1bx", True),
    ],
)
def test_escape_complete(sequence, complete):
    assert escape_complete(sequence) is complete


def test_normalize_key_passes_names_controls_and_text():
    assert normalize_key("left") == "left"
    assert normalize_key("\x01") == "ctrl_a"
    assert normalize_key("hola") == "hola"
    assert normalize_key("\x1b[Z") == "shift_tab"


def test_raw_key_reader_exists_with_termios():
    reader = raw_key_reader()
    assert reader is not None
    assert hasattr(reader, "session")


def test_raw_key_reader_is_none_without_termios(monkeypatch):
    monkeypatch.setattr(editor, "termios", None)
    assert raw_key_reader() is None
