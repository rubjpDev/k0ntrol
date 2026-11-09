"""A tiny line editor: a closed rounded box whose border shows the mode.

``input()`` and readline cannot draw a right border, nor recolour while the
user types, so this module reads keys itself and redraws the whole box after
every key. Everything that decides something is pure and tests without a
terminal: :func:`apply_key` (one key in, new state out), :func:`render_box`
(state in, lines out) and :func:`paint_frame` (lines in, escape codes out).
Only the byte read at the bottom of the file touches the terminal.

```
╭─ chat ─────────────────────────────────────────────────╮
│ ❯ what does slugify do?                                │
╰────────────────────────────────────────────────────────╯
  bedrock · 25s · ↑ 3.9k ↓ 473 · ≈ $0.0012 · shift+tab mode · ? shortcuts
```
"""

import os
import re
import select
import shutil
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TextIO

from rich.cells import cell_len

from k0ntrol.frontends.theme import ACCENT, BRAND, DIM, POINTER, WARN

try:
    import termios
    import tty as tty_mode
except ImportError:  # pragma: no cover - Windows
    termios = None  # type: ignore[assignment]
    tty_mode = None  # type: ignore[assignment]

HISTORY_LIMIT = 1000
MIN_WIDTH = 20
FRAME_PADDING = 6  # "│ ❯ " on the left, " │" on the right
ESC = "\x1b"
PASTE_START = "\x1b[200~"
PASTE_END = "\x1b[201~"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
CLEAR_ROW = "\x1b[2K"
CLEAR_BELOW = "\x1b[J"
PASTE_ON = "\x1b[?2004h"
PASTE_OFF = "\x1b[?2004l"
_ESCAPE_TIMEOUT = 0.05


@dataclass(frozen=True)
class Mode:
    name: str
    color: str
    hint: str = ""


DEFAULT_MODES = (
    Mode("chat", BRAND, "ask about this repo"),
    Mode("fast", WARN, "quick test-first change"),
    Mode("full", ACCENT, "new work, spec first"),
)
_PREFIX_MODES = {"/fast": "fast", "/full": "full", "/ask": "chat"}
_ISSUE_REF = re.compile(r"#\d+")


@dataclass(frozen=True)
class EditorState:
    buffer: str = ""
    cursor: int = 0
    history_index: int | None = None  # None: editing the draft, not browsing
    draft: str = ""  # what was typed before the first Up
    mode_index: int = 0


@dataclass(frozen=True)
class Submitted:
    text: str


@dataclass(frozen=True)
class Cancelled:
    eof: bool  # True for Ctrl-D (EOFError), False for Ctrl-C (KeyboardInterrupt)


@dataclass(frozen=True)
class Env:
    """What a key press needs besides the state: history, commands, modes."""

    history: tuple[str, ...] = ()
    commands: Mapping[str, str] = field(default_factory=dict)
    mode_count: int = 1


@dataclass(frozen=True)
class Geometry:
    """Where the last frame left the cursor: its size and the caret's row."""

    rows: int
    caret_row: int


Outcome = EditorState | Submitted | Cancelled
Handler = Callable[[EditorState, Env], Outcome]


# --- modes -----------------------------------------------------------------


def prefix_mode_name(buffer: str) -> str | None:
    """The mode a typed prefix forces (``/fast``, ``/full``, ``#12``, ``/ask``)."""
    words = buffer.split(maxsplit=1)
    if not words:
        return None
    if _ISSUE_REF.fullmatch(words[0]):
        return "full"
    return _PREFIX_MODES.get(words[0])


def effective_mode(buffer: str, mode_index: int, modes: list[Mode]) -> Mode:
    """The mode Enter would use: a typed prefix wins over the selected mode."""
    selected = modes[mode_index % len(modes)]
    forced = prefix_mode_name(buffer)
    return next((mode for mode in modes if mode.name == forced), selected)


# --- history ---------------------------------------------------------------


def add_entry(entries: list[str], text: str) -> list[str]:
    """Append ``text`` unless it is blank or repeats the last entry."""
    if not text.strip() or (entries and entries[-1] == text):
        return entries
    return (entries + [text])[-HISTORY_LIMIT:]


def load_history(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries: list[str] = []
    for line in lines:
        entries = add_entry(entries, line)
    return entries


def save_history(path: Path, entries: list[str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(entries[-HISTORY_LIMIT:]) + "\n", encoding="utf-8")
    except OSError:
        pass


# --- keys ------------------------------------------------------------------

_CONTROLS = {
    "\x01": "ctrl_a",
    "\x03": "ctrl_c",
    "\x04": "ctrl_d",
    "\x05": "ctrl_e",
    "\x15": "ctrl_u",
    "\x17": "ctrl_w",
    "\t": "tab",
    "\r": "enter",
    "\n": "enter",
    "\x7f": "backspace",
    "\x08": "backspace",
}
_ESCAPES = {
    "[A": "up",
    "OA": "up",
    "[B": "down",
    "OB": "down",
    "[C": "right",
    "OC": "right",
    "[D": "left",
    "OD": "left",
    "[H": "home",
    "OH": "home",
    "[1~": "home",
    "[7~": "home",
    "[F": "end",
    "OF": "end",
    "[4~": "end",
    "[8~": "end",
    "[3~": "delete",
    "[Z": "shift_tab",
}
_NEWLINES = re.compile(r"[\r\n\t]+")


def sanitize_text(text: str) -> str:
    """Text safe to insert: newlines and tabs become one space, controls go."""
    flat = _NEWLINES.sub(" ", text)
    return "".join(char for char in flat if char.isprintable())


def normalize_key(key: str) -> str:
    """A key name, the text to insert, or "" for a key to ignore.

    Accepts key names, raw control characters and raw escape sequences, so the
    reader and the tests can feed either.
    """
    if key in KEY_NAMES:
        return key
    if key.startswith(ESC):
        return _ESCAPES.get(key[1:], "")
    if key in _CONTROLS:
        return _CONTROLS[key]
    return sanitize_text(key)


# --- editing ---------------------------------------------------------------


def _edit(state: EditorState, buffer: str, cursor: int) -> EditorState:
    return replace(state, buffer=buffer, cursor=cursor)


def _move(state: EditorState, cursor: int) -> EditorState:
    return replace(state, cursor=max(0, min(len(state.buffer), cursor)))


def insert_text(state: EditorState, text: str) -> EditorState:
    head, tail = state.buffer[: state.cursor], state.buffer[state.cursor :]
    return _edit(state, head + text + tail, state.cursor + len(text))


def _word_start(buffer: str, cursor: int) -> int:
    start = cursor
    while start > 0 and buffer[start - 1] == " ":
        start -= 1
    while start > 0 and buffer[start - 1] != " ":
        start -= 1
    return start


def _left(state: EditorState, env: Env) -> Outcome:
    return _move(state, state.cursor - 1)


def _right(state: EditorState, env: Env) -> Outcome:
    return _move(state, state.cursor + 1)


def _home(state: EditorState, env: Env) -> Outcome:
    return _move(state, 0)


def _end(state: EditorState, env: Env) -> Outcome:
    return _move(state, len(state.buffer))


def _backspace(state: EditorState, env: Env) -> Outcome:
    if state.cursor == 0:
        return state
    buffer = state.buffer[: state.cursor - 1] + state.buffer[state.cursor :]
    return _edit(state, buffer, state.cursor - 1)


def _delete(state: EditorState, env: Env) -> Outcome:
    buffer = state.buffer[: state.cursor] + state.buffer[state.cursor + 1 :]
    return _edit(state, buffer, state.cursor)


def _kill_to_start(state: EditorState, env: Env) -> Outcome:
    return _edit(state, state.buffer[state.cursor :], 0)


def _kill_word(state: EditorState, env: Env) -> Outcome:
    start = _word_start(state.buffer, state.cursor)
    return _edit(state, state.buffer[:start] + state.buffer[state.cursor :], start)


def _ctrl_c(state: EditorState, env: Env) -> Outcome:
    return Cancelled(eof=False)


def _ctrl_d(state: EditorState, env: Env) -> Outcome:
    return _delete(state, env) if state.buffer else Cancelled(eof=True)


def _enter(state: EditorState, env: Env) -> Outcome:
    return Submitted(state.buffer)


def _next_mode(state: EditorState, env: Env) -> Outcome:
    return replace(state, mode_index=(state.mode_index + 1) % max(1, env.mode_count))


def _tab(state: EditorState, env: Env) -> Outcome:
    text = state.buffer
    if not text.startswith("/") or " " in text:
        return state
    matches = sorted(name for name in env.commands if name.startswith(text))
    if not matches:
        return state
    done = os.path.commonprefix(matches)
    return _edit(state, done, len(done))


def _show_history(state: EditorState, env: Env, index: int, draft: str) -> EditorState:
    entry = env.history[index]
    return replace(state, buffer=entry, cursor=len(entry), history_index=index, draft=draft)


def _up(state: EditorState, env: Env) -> Outcome:
    if not env.history:
        return state
    if state.history_index is None:
        return _show_history(state, env, len(env.history) - 1, state.buffer)
    return _show_history(state, env, max(0, state.history_index - 1), state.draft)


def _down(state: EditorState, env: Env) -> Outcome:
    index = state.history_index
    if index is None:
        return state
    if index + 1 < len(env.history):
        return _show_history(state, env, index + 1, state.draft)
    return replace(
        state, buffer=state.draft, cursor=len(state.draft), history_index=None, draft=""
    )


_HANDLERS: dict[str, Handler] = {
    "left": _left,
    "right": _right,
    "home": _home,
    "end": _end,
    "ctrl_a": _home,
    "ctrl_e": _end,
    "backspace": _backspace,
    "delete": _delete,
    "ctrl_u": _kill_to_start,
    "ctrl_w": _kill_word,
    "ctrl_c": _ctrl_c,
    "ctrl_d": _ctrl_d,
    "enter": _enter,
    "shift_tab": _next_mode,
    "tab": _tab,
    "up": _up,
    "down": _down,
}
KEY_NAMES = frozenset(_HANDLERS)


def apply_key(state: EditorState, key: str, env: Env | None = None) -> Outcome:
    """One key press: the next state, or Submitted / Cancelled when it ends."""
    name = normalize_key(key)
    if not name:
        return state
    handler = _HANDLERS.get(name)
    if handler is None:
        return insert_text(state, name)
    return handler(state, env or Env())


# --- layout and rendering --------------------------------------------------


def text_width(width: int) -> int:
    """Columns available for text inside a box ``width`` columns wide."""
    return max(width, MIN_WIDTH) - FRAME_PADDING


def wrap_segments(text: str, columns: int) -> list[tuple[int, int]]:
    """Split ``text`` into (start, end) index ranges that fit ``columns`` cells."""
    segments: list[tuple[int, int]] = []
    start = used = 0
    for index, char in enumerate(text):
        size = cell_len(char)
        if used + size > columns:
            segments.append((start, index))
            start, used = index, 0
        used += size
    segments.append((start, len(text)))
    return segments


def layout(state: EditorState, width: int) -> list[tuple[int, int]]:
    """The rows of the input, with an empty one when the caret needs it."""
    columns = text_width(width)
    segments = wrap_segments(state.buffer, columns)
    start, end = segments[-1]
    if state.cursor == len(state.buffer) and cell_len(state.buffer[start:end]) >= columns:
        segments.append((end, end))
    return segments


def caret_position(state: EditorState, width: int) -> tuple[int, int]:
    """The caret as (row from the box's top border, 0-based column)."""
    segments = layout(state, width)
    row = max(i for i, (start, _) in enumerate(segments) if start <= state.cursor)
    column = cell_len(state.buffer[segments[row][0] : state.cursor])
    return row + 1, column + 4


def paint(text: str, color: str, enabled: bool) -> str:
    """Wrap ``text`` in a 24-bit foreground colour (``#RRGGBB``)."""
    if not enabled or not text:
        return text
    red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"\x1b[38;2;{red};{green};{blue}m{text}\x1b[0m"


def crop(text: str, columns: int) -> str:
    """``text`` cut to ``columns`` characters, ending in ``…`` when cut."""
    if len(text) <= columns:
        return text
    return text[: max(0, columns - 1)] + "…"


def _top_border(mode: Mode, width: int, color: bool) -> str:
    label = f" {mode.name} "
    fill = "─" * max(0, width - 3 - len(label))
    return paint(f"╭─{label}{fill}╮", mode.color, color)


def _bottom_border(mode: Mode, width: int, color: bool) -> str:
    return paint("╰" + "─" * (width - 2) + "╯", mode.color, color)


def _row(index: int, text: str, shown: str, mode: Mode, width: int, color: bool) -> str:
    edge = paint("│", mode.color, color)
    lead = paint(POINTER, mode.color, color) if index == 0 else " "
    pad = " " * (text_width(width) - cell_len(text))
    return f"{edge} {lead} {shown}{pad} {edge}"


def _content_rows(state: EditorState, mode: Mode, width: int, color: bool) -> list[str]:
    if not state.buffer:
        hint = crop(mode.hint, text_width(width))
        return [_row(0, hint, paint(hint, DIM, color), mode, width, color)]
    rows = layout(state, width)
    return [
        _row(i, state.buffer[a:b], state.buffer[a:b], mode, width, color)
        for i, (a, b) in enumerate(rows)
    ]


def render_box(
    state: EditorState,
    *,
    width: int,
    status: str,
    modes: list[Mode],
    color: bool = True,
) -> list[str]:
    """The frame: top border, input rows, bottom border, dim status line."""
    width = max(width, MIN_WIDTH)
    mode = effective_mode(state.buffer, state.mode_index, modes)
    return [
        _top_border(mode, width, color),
        *_content_rows(state, mode, width, color),
        _bottom_border(mode, width, color),
        paint("  " + crop(status, width - 2), DIM, color),
    ]


def _up_rows(count: int) -> str:
    return f"\x1b[{count}A" if count > 0 else ""


def paint_frame(
    lines: list[str], caret: tuple[int, int], previous: Geometry | None
) -> str:
    """Escape codes that replace the previous frame with ``lines``.

    Moves up to the old frame's top, clears every row either frame uses,
    writes the new rows and parks the cursor at ``caret``.
    """
    rows = max(len(lines), previous.rows if previous else 0)
    padded = lines + [""] * (rows - len(lines))
    top = _up_rows(previous.caret_row) + "\r" if previous else ""
    body = "\n".join(CLEAR_ROW + line for line in padded)
    park = _up_rows(rows - 1 - caret[0]) + f"\x1b[{caret[1] + 1}G"
    return HIDE_CURSOR + top + body + park + SHOW_CURSOR


def erase_frame(previous: Geometry | None) -> str:
    """Escape codes that wipe the frame, leaving the cursor at its top."""
    if previous is None:
        return ""
    return _up_rows(previous.caret_row) + "\r" + CLEAR_BELOW


# --- the editor ------------------------------------------------------------


class LineEditor:
    def __init__(
        self,
        out: TextIO,
        *,
        read_key: Callable[[], str],
        commands: dict[str, str],
        modes: list[Mode] | None = None,
        history_path: Path | None = None,
        width: Callable[[], int] | None = None,
        color: bool = True,
        history: list[str] | None = None,
    ) -> None:
        self._out = out
        self._read_key = read_key
        self._commands = commands
        self._modes = modes or list(DEFAULT_MODES)
        self._history_path = history_path
        self._width = width or _terminal_width
        self._color = color
        self._mode_index = 0
        self._geometry: Geometry | None = None
        self._history = self._initial_history(history or [])

    @property
    def mode(self) -> str:
        """The mode selected with Shift+Tab (a typed prefix may override it)."""
        return self._modes[self._mode_index].name

    @property
    def history(self) -> list[str]:
        return list(self._history)

    def read(self, *, status: Callable[[], str]) -> tuple[str, str]:
        """Block for one line: ``(mode_name, text)``. Ctrl-C / Ctrl-D raise."""
        self._geometry = None
        try:
            with self._terminal_session():
                text, mode = self._loop(status)
        except (KeyboardInterrupt, EOFError):
            self._erase()
            raise
        self._erase()
        self._write(paint(f"{POINTER} {text}", DIM, self._color) + "\n")
        self._remember(text)
        return mode, text

    def _loop(self, status: Callable[[], str]) -> tuple[str, str]:
        state = EditorState(mode_index=self._mode_index)
        env = Env(tuple(self._history), self._commands, len(self._modes))
        while True:
            self._draw(state, status())
            outcome = apply_key(state, self._read_key(), env)
            if not isinstance(outcome, EditorState):
                return self._finish(outcome, state)
            state = outcome
            self._mode_index = state.mode_index

    def _finish(self, outcome: Submitted | Cancelled, state: EditorState) -> tuple[str, str]:
        if isinstance(outcome, Cancelled):
            raise EOFError if outcome.eof else KeyboardInterrupt
        mode = effective_mode(outcome.text, state.mode_index, self._modes)
        return outcome.text, mode.name

    def _terminal_session(self) -> AbstractContextManager[object]:
        opener = getattr(self._read_key, "session", None)
        return opener() if opener is not None else nullcontext()

    def _draw(self, state: EditorState, status: str) -> None:
        width = self._width()
        lines = render_box(
            state, width=width, status=status, modes=self._modes, color=self._color
        )
        caret = caret_position(state, width)
        self._write(paint_frame(lines, caret, self._geometry))
        self._geometry = Geometry(rows=len(lines), caret_row=caret[0])

    def _erase(self) -> None:
        self._write(erase_frame(self._geometry) + SHOW_CURSOR)
        self._geometry = None

    def _write(self, text: str) -> None:
        self._out.write(text)
        self._out.flush()

    def _initial_history(self, seed: list[str]) -> list[str]:
        entries: list[str] = []
        loaded = load_history(self._history_path) if self._history_path else []
        for entry in [*seed, *loaded]:
            entries = add_entry(entries, entry)
        return entries

    def _remember(self, text: str) -> None:
        self._history = add_entry(self._history, text)
        if self._history_path is not None:
            save_history(self._history_path, self._history)


def _terminal_width() -> int:
    return shutil.get_terminal_size().columns


# --- raw terminal input ----------------------------------------------------


def utf8_length(lead: int) -> int:
    """How many bytes a UTF-8 character has, from its first byte."""
    if lead >= 0xF0:
        return 4
    if lead >= 0xE0:
        return 3
    if lead >= 0xC0:
        return 2
    return 1


def escape_complete(sequence: str) -> bool:
    """True when ``sequence`` (starting with ESC) has all its characters."""
    if len(sequence) < 2:
        return False
    if sequence[1] == "[":
        return len(sequence) > 2 and "@" <= sequence[-1] <= "~"
    if sequence[1] == "O":
        return len(sequence) >= 3
    return True


def _read_paste(read_char: Callable[[], str]) -> str:
    text = ""
    while not text.endswith(PASTE_END):
        text += read_char()
    return sanitize_text(text[: -len(PASTE_END)])


def _read_escape(read_char: Callable[[], str], pending: Callable[[], bool]) -> str:
    sequence = ESC
    while pending() and not escape_complete(sequence):
        sequence += read_char()
    if sequence == PASTE_START:
        return _read_paste(read_char)
    return normalize_key(sequence)


def read_one(read_char: Callable[[], str], pending: Callable[[], bool]) -> str:
    """One key from a character source: a key name or the text typed.

    A newline with more input already waiting is part of a paste, so it
    becomes a space instead of submitting.
    """
    first = read_char()
    if first == ESC:
        return _read_escape(read_char, pending)
    if first in "\r\n" and pending():
        return " "
    return normalize_key(first)


def _read_char(fd: int) -> str:  # pragma: no cover - raw terminal I/O
    lead = os.read(fd, 1)
    if not lead:
        raise EOFError
    rest = b"".join(os.read(fd, 1) for _ in range(utf8_length(lead[0]) - 1))
    return (lead + rest).decode(errors="ignore")


def _pending(fd: int) -> bool:  # pragma: no cover - raw terminal I/O
    return bool(select.select([fd], [], [], _ESCAPE_TIMEOUT)[0])


class RawKeyReader:
    """Reads key names from the terminal; cbreak holds for a whole ``session``."""

    def __init__(self) -> None:
        self._active = False

    @contextmanager
    def session(self) -> Iterator[None]:  # pragma: no cover - raw terminal I/O
        """Keep the terminal in cbreak (no echo) with bracketed paste on."""
        if self._active:
            yield
            return
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        tty_mode.setcbreak(fd, termios.TCSANOW)  # TCSAFLUSH would drop typed-ahead keys
        sys.stdout.write(PASTE_ON)
        sys.stdout.flush()
        self._active = True
        try:
            yield
        finally:
            self._active = False
            sys.stdout.write(PASTE_OFF)
            sys.stdout.flush()
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    def __call__(self) -> str:  # pragma: no cover - raw terminal I/O
        fd = sys.stdin.fileno()
        with self.session():
            return read_one(lambda: _read_char(fd), lambda: _pending(fd))


def raw_key_reader() -> Callable[[], str] | None:
    """The key reader to pass to :class:`LineEditor`, or None without termios."""
    return RawKeyReader() if termios is not None else None
