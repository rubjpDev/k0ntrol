"""The REPL's input side: welcome screen, ruled prompt, option selector.

Modelled on Claude Code's input area: a rule, a ``❯`` prompt, a rule and a
hint. All raw-terminal work lives in :func:`read_key`; everything else is
plain functions over strings and indexes, so it tests without a terminal.
"""

import os
import select
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from rich.console import Console
from rich.text import Text

from k0ntrol.frontends.banner import render_banner
from k0ntrol.frontends.theme import ACCENT, BORDER, BRAND, DIM, POINTER

try:
    import readline
except ImportError:  # pragma: no cover - Windows
    readline = None  # type: ignore[assignment]

try:
    import termios
    import tty as tty_mode
except ImportError:  # pragma: no cover - Windows
    termios = None  # type: ignore[assignment]
    tty_mode = None  # type: ignore[assignment]

HISTORY_LIMIT = 1000
PLAIN_PROMPT = "› "
INPUT_HINT = "? for shortcuts"
CHOOSE_HINT = "↑/↓ to move · enter to select · 1-9 to pick"
SHORTCUTS = "↑/↓ history · tab completes /commands · ctrl-c pauses a run · ctrl-d quits"
WELCOME_LINES = (
    "",
    "  Just type to ask about this repo · /help for commands · ? for shortcuts",
    "",
    "  Tips for getting started:",
    "  1. type a question: k0 answers and remembers the conversation",
    "  2. #<issue> takes a work item to a pull request",
    "  3. /fast <task> for a quick test-first change · /full <task> for new work",
)
_UP_KEYS = {"up", "k"}
_DOWN_KEYS = {"down", "j"}
_ESCAPE_KEYS = {"[A": "up", "OA": "up", "[B": "down", "OB": "down"}
_ARROW_TIMEOUT = 0.05


def render_welcome(
    console: Console, *, version: str, provider: str | None, cwd: str, color: bool
) -> None:
    banner = render_banner(version=version, provider=provider, cwd=cwd, color=color)
    console.print(Text.from_ansi(banner), end="")
    for line in WELCOME_LINES:
        console.print(Text(line, style=DIM))


def complete_command(text: str, commands: Mapping[str, str]) -> list[str]:
    if not text.startswith("/"):
        return []
    return sorted(name for name in commands if name.startswith(text))


def next_index(index: int, key: str, count: int) -> int:
    if key in _UP_KEYS:
        return (index - 1) % count
    if key in _DOWN_KEYS:
        return (index + 1) % count
    return index


def digit_choice(key: str, count: int) -> int | None:
    if len(key) == 1 and key in "123456789" and int(key) <= count:
        return int(key) - 1
    return None


def choice_step(index: int, key: str, count: int) -> tuple[int, bool]:
    """One key in the picker: the new selection and whether it is final."""
    picked = digit_choice(key, count)
    if picked is not None:
        return picked, True
    if key == "enter":
        return index, True
    return next_index(index, key, count), False


def parse_choice(answer: str, options: list[tuple[str, str]]) -> str | None:
    answer = answer.strip().lower()
    lowered = [key.lower() for key, _ in options]
    if answer in lowered:
        return options[lowered.index(answer)][0]
    number = digit_choice(answer, len(options))
    return None if number is None else options[number][0]


def decode_key(sequence: str) -> str:
    """Turn the characters of one key press into a key name."""
    if sequence == "\x03":
        raise KeyboardInterrupt
    if sequence in ("\r", "\n"):
        return "enter"
    if sequence.startswith("\x1b"):
        return _ESCAPE_KEYS.get(sequence[1:], "")
    return sequence[:1]


def _read_sequence() -> str:  # pragma: no cover - raw terminal I/O
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty_mode.setcbreak(fd)
        text = os.read(fd, 1).decode(errors="ignore")
        while text.startswith("\x1b") and len(text) < 3 and _key_pending(fd):
            text += os.read(fd, 1).decode(errors="ignore")
        return text
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _key_pending(fd: int) -> bool:
    return bool(select.select([fd], [], [], _ARROW_TIMEOUT)[0])


def read_key() -> str:
    """Block for one key press: "up", "down", "enter", or the character."""
    return decode_key(_read_sequence())


def terminal_key_reader() -> Callable[[], str] | None:
    """The key reader to pass to :class:`Prompt`, or None without termios."""
    return read_key if termios is not None else None


def _ansi(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return f"\001\x1b[38;2;{red};{green};{blue}m\002{text}\001\x1b[0m\002"


def _bind_tab() -> None:
    binding = "bind ^I rl_complete" if "libedit" in (readline.__doc__ or "") else "tab: complete"
    readline.parse_and_bind(binding)


class Prompt:
    def __init__(
        self,
        console: Console,
        *,
        commands: dict[str, str],
        history: Path | None = None,
        tty: bool | None = None,
        read_input: Callable[[str], str] = input,
        read_key: Callable[[], str] | None = None,
        status: Callable[[], str] | None = None,
    ) -> None:
        self._console = console
        self._status = status
        self._commands = commands
        self._history = history
        self._tty = _is_tty() if tty is None else tty
        self._read_input = read_input
        self._read_key = read_key
        if readline is not None:
            self._setup_readline()

    def read(self) -> str:
        if not self._tty:
            return self._read_input(PLAIN_PROMPT)
        self._draw_frame()
        try:
            line = self._read_input(self._styled(POINTER + " ", BRAND))
        except BaseException:
            self._raw("\x1b[3B")  # below the frame, so the next output is clean
            raise
        self._replace_frame(line)
        self._save_history()
        return line

    def ask_text(self, label: str) -> str:
        if not self._tty:
            return self._read_input(f"{label} {PLAIN_PROMPT}")
        return self._read_input(self._styled(label, DIM) + f" {POINTER} ")

    def choose(self, question: str, options: list[tuple[str, str]]) -> str:
        if self._tty and self._read_key is not None:
            return self._choose_with_keys(question, options, self._read_key)
        return self._choose_by_line(question, options)

    def help_table(self) -> None:
        width = max((len(name) for name in self._commands), default=0)
        for name, description in sorted(self._commands.items()):
            line = Text("  ")
            line.append(name.ljust(width), style=ACCENT)
            line.append(f"  {description}")
            self._console.print(line)
        self._console.print()
        self._console.print(Text("  " + SHORTCUTS, style=DIM))

    def complete(self, text: str, state: int) -> str | None:
        matches = complete_command(text, self._commands)
        return matches[state] if state < len(matches) else None

    def _styled(self, text: str, color: str) -> str:
        return _ansi(text, color, self._console.color_system is not None)

    def _raw(self, text: str) -> None:
        self._console.file.write(text)
        self._console.file.flush()

    def _draw_frame(self) -> None:
        rule = Text("─" * self._console.width, style=BORDER)
        self._console.print(rule, overflow="crop", no_wrap=True)
        self._raw("\n")
        self._console.print(rule, overflow="crop", no_wrap=True)
        hint = self._status() if self._status is not None else INPUT_HINT
        self._console.print(Text("  " + hint, style=DIM), overflow="ellipsis", no_wrap=True)
        self._raw("\x1b[3A")

    def _replace_frame(self, line: str) -> None:
        """The submitted line stays in the scrollback as one dim ``❯`` line,
        without the rules and the hint (Claude Code's history)."""
        self._raw("\x1b[2A" + "\x1b[2K\n" * 4 + "\x1b[4A")
        self._console.print(Text(f"{POINTER} {line}", style=DIM))

    def _choose_by_line(self, question: str, options: list[tuple[str, str]]) -> str:
        keys = "/".join(key for key, _ in options)
        self._console.print(Text(f"{question} [{keys}]"))
        while True:
            choice = parse_choice(self._read_input(PLAIN_PROMPT), options)
            if choice is not None:
                return choice
            self._console.print(Text(f"Answer {keys} or 1-{len(options)}.", style=DIM))

    def _choose_with_keys(
        self,
        question: str,
        options: list[tuple[str, str]],
        read_key: Callable[[], str],
    ) -> str:
        self._console.print(Text(question, style="bold"))
        index = 0
        self._draw_options(options, index)
        while True:
            index, chosen = choice_step(index, read_key(), len(options))
            if chosen:
                self._collapse_options(options, index)
                return options[index][0]
            self._redraw_options(options, index)

    def _draw_options(self, options: list[tuple[str, str]], index: int) -> None:
        for position, (_, label) in enumerate(options):
            self._raw("\x1b[2K")
            self._console.print(_option_line(position, label, position == index))
        self._raw("\x1b[2K")
        self._console.print(Text("  " + CHOOSE_HINT, style=DIM))

    def _redraw_options(self, options: list[tuple[str, str]], index: int) -> None:
        self._raw(f"\x1b[{len(options) + 1}A")
        self._draw_options(options, index)

    def _collapse_options(self, options: list[tuple[str, str]], index: int) -> None:
        """Replace the menu and its hint with the chosen answer."""
        lines = len(options) + 1
        self._raw(f"\x1b[{lines}A" + "\x1b[2K\n" * lines + f"\x1b[{lines}A")
        self._console.print(Text(f"  {POINTER} {options[index][1]}", style=ACCENT))

    def _setup_readline(self) -> None:
        readline.set_completer(self.complete)
        readline.set_completer_delims(" \t\n")
        _bind_tab()
        readline.set_history_length(HISTORY_LIMIT)
        if self._history is not None:
            self._load_history(self._history)

    def _load_history(self, path: Path) -> None:
        try:
            readline.read_history_file(str(path))
        except OSError:
            pass

    def _save_history(self) -> None:
        if readline is None or self._history is None:
            return
        try:
            self._history.parent.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(str(self._history))
        except OSError:
            pass


def _option_line(position: int, label: str, selected: bool) -> Text:
    marker = f"{POINTER} " if selected else "  "
    return Text(f"{marker}{position + 1}. {label}", style=ACCENT if selected else "")


def _is_tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()
