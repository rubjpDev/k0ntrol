"""The terminal front: the REPL as it looks in a real terminal.

Claude Code's look on top of k0's run module: a welcome screen, a ruled
``❯`` prompt, live progress with a spinner, gates as a panel plus an
option picker, and ``/ask`` answers behind a bullet. The session drives it
through a handful of methods (``read``, ``line``, ``event``, ``stream``,
``gate``, ``confirm``); without a terminal the session keeps plain lines.
"""

import time
from collections.abc import Callable
from pathlib import Path

import yaml
from rich.console import Console
from rich.text import Text

from k0ntrol.frontends.prompt import Prompt, render_welcome, terminal_key_reader
from k0ntrol.credentials import load_credentials
from k0ntrol.frontends.editor import LineEditor, raw_key_reader
from k0ntrol.frontends.render import LiveRenderer, format_elapsed, format_tokens, render_gate
from k0ntrol.frontends.setup import run_project_setup, run_provider_setup
from k0ntrol.pricing import estimate_cost
from k0ntrol.frontends.theme import ACCENT, BULLET, DIM, ERROR, WARN

COMMANDS = {
    "/ask": "ask a question about this repo",
    "/full": "take #<issue> or a task to a ready pull request",
    "/fast": "a quick test-first change in the working tree",
    "/status": "provider, models and project settings",
    "/config": "change the models or this project's setup",
    "/help": "commands and shortcuts",
    "/quit": "leave k0",
}

# Each gate's answers in plain words, in the order the gate lists them.
GATE_OPTIONS: dict[str, list[tuple[str, str]]] = {
    "spec": [("ok", "Approve the brief"), ("redo", "Send back to SPEC with feedback")],
    "test": [("ok", "Approve the tests"), ("redo", "Send back to TESTER with feedback")],
    "review": [
        ("ok", "Approve: mark the pull request ready"),
        ("redo", "Send back to CODER with feedback"),
        ("spec", "Send back to SPEC with feedback"),
    ],
    "ci": [("ok", "Allow the .github/ change"), ("redo", "Send back to CODER with feedback")],
    "diff": [("y", "Keep the change"), ("n", "Reject it")],
    "limit_hit": [
        ("retry", "Retry with fresh context (files are kept)"),
        ("stop", "Stop the run"),
    ],
}

# Answers that carry the human's words to the agent that runs next.
FEEDBACK_LABELS = {
    "redo": "feedback",
    "spec": "feedback for SPEC",
    "retry": "feedback (enter to skip)",
}


class SessionStats:
    """What this session has spent so far: time, tokens, estimated cost."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._start = clock()
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost = 0.0
        self.cost_complete = True  # False once a call had no known price

    def add(self, event: dict) -> None:
        if event.get("kind") != "model":
            return
        tokens_in, tokens_out = event.get("tokens_in") or 0, event.get("tokens_out") or 0
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        cost = estimate_cost(event.get("provider"), event.get("model"), tokens_in, tokens_out)
        if cost is None:
            self.cost_complete = False
            return
        self.cost += cost

    def summary(self) -> list[str]:
        """``12m``, ``↑ 45.2k ↓ 3.1k``, ``≈ $0.04``: the parts there is data for."""
        parts = [format_elapsed(self._clock() - self._start)]
        if self.tokens_in or self.tokens_out:
            parts.append(f"↑ {format_tokens(self.tokens_in)} ↓ {format_tokens(self.tokens_out)}")
        if self.cost or self.cost_complete and self.tokens_in:
            parts.append(f"≈ {format_cost(self.cost)}" + ("" if self.cost_complete else "+"))
        return parts


def format_cost(usd: float) -> str:
    """Cents for real spend, four decimals while it is still under a cent."""
    return f"${usd:.2f}" if usd >= 0.01 or usd == 0 else f"${usd:.4f}"


class Terminal:
    """The REPL front for an interactive terminal."""

    def __init__(
        self,
        console: Console,
        prompt: Prompt,
        renderer: LiveRenderer,
        stats: SessionStats | None = None,
        editor: LineEditor | None = None,
    ) -> None:
        self._console = console
        self._prompt = prompt
        self._editor = editor
        self._renderer = renderer
        self._stats = stats or SessionStats()
        self._provider: str | None = None
        self._answering = False
        self._at_line_start = True
        self._answered_gate = False
        self._welcomed = False

    @classmethod
    def open(cls, history: Path | None = None) -> "Terminal":
        """The terminal front on the real stdin/stdout."""
        console = Console(highlight=False)
        stats = SessionStats()
        terminal: Terminal | None = None
        prompt = Prompt(
            console, commands=COMMANDS, history=history, read_key=terminal_key_reader(),
            status=lambda: terminal.status_line() if terminal else "",
        )
        terminal = cls(console, prompt, LiveRenderer(console), stats, _editor(console, history))
        return terminal

    def welcome(self, *, version: str, provider: str | None, cwd: str) -> None:
        """The welcome screen, once per session (first-run setup shows it early)."""
        if self._welcomed:
            return
        self._welcomed = True
        self._provider = provider
        self._console.clear()  # k0 starts on a clean screen (no-op off a terminal)
        color = self._console.color_system is not None
        short = short_path(cwd, self._console.width - 40)
        render_welcome(self._console, version=version, provider=provider, cwd=short, color=color)

    def read(self) -> str:
        self._settle()
        self._console.set_window_title(" · ".join(["k0", *self._stats.summary()]))
        if self._editor is not None:
            line = as_command(*self._editor.read(status=self.status_line))
        else:
            line = self._prompt.read()
        self._renderer.new_turn()
        return line

    def status_line(self) -> str:
        """The bar under the input: provider · time · tokens · cost · hint."""
        parts = [self._provider] if self._provider else []
        hints = ["shift+tab mode", "? shortcuts"] if self._editor else ["? for shortcuts"]
        return " · ".join([*parts, *self._stats.summary(), *hints])

    def line(self, text: str) -> None:
        self._settle()
        self._console.print(Text.from_ansi(text, style=line_style(text)))

    def event(self, event: dict) -> None:
        self._stats.add(event)
        if self._continues_after_gate(event):
            return
        self._end_answer()
        self._renderer(event)

    def stream(self, delta: str) -> None:
        if not self._answering:
            self._settle()
            self._console.print(Text(f"{BULLET} ", style=ACCENT), end="")
            self._answering = True
            delta = delta.lstrip()
        # soft_wrap: rich cannot know the cursor column between chunks, so
        # the terminal wraps the line instead.
        self._console.print(Text(delta), end="", soft_wrap=True)
        self._at_line_start = delta.endswith("\n")

    def gate(self, pending) -> str:
        """Show the gate and pick an answer; feedback rides on redo/retry."""
        self._settle()
        render_gate(
            self._console, gate=pending.gate, question=pending.question, detail=pending.detail
        )
        key = self._prompt.choose(pending.question, gate_options(pending.gate, pending.answers))
        self._answered_gate = True
        return self._with_feedback(key)

    def confirm(self, question: str) -> str:
        self._settle()
        return self._prompt.choose(question, [("y", "Yes"), ("n", "No")])

    def answer_line(self, label: str) -> str:
        self._settle()
        return self._prompt.ask_text(label)

    def help(self) -> None:
        self._settle()
        self._prompt.help_table()

    def status(self, rows: list[tuple[str, str]]) -> None:
        self._settle()
        width = max((len(name) for name, _ in rows), default=0)
        for name, value in rows:
            line = Text("  ")
            line.append(name.ljust(width), style=DIM)
            line.append(f"  {value}")
            self._console.print(line)

    def first_run(self, root: Path, home: Path) -> None:
        """The setup screens k0 needs before its first run here."""
        if needs_project_setup(root):
            run_project_setup(root, self._prompt, self._console)
        if needs_provider_setup(home):
            self._setup_provider(root, home)

    def configure(self, root: Path, home: Path) -> None:
        """``/config``: the provider and models, or this project's setup."""
        self._settle()
        choice = self._prompt.choose(
            "What do you want to change?",
            [("models", "Models and provider"),
             ("project", "This project: code host, tracker, base branch, test command")],
        )
        if choice == "project":
            run_project_setup(root, self._prompt, self._console)
            return
        self._setup_provider(root, home)

    def _setup_provider(self, root: Path, home: Path) -> None:
        run_provider_setup(home, self._prompt, self._console, root=root)
        load_credentials(home)  # a key saved just now works without a restart

    def _continues_after_gate(self, event: dict) -> bool:
        """The run resuming right after a gate answer is not news."""
        answered, self._answered_gate = self._answered_gate, False
        return answered and event.get("kind") == "run_start" and bool(event.get("resumed"))

    def _with_feedback(self, key: str) -> str:
        label = FEEDBACK_LABELS.get(key)
        if label is None:
            return key
        feedback = self._prompt.ask_text(label).strip()
        return f"{key} {feedback}".strip()

    def _settle(self) -> None:
        """Stop the spinner and close an answer before anything else prints."""
        self._end_answer()
        self._renderer.pause()

    def _end_answer(self) -> None:
        if self._answering and not self._at_line_start:
            self._console.print()
        self._answering = False
        self._at_line_start = True


def _editor(console: Console, history: Path | None) -> LineEditor | None:
    """The boxed editor where keys can be read raw; None falls back to readline."""
    read_key = raw_key_reader()
    if read_key is None:
        return None
    return LineEditor(
        console.file, read_key=read_key, commands=COMMANDS,
        history_path=history.with_name("prompt_history") if history else None,
        color=console.color_system is not None,
    )


def as_command(mode: str, text: str) -> str:
    """What Enter sends: plain text follows the mode; commands, ``#12`` and
    ``?`` go through as typed."""
    if not text or text.startswith(("/", "#", "?")) or mode == "chat":
        return text
    return f"/{mode} {text}"


def short_path(path: str, width: int, home: str | None = None) -> str:
    """``path`` with the home directory as ``~``, cut from the left to fit."""
    home = home if home is not None else str(Path.home())
    if path == home or path.startswith(home + "/"):
        path = "~" + path[len(home):]
    if len(path) <= width:
        return path
    return "…" + path[-(max(width, 10) - 1):]


def needs_project_setup(root: Path) -> bool:
    """A git repo k0 has not been set up in yet."""
    return (root / ".git").exists() and not (root / ".k0-mem" / "config.yaml").exists()


def needs_provider_setup(home: Path) -> bool:
    """No models chosen yet on this machine (an unreadable file is left to
    the config loader, which says what is wrong with it)."""
    path = home / ".k0ntrol" / "config.yaml"
    if not path.exists():
        return True
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return False
    return not (isinstance(data, dict) and data.get("agents"))


def line_style(text: str) -> str:
    """Failures read red, pauses amber; every other session line is plain."""
    head = text.split(":", 1)[0]
    if "failed" in head or text.startswith("refusing"):
        return ERROR
    if text.startswith(("run paused", "answer interrupted")):
        return WARN
    return ""


def gate_options(gate: str, answers: tuple[str, ...]) -> list[tuple[str, str]]:
    """The gate's answers as ``(key, label)``; unknown gates show the keys."""
    known = GATE_OPTIONS.get(gate)
    if known and [key for key, _ in known] == list(answers):
        return known
    return [(answer, answer) for answer in answers]
