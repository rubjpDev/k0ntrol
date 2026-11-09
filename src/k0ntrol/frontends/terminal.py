"""The terminal front: the REPL as it looks in a real terminal.

Claude Code's look on top of k0's run module: a welcome screen, a ruled
``❯`` prompt, live progress with a spinner, gates as a panel plus an
option picker, and ``/ask`` answers behind a bullet. The session drives it
through a handful of methods (``read``, ``line``, ``event``, ``stream``,
``gate``, ``confirm``); without a terminal the session keeps plain lines.
"""

from pathlib import Path

from rich.console import Console
from rich.text import Text

from k0ntrol.frontends.prompt import Prompt, render_welcome, terminal_key_reader
from k0ntrol.frontends.render import LiveRenderer, render_gate
from k0ntrol.frontends.theme import ACCENT, BULLET, DIM

COMMANDS = {
    "/ask": "ask a question about this repo",
    "/full": "take #<issue> or a task to a ready pull request",
    "/fast": "a quick test-first change in the working tree",
    "/status": "provider, models and project settings",
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


class Terminal:
    """The REPL front for an interactive terminal."""

    def __init__(self, console: Console, prompt: Prompt, renderer: LiveRenderer) -> None:
        self._console = console
        self._prompt = prompt
        self._renderer = renderer
        self._answering = False
        self._at_line_start = True

    @classmethod
    def open(cls, history: Path | None = None) -> "Terminal":
        """The terminal front on the real stdin/stdout."""
        console = Console(highlight=False)
        prompt = Prompt(
            console, commands=COMMANDS, history=history, read_key=terminal_key_reader()
        )
        return cls(console, prompt, LiveRenderer(console))

    def welcome(self, *, version: str, provider: str | None, cwd: str) -> None:
        color = self._console.color_system is not None
        render_welcome(self._console, version=version, provider=provider, cwd=cwd, color=color)

    def read(self) -> str:
        self._settle()
        return self._prompt.read()

    def line(self, text: str) -> None:
        self._settle()
        self._console.print(Text.from_ansi(text))

    def event(self, event: dict) -> None:
        self._end_answer()
        self._renderer(event)

    def stream(self, delta: str) -> None:
        if not self._answering:
            self._settle()
            self._console.print(Text(f"{BULLET} ", style=ACCENT), end="")
            self._answering = True
        self._console.print(Text(delta), end="")
        self._at_line_start = delta.endswith("\n")

    def gate(self, pending) -> str:
        """Show the gate and pick an answer; feedback rides on redo/retry."""
        self._settle()
        render_gate(
            self._console, gate=pending.gate, question=pending.question, detail=pending.detail
        )
        key = self._prompt.choose(pending.question, gate_options(pending.gate, pending.answers))
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


def gate_options(gate: str, answers: tuple[str, ...]) -> list[tuple[str, str]]:
    """The gate's answers as ``(key, label)``; unknown gates show the keys."""
    known = GATE_OPTIONS.get(gate)
    if known and [key for key, _ in known] == list(answers):
        return known
    return [(answer, answer) for answer in answers]
