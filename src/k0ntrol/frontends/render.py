"""Live progress rendering in the style of Claude Code.

`LiveRenderer` is a progress sink: called once per k0 event, in order, it
prints a bullet per action, an elbow under it for results, and keeps a
spinner alive while an agent works. `render_gate` draws a human gate as a
rounded panel. Palette and glyphs come from `theme`.
"""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rich.box import ROUNDED
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from k0ntrol.frontends.theme import (
    ACCENT,
    BORDER,
    BULLET,
    CHECK,
    CROSS,
    DIM,
    ELBOW,
    ERROR,
    OK,
    SPINNER_FRAMES,
    WARN,
)

_ERR_MAX = 160
_GATE_MAX_LINES = 300
_SPINNER_FPS = 8

_TOOL_NAMES = {
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "list_files": "List",
    "grep": "Search",
    "bulk_read": "BulkRead",
    "search_memory": "SearchMemory",
    "run_tests": "RunTests",
    "read_work_item": "ReadWorkItem",
    "read_change_diff": "ReadDiff",
}

_CHECK_NAMES = {"red_check": "Red check", "validator": "Validator", "ci": "CI"}

_GATE_TITLES = {
    "spec": "Spec gate",
    "test": "Test gate",
    "review": "Review gate",
    "ci": "CI gate",
    "diff": "Diff gate",
    "limit_hit": "Limit hit",
}

_DECISIONS = {
    "approve": "approved",
    "reject": "sent back",
    "redo": "sent back",
    "retry": "retry",
    "stop": "stopped",
}

_OUTCOME_STYLES = {"done": OK, "stuck": ERROR}
_OUTCOME_GLYPHS = {"done": f"{CHECK} ", "stuck": f"{CROSS} "}


# --- pure helpers -----------------------------------------------------------


def format_tokens(count: int) -> str:
    """950 -> `950`, 34_500 -> `34.5k`, 1_200_000 -> `1.2M`."""
    if count < 1000:
        return str(count)
    if count < 999_950:
        return _scaled(count / 1000, "k")
    return _scaled(count / 1_000_000, "M")


def _scaled(value: float, suffix: str) -> str:
    return f"{value:.1f}".removesuffix(".0") + suffix


def format_elapsed(seconds: float) -> str:
    """45 -> `45s`, 192 -> `3m 12s`, 3720 -> `1h 2m`."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_ms(ms: float) -> str:
    """28 -> `28ms`, 1500 -> `1.5s`."""
    if ms < 1000:
        return f"{int(ms)}ms"
    return _scaled(ms / 1000, "s")


def tool_label(tool: str | None, path: str | None = None) -> str:
    """`read_file` + path -> `Read(src/x.py)`; no path, no parentheses."""
    name = _TOOL_NAMES.get(tool or "", tool or "tool")
    return f"{name}({path})" if path else name


def _first_line(value: Any, limit: int = _ERR_MAX) -> str:
    lines = str(value or "").strip().splitlines()
    line = lines[0].strip() if lines else ""
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _human(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _elbow(*parts: tuple[str, str]) -> Text:
    return Text.assemble((f"  {ELBOW}  ", DIM), *parts)


def _as_int(value: Any) -> int:
    return value if isinstance(value, int) else 0


# --- one line per event kind ------------------------------------------------


def run_start_line(event: dict) -> Text:
    line = Text.assemble((f"{BULLET} ", ACCENT), f"/{event.get('run_kind', '?')}")
    shown = [str(event[key]) for key in ("work_item", "branch") if event.get(key)]
    if shown:
        line.append(" " + " · ".join(shown), style=DIM)
    if event.get("resumed"):
        line.append(" (resumed)", style=DIM)
    return line


def header_line(key: str, *, is_agent: bool) -> Text:
    if is_agent:
        return Text.assemble((f"{BULLET} ", ACCENT), (key.upper(), f"bold {ACCENT}"))
    return Text(f"{BULLET} {key}", style=DIM)


def model_failure_line(event: dict) -> Text:
    reason = _first_line(event.get("err"))
    return _elbow((f"{CROSS} model call failed: {reason}", ERROR))


def tool_line(event: dict) -> Text:
    label = tool_label(event.get("tool"), event.get("path"))
    if event.get("denied"):
        reason = event["denied"]
        suffix = f" ({reason})" if isinstance(reason, str) else ""
        return _elbow((f"{CROSS} {label} denied{suffix}", ERROR))
    if event.get("ok") is False:
        detail = _first_line(event.get("err"))
        return _elbow((f"{CROSS} {label}{': ' + detail if detail else ''}", WARN))
    line = _elbow((label, ""))
    if isinstance(event.get("ms"), int | float):
        line.append(f" · {format_ms(event['ms'])}", style=DIM)
    return line


def check_line(gate: str, result: str) -> Text:
    name = _CHECK_NAMES.get(gate, _human(gate))
    if result == "pass":
        return Text(f"{BULLET} {name} passed", style=OK)
    return Text(f"{BULLET} {name} failed", style=ERROR)


def decision_line(gate: str, result: str) -> Text:
    title = _GATE_TITLES.get(gate, _human(gate))
    verdict = _DECISIONS.get(result, result)
    return Text.assemble((f"{BULLET} ", ACCENT), f"{title} · {verdict}")


def run_end_line(event: dict) -> Text:
    outcome = str(event.get("outcome", "?"))
    parts = [f"Run {outcome}"]
    if event.get("reason"):
        where = f" at {event['node']}" if event.get("node") else ""
        parts.append(f"{event['reason']}{where}")
    if isinstance(event.get("ms"), int | float):
        parts.append(format_elapsed(event["ms"] / 1000))
    style = _OUTCOME_STYLES.get(outcome, WARN)
    return Text(_OUTCOME_GLYPHS.get(outcome, "") + " · ".join(parts), style=style)


# --- the renderer -----------------------------------------------------------


@dataclass
class _Stretch:
    """One agent's (or helper's) uninterrupted run of activity."""

    key: str
    is_agent: bool
    started: float
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    activity: str = "thinking"  # what the spinner says the agent is doing
    last_seen: float = 0.0  # clock time of the latest sign of life

    @property
    def label(self) -> str:
        return self.key.upper() if self.is_agent else self.key


class LiveRenderer:
    """Progress sink: `renderer(event)` once per event, in order."""

    def __init__(
        self,
        console: Console,
        *,
        spinner: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._console = console
        self._spinner = spinner
        self._clock = clock
        self._live: Live | None = None
        self._stretch: _Stretch | None = None
        self._handlers: dict[str, Callable[[dict], None]] = {
            "run_start": self._on_run_start,
            "working": self._on_working,
            "model": self._on_model,
            "tool": self._on_tool,
            "gate": self._on_gate,
            "run_end": self._on_run_end,
        }

    def __call__(self, event: dict) -> None:
        handler = self._handlers.get(str(event.get("kind")))
        if handler is not None:
            handler(event)

    def pause(self) -> None:
        """Stop the spinner so the caller can read input; it restarts on
        the next model or tool event."""
        self._stop_spinner()

    def close(self) -> None:
        self._stop_spinner()

    def new_turn(self) -> None:
        """The human spoke: the next agent prints its header again."""
        self._stop_spinner()
        self._stretch = None

    # event handlers

    def _on_run_start(self, event: dict) -> None:
        self._stop_spinner()
        self._stretch = None
        self._console.print(run_start_line(event))

    def _on_working(self, event: dict) -> None:
        """A model or tool call just started: show who and what, now."""
        stretch = self._enter(event)
        if stretch is not None:
            stretch.activity = tool_label(event.get("tool"), event.get("path")) if event.get("tool") else "thinking"
            stretch.last_seen = self._clock()

    def _on_model(self, event: dict) -> None:
        stretch = self._enter(event)
        if stretch is not None:
            stretch.last_seen = self._clock()
            stretch.calls += 1
            stretch.tokens_in += _as_int(event.get("tokens_in"))
            stretch.tokens_out += _as_int(event.get("tokens_out"))
        if event.get("ok") is False:
            self._console.print(model_failure_line(event))

    def _on_tool(self, event: dict) -> None:
        stretch = self._enter(event)
        if stretch is not None:
            stretch.last_seen = self._clock()
            stretch.activity = "thinking"
        self._console.print(tool_line(event))

    def _on_gate(self, event: dict) -> None:
        result = str(event.get("result", ""))
        gate = str(event.get("gate") or "gate")
        if result == "waiting":
            self._stop_spinner()
        elif result in ("pass", "fail"):
            self._console.print(check_line(gate, result))
        else:
            self._console.print(decision_line(gate, result))
            self._print_redo(event.get("redo"))

    def _on_run_end(self, event: dict) -> None:
        self._stop_spinner()
        self._stretch = None
        self._console.print(run_end_line(event))

    def _print_redo(self, redo: Any) -> None:
        for item in redo or ():
            self._console.print(_elbow((_first_line(item), DIM)))

    # active agent and spinner

    def _enter(self, event: dict) -> _Stretch | None:
        """Switch to the event's agent/helper (printing its header when it
        changes) and make sure the spinner is running."""
        key = event.get("agent") or event.get("helper")
        if key:
            self._switch_to(str(key), is_agent="agent" in event)
        self._start_spinner()
        return self._stretch

    def _switch_to(self, key: str, *, is_agent: bool) -> None:
        current = self._stretch
        if current is not None and current.key == key:
            return
        self._stretch = _Stretch(key, is_agent, self._clock())
        self._console.print(header_line(key, is_agent=is_agent))

    def _start_spinner(self) -> None:
        if not self._spinner or self._live is not None:
            return
        self._live = Live(
            get_renderable=self._spinner_text,
            console=self._console,
            transient=True,
            refresh_per_second=_SPINNER_FPS,
        )
        self._live.start()

    def _stop_spinner(self) -> None:
        live, self._live = self._live, None
        if live is not None:
            live.stop()

    def _spinner_text(self) -> Text:
        stretch = self._stretch
        if stretch is None:
            return Text("")
        now = self._clock()
        frame = SPINNER_FRAMES[int(now * _SPINNER_FPS) % len(SPINNER_FRAMES)]
        quiet = now - (stretch.last_seen or stretch.started)
        if quiet >= STALL_SECONDS:
            return _stalled_text(frame, stretch, quiet)
        stats = _spinner_stats(stretch, now - stretch.started)
        return Text.assemble(
            (f"{frame} ", ACCENT),
            (f"{stretch.label} · {stretch.activity}…", ACCENT),
            (f" ({stats})", DIM),
        )


# No sign of life for this long and the spinner says so, in amber.
STALL_SECONDS = 90


def _stalled_text(frame: str, stretch: _Stretch, quiet: float) -> Text:
    return Text.assemble(
        (f"{frame} ", WARN),
        (f"{stretch.label} · no response for {format_elapsed(quiet)}", WARN),
        (" · ctrl-c pauses the run", DIM),
    )


def _spinner_stats(stretch: _Stretch, elapsed: float) -> str:
    noun = "call" if stretch.calls == 1 else "calls"
    return (
        f"{format_elapsed(elapsed)} · {stretch.calls} {noun}"
        f" · ↑ {format_tokens(stretch.tokens_in)}"
        f" ↓ {format_tokens(stretch.tokens_out)}"
    )


# --- gate panel -------------------------------------------------------------

_MARKDOWN_STARTS = ("## Agent Brief", "## Review")
_PR_PREFIX = re.compile(r"^PR \d+\n\n")
_STATUS_LINE = re.compile(r"^(\?\?| [MAD]|[MAD] |[MAD]{2}) ")
_DIFF_HEADERS = ("diff --git", "+++", "---", "@@")


def _gate_title(gate: str) -> str:
    return _GATE_TITLES.get(gate, _human(gate))


def _is_markdown(detail: str) -> bool:
    return _PR_PREFIX.sub("", detail, count=1).startswith(_MARKDOWN_STARTS)


_LINE_RULES: tuple[tuple[Callable[[str], bool], str], ...] = (
    (lambda line: line.startswith(_DIFF_HEADERS), f"bold {DIM}"),
    (lambda line: line.startswith("+"), OK),
    (lambda line: line.startswith("-"), ERROR),
    (lambda line: line.startswith("## "), f"bold {ACCENT}"),
    (lambda line: "FAILED" in line or "Error" in line, ERROR),
    (lambda line: _STATUS_LINE.match(line) is not None, WARN),
)


def _line_style(line: str) -> str:
    return next((style for matches, style in _LINE_RULES if matches(line)), "")


def style_detail(lines: list[str]) -> Text:
    """One `Text`, each line styled by what it looks like."""
    body = Text()
    for index, line in enumerate(lines):
        body.append("\n" if index else "")
        body.append(line, style=_line_style(line))
    return body


def _gate_body(detail: str) -> RenderableType:
    lines = detail.splitlines()
    shown, hidden = lines[:_GATE_MAX_LINES], len(lines) - _GATE_MAX_LINES
    if _is_markdown(detail):
        body: RenderableType = Markdown("\n".join(shown))
    else:
        body = style_detail(shown)
    if hidden <= 0:
        return body
    return Group(body, Text(f"… {hidden} more lines", style=DIM))


def render_gate(console: Console, *, gate: str, question: str, detail: str) -> None:
    """Print a human gate as a rounded panel holding `detail`.

    `question` is not printed: the option selector shows it.
    """
    console.print(
        Panel(
            _gate_body(detail),
            box=ROUNDED,
            border_style=BORDER,
            title=Text(_gate_title(gate), style=ACCENT),
            title_align="left",
            padding=(0, 1),
        )
    )
