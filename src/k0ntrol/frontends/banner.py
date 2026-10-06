import io
import os
from typing import TextIO

from rich.box import ROUNDED
from rich.cells import cell_len
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from k0ntrol.frontends.logo import LOGO

_BORDER_COLOR = "#374151"
_LABEL_COLOR = "#9CA3AF"
_GRADIENT_START = (94, 234, 212)
_GRADIENT_END = (167, 139, 250)


def render_banner(*, version: str, backend: str, cwd: str, color: bool) -> str:
    body = _build_body(version, backend, cwd)
    panel = Panel(
        body,
        box=ROUNDED,
        border_style=_BORDER_COLOR,
        padding=(0, 1),
        expand=False,
    )
    output = io.StringIO()
    console = Console(
        file=output,
        no_color=not color,
        force_terminal=color,
        color_system="truecolor" if color else None,
        width=max(80, _body_width(body) + 4),
    )
    console.print(panel)
    return output.getvalue()


def color_enabled(stream: TextIO) -> bool:
    return "NO_COLOR" not in os.environ and stream.isatty()


def _body_width(body: Text) -> int:
    return max(cell_len(line) for line in body.plain.splitlines())


def _build_body(version: str, backend: str, cwd: str) -> Text:
    labels = {
        2: f"k0ntrol v{version}",
        3: backend,
        4: cwd,
    }
    body = Text()
    for index, logo_line in enumerate(LOGO.splitlines()):
        if index:
            body.append("\n")
        body.append(_logo_line(logo_line))
        label = labels.get(index)
        if label is not None:
            body.append("  ")
            body.append(label, style=_LABEL_COLOR)
    return body


def _logo_line(line: str) -> Text:
    rendered = Text()
    for column, character in enumerate(line):
        if character == " ":
            rendered.append(character)
        else:
            rendered.append(character, style=_logo_color(column))
    return rendered


def _logo_color(column: int) -> str:
    ratio = column / 24
    channels = [
        round(start + (end - start) * ratio)
        for start, end in zip(_GRADIENT_START, _GRADIENT_END)
    ]
    return f"rgb({channels[0]},{channels[1]},{channels[2]})"
