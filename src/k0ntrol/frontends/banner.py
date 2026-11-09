import io
import os
from typing import TextIO

from rich.box import ROUNDED
from rich.cells import cell_len
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from k0ntrol.frontends.badges import Badge, badge_block, block_width
from k0ntrol.frontends.logo import LOGO

_BORDER_COLOR = "#374151"
_LABEL_COLOR = "#9CA3AF"
_GRADIENT_START = (94, 234, 212)
_GRADIENT_END = (167, 139, 250)
# Panel border and padding (4), the logo (25) and the gap before the labels (2).
_LEFT_FIXED = 4 + 25 + 2
_BADGE_GAP = 3  # between the labels and the badges
_MIN_LABEL_ROOM = 20  # below this the badges give way to the path


def banner_layout(width: int, badges: list[Badge]) -> tuple[list[Badge], int]:
    """The badges that fit a ``width``-column banner and the room left for
    the path label; a narrow terminal drops the badges, never the path."""
    room = width - _LEFT_FIXED - _BADGE_GAP - block_width(badges)
    if not badges or room < _MIN_LABEL_ROOM:
        return [], width - 40
    return badges, room


def render_banner(
    *,
    version: str,
    provider: str | None,
    cwd: str,
    color: bool,
    badges: list[Badge] | None = None,
    width: int | None = None,
) -> str:
    """The welcome panel. With ``width`` it spans the terminal and the
    ``badges`` sit flush right; without, it takes its natural width."""
    left = _build_body(version, provider, cwd)
    badges = badges or []
    natural = _body_width(left) + (_BADGE_GAP + block_width(badges) if badges else 0)
    panel = Panel(
        _with_badges(left, badges),
        box=ROUNDED,
        border_style=_BORDER_COLOR,
        padding=(0, 1),
        expand=width is not None,
    )
    output = io.StringIO()
    console = Console(
        file=output,
        no_color=not color,
        force_terminal=color,
        color_system="truecolor" if color else None,
        width=width or max(80, natural + 4),
    )
    console.print(panel)
    return output.getvalue()


def _with_badges(left: Text, badges: list[Badge]) -> Text | Table:
    """The logo and labels, with the badge block on the right, one row down."""
    if not badges:
        return left
    # A stretching middle column pushes the block right; the block itself
    # stays left-justified so centred names and roles keep their columns.
    grid = Table.grid(padding=(0, _BADGE_GAP), expand=True)
    grid.add_column(no_wrap=True)
    grid.add_column(ratio=1)
    grid.add_column(no_wrap=True)
    grid.add_row(left, "", Text("\n").join([Text(""), *badge_block(badges)]))
    return grid


def color_enabled(stream: TextIO) -> bool:
    return "NO_COLOR" not in os.environ and stream.isatty()


def _body_width(body: Text) -> int:
    return max(cell_len(line) for line in body.plain.splitlines())


def _build_body(version: str, provider: str | None, cwd: str) -> Text:
    labels = {
        2: f"k0ntrol v{version}",
        4: cwd,
    }
    if provider:
        labels[3] = provider
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
