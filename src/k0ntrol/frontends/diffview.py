"""A git diff drawn for a human gate, one panel per file.

A file that only gains lines (or only loses them) shows one column; a file
whose existing code changed shows old and new side by side, each removed
run paired with the added run that replaced it.
"""

import re
from dataclasses import dataclass
from itertools import groupby, zip_longest

from rich.box import ROUNDED
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from k0ntrol.frontends.theme import BORDER, DIM, ERROR, OK

_DIFF_MAX_LINES = 400
_HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_FILE_START = re.compile(r"(?m)^(?=diff --git )")
_PATH_PREFIXES = ("+++ b/", "--- a/")  # both six characters long
_MARKS = {" ": ("  ", ""), "-": ("- ", ERROR), "+": ("+ ", OK), "⋯": ("⋯", DIM)}


@dataclass(frozen=True)
class Line:
    """One diff line: kind ``" "``, ``"-"``, ``"+"`` or ``"⋯"`` (hunk break)."""

    kind: str
    text: str
    old: int | None
    new: int | None


_BREAK = Line("⋯", "", None, None)


@dataclass(frozen=True)
class FileDiff:
    path: str
    lines: tuple[Line, ...]

    def count(self, kind: str) -> int:
        return sum(1 for line in self.lines if line.kind == kind)

    @property
    def one_sided(self) -> bool:
        """Only additions or only removals: nothing to put side by side."""
        return not (self.count("+") and self.count("-"))


class _Numbering:
    """Old and new line numbers as a hunk is read top to bottom."""

    def __init__(self) -> None:
        self.old = self.new = 0

    def read(self, raw: str) -> Line:
        match = _HUNK.match(raw)
        if match:
            self.old, self.new = int(match[1]), int(match[2])
            return _BREAK
        kind = raw[:1] if raw[:1] in ("-", "+") else " "
        line = Line(kind, raw[1:], None if kind == "+" else self.old, None if kind == "-" else self.new)
        self.old += kind != "+"
        self.new += kind != "-"
        return line


def parse_diff(diff: str) -> list[FileDiff]:
    """A ``git diff`` split into files, each with numbered lines."""
    chunks = _FILE_START.split(diff)
    return [_parse_file(chunk) for chunk in chunks if chunk.startswith("diff --git ")]


def _parse_file(chunk: str) -> FileDiff:
    raw = chunk.splitlines()
    start = next((i for i, line in enumerate(raw) if line.startswith("@@")), len(raw))
    numbering = _Numbering()
    lines = [numbering.read(line) for line in raw[start:] if not line.startswith("\\")]
    return FileDiff(_path(raw[:start]), tuple(lines[1:]))  # [1:]: no break before the first hunk


def _path(header: list[str]) -> str:
    """``+++ b/<path>``, else ``--- a/<path>`` (deleted), else the git line."""
    marked = [line[6:] for line in header if line.startswith(_PATH_PREFIXES)]
    return marked[-1] if marked else header[0].rpartition(" b/")[2]


def side_by_side(lines: tuple[Line, ...]) -> list[tuple[Line | None, Line | None]]:
    """``(old, new)`` rows: context on both sides, removed runs beside the
    added runs that replaced them, a blank where one side is shorter."""
    rows: list[tuple[Line | None, Line | None]] = []
    for changed, run in groupby(lines, key=lambda line: line.kind in ("-", "+")):
        block = list(run)
        rows += _paired(block) if changed else [(line, line) for line in block]
    return rows


def _paired(block: list[Line]) -> list[tuple[Line | None, Line | None]]:
    removed = [line for line in block if line.kind == "-"]
    added = [line for line in block if line.kind == "+"]
    return list(zip_longest(removed, added))


# --- rendering --------------------------------------------------------------


def render_diff(diff: str) -> RenderableType:
    """Every file of ``diff`` as a panel, cut at ``_DIFF_MAX_LINES`` lines."""
    lines = diff.splitlines()
    shown = "\n".join(lines[:_DIFF_MAX_LINES])
    parts: list[RenderableType] = [_file_panel(file) for file in parse_diff(shown)]
    hidden = len(lines) - _DIFF_MAX_LINES
    if hidden > 0:
        parts.append(Text(f"… {hidden} more diff lines", style=DIM))
    return Group(*parts)


def _file_panel(file: FileDiff) -> Panel:
    title = Text.assemble(
        (file.path, "bold default"), (f" +{file.count('+')}", OK), (f" -{file.count('-')}", ERROR)
    )
    return Panel(
        _file_body(file),
        box=ROUNDED,
        border_style=BORDER,
        title=title,
        title_align="left",
        padding=(0, 1),
    )


def _file_body(file: FileDiff) -> RenderableType:
    if not file.lines:
        return Text("no text changes", style=DIM)
    return _one_column(file) if file.one_sided else _two_columns(file)


def _one_column(file: FileDiff) -> Table:
    side = "new" if file.count("+") else "old"
    grid = _grid(columns=1)
    for line in file.lines:
        grid.add_row(*_cell(line, side))
    return grid


def _two_columns(file: FileDiff) -> Table:
    grid = _grid(columns=2)
    for old, new in side_by_side(file.lines):
        grid.add_row(*_cell(old, "old"), Text("│", style=BORDER), *_cell(new, "new"))
    return grid


def _grid(*, columns: int) -> Table:
    """A number column and a text column per side, a rule between sides."""
    grid = Table.grid(padding=(0, 1), expand=True)
    for index in range(columns):
        if index:
            grid.add_column()  # the rule between old and new
        grid.add_column(justify="right")
        grid.add_column(ratio=1, overflow="fold")
    return grid


def _cell(line: Line | None, side: str) -> tuple[Text, Text]:
    if line is None:
        return Text(""), Text("")
    number = line.old if side == "old" else line.new
    sign, style = _MARKS[line.kind]
    shown = "" if number is None else str(number)
    return Text(shown, style=DIM), Text(sign + line.text, style=style)
