"""The slice guard: IMPLEMENTER's write tools hold red-before-green (ADR 0011).

One ``SliceGuard`` is shared by an IMPLEMENTER run's ``write_file``,
``edit_file`` and ``run_tests``. Every run starts **green**: test files may
be written and code writes are refused. A red ``run_tests`` result after a
test write moves to **red**: code may be written and test writes are
refused, so a test cannot be weakened to go green. A green result moves
back. Markdown under ``docs/`` and ``GLOSSARY.md`` pass in either state.
"""

from pathlib import PurePosixPath
from typing import Literal

from k0ntrol.tools.errors import ToolError
from k0ntrol.tools.path_guard import is_test_path

SliceState = Literal["green", "red"]
PathKind = Literal["test", "docs", "code"]


def _is_docs_path(relative: str) -> bool:
    """Markdown under ``docs/`` and ``GLOSSARY.md`` are docs, not code."""
    return relative == "GLOSSARY.md" or (
        relative.startswith("docs/") and PurePosixPath(relative).suffix == ".md"
    )


def _path_kind(relative: str) -> PathKind:
    """Classify a root-relative write path; docs win over the test rule."""
    if _is_docs_path(relative):
        return "docs"
    if is_test_path(relative):
        return "test"
    return "code"


class SliceGuard:
    """The green/red state one IMPLEMENTER run's write tools share."""

    def __init__(self) -> None:
        self._state: SliceState = "green"
        self._test_written = False

    @property
    def state(self) -> SliceState:
        return self._state

    def check_write(self, relative: str) -> None:
        """Refuse a write the current state does not allow."""
        kind = _path_kind(relative)
        if kind == "docs":
            return
        if self._state == "green" and kind == "code":
            raise ToolError(
                f"slice guard is green: {relative} is a code write — "
                "write a failing test, then a red run_tests unlocks code",
                denied="slice_guard",
            )
        if self._state == "red" and kind == "test":
            raise ToolError(
                f"slice guard is red: {relative} is a test write — "
                "a green run_tests unlocks test writes",
                denied="slice_guard",
            )

    def note_write(self, relative: str) -> None:
        """Record a successful write: a test-path write arms the guard."""
        if _path_kind(relative) == "test":
            self._test_written = True

    def note_test_result(self, ok: bool) -> None:
        """Flip the state on a ``run_tests`` verdict, then re-arm."""
        if ok:
            self._state = "green"
        elif self._test_written:
            self._state = "red"
        self._test_written = False
