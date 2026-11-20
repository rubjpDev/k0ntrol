"""The slice guard: IMPLEMENTER's write tools hold red-before-green (ADR 0011).

One ``SliceGuard`` is shared by an IMPLEMENTER run's ``write_file``,
``edit_file`` and ``run_tests``. Runs start **green**, or **refactor** for a
review redo: test files may be written and code writes are refused in green;
code may be written and test writes are refused in refactor. A red
``run_tests`` result after a test write moves green to **red**: code may be
written and test writes are refused, so a test cannot be weakened to go
green. A green result moves back. ``run_tests`` verdicts do not leave
refactor. The first test write attempted in refactor returns the guard to
green. Markdown under ``docs/`` and ``GLOSSARY.md`` pass in any state.
"""

import threading
from pathlib import Path, PurePosixPath
from typing import Literal

from k0ntrol.tools.errors import SliceRecoveryNeeded, ToolError
from k0ntrol.tools.path_guard import is_test_path
from k0ntrol.tools.slice_journal import SliceJournal

SliceState = Literal["green", "red", "refactor"]
SliceStart = Literal["green", "refactor"]
PathKind = Literal["test", "docs", "code"]

# Denied test writes in one red slice before the guard stops answering the
# model and escalates for a human recovery decision (#174).
MAX_SLICE_DENIALS = 3


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
    """The green/red/refactor state one IMPLEMENTER run's write tools share.

    ``lock`` serializes those tools: a write holds it across its check,
    filesystem mutation and note, and ``run_tests`` holds it across the
    whole command and verdict. The tool node runs one model turn's tool
    calls on a thread pool (ADR 0007) — without it a write's check could
    pass green while a concurrent red verdict lands before its mutation.
    """

    def __init__(
        self, start: SliceStart = "green", journal: SliceJournal | None = None
    ) -> None:
        self._state: SliceState = start
        self._test_written = False
        self._red_denials = 0
        self._journal = journal
        self.lock = threading.Lock()

    @property
    def state(self) -> SliceState:
        return self._state

    def check_write(self, relative: str) -> None:
        """Refuse a write the current state does not allow."""
        kind = _path_kind(relative)
        if kind == "docs":
            return
        if self._state == "refactor":
            self._check_refactor_write(kind, relative)
            return
        self._check_cycle_write(kind, relative)

    def _check_refactor_write(self, kind: PathKind, relative: str) -> None:
        if kind != "test":
            return
        self._state = "green"
        if self._journal is not None:
            self._journal.clear()
        raise ToolError(
            f"slice guard is refactor: {relative} is a test write — "
            "the first test write attempt returns the guard to green",
            denied="slice_guard",
        )

    def _check_cycle_write(self, kind: PathKind, relative: str) -> None:
        if self._state == "green" and kind == "code":
            raise ToolError(
                f"slice guard is green: {relative} is a code write — "
                "write a failing test, then a red run_tests unlocks code",
                denied="slice_guard",
            )
        if self._state == "red" and kind == "test":
            self._red_denials += 1
            if self._red_denials >= MAX_SLICE_DENIALS:
                journal = self._journal
                raise SliceRecoveryNeeded(
                    f"slice guard refused {MAX_SLICE_DENIALS} test writes in "
                    "one red slice — the test file itself may be broken; "
                    "stopping for a human recovery decision",
                    slice=journal.slice if journal else None,
                    paths=journal.paths if journal else [],
                )
            raise ToolError(
                f"slice guard is red: {relative} is a test write — "
                "a green run_tests unlocks test writes",
                denied="slice_guard",
            )

    def record_write(self, resolved: Path, relative: str) -> None:
        """Capture the pre-write image in the slice journal. A journaled
        path whose bytes no longer match what the slice last wrote —
        someone else changed it — refuses the write rather than carrying
        a stale pre-image over it (#174)."""
        if self._journal is None:
            return
        conflict = self._journal.record(resolved, relative)
        if conflict is None:
            return
        if self._state == "red" and _path_kind(relative) == "test":
            self._red_denials += 1
            if self._red_denials >= MAX_SLICE_DENIALS:
                raise SliceRecoveryNeeded(
                    f"slice guard refused {MAX_SLICE_DENIALS} test writes in "
                    "one red slice — the test file itself may be broken; "
                    "stopping for a human recovery decision",
                    slice=self._journal.slice,
                    paths=self._journal.paths,
                )
        raise ToolError(f"slice journal: {conflict}", denied="slice_guard")

    def note_write(self, resolved: Path, relative: str) -> None:
        """Record a successful write: journal its post-image and, for a
        test-path write, arm the guard."""
        if self._journal is not None:
            self._journal.note(resolved, relative)
        if _path_kind(relative) == "test":
            self._test_written = True

    def note_test_result(self, ok: bool) -> None:
        """Flip green/red on a ``run_tests`` verdict, then re-arm."""
        if self._state == "refactor":
            return
        if ok:
            self._state = "green"
            self._red_denials = 0
            if self._journal is not None:
                self._journal.clear()
        elif self._test_written:
            self._state = "red"
        self._test_written = False
