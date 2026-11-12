"""Write tools: ``write_file`` and ``edit_file``.

The ``"test"`` flavour allows test paths only and ``"non_test"`` the rest,
so the two batch agents cannot cheat each other (spec user story 9);
IMPLEMENTER gets a ``SliceGuard`` instead, whose green/red state decides
which path kinds may be written (ADR 0011). New files are allowed. The
check runs on the resolved path, after the path guard's root and
protected-path checks.
"""

from pathlib import Path
from typing import Literal

from langchain_core.tools import BaseTool, tool

from k0ntrol.tools.errors import ToolError
from k0ntrol.tools.path_guard import guard_path, is_test_path
from k0ntrol.tools.slice_guard import SliceGuard

WriteScope = Literal["test", "non_test"]
_SCOPES: tuple[WriteScope, ...] = ("test", "non_test")
WritePolicy = WriteScope | SliceGuard


class _StaticScope:
    """A fixed path flavour as a write policy: test-only or non-test-only."""

    def __init__(self, scope: WriteScope) -> None:
        self._scope = scope

    def check_write(self, relative: str) -> None:
        if self._scope == "test" and not is_test_path(relative):
            raise ToolError(
                f"TESTER writes test paths only: {relative}", denied="path_guard"
            )
        if self._scope == "non_test" and is_test_path(relative):
            raise ToolError(
                f"CODER writes non-test paths only: {relative}", denied="path_guard"
            )

    def note_write(self, relative: str) -> None:
        """Nothing to record: a static flavour has no state."""


def _check_scope(scope: WritePolicy) -> SliceGuard | _StaticScope:
    if isinstance(scope, SliceGuard):
        return scope
    for candidate in _SCOPES:
        if scope == candidate:
            return _StaticScope(candidate)
    raise ValueError(f"scope must be one of {_SCOPES} or a SliceGuard, got {scope!r}")


def _guard_writable(
    path: str, root: Path, policy: SliceGuard | _StaticScope
) -> tuple[Path, str]:
    resolved = guard_path(path, root)
    relative = resolved.relative_to(root.resolve()).as_posix()
    policy.check_write(relative)
    return resolved, relative


def make_write_file(root: Path, scope: WritePolicy) -> BaseTool:
    """Build the ``write_file`` tool: write a whole file inside ``root``."""
    checked = _check_scope(scope)

    @tool
    def write_file(path: str, content: str) -> str:
        """Write `content` to `path`, relative to the repo root. Creates dirs."""
        resolved, relative = _guard_writable(path, root, checked)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        checked.note_write(relative)
        return f"wrote {relative}"

    return write_file


def make_edit_file(root: Path, scope: WritePolicy) -> BaseTool:
    """Build the ``edit_file`` tool: replace one unique text span in a file."""
    checked = _check_scope(scope)

    @tool
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace the single occurrence of `old` with `new` in `path`."""
        resolved, relative = _guard_writable(path, root, checked)
        text = _existing_text(resolved, path)
        replaced = _replace_once(text, old, new, path)
        resolved.write_text(replaced, encoding="utf-8")
        checked.note_write(relative)
        return f"edited {relative}"

    return edit_file


def _existing_text(path: Path, relative: str) -> str:
    if not path.is_file():
        raise ToolError(f"no such file: {relative}")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ToolError(f"unable to read {relative}: {error}") from error


def _replace_once(text: str, old: str, new: str, path: str) -> str:
    count = text.count(old)
    if count == 0:
        raise ToolError(f"old text not found in {path}")
    if count > 1:
        raise ToolError(f"old text is ambiguous in {path} ({count} matches)")
    return text.replace(old, new, 1)
