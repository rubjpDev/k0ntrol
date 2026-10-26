"""Write tools: ``write_file`` and ``edit_file`` in two flavours.

TESTER gets the ``"test"`` flavour (test paths only) and CODER the
``"non_test"`` flavour, so neither can cheat the other (spec user story 9).
New files are allowed. The flavour check runs on the resolved path, after
the path guard's root and protected-path checks.
"""

from pathlib import Path
from typing import Literal

from langchain_core.tools import BaseTool, tool

from k0ntrol.tools.errors import ToolError
from k0ntrol.tools.path_guard import guard_path, is_test_path

WriteScope = Literal["test", "non_test"]
_SCOPES: tuple[WriteScope, ...] = ("test", "non_test")


def _guard_writable(path: str, root: Path, scope: WriteScope) -> Path:
    resolved = guard_path(path, root)
    relative = resolved.relative_to(root.resolve()).as_posix()
    if scope == "test" and not is_test_path(relative):
        raise ToolError(f"TESTER writes test paths only: {path}", denied="path_guard")
    if scope == "non_test" and is_test_path(relative):
        raise ToolError(f"CODER writes non-test paths only: {path}", denied="path_guard")
    return resolved


def _check_scope(scope: str) -> WriteScope:
    if scope not in _SCOPES:
        raise ValueError(f"scope must be one of {_SCOPES}, got {scope!r}")
    return scope  # type: ignore[return-value]


def make_write_file(root: Path, scope: WriteScope) -> BaseTool:
    """Build the ``write_file`` tool: write a whole file inside ``root``."""
    checked = _check_scope(scope)

    @tool
    def write_file(path: str, content: str) -> str:
        """Write `content` to `path`, relative to the repo root. Creates dirs."""
        resolved = _guard_writable(path, root, checked)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"wrote {resolved.relative_to(root.resolve()).as_posix()}"

    return write_file


def make_edit_file(root: Path, scope: WriteScope) -> BaseTool:
    """Build the ``edit_file`` tool: replace one unique text span in a file."""
    checked = _check_scope(scope)

    @tool
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace the single occurrence of `old` with `new` in `path`."""
        resolved = _guard_writable(path, root, checked)
        text = _existing_text(resolved, path)
        replaced = _replace_once(text, old, new, path)
        resolved.write_text(replaced, encoding="utf-8")
        return f"edited {resolved.relative_to(root.resolve()).as_posix()}"

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
