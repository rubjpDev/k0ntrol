"""Read tools: ``read_file``, ``list_files`` and ``grep``.

Read tools see only the files git sees: ``git ls-files --cached --others
--exclude-standard`` under the root (ADR 0006). An ignored ``.env`` never
reaches a provider. Every tool is a ``BaseTool`` built by a factory that
closes over the repo root.
"""

import re
import subprocess
from pathlib import Path, PurePosixPath

from langchain_core.tools import BaseTool, tool

from k0ntrol.tools.errors import ToolError
from k0ntrol.tools.path_guard import guard_path

_LIST_LIMIT = 500
_MATCH_LIMIT = 200


def visible_files(root: Path) -> set[str]:
    """Root-relative paths git tracks or would track (not ignored)."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ToolError(f"git ls-files failed: {result.stderr.strip()}")
    return {line for line in result.stdout.splitlines() if line}


def _guard_readable(path: str, root: Path) -> Path:
    resolved = guard_path(path, root)
    relative = resolved.relative_to(root.resolve()).as_posix()
    if relative not in visible_files(root):
        raise ToolError(f"not a readable file: {path}", denied="path_guard")
    return resolved


def _read_lines(path: Path) -> list[str] | None:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None


def _filtered_glob(pattern: str, root: Path) -> list[str]:
    try:
        candidates = root.glob(pattern)
    except (NotImplementedError, ValueError) as error:
        raise ToolError(f"bad glob pattern: {pattern}") from error
    visible = visible_files(root)
    matches = []
    for candidate in candidates:
        relative = _relative_if_visible(candidate, root, visible)
        if relative is not None:
            matches.append(relative)
    return sorted(matches)


def _relative_if_visible(path: Path, root: Path, visible: set[str]) -> str | None:
    if not path.is_file():
        return None
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None
    return relative if relative in visible else None


def make_read_file(root: Path) -> BaseTool:
    """Build the ``read_file`` tool bound to ``root``."""

    @tool
    def read_file(path: str) -> str:
        """Read one file in the repository. `path` is relative to the repo root."""
        resolved = _guard_readable(path, root)
        if not resolved.is_file():
            raise ToolError(f"not a readable file: {path}")
        lines = _read_lines(resolved)
        if lines is None:
            raise ToolError(f"unable to read {path}")
        return "\n".join(lines)

    return read_file


def make_list_files(root: Path) -> BaseTool:
    """Build the ``list_files`` tool bound to ``root``."""

    @tool
    def list_files(pattern: str = "**/*") -> str:
        """List repository files matching a glob, relative to the repo root."""
        matches = _filtered_glob(pattern, root)
        if len(matches) > _LIST_LIMIT:
            matches = matches[:_LIST_LIMIT] + [f"... truncated at {_LIST_LIMIT}"]
        return "\n".join(matches)

    return list_files


def make_grep(root: Path) -> BaseTool:
    """Build the ``grep`` tool bound to ``root``."""

    @tool
    def grep(pattern: str, path_glob: str | None = None) -> str:
        """Search repository files for a regex. Returns `path:line: text` matches."""
        regex = _compile(pattern)
        files = _search_targets(path_glob, root)
        return _collect_matches(regex, files, root)

    return grep


def _compile(pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as error:
        raise ToolError(f"bad regex: {error}") from error


def _search_targets(path_glob: str | None, root: Path) -> list[str]:
    visible = sorted(visible_files(root))
    if path_glob is None:
        return visible
    return [name for name in visible if PurePosixPath(name).match(path_glob)]


def _collect_matches(
    regex: re.Pattern[str], files: list[str], root: Path
) -> str:
    matches = []
    for name in files:
        matches.extend(_file_matches(regex, name, root))
        if len(matches) >= _MATCH_LIMIT:
            break
    if len(matches) > _MATCH_LIMIT:
        matches = matches[:_MATCH_LIMIT] + [f"... truncated at {_MATCH_LIMIT}"]
    return "\n".join(matches)


def _file_matches(regex: re.Pattern[str], name: str, root: Path) -> list[str]:
    lines = _read_lines(root / name)
    if lines is None:
        return []
    return [
        f"{name}:{number}:{line}"
        for number, line in enumerate(lines, start=1)
        if regex.search(line)
    ]
