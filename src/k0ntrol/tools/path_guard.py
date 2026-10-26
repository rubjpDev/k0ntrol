"""The path guard: resolve symlinks, then check root and protected paths.

ADR 0006: every read and write tool resolves symlinks first, then checks the
resolved path against the root, the test/non-test class and the protected
paths (``.git/``, ``.k0-mem/``). A refusal is a ``ToolError`` the agent sees,
tagged ``denied="path_guard"`` for the event log.
"""

from pathlib import Path, PurePosixPath

from k0ntrol.tools.errors import ToolError

PROTECTED_DIRS = (".git", ".k0-mem")


def is_test_path(relative: str) -> bool:
    """Classify a root-relative path as a test path.

    Today's rule (a name starting ``test_``, ending ``_test.py``, or a
    ``tests`` path part) plus ``conftest.py`` anywhere.
    """
    path = PurePosixPath(relative)
    return (
        path.name == "conftest.py"
        or path.name.startswith("test_")
        or path.name.endswith("_test.py")
        or "tests" in path.parts
    )


def guard_path(path: str | Path, root: Path) -> Path:
    """Resolve ``path`` under ``root`` and refuse escapes and protected paths.

    Returns the resolved absolute path. Raises ``ToolError`` with
    ``denied="path_guard"`` when the resolved path lands outside the root or
    inside a protected directory. Works for files that do not exist yet.
    """
    root_resolved = root.resolve()
    resolved = (root_resolved / path).resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ToolError(f"path outside root: {path}", denied="path_guard")
    relative = resolved.relative_to(root_resolved)
    if relative.parts and relative.parts[0] in PROTECTED_DIRS:
        raise ToolError(f"protected path: {path}", denied="path_guard")
    return resolved
