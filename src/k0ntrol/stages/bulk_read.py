from collections.abc import Callable
from pathlib import Path

from k0ntrol.modespec import ModeSpec


class BulkReadError(Exception):
    """Raised when bulk-read cannot prepare a file for the mode."""


Invoke = Callable[[ModeSpec, str], str]


def bulk_read(
    question: str,
    paths: list[Path],
    *,
    root: Path,
    repo: str,
    mode: ModeSpec,
    invoke: Invoke,
) -> str:
    """Read files, wrap them in XML, and invoke the bulk-reader mode."""
    root_path = root.resolve()
    wrapped_files = [
        _wrap_file(path, root_path, repo)
        for path in paths
    ]
    prompt = f"{question}\n\n" + "\n".join(wrapped_files)
    return invoke(mode, prompt)


def _wrap_file(path: Path, root: Path, repo: str) -> str:
    relative_path = _relative_path(path, root)
    content = _read_file(path)
    return (
        f'<file repo="{_escape(repo)}" path="{_escape(relative_path)}">'
        f"{_escape(content)}</file>"
    )


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise BulkReadError(f"path outside root: {path}") from error


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise BulkReadError(f"unable to read {path}: {error}") from error


def _escape(value: str) -> str:
    replacements = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
    }
    return "".join(replacements.get(character, character) for character in value)
