"""The bulk_read helper and its agent tool.

``bulk_read`` is the stage: wrap files in XML and ask a chat model about
them. ``make_bulk_read`` wraps it as a ``BaseTool`` every agent may call:
paths go through the readable-file path guard, and the model call is a
one-shot ``helpers.bulk_read`` invocation whose ``model`` events carry
``helper`` (see ``k0ntrol.helpers``).
"""

from collections.abc import Callable
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool, tool

from k0ntrol.helpers import helper_invoker
from k0ntrol.tools.read import guard_readable


class BulkReadError(Exception):
    """Raised when bulk-read cannot prepare a file for the model."""


Invoke = Callable[[str], str]


def make_bulk_read(
    root: Path, cfg: dict | None, *, init: Callable = init_chat_model
) -> BaseTool:
    """Build the ``bulk_read`` tool bound to ``root`` and its helper config."""
    invoke = helper_invoker(cfg, "bulk_read", init=init)

    @tool("bulk_read")
    def read_files(question: str, paths: list[str]) -> str:
        """Answer `question` about several files at once via the bulk_read
        helper. `paths` are relative to the repository root."""
        resolved = [guard_readable(path, root) for path in paths]
        return bulk_read(
            question, resolved, root=root, repo=root.name, invoke=invoke
        )

    return read_files


def bulk_read(
    question: str,
    paths: list[Path],
    *,
    root: Path,
    repo: str,
    invoke: Invoke,
) -> str:
    """Read files, wrap them in XML, and invoke the model with the question."""
    root_path = root.resolve()
    wrapped_files = [
        _wrap_file(path, root_path, repo)
        for path in paths
    ]
    prompt = f"{question}\n\n" + "\n".join(wrapped_files)
    return invoke(prompt)


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
