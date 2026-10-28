"""The ``change`` module: deterministic git side effects for the graphs.

#41 needs only reads — the current branch for ``run_start`` and the
working-tree summary the diff gate shows. #44/#45 add the write side
(branch, memory commits, commit + push + draft PR, stuck side effects)
as marker-idempotent nodes through this module.
"""

import subprocess
from pathlib import Path

# Cap on the diff text shown at a gate.
_SUMMARY_MAX_CHARS = 12_000


def current_branch(root: Path) -> str | None:
    """The checkout's branch name, or ``None`` outside a git repo."""
    result = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def worktree_summary(root: Path) -> str:
    """Status plus the tracked diff and excerpts of new untracked files.

    ``git diff`` never shows untracked files — which is exactly what
    agents write — so new files get a ``--- /dev/null``-style excerpt.
    """
    status = _git(root, "status", "--porcelain").stdout
    parts = [status.rstrip(), _git(root, "diff", "HEAD", "--").stdout.rstrip()]
    for path in _untracked(status):
        parts.append(_new_file_excerpt(root, path))
    text = "\n".join(part for part in parts if part)
    if len(text) > _SUMMARY_MAX_CHARS:
        text = text[:_SUMMARY_MAX_CHARS] + "\n... diff truncated ..."
    return text


def _untracked(status: str) -> list[str]:
    return [
        line[3:] for line in status.splitlines() if line.startswith("?? ")
    ]


def _new_file_excerpt(root: Path, path: str, *, lines: int = 60) -> str:
    try:
        text = (root / path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return f"--- /dev/null\n+++ b/{path} (binary or unreadable)"
    body = "\n".join("+" + line for line in text.splitlines()[:lines])
    return f"--- /dev/null\n+++ b/{path}\n{body}"


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
