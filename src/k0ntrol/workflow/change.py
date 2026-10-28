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


def diff_stat(root: Path) -> str:
    """``git diff --stat HEAD`` plus untracked paths, for gate details.

    Agent-written files are untracked and never appear in ``--stat``, so
    they are listed as ``(new)`` lines — otherwise the stat would show
    nothing while the tree is full of new work. ``.k0-mem/`` is runtime
    state, not part of the change, so it is left out.
    """
    stat = _git(root, "diff", "--stat", "HEAD", "--").stdout.strip()
    lines = [stat] if stat else []
    lines += [f" {path} | (new)" for path in _new_files(root)]
    return "\n".join(lines) or "no changes"


def _new_files(root: Path) -> list[str]:
    """Untracked, non-ignored file paths, without ``.k0-mem/``."""
    out = _git(root, "ls-files", "--others", "--exclude-standard").stdout
    return [p for p in out.splitlines() if p and not p.startswith(".k0-mem/")]


def _status(root: Path) -> str:
    return _git(root, "status", "--porcelain").stdout


def worktree_summary(root: Path) -> str:
    """Status plus the tracked diff and excerpts of new untracked files.

    ``git diff`` never shows untracked files — which is exactly what
    agents write — so new files get a ``--- /dev/null``-style excerpt.
    """
    status = _status(root)
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
