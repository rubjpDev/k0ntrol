"""The ``change`` module: deterministic git side effects for the graphs.

#41 needs only reads — the current branch for ``run_start`` and the
working-tree summary the diff gate shows. #44/#45 add the write side
(branch, memory commits, commit + push + draft PR, stuck side effects)
as marker-idempotent nodes through this module.
"""

import re
import subprocess
from pathlib import Path

# Cap on the diff text shown at a gate.
_SUMMARY_MAX_CHARS = 12_000


def dirty_paths(root: Path, *, ignored_prefix: str = ".k0-mem/") -> list[str]:
    """Porcelain lines for the working tree, minus ``.k0-mem/`` entries.

    Local memory is the harness's own scratch space — it never makes the
    tree dirty for ``/full``.
    """
    status = _git(root, "status", "--porcelain", "-uall").stdout
    return [
        line
        for line in status.splitlines()
        if not _ignored_path(line, ignored_prefix)
    ]


def _ignored_path(porcelain_line: str, prefix: str) -> bool:
    path = porcelain_line[3:] if len(porcelain_line) > 3 else ""
    path = path.split(" -> ")[-1].strip().strip('"')
    return path == prefix.rstrip("/") or path.startswith(prefix)


def slugify(title: str) -> str:
    """``k0/<n>-<slug>``'s slug: lowercase alnum runs joined by dashes."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:48].strip("-") or "work"


def default_base_branch(root: Path) -> str:
    """origin's default branch, ``main`` when there is no remote."""
    result = _git(
        root, "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"
    )
    name = result.stdout.strip()
    if result.returncode == 0 and name:
        return name.removeprefix("origin/")
    return "main"


def decider(root: Path) -> str:
    """Who is answering the gates: the git user.name, else ``human``."""
    return _git(root, "config", "user.name").stdout.strip() or "human"


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
    status = _git(root, "status", "--porcelain", "-uall").stdout
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
