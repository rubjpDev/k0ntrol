"""The ``change`` module: deterministic git side effects for the graphs.

#41 needs only reads — the current branch for ``run_start`` and the
working-tree summary the diff gate shows. #44/#45 add the write side
(branch, memory commits, commit + push + draft PR)
as marker-idempotent nodes through this module.
"""

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

# Cap on the diff text shown at a gate.
_SUMMARY_MAX_CHARS = 12_000


def dirty_paths(root: Path, *, ignored_prefix: str = ".k0-mem/") -> list[str]:
    """Porcelain lines for the working tree, minus ``.k0-mem/`` entries.

    Local memory is the harness's own scratch space — it never makes the
    tree dirty for a run.
    """
    status = _status(root)
    return [
        line
        for line in status.splitlines()
        if not _ignored_path(line, ignored_prefix)
    ]


def reset_scan_output(root: Path) -> None:
    """Drop local changes to the scanner's output (``Project_context/``).

    The run-start rescan rewrites the code graph on whatever branch the run
    starts from; left there, it blocks the checkout of the run branch. It is
    derived data the next scan rewrites. ``Mem.md`` and ``Rules.md`` stay.
    """
    _git(root, "checkout", "--", ".k0-mem/Project_context")


def changed_paths(root: Path) -> list[str]:
    """The paths ``dirty_paths`` reports: what the working tree changed."""
    return [_porcelain_path(line) for line in dirty_paths(root)]


def _ignored_path(porcelain_line: str, prefix: str) -> bool:
    path = _porcelain_path(porcelain_line)
    return path == prefix.rstrip("/") or path.startswith(prefix)


def _porcelain_path(porcelain_line: str) -> str:
    path = porcelain_line[3:] if len(porcelain_line) > 3 else ""
    return path.split(" -> ")[-1].strip().strip('"')


def slugify(title: str) -> str:
    """``k0/<n>-<slug>``'s slug: lowercase alnum runs joined by dashes."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:48].strip("-") or "work"


def existing_branch(root: Path, ref: str) -> str | None:
    """The ``k0/<ref>-*`` branch an earlier run cut, if any.

    A local branch wins; a branch only on ``origin`` (a fresh clone) counts
    too, named without the remote.
    """
    local = _branches(root, f"refs/heads/k0/{ref}-*")
    remote = _branches(root, f"refs/remotes/origin/k0/{ref}-*")
    return next(iter(local + [b.removeprefix("origin/") for b in remote]), None)


def _branches(root: Path, pattern: str) -> list[str]:
    out = _git(root, "for-each-ref", "--format=%(refname:short)", pattern).stdout
    return out.split()


def remote_tip(root: Path, branch: str) -> str | None:
    """The commit ``origin/<branch>`` points at, if the branch was pushed."""
    result = _git(root, "rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{branch}")
    return result.stdout.strip() or None


def default_base_branch(root: Path) -> str:
    """origin's default branch, ``main`` when there is no remote."""
    result = _git(
        root, "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"
    )
    name = result.stdout.strip()
    if result.returncode == 0 and name:
        return name.removeprefix("origin/")
    return "main"


def origin_url(root: Path) -> str:
    """The URL of the ``origin`` remote, empty when there is none."""
    return _git(root, "remote", "get-url", "origin").stdout.strip()


def decider(root: Path) -> str:
    """Who is answering the gates: the git user.name, else ``human``."""
    return _git(root, "config", "user.name").stdout.strip() or "human"


def current_branch(root: Path) -> str | None:
    """The checkout's branch name, or ``None`` outside a git repo."""
    result = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def head_sha(root: Path) -> str:
    """The commit ``HEAD`` points at, or "" outside a git repo."""
    return _git(root, "rev-parse", "HEAD").stdout.strip()


def merge_base(root: Path, base: str) -> str:
    """Where HEAD left ``origin/<base>`` (else local ``<base>``).

    Falls back to git's empty tree so a diff against it shows everything.
    """
    for ref in (f"origin/{base}", base):
        found = _git(root, "merge-base", "HEAD", ref).stdout.strip()
        if found:
            return found
    return _empty_tree(root)


def _empty_tree(root: Path) -> str:
    """The empty tree object (written if missing): diff against it shows all."""
    return _git(root, "hash-object", "-w", "-t", "tree", "/dev/null").stdout.strip()


class DiffError(RuntimeError):
    """``git diff`` failed, so a gate cannot tell what changed."""


def diff_since(root: Path, since: str, path: str) -> str:
    """``git diff <since> HEAD -- <path>``, capped like every gate detail.

    Raises ``DiffError`` when git fails (say ``since`` is missing from the
    repo): a diff that cannot be computed is never an empty diff.
    """
    proc = _git(root, "diff", since, "HEAD", "--", path)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or f"git diff exited {proc.returncode}"
        raise DiffError(f"cannot diff {path} since {since}: {detail}")
    return _truncate(proc.stdout.rstrip())


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
    """``git status --porcelain`` with ``-uall``: every untracked file."""
    return _git(root, "status", "--porcelain", "-uall").stdout


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


# --- working-tree snapshots (/implement's diff-gate ``n``, #116) ---------------

# Backed-up repository files live under this payload subdirectory, apart
# from the snapshot's own ``manifest.json`` at the root — a repo file of
# the same name must never be clobbered by the metadata.
_PAYLOAD = "tree"


def worktree_snapshot_dir(root: Path, run_id: str) -> Path:
    """Where run ``run_id``'s pre-run working-tree snapshot lives."""
    return root / ".k0-mem" / "snapshots" / run_id


def is_git_repo(root: Path) -> bool:
    """Whether ``root`` sits inside a git work tree.

    The snapshot is built on ``git ls-files`` — outside a repo it would
    be empty and ``n`` would silently keep every change, so ``/implement``
    refuses such roots up front.
    """
    result = _git(root, "rev-parse", "--is-inside-work-tree")
    return result.returncode == 0 and result.stdout.strip() == "true"


def snapshot_worktree(root: Path, dest: Path) -> None:
    """Copy the working tree's files into ``dest`` for ``restore_worktree``.

    The set is what git sees — tracked files plus untracked non-ignored
    ones — minus ``.k0-mem/``: agents cannot write protected paths, so
    everything under it is harness bookkeeping, never the change. Files
    land under ``dest / _PAYLOAD`` so the snapshot's own ``manifest.json``
    can never overwrite a backed-up repository file of the same name.
    """
    _ensure_snapshot_ignored(root)
    dest.mkdir(parents=True, exist_ok=True)
    payload = dest / _PAYLOAD
    manifest: dict[str, str] = {}
    links: dict[str, str] = {}
    dirs: list[str] = []
    for relative in _worktree_files(root):
        source = root / relative
        if source.is_symlink():
            links[relative] = os.readlink(source)
        elif source.is_file():
            target = payload / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            manifest[relative] = _sha256(source)
        elif source.is_dir():
            # a gitlink (submodule): recorded so restore leaves it alone
            dirs.append(relative)
    (dest / "manifest.json").write_text(
        json.dumps(
            {"files": manifest, "links": links, "dirs": dirs},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def restore_worktree(root: Path, snapshot: Path) -> None:
    """Put the tree back as the snapshot found it.

    Files the run added are deleted (emptying their dirs), files it
    edited or removed are copied back from the snapshot. ``.k0-mem/`` is
    never in the file set, so events and checkpoints survive.
    """
    recorded = _snapshot_manifest(snapshot)
    manifest: dict[str, str] = recorded["files"]
    links: dict[str, str] = recorded.get("links", {})
    dirs: list[str] = recorded.get("dirs", [])
    known = set(manifest) | set(links) | set(dirs)
    _remove_new_files(root, known)
    for relative, digest in manifest.items():
        _restore_file(root, snapshot, relative, digest)
    for relative, link_target in links.items():
        _restore_link(root, relative, link_target)


def _remove_new_files(root: Path, known: set[str]) -> None:
    """Delete what the run added; empty dirs it leaves are pruned."""
    for relative in _worktree_files(root):
        if relative not in known:
            _remove(root / relative)
            _prune_empty_dirs(root, relative)


def _restore_file(root: Path, snapshot: Path, relative: str, digest: str) -> None:
    """Copy one snapshotted file back unless the tree already matches."""
    target = root / relative
    unchanged = (
        not target.is_symlink()
        and target.is_file()
        and _sha256(target) == digest
    )
    if unchanged:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    _remove(target)  # a symlink or dir here must go before copy2 writes
    shutil.copy2(snapshot / _PAYLOAD / relative, target)


def drop_worktree_snapshot(root: Path, snapshot: Path) -> None:
    """A finished run keeps no snapshot."""
    shutil.rmtree(snapshot, ignore_errors=True)


def _restore_link(root: Path, relative: str, link_target: str) -> None:
    """Recreate a symlink the run removed or retargeted; leave a correct one."""
    path = root / relative
    if path.is_symlink() and os.readlink(path) == link_target:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _remove(path)
    path.symlink_to(link_target)


def _worktree_files(root: Path) -> list[str]:
    """Tracked plus untracked-nonignored paths, minus ``.k0-mem/``."""
    seen = dict.fromkeys(
        _git_paths(root, "ls-files", "-z")
        + _git_paths(root, "ls-files", "--others", "--exclude-standard", "-z")
    )
    return sorted(p for p in seen if p and not p.startswith(".k0-mem/"))


def _git_paths(root: Path, *args: str) -> list[str]:
    return [p for p in _git(root, *args).stdout.split("\0") if p]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _prune_empty_dirs(root: Path, relative: str) -> None:
    """Remove ancestor dirs the run emptied, stopping at the first non-empty."""
    base = root.resolve()
    current = (base / relative).parent
    while current != base:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _ensure_snapshot_ignored(root: Path) -> None:
    """Keep ``.k0-mem/snapshots/`` out of ``git status`` like the rest of
    the harness's scratch (same append-only pattern as event rotation)."""
    gitignore = root / ".k0-mem" / ".gitignore"
    text = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    if "snapshots/" in text.splitlines():
        return
    prefix = "" if not text or text.endswith("\n") else "\n"
    with open(gitignore, "a", encoding="utf-8") as fh:
        fh.write(f"{prefix}snapshots/\n")


# --- the diff gate's run-change listing (#149) ---------------------------------


def gate_change(root: Path, snapshot: Path) -> tuple[str, str]:
    """The diff gate's ``(detail, diff)`` against the pre-run snapshot.

    The baseline is the tree as the run found it, not ``HEAD``: the
    detail lists the pre-run files the run modified or deleted, apart
    from the files it added, so edits to shared code stand out — a file
    already uncommitted before the run and untouched by it matches the
    snapshot and is never listed. The diff is a git-style unified diff
    the frontend draws file by file.
    """
    recorded = _snapshot_manifest(snapshot)
    changed = _changed_entries(root, recorded)
    added = _added_paths(root, recorded)
    detail = _change_lines(changed, added)
    chunks = [_entry_diff(snapshot, root, entry) for entry in changed]
    chunks += [_added_diff(root, path) for path in added]
    return detail, _truncate("\n".join(chunks).rstrip())


def _snapshot_manifest(snapshot: Path) -> dict:
    return json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))


def _changed_entries(
    root: Path, recorded: dict
) -> list[tuple[str, str, str]]:
    """``(path, mark, kind)`` for each snapshot path the run touched."""
    entries: list[tuple[str, str, str]] = []
    for relative, digest in recorded["files"].items():
        _note(entries, relative, _file_mark(root / relative, digest), "file")
    for relative, target in recorded.get("links", {}).items():
        _note(entries, relative, _link_mark(root / relative, target), "link")
    for relative in recorded.get("dirs", []):
        _note(entries, relative, _dir_mark(root / relative), "dir")
    return sorted(entries)


def _note(
    entries: list[tuple[str, str, str]],
    relative: str,
    mark: str | None,
    kind: str,
) -> None:
    if mark:
        entries.append((relative, mark, kind))


def _file_mark(path: Path, digest: str) -> str | None:
    """``D`` gone, ``M`` different bytes or kind, ``None`` untouched."""
    if path.is_symlink() or path.is_dir():
        return "M"  # a file became a link or a dir
    if not path.is_file():
        return "D"
    return None if _sha256(path) == digest else "M"


def _link_mark(path: Path, target: str) -> str | None:
    """``None`` while the symlink still points at ``target``."""
    if not path.is_symlink():
        return "M" if path.exists() else "D"
    return None if os.readlink(path) == target else "M"


def _dir_mark(path: Path) -> str | None:
    """A gitlink's checkout: ``None`` while it is still a directory."""
    if path.is_dir():
        return None
    return "M" if path.exists() else "D"


def _added_paths(root: Path, recorded: dict) -> list[str]:
    """Worktree files the snapshot does not know: what the run added."""
    known = (
        set(recorded["files"])
        | set(recorded.get("links", {}))
        | set(recorded.get("dirs", []))
    )
    return [p for p in _worktree_files(root) if p not in known]


def _change_lines(
    changed: list[tuple[str, str, str]], added: list[str]
) -> str:
    """Porcelain-style lines: the pre-run files first, then the adds."""
    lines = [f" {mark} {path}" for path, mark, _ in changed]
    lines += [f"?? {path}" for path in added]
    return "\n".join(lines) or "no changes"


def _entry_diff(
    snapshot: Path, root: Path, entry: tuple[str, str, str]
) -> str:
    """One ``diff --git`` chunk for a changed pre-run path."""
    relative, mark, kind = entry
    header = f"diff --git a/{relative} b/{relative}\n"
    if kind != "file":
        return header + f"{kind} changed\n"
    old = _read_lines(snapshot / _PAYLOAD / relative)
    new = [] if mark == "D" else _read_lines(root / relative)
    if old is None or new is None:
        return header + "binary or unreadable file changed\n"
    return header + "".join(
        difflib.unified_diff(
            old, new, fromfile=f"a/{relative}", tofile=f"b/{relative}"
        )
    )


def _added_diff(root: Path, relative: str) -> str:
    """One ``diff --git`` chunk for a file the run added."""
    header = f"diff --git a/{relative} b/{relative}\nnew file\n"
    lines = _read_lines(root / relative)
    if lines is None:
        return header + "binary or unreadable file\n"
    return header + "".join(
        difflib.unified_diff(
            [], lines, fromfile="/dev/null", tofile=f"b/{relative}"
        )
    )


def _read_lines(path: Path) -> list[str] | None:
    try:
        return path.read_text(encoding="utf-8").splitlines(keepends=True)
    except (OSError, UnicodeError):
        return None


def _truncate(text: str) -> str:
    if len(text) > _SUMMARY_MAX_CHARS:
        text = text[:_SUMMARY_MAX_CHARS] + "\n... diff truncated ..."
    return text
