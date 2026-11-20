"""Local-only run files: the ``.k0-mem/`` paths that must not leave the
checkout (#173).

``k0 init`` writes ``.k0-mem/.gitignore`` so transcripts (``sessions/``),
event logs (``events*.jsonl``), checkpoint and index databases
(``*.sqlite``, ``*.sqlite-*``) and worktree snapshots (``snapshots/``)
stay out of commits. That file can be absent and an already-staged path
ignores it, so the same set is enforced here in code at the tracker's
two boundaries — never as a blanket ban on ``.k0-mem/``: shared project
memory and config still ship.

Fail closed on both boundaries. ``commit`` stages through glob-exclude
pathspecs so a private path is never staged, and refuses — before the
index is touched — when one is already staged, conflicted, or inside
``HEAD``'s tree. ``checked_push`` inspects every outgoing commit's tree
(not just touched paths, so a merge or an add-then-remove still shows)
against the live origin refs — a forged or stale remote-tracking ref
cannot shrink the range, and neither can a distinct push endpoint: when
the effective push URL is not the fetch URL ``ls-remote`` asks, the
whole branch counts as outgoing. Refusals name paths, never contents,
and k0 never rewrites history, refs, or the developer's index.
"""

import subprocess
from collections.abc import Callable
from fnmatch import fnmatch

from k0ntrol.tracker.protocol import TrackerError

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess]

_MEM = ".k0-mem"
_LOCAL_DIRS = {"sessions", "snapshots"}
_LOCAL_NAMES = ("events*.jsonl", "*.sqlite", "*.sqlite-*")

#: ``git add`` pathspecs keeping the private set out of the index — the
#: staging-time twin of ``is_local_only`` (glob ``*`` stays within a
#: path component; ``**/`` matches any depth including none).
_EXCLUDES = tuple(
    f":(exclude,glob).k0-mem/**/{pattern}"
    for pattern in (
        "sessions",
        "sessions/**",
        "snapshots",
        "snapshots/**",
        *_LOCAL_NAMES,
    )
)


def is_local_only(path: str) -> bool:
    """Whether a repo-relative path is a local-only run file.

    A ``sessions`` or ``snapshots`` component anywhere under ``.k0-mem/``
    counts, as does a basename matching ``events*.jsonl``, ``*.sqlite``
    or ``*.sqlite-*`` — the same patterns ``MEMORY_GITIGNORE`` writes.
    """
    parts = path.split("/")
    if len(parts) < 2 or parts[0] != _MEM:
        return False
    if any(part in _LOCAL_DIRS for part in parts[1:]):
        return True
    base = parts[-1]
    return any(fnmatch(base, pattern) for pattern in _LOCAL_NAMES)


def commit(run: CommandRunner, message: str) -> bool:
    """Whole-tree commit minus local-only run files; False when empty.

    A private path is never staged — ``git add`` runs with the exclusion
    pathspecs. The commit refuses first, leaving index and worktree
    exactly as found, when a private path is conflicted, already staged,
    or already inside ``HEAD``'s tree (the new commit would keep it).
    """
    status = _must(run, ["git", "status", "--porcelain", "-z", "-uall"])
    if not status.stdout.strip("\0").strip():
        return False
    _refuse_unsafe(run)
    _must(run, ["git", "add", "-A", "--", ".", *_EXCLUDES])
    if run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        return False
    _must(run, ["git", "commit", "-m", message])
    return True


def _refuse_unsafe(run: CommandRunner) -> None:
    conflicted = _conflicted_local_only(run)
    if conflicted:
        raise _refuse(
            conflicted,
            "commit refused: unresolved index entries for local-only run files",
            "k0 does not stage or stash them — resolve the merge yourself",
        )
    head = _head_paths(run)
    if head:
        raise _refuse(
            sorted(head),
            "commit refused: local-only run files already in HEAD's tree",
            "the new commit would keep them — untrack and commit the "
            "removal yourself (git rm --cached <path>, then git commit): "
            "k0 refuses even the staged untrack, so that cleanup commit "
            "is yours by hand. If the commit that leaked them is still "
            "outgoing the push refuses it too — k0 never rewrites "
            "history, untangle it by hand",
        )
    staged = _staged_local_only(run)
    if staged:
        raise _refuse(
            staged,
            "commit refused: local-only run files staged",
            "k0 left the index untouched — unstage them yourself: "
            "git reset -q -- <path>",
        )


def _refuse(paths: list[str], problem: str, fix: str) -> TrackerError:
    listing = "\n".join(f"  {path}" for path in paths)
    return TrackerError(f"{problem}:\n{listing}\n{fix}")


def _conflicted_local_only(run: CommandRunner) -> list[str]:
    """Local-only paths with unmerged index entries."""
    out = _must(run, ["git", "ls-files", "-u", "-z", "--", _MEM]).stdout
    paths = {
        record.partition("\t")[2]
        for record in out.split("\0")
        if "\t" in record
    }
    return sorted(path for path in paths if is_local_only(path))


def _head_paths(run: CommandRunner) -> set[str]:
    """Local-only paths in ``HEAD``'s tree; empty on an unborn HEAD."""
    if (
        run(["git", "rev-parse", "--verify", "--quiet", "HEAD"]).returncode
        != 0
    ):
        return set()
    out = _must(
        run, ["git", "ls-tree", "-r", "HEAD", "--name-only", "-z", "--", _MEM]
    ).stdout
    return {p for p in out.split("\0") if is_local_only(p)}


def _staged_local_only(run: CommandRunner) -> list[str]:
    out = _must(
        run,
        ["git", "diff", "--cached", "--name-only", "-z", "--", _MEM],
    ).stdout
    return sorted(p for p in out.split("\0") if is_local_only(p))


def checked_push(run: CommandRunner, branch: str) -> None:
    """Refuse a push whose outgoing commits carry a local-only path.

    Every outgoing commit's tree is inspected — a merge or an
    add-then-remove still shows the path, since the net diff would hide
    it. Only paths are reported, never contents; the history and refs
    are left alone.
    """
    paths = _outgoing_local_only(run, branch)
    if paths:
        raise _refuse(
            paths,
            f"push refused: local-only run files in outgoing commits "
            f"of {branch}",
            "k0 never rewrites history — untangle it yourself, then resume",
        )


def _outgoing_local_only(run: CommandRunner, branch: str) -> list[str]:
    found: set[str] = set()
    for sha in _outgoing_shas(run, branch):
        names = _must(
            run,
            ["git", "ls-tree", "-r", sha, "--name-only", "-z", "--", _MEM],
        ).stdout
        found.update(p for p in names.split("\0") if is_local_only(p))
    return sorted(found)


def _outgoing_shas(run: CommandRunner, branch: str) -> list[str]:
    """Commits on ``branch`` not reachable from the live origin refs.

    ``refs/remotes/origin/*`` is local state a human or a stale fetch
    can fake, so the baseline is ``git ls-remote``: when it cannot be
    verified — or when push may land somewhere it never asked — the
    whole branch counts as outgoing. Fail closed.
    """
    remote = _remote_shas(run)
    argv = ["git", "rev-list", branch]
    if remote:
        argv += ["--not", *remote]
    return _must(run, argv).stdout.split()


def _remote_shas(run: CommandRunner) -> list[str]:
    """Commit shas the real ``origin`` advertises, filtered to objects
    held locally; empty when the remote cannot be asked or its answer
    says nothing about the push destination."""
    if _push_targets_differ(run):
        return []
    out = run(["git", "ls-remote", "origin"])
    if out.returncode != 0:
        return []
    shas = {
        line.partition("\t")[0]
        for line in out.stdout.splitlines()
        if "\t" in line
    }
    return [
        sha
        for sha in sorted(shas)
        if run(["git", "cat-file", "-e", f"{sha}^{{commit}}"]).returncode == 0
    ]


def _push_targets_differ(run: CommandRunner) -> bool:
    """Whether push may land somewhere ``ls-remote origin`` never asked.

    ``ls-remote`` reads the fetch endpoint; configured ``pushurl``s can
    send the history elsewhere, and an endpoint's advertised refs cannot
    prove a commit is already at a destination it is not. Anything that
    cannot be verified counts as different — a missing or unreadable
    answer must never shrink the outgoing range.
    """
    fetch = run(["git", "remote", "get-url", "origin"])
    push = run(["git", "remote", "get-url", "--push", "--all", "origin"])
    if fetch.returncode != 0 or push.returncode != 0:
        return True
    return push.stdout.split() != fetch.stdout.split()[:1]


def _must(run: CommandRunner, argv: list[str]) -> subprocess.CompletedProcess:
    result = run(argv)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise TrackerError(
            f"{argv[0]} {argv[1]} failed (exit {result.returncode}): {detail}"
        )
    return result
