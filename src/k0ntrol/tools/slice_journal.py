"""The slice journal: the durable record of one active slice's writes (#174).

While a slice is open, every write the guard lets through is journaled under
``.k0-mem/snapshots/<run>/slice/`` — the path's pre-slice bytes (or its
absence) under ``tree/`` plus the digest of what the agent last wrote, all in
``manifest.json``. The manifest is the crash-safe contract: the recovery
gate's ``recover`` restores exactly the journaled paths — pre-slice bytes
back, slice-created files removed — and refuses a journal that is missing,
malformed, another slice's (stale), or conflicting (current bytes are
neither the pre- nor the post-write image, so something outside the journal
touched the path) — always before mutating anything for a detected
conflict. ``restore`` marks progress: ``in_progress`` while an apply runs,
``applied`` once it finished, so a replay never restores twice nor reads a
developer edit as recovery progress. The apply is ordered and per-file
atomic — never a multi-file transaction — and it restores through paths
validated against traversal and substituted symlinks only.
"""

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

# Backed-up pre-slice bytes live under this payload subdirectory, apart
# from the manifest at the journal's root (same layout as the run snapshot).
_PAYLOAD = "tree"

_MANIFEST = "manifest.json"

# A path's recorded or observed state: ``file`` carries the content digest,
# ``absent`` means no file, dir or link, ``other`` is anything else — a
# directory, a link, a fifo — that the journal can never treat as expected.
_STATES = ("file", "absent", "other")


@dataclass(frozen=True)
class RestoreResult:
    """The apply's outcome: ``applied`` — the journal is consumed — or a
    refusal with the reason in ``detail`` and the conflicting paths."""

    applied: bool
    detail: str
    conflicts: tuple[str, ...] = field(default_factory=tuple)


class SliceJournal:
    """The write side of the journal, bound to one slice by a fresh id.

    ``record`` runs before the mutation it describes and ``note`` after it,
    both under the guard's lock: a crash between them leaves an entry whose
    ``after`` is missing, which later writes and restores both treat as
    unrestorable rather than guessing the post-write bytes. A repeat write
    to a journaled path is refused when its current bytes no longer match
    the recorded post-write image — the journal never carries a stale
    pre-image over someone else's edit. ``clear`` drops the journal when a
    slice ends green.
    """

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.slice = uuid.uuid4().hex

    @property
    def paths(self) -> list[str]:
        manifest = _read_manifest(self.dir)
        if manifest is None or _manifest_error(manifest) is not None:
            return []
        return sorted(manifest["paths"])

    def record(self, resolved: Path, relative: str) -> str | None:
        """Capture the pre-write image: first touch of ``relative`` only.

        Returns a refusal reason when the path is already journaled and its
        bytes no longer match the recorded post-write image — someone else
        changed it — or when there is no file to safely journal.
        """
        manifest = self._open()
        entry = manifest["paths"].get(relative)
        now = _state_of(resolved)
        if entry is not None:
            if entry["after"] is not None and entry["after"] == now:
                return None
            return f"{relative} changed outside the slice"
        if now["kind"] == "other":
            return f"{relative} is not a regular file"
        manifest["paths"][relative] = {
            "before": now,
            "after": None,
            "parents": _created_parents(resolved, relative),
        }
        if now["kind"] == "file":
            payload = self.dir / _PAYLOAD / relative
            payload.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved, payload)
        _write_manifest(self.dir, manifest)
        return None

    def note(self, resolved: Path, relative: str) -> None:
        """Record the post-write state; ``record`` ran first, same op."""
        manifest = _read_manifest(self.dir) or {}
        entry = manifest.get("paths", {}).get(relative)
        assert entry is not None, f"unrecorded journal write: {relative}"
        entry["after"] = _state_of(resolved)
        _write_manifest(self.dir, manifest)

    def clear(self) -> None:
        """The slice ended green: nothing left to recover — drop the
        journal plus the ``snapshots/<run>`` dirs only it created (a
        run snapshot or another journal's files keep them)."""
        shutil.rmtree(self.dir, ignore_errors=True)
        for parent in (self.dir.parent, self.dir.parent.parent):
            try:
                parent.rmdir()
            except OSError:
                break

    def _open(self) -> dict:
        """The open manifest for this slice, reinitialised when the one on
        disk is malformed, belongs to another slice, or holds an old
        restore."""
        manifest = _read_manifest(self.dir)
        if (
            manifest is None
            or _manifest_error(manifest) is not None
            or manifest["slice"] != self.slice
            or manifest["restore"] is not None
        ):
            shutil.rmtree(self.dir, ignore_errors=True)
            manifest = {"slice": self.slice, "restore": None, "paths": {}}
        return manifest


def restore_slice(
    root: Path, journal_dir: Path, *, expected_slice: str
) -> RestoreResult:
    """Restore the journaled paths, or refuse without mutating anything.

    Validation is total before the first mutation: the manifest's shape,
    the slice id, each journaled path's containment and file kind, every
    backup's digest, and every target's current state. A path restores only
    while its current bytes are still what the agent's last write left; on
    a replay after durable intent (``restore: in_progress``) a path already
    at its pre-slice image counts as done — on a first apply that same
    match is a conflict, since something outside the journal moved it back.
    Each path is rechecked as it is mutated, so a concurrent edit is never
    overwritten; a refusal leaves the intent marker for a later replay.
    """
    manifest = _read_manifest(journal_dir)
    if manifest is None:
        return _refused(journal_dir)
    refusal = _identity_refusal(manifest, expected_slice)
    if refusal is not None:
        return refusal
    pending, conflicts = _plan(
        root, journal_dir, manifest, replay=manifest["restore"] == "in_progress"
    )
    if conflicts:
        return _conflicted(conflicts)
    return _apply(root, journal_dir, manifest, pending)


def _identity_refusal(manifest: dict, expected_slice: str) -> RestoreResult | None:
    """Shape and identity checks before the plan: a malformed journal, a
    journal that is not this slice's, or a consumed one never reach the
    tree. The slice id is checked before ``applied`` so a foreign consumed
    journal can never fast-path."""
    error = _manifest_error(manifest)
    if error is not None:
        return RestoreResult(False, f"slice journal malformed: {error}")
    if manifest["slice"] != expected_slice:
        return RestoreResult(False, "stale slice journal: a different slice")
    if manifest["restore"] == "applied":
        return RestoreResult(True, "slice already restored")
    return None


def _conflicted(conflicts: list[str]) -> RestoreResult:
    return RestoreResult(
        False,
        "paths changed since the slice was written: " + ", ".join(sorted(conflicts)),
        tuple(sorted(conflicts)),
    )


def _apply(
    root: Path, journal_dir: Path, manifest: dict, pending: list[str]
) -> RestoreResult:
    """Intent on disk, then each path rechecked as it is restored."""
    _write_manifest(journal_dir, {**manifest, "restore": "in_progress"})
    restored: list[str] = []
    for relative in pending:
        if not _restore_path(root, journal_dir, relative, manifest["paths"][relative]):
            return RestoreResult(
                False,
                f"{relative} changed while the restore ran",
                (relative,),
            )
        restored.append(relative)
    _write_manifest(journal_dir, {**manifest, "restore": "applied"})
    return RestoreResult(
        True, f"restored {len(restored)} path(s): {', '.join(restored)}"
    )


def _refused(journal_dir: Path) -> RestoreResult:
    if not (journal_dir / _MANIFEST).is_file():
        return RestoreResult(False, "slice journal missing")
    return RestoreResult(False, "slice journal unreadable")


# --- validation: the manifest must earn trust before the plan ---------------


def _manifest_error(manifest: object) -> str | None:
    """None when the manifest's shape is what the journal writes."""
    if not isinstance(manifest, dict):
        return "not an object"
    if not isinstance(manifest.get("slice"), str) or not manifest["slice"]:
        return "bad slice id"
    if manifest.get("restore") not in (None, "in_progress", "applied"):
        return "bad restore marker"
    paths = manifest.get("paths")
    if not isinstance(paths, dict):
        return "bad paths"
    return _paths_error(paths)


def _paths_error(paths: dict) -> str | None:
    for relative, entry in paths.items():
        error = _entry_error(relative, entry)
        if error is not None:
            return error
    return None


def _entry_error(relative: object, entry: object) -> str | None:
    if not isinstance(relative, str) or not _relative_ok(relative):
        return f"bad path {relative!r}"
    if not isinstance(entry, dict):
        return f"bad entry for {relative}"
    return _images_error(relative, entry) or _parents_error(relative, entry)


def _images_error(relative: str, entry: dict) -> str | None:
    """The recorded images: ``before`` is a file or absent, ``after`` is
    any state or null while a crash leaves the write unknown."""
    if _state_error(entry.get("before"), allow_other=False) is not None:
        return f"bad before-image for {relative}"
    after = entry.get("after")
    if after is not None and _state_error(after, allow_other=True) is not None:
        return f"bad after-image for {relative}"
    return None


def _parents_error(relative: str, entry: dict) -> str | None:
    parents = entry.get("parents")
    ok = isinstance(parents, list) and all(
        isinstance(parent, str) and _relative_ok(parent) for parent in parents
    )
    return None if ok else f"bad parents for {relative}"


def _relative_ok(relative: str) -> bool:
    """A safe root-relative path: no anchor, no dot parts, not in .k0-mem."""
    parts = PurePosixPath(relative).parts
    return (
        bool(parts)
        and not PurePosixPath(relative).is_absolute()
        and all(part not in ("", ".", "..") for part in parts)
        and parts[0] != ".k0-mem"
    )


def _state_error(state: object, *, allow_other: bool) -> str | None:
    """A recorded state: ``file`` with its digest, ``absent``, or — where
    the caller allows — ``other``. Anything else is corruption."""
    if not isinstance(state, dict):
        return "bad state"
    kinds = _STATES if allow_other else ("file", "absent")
    if state.get("kind") not in kinds:
        return "bad state"
    if state["kind"] == "file" and _digest_bad(state.get("sha")):
        return "bad digest"
    return None


def _digest_bad(sha: object) -> bool:
    return not (isinstance(sha, str) and len(sha) == 64)


# --- the plan: every check, before any mutation ------------------------------


def _plan(
    root: Path, journal_dir: Path, manifest: dict, *, replay: bool
) -> tuple[list[str], list[str]]:
    """Split the journal into paths still to restore and conflicts."""
    pending: list[str] = []
    conflicts: list[str] = []
    for relative, entry in sorted(manifest["paths"].items()):
        if _path_conflict(root, journal_dir, relative, entry, replay):
            conflicts.append(relative)
        elif _still_at_after(root, relative, entry):
            pending.append(relative)
    return pending, conflicts


def _path_conflict(
    root: Path, journal_dir: Path, relative: str, entry: dict, replay: bool
) -> bool:
    """One journaled path's verdict: anything the plan cannot prove safe
    is a conflict."""
    now = _target_state(root, relative)
    if now["kind"] == "other":
        return True
    if replay and now == entry["before"]:
        return False  # durable intent already restored this path
    if entry["after"] is None or now != entry["after"]:
        return True  # a crash left the write unknown, or someone moved it
    return not _restorable(root, journal_dir, relative, entry)


def _still_at_after(root: Path, relative: str, entry: dict) -> bool:
    return _target_state(root, relative) == entry["after"]


def _restorable(
    root: Path, journal_dir: Path, relative: str, entry: dict
) -> bool:
    """Everything a pending restore needs, checked pre-mutation: clean
    target ancestors (no substituted symlink) and — for a file pre-image —
    a backup payload that is contained, regular, and digests correctly."""
    if not _ancestors_clean(root, relative):
        return False
    if entry["before"]["kind"] == "absent":
        return True
    return _payload_ok(journal_dir, relative, entry["before"]["sha"])


def _target_state(root: Path, relative: str) -> dict:
    """The observed state of a journaled target: file+sha, absent, other.

    A symlink — even a working one — and a directory are ``other``: never
    the regular file a write left nor the absence a created file expects.
    """
    target = root / relative
    if target.is_symlink() or (target.exists() and not target.is_file()):
        return {"kind": "other"}
    return _state_of(target)


def _state_of(path: Path) -> dict:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        return {"kind": "other"}
    if not path.exists():
        return {"kind": "absent"}
    return {"kind": "file", "sha": _sha256(path)}


def _ancestors_clean(root: Path, relative: str) -> bool:
    """No ancestor of ``root/relative`` may be a symlink: a swapped link
    would redirect the write or the delete outside the recorded location."""
    base = root.resolve()
    node = base
    for part in PurePosixPath(relative).parts[:-1]:
        node = node / part
        if node.is_symlink():
            return False
        if not node.exists():
            return True  # missing ancestors get created as real dirs
    return True


def _payload_ok(journal_dir: Path, relative: str, sha: str) -> bool:
    """The backup is usable: inside the payload dir, a regular file whose
    bytes still digest to the recorded pre-image."""
    payload = journal_dir / _PAYLOAD / relative
    if payload.is_symlink() or not payload.is_file():
        return False
    try:
        payload.resolve().relative_to((journal_dir / _PAYLOAD).resolve())
    except ValueError:
        return False
    return _sha256(payload) == sha


# --- the apply: per-file atomic, rechecked as it goes -------------------------


def _restore_path(
    root: Path, journal_dir: Path, relative: str, entry: dict
) -> bool:
    """Restore one journaled path; False when it no longer stands where
    the plan left it (a concurrent edit — refuse rather than overwrite)."""
    target = root / relative
    if not _ancestors_clean(root, relative):
        return False
    if _target_state(root, relative) != entry["after"]:
        return False
    if entry["before"]["kind"] == "absent":
        target.unlink(missing_ok=True)
        _prune_created(root, entry["parents"])
        return True
    return _replace_from_payload(root, journal_dir, relative, entry["before"])


def _replace_from_payload(
    root: Path, journal_dir: Path, relative: str, before: dict
) -> bool:
    """Atomically put the pre-image back: a verified backup staged in the
    target's own directory, then renamed — never unlink-then-copy, which a
    crash could leave as an unexplained missing file."""
    payload = journal_dir / _PAYLOAD / relative
    if not _payload_ok(journal_dir, relative, before["sha"]):
        return False
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".k0-restore-")
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(payload.read_bytes())
        os.replace(tmp, target)
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise
    return True


def _prune_created(root: Path, parents: list[str]) -> None:
    """Remove ancestors the slice's write created, deepest first and only
    while still empty — a directory that predates the slice or holds a
    developer file is never the journal's to take."""
    for prefix in sorted(parents, key=len, reverse=True):
        node = root / prefix
        try:
            node.rmdir()
        except OSError:
            continue


def _created_parents(resolved: Path, relative: str) -> list[str]:
    """The ancestors of ``relative`` absent at record time — exactly the
    dirs the coming write's ``mkdir -p`` creates (relative, deepest last
    in path order)."""
    parts = PurePosixPath(relative).parts
    root = resolved.parents[len(parts) - 1]
    created: list[str] = []
    node = root
    for part in parts[:-1]:
        node = node / part
        if node.exists() or node.is_symlink():
            continue
        created.append(node.relative_to(root).as_posix())
    return created


# --- manifest I/O ---------------------------------------------------------------


def _read_manifest(journal_dir: Path) -> dict | None:
    try:
        return json.loads((journal_dir / _MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_manifest(journal_dir: Path, manifest: dict) -> None:
    journal_dir.mkdir(parents=True, exist_ok=True)
    tmp = journal_dir / f"{_MANIFEST}.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, journal_dir / _MANIFEST)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
