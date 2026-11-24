"""The slice journal: the durable per-path record the recovery gate restores (#174).

While the guard is in a red slice, every write the tools land is journaled:
the path's pre-slice bytes (or its absence) and the digest of what the agent
last wrote. The recovery gate's ``recover`` restores exactly those paths —
pre-slice bytes back, slice-created files removed — and refuses a journal
that is missing, stale (another slice's), malformed or conflicting (bytes
neither the pre- nor the post-write image), before mutating anything.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from k0ntrol.tools import slice_journal
from k0ntrol.tools.slice_journal import SliceJournal, restore_slice

RED_CMD = f'{sys.executable} -c "import sys; sys.exit(1)"'


@pytest.fixture
def root(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    return 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_main():\n    assert True\n")
    return tmp_path


def _journal(root: Path) -> SliceJournal:
    return SliceJournal(root / ".k0-mem" / "snapshots" / "run-1" / "slice")


def _write_manifest(journal: SliceJournal, manifest: dict) -> None:
    journal.dir.mkdir(parents=True, exist_ok=True)
    (journal.dir / "manifest.json").write_text(json.dumps(manifest))


def _journal_manifest(journal: SliceJournal) -> dict:
    return json.loads((journal.dir / "manifest.json").read_text())


def test_record_keeps_the_pre_image_and_the_post_write_digest(root):
    journal = _journal(root)
    target = root / "src" / "app.py"
    journal.record(target, "src/app.py")
    target.write_text("def main():\n    return 2\n")
    journal.note(target, "src/app.py")

    entry = _journal_manifest(journal)["paths"]["src/app.py"]
    assert entry["before"]["kind"] == "file" and entry["after"]["kind"] == "file"
    assert entry["before"]["sha"] != entry["after"]["sha"]
    assert (journal.dir / "tree" / "src" / "app.py").read_text() == (
        "def main():\n    return 1\n"
    )


def test_record_marks_a_file_the_slice_created_as_absent_before(root):
    journal = _journal(root)
    created = root / "tests" / "test_new.py"
    journal.record(created, "tests/test_new.py")
    created.write_text("def test_x():\n    pass\n")
    journal.note(created, "tests/test_new.py")

    entry = _journal_manifest(journal)["paths"]["tests/test_new.py"]
    assert entry["before"]["kind"] == "absent"


def test_record_refuses_a_repeat_write_after_a_developer_edit(root):
    """The first-write pre-image cannot stand when the file moved on:
    a developer edit between the slice's writes makes the write refuse
    rather than journal stale bytes (#174)."""
    journal = _journal(root)
    target = root / "src" / "app.py"
    assert journal.record(target, "src/app.py") is None
    target.write_text("def main():\n    return 2\n")
    journal.note(target, "src/app.py")
    target.write_text("def main():\n    return 3\n")  # developer edit

    conflict = journal.record(target, "src/app.py")

    assert conflict is not None and "src/app.py" in conflict


def test_record_refuses_after_a_crash_between_record_and_note(root):
    """A record with no note left the write unknown: the next write to the
    path cannot assume its bytes are the agent's (#174)."""
    journal = _journal(root)
    target = root / "src" / "app.py"
    assert journal.record(target, "src/app.py") is None
    target.write_text("def main():\n    return 2\n")  # noted write crashed

    assert journal.record(target, "src/app.py") is not None


def test_record_notes_the_dirs_the_slice_created(root):
    """Only ancestors absent at record time may be pruned on restore."""
    journal = _journal(root)
    deep = root / "new" / "deep" / "made.py"
    journal.record(deep, "new/deep/made.py")
    deep.parent.mkdir(parents=True)
    deep.write_text("x = 1\n")
    journal.note(deep, "new/deep/made.py")

    entry = _journal_manifest(journal)["paths"]["new/deep/made.py"]
    assert entry["parents"] == ["new", "new/deep"]


def test_apply_restores_edited_files_and_removes_created_ones(root):
    journal = _journal(root)
    edited = root / "src" / "app.py"
    journal.record(edited, "src/app.py")
    edited.write_text("def main():\n    return 2\n")
    journal.note(edited, "src/app.py")
    created = root / "tests" / "test_new.py"
    journal.record(created, "tests/test_new.py")
    created.write_text("def test_x():\n    pass\n")
    journal.note(created, "tests/test_new.py")

    result = restore_slice(root, journal.dir, expected_slice=journal.slice)

    assert result.applied
    assert edited.read_text() == "def main():\n    return 1\n"
    assert not created.exists()


def test_apply_refuses_a_missing_journal(root):
    result = restore_slice(
        root, root / ".k0-mem" / "snapshots" / "run-1" / "slice", expected_slice="x"
    )
    assert not result.applied
    assert "missing" in result.detail


def test_apply_refuses_a_malformed_manifest(root):
    """A manifest that parses but has the wrong shape refuses rather than
    trusting its entries (#174)."""
    journal = _journal(root)
    _write_manifest(journal, {"slice": journal.slice, "paths": {"../evil": {}}})

    result = restore_slice(root, journal.dir, expected_slice=journal.slice)

    assert not result.applied
    assert "malformed" in result.detail


def test_apply_refuses_a_journal_from_another_slice(root):
    journal = _journal(root)
    target = root / "src" / "app.py"
    journal.record(target, "src/app.py")
    journal.note(target, "src/app.py")

    result = restore_slice(root, journal.dir, expected_slice="another-slice")

    assert not result.applied
    assert "stale" in result.detail


def test_an_applied_journal_from_another_slice_is_stale_not_done(root):
    """The slice id is checked before the applied marker: another slice's
    consumed journal can never fast-path this run's recovery (#174)."""
    journal = _journal(root)
    target = root / "src" / "app.py"
    journal.record(target, "src/app.py")
    journal.note(target, "src/app.py")
    manifest = _journal_manifest(journal)
    manifest["restore"] = "applied"
    _write_manifest(journal, manifest)

    result = restore_slice(root, journal.dir, expected_slice="another-slice")

    assert not result.applied
    assert "stale" in result.detail
    assert target.read_text() == "def main():\n    return 1\n"


def test_apply_refuses_a_conflicting_path_before_any_mutation(root):
    journal = _journal(root)
    edited = root / "src" / "app.py"
    journal.record(edited, "src/app.py")
    edited.write_text("def main():\n    return 2\n")
    journal.note(edited, "src/app.py")
    created = root / "tests" / "test_new.py"
    journal.record(created, "tests/test_new.py")
    created.write_text("def test_x():\n    pass\n")
    journal.note(created, "tests/test_new.py")
    # A developer edit after the slice: neither the before- nor the after-image.
    edited.write_text("def main():\n    return 3\n")

    result = restore_slice(root, journal.dir, expected_slice=journal.slice)

    assert not result.applied
    assert "src/app.py" in result.conflicts
    # Nothing mutated: the refusal happens before any restore.
    assert edited.read_text() == "def main():\n    return 3\n"
    assert created.read_text() == "def test_x():\n    pass\n"


def test_apply_refuses_a_first_apply_path_at_its_pre_image(root):
    """The pre-image only counts as progress after durable restore intent:
    on a first apply, a path already at its pre-slice bytes means someone
    else touched it — refuse, do not silently skip (#174)."""
    journal = _journal(root)
    target = root / "src" / "app.py"
    journal.record(target, "src/app.py")
    target.write_text("def main():\n    return 2\n")
    journal.note(target, "src/app.py")
    target.write_text("def main():\n    return 1\n")  # developer revert

    result = restore_slice(root, journal.dir, expected_slice=journal.slice)

    assert not result.applied
    assert "src/app.py" in result.conflicts


def test_apply_never_treats_a_dir_or_broken_link_as_absent(root):
    """A slice-created path whose target is now a directory or a broken
    symlink is a conflict, not 'already gone' (#174)."""
    journal = _journal(root)
    created = root / "tests" / "test_new.py"
    journal.record(created, "tests/test_new.py")
    created.write_text("def test_x():\n    pass\n")
    journal.note(created, "tests/test_new.py")
    created.unlink()
    created.mkdir()
    (created / "keep.py").write_text("x = 1\n")

    result = restore_slice(root, journal.dir, expected_slice=journal.slice)

    assert not result.applied
    assert "tests/test_new.py" in result.conflicts
    # Never a recursive delete: the directory and its contents survive.
    assert (created / "keep.py").exists()


def test_apply_refuses_a_target_behind_a_substituted_symlink(root):
    """If a journaled path's ancestor was swapped for a symlink, restoring
    through it could write outside the recorded location: refuse (#174)."""
    journal = _journal(root)
    target = root / "src" / "app.py"
    journal.record(target, "src/app.py")
    target.write_text("def main():\n    return 2\n")
    journal.note(target, "src/app.py")
    elsewhere = root / "elsewhere"
    elsewhere.mkdir()
    moved = elsewhere / "app.py"
    target.rename(moved)
    target.parent.rmdir()
    target.parent.symlink_to(elsewhere)

    result = restore_slice(root, journal.dir, expected_slice=journal.slice)

    assert not result.applied
    assert "src/app.py" in result.conflicts
    # Nothing was written through the link.
    assert moved.read_text() == "def main():\n    return 2\n"


def test_apply_refuses_a_payload_behind_a_symlink(root):
    """A swapped symlink inside the journal's payload dir cannot redirect
    the backup read (#174)."""
    journal = _journal(root)
    target = root / "src" / "app.py"
    journal.record(target, "src/app.py")
    target.write_text("def main():\n    return 2\n")
    journal.note(target, "src/app.py")
    payload_dir = journal.dir / "tree" / "src"
    planted = root / "planted"
    planted.mkdir()
    (planted / "app.py").write_text("evil()\n")
    payload_target = payload_dir / "app.py"
    payload_target.rename(planted / "app.py")
    payload_dir.rmdir()
    payload_dir.symlink_to(planted)

    result = restore_slice(root, journal.dir, expected_slice=journal.slice)

    assert not result.applied
    assert "src/app.py" in result.conflicts
    assert target.read_text() == "def main():\n    return 2\n"


def test_apply_refuses_a_corrupt_backup_payload(root):
    """The backup's bytes must digest to the recorded pre-image before it
    is ever trusted (#174)."""
    journal = _journal(root)
    target = root / "src" / "app.py"
    journal.record(target, "src/app.py")
    target.write_text("def main():\n    return 2\n")
    journal.note(target, "src/app.py")
    (journal.dir / "tree" / "src" / "app.py").write_text("corrupted\n")

    result = restore_slice(root, journal.dir, expected_slice=journal.slice)

    assert not result.applied
    assert "src/app.py" in result.conflicts
    assert target.read_text() == "def main():\n    return 2\n"


def test_apply_rechecks_a_path_before_mutating_it(root, monkeypatch):
    """The plan is not the mutation: a file changed between intent and its
    own restore step must not be overwritten mid-apply (#174)."""
    journal = _journal(root)
    edited = root / "src" / "app.py"
    journal.record(edited, "src/app.py")
    edited.write_text("def main():\n    return 2\n")
    journal.note(edited, "src/app.py")
    created = root / "tests" / "test_new.py"
    journal.record(created, "tests/test_new.py")
    created.write_text("def test_x():\n    pass\n")
    journal.note(created, "tests/test_new.py")
    real_write = slice_journal._write_manifest

    def race(directory, manifest):
        real_write(directory, manifest)
        if manifest.get("restore") == "in_progress":
            edited.write_text("def main():\n    return 3\n")  # concurrent edit

    monkeypatch.setattr(slice_journal, "_write_manifest", race)

    result = restore_slice(root, journal.dir, expected_slice=journal.slice)

    assert not result.applied
    assert "src/app.py" in result.conflicts
    assert edited.read_text() == "def main():\n    return 3\n"
    assert created.exists()  # the conflict aborted before it was removed


def test_apply_prunes_only_dirs_the_slice_created(root):
    """A directory that predates the slice is kept even when the restore
    empties it (#174)."""
    journal = _journal(root)
    (root / "new").mkdir()  # existed before the slice
    deep = root / "new" / "deep" / "made.py"
    journal.record(deep, "new/deep/made.py")
    deep.parent.mkdir()
    deep.write_text("x = 1\n")
    journal.note(deep, "new/deep/made.py")

    result = restore_slice(root, journal.dir, expected_slice=journal.slice)

    assert result.applied
    assert not deep.exists()
    assert not (root / "new" / "deep").exists()
    assert (root / "new").is_dir()  # recorded parents only: "new/deep"


def test_apply_keeps_developer_files_inside_created_dirs(root):
    journal = _journal(root)
    deep = root / "new" / "made.py"
    journal.record(deep, "new/made.py")
    deep.parent.mkdir()
    deep.write_text("x = 1\n")
    journal.note(deep, "new/made.py")
    (root / "new" / "developer.txt").write_text("keep me\n")

    result = restore_slice(root, journal.dir, expected_slice=journal.slice)

    assert result.applied
    assert (root / "new" / "developer.txt").read_text() == "keep me\n"


def test_apply_skips_what_an_interrupted_restore_already_did(root):
    """A restore that died mid-apply resumes: paths already at their
    pre-slice bytes are done, the rest still restore (#174)."""
    journal = _journal(root)
    first = root / "src" / "app.py"
    journal.record(first, "src/app.py")
    first.write_text("def main():\n    return 2\n")
    journal.note(first, "src/app.py")
    second = root / "tests" / "test_new.py"
    journal.record(second, "tests/test_new.py")
    second.write_text("def test_x():\n    pass\n")
    journal.note(second, "tests/test_new.py")
    # The crash left one path restored and the intent marker on disk.
    first.write_text("def main():\n    return 1\n")
    manifest = _journal_manifest(journal)
    manifest["restore"] = "in_progress"
    _write_manifest(journal, manifest)

    result = restore_slice(root, journal.dir, expected_slice=journal.slice)

    assert result.applied
    assert first.read_text() == "def main():\n    return 1\n"
    assert not second.exists()


def test_an_applied_journal_is_a_no_op_even_when_the_tree_moves_on(root):
    """After a completed restore a developer edit is not recovery progress:
    replaying the apply must not touch it or refuse it (#174)."""
    journal = _journal(root)
    target = root / "src" / "app.py"
    journal.record(target, "src/app.py")
    target.write_text("def main():\n    return 2\n")
    journal.note(target, "src/app.py")
    first = restore_slice(root, journal.dir, expected_slice=journal.slice)
    assert first.applied

    target.write_text("def main():\n    return 9\n")
    second = restore_slice(root, journal.dir, expected_slice=journal.slice)

    assert second.applied
    assert target.read_text() == "def main():\n    return 9\n"


def test_the_first_write_of_a_new_slice_replaces_a_consumed_journal(root):
    old = _journal(root)
    target = root / "src" / "app.py"
    old.record(target, "src/app.py")
    target.write_text("def main():\n    return 2\n")
    old.note(target, "src/app.py")
    assert restore_slice(root, old.dir, expected_slice=old.slice).applied

    fresh = _journal(root)
    fresh.record(target, "src/app.py")

    manifest = _journal_manifest(fresh)
    assert manifest["slice"] == fresh.slice
    assert list(manifest["paths"]) == ["src/app.py"]
    assert manifest["paths"]["src/app.py"]["before"]["kind"] == "file"


def test_journal_operations_never_write_the_developers_gitignore(root):
    """``.k0-mem/.gitignore`` is the developer's file: the journal neither
    creates nor edits it — excluding snapshots/ at publication is #173."""
    journal = _journal(root)
    target = root / "src" / "app.py"
    journal.record(target, "src/app.py")
    target.write_text("def main():\n    return 2\n")
    journal.note(target, "src/app.py")
    restore_slice(root, journal.dir, expected_slice=journal.slice)

    assert not (root / ".k0-mem" / ".gitignore").exists()
