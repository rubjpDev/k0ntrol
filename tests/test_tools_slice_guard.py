"""IMPLEMENTER's write tools hold the slice guard's green/red states (ADR 0011).

Green is where every run starts: test files may be written, code writes are
refused. A red ``run_tests`` after a test write moves to red: code may be
written, test writes are refused. A green result moves back. Markdown under
``docs/`` and ``GLOSSARY.md`` pass in either state.
"""

import subprocess
import sys
import threading
from pathlib import Path

import pytest

from k0ntrol.tools.errors import SliceRecoveryNeeded, ToolError
from k0ntrol.tools.run_tests import make_run_tests
from k0ntrol.tools.runner import CommandResult
from k0ntrol.tools.slice_guard import SliceGuard
from k0ntrol.tools.slice_journal import SliceJournal
from k0ntrol.tools.write import make_edit_file, make_write_file

RED_CMD = f'{sys.executable} -c "import sys; sys.exit(1)"'
GREEN_CMD = f'{sys.executable} -c "print(\'ok\')"'


@pytest.fixture
def root(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    return 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_main():\n    assert True\n")
    return tmp_path


def test_every_run_starts_green(root):
    guard = SliceGuard()
    assert guard.state == "green"


def test_green_refuses_a_code_write_and_names_the_state_and_unlock(root):
    tool = make_write_file(root, SliceGuard())
    with pytest.raises(ToolError) as error:
        tool.invoke({"path": "src/app.py", "content": "X = 2\n"})
    assert "green" in str(error.value)
    assert "run_tests" in str(error.value)
    assert (root / "src" / "app.py").read_text() == "def main():\n    return 1\n"


def test_green_allows_a_test_write(root):
    tool = make_write_file(root, SliceGuard())
    tool.invoke({"path": "tests/test_new.py", "content": "def test_x():\n    pass\n"})
    assert (root / "tests" / "test_new.py").read_text() == "def test_x():\n    pass\n"


@pytest.mark.parametrize(
    "path", ["docs/adr/0001-x.md", "docs/research/note.md", "GLOSSARY.md"]
)
def test_docs_writes_pass_in_green(root, path):
    tool = make_write_file(root, SliceGuard())
    tool.invoke({"path": path, "content": "# doc\n"})
    assert (root / path).read_text() == "# doc\n"


def test_edit_file_is_held_to_the_same_guard(root):
    guard = SliceGuard()
    edit = make_edit_file(root, guard)
    with pytest.raises(ToolError) as error:
        edit.invoke({"path": "src/app.py", "old": "return 1", "new": "return 2"})
    assert "green" in str(error.value)
    edit.invoke({"path": "tests/test_app.py", "old": "assert True", "new": "assert True"})
    assert "assert True" in (root / "tests" / "test_app.py").read_text()


def _tools(root, guard, test_cmd):
    return (
        make_write_file(root, guard),
        make_edit_file(root, guard),
        make_run_tests(root, {"test_cmd": test_cmd}, guard),
    )


def test_a_refactor_start_allows_code_and_the_first_test_write_returns_to_green(root):
    guard = SliceGuard(start="refactor")
    write, edit, run_tests = _tools(root, guard, RED_CMD)

    edit.invoke({"path": "src/app.py", "old": "return 1", "new": "return 2"})
    assert "return 2" in (root / "src" / "app.py").read_text()

    write.invoke({"path": "docs/refactor.md", "content": "# refactor\n"})
    assert (root / "docs" / "refactor.md").read_text() == "# refactor\n"

    edit.invoke({"path": "src/app.py", "old": "return 2", "new": "return 3"})
    assert "return 3" in (root / "src" / "app.py").read_text()

    with pytest.raises(ToolError) as error:
        edit.invoke(
            {"path": "tests/test_app.py", "old": "assert True", "new": "assert False"}
        )
    assert "refactor" in str(error.value)
    assert "assert True" in (root / "tests" / "test_app.py").read_text()

    with pytest.raises(ToolError):
        write.invoke({"path": "src/app.py", "content": "X = 3\n"})
    write.invoke({"path": "tests/test_new.py", "content": "x\n"})
    run_tests.invoke({"paths": []})

    edit.invoke({"path": "src/app.py", "old": "return 3", "new": "return 4"})
    assert "return 4" in (root / "src" / "app.py").read_text()
    with pytest.raises(ToolError) as error:
        write.invoke({"path": "tests/test_other.py", "content": "x\n"})
    assert "red" in str(error.value)


def test_run_tests_does_not_leave_a_refactor_start(root):
    guard = SliceGuard(start="refactor")
    write, _, run_tests = _tools(root, guard, GREEN_CMD)

    run_tests.invoke({"paths": []})
    write.invoke({"path": "src/app.py", "content": "X = 2\n"})
    assert (root / "src" / "app.py").read_text() == "X = 2\n"

    with pytest.raises(ToolError) as error:
        write.invoke({"path": "tests/test_new.py", "content": "x\n"})
    assert "refactor" in str(error.value)
    assert not (root / "tests" / "test_new.py").exists()


def test_a_red_run_tests_after_a_test_write_moves_to_red(root):
    guard = SliceGuard()
    write, _, run_tests = _tools(root, guard, RED_CMD)
    write.invoke({"path": "tests/test_new.py", "content": "def test_x():\n    pass\n"})
    run_tests.invoke({"paths": []})
    assert guard.state == "red"

    write.invoke({"path": "src/app.py", "content": "X = 2\n"})
    assert (root / "src" / "app.py").read_text() == "X = 2\n"
    with pytest.raises(ToolError) as error:
        write.invoke({"path": "tests/test_other.py", "content": "x\n"})
    assert "red" in str(error.value)
    assert "run_tests" in str(error.value)


def test_a_green_run_tests_moves_back_to_green(root):
    guard = SliceGuard()
    write, _, run_tests = _tools(root, guard, RED_CMD)
    write.invoke({"path": "tests/test_new.py", "content": "x\n"})
    run_tests.invoke({"paths": []})
    assert guard.state == "red"

    green = make_run_tests(root, {"test_cmd": GREEN_CMD}, guard)
    green.invoke({"paths": []})
    assert guard.state == "green"
    with pytest.raises(ToolError) as error:
        write.invoke({"path": "src/app.py", "content": "X = 2\n"})
    assert "green" in str(error.value)
    write.invoke({"path": "tests/test_next.py", "content": "x\n"})
    assert (root / "tests" / "test_next.py").exists()


def test_a_red_run_tests_without_a_test_write_stays_green(root):
    guard = SliceGuard()
    write, _, run_tests = _tools(root, guard, RED_CMD)
    run_tests.invoke({"paths": []})
    assert guard.state == "green"
    with pytest.raises(ToolError):
        write.invoke({"path": "src/app.py", "content": "X = 2\n"})


def test_docs_writes_pass_in_red_too(root):
    guard = SliceGuard()
    write, _, run_tests = _tools(root, guard, RED_CMD)
    write.invoke({"path": "tests/test_new.py", "content": "x\n"})
    run_tests.invoke({"paths": []})
    write.invoke({"path": "docs/notes.md", "content": "# notes\n"})
    write.invoke({"path": "GLOSSARY.md", "content": "# glossary\n"})
    assert (root / "docs" / "notes.md").exists()
    assert (root / "GLOSSARY.md").exists()


def test_the_guard_holds_the_whole_red_green_cycle(root):
    """Slice after slice: test, red, code, green, test, red, code, green."""
    guard = SliceGuard()
    write, edit, run_tests = _tools(root, guard, RED_CMD)
    green = make_run_tests(root, {"test_cmd": GREEN_CMD}, guard)

    for index in range(2):
        write.invoke({"path": f"tests/test_slice{index}.py", "content": "x\n"})
        run_tests.invoke({"paths": []})
        assert guard.state == "red"
        edit.invoke(
            {"path": "src/app.py", "old": f"return {index + 1}", "new": f"return {index + 2}"}
        )
        green.invoke({"paths": []})
        assert guard.state == "green"

    assert "return 3" in (root / "src" / "app.py").read_text()


def test_markdown_under_docs_does_not_arm_the_guard(root):
    """``docs/test_guide.md`` looks like a test path but is a docs write."""
    guard = SliceGuard()
    write, _, run_tests = _tools(root, guard, RED_CMD)
    write.invoke({"path": "docs/test_guide.md", "content": "# guide\n"})
    run_tests.invoke({"paths": []})
    assert guard.state == "green"
    with pytest.raises(ToolError):
        write.invoke({"path": "src/app.py", "content": "X = 2\n"})


def test_a_green_result_consumes_the_test_write_signal(root):
    """A passing test write does not linger: the next red needs a new test."""
    guard = SliceGuard()
    write, _, run_tests = _tools(root, guard, GREEN_CMD)
    write.invoke({"path": "tests/test_new.py", "content": "x\n"})
    run_tests.invoke({"paths": []})
    assert guard.state == "green"

    red = make_run_tests(root, {"test_cmd": RED_CMD}, guard)
    red.invoke({"paths": []})
    assert guard.state == "green"
    with pytest.raises(ToolError):
        write.invoke({"path": "src/app.py", "content": "X = 2\n"})


def test_the_path_guard_still_runs_before_the_slice_guard(root):
    tool = make_write_file(root, SliceGuard())
    with pytest.raises(ToolError) as error:
        tool.invoke({"path": "../escape.py", "content": "x\n"})
    assert error.value.denied == "path_guard"


def test_a_run_tests_that_cannot_launch_still_counts_as_red(root):
    """``result.ok`` is the verdict everywhere: no exit 0 means red."""
    guard = SliceGuard()
    write, _, run_tests = _tools(
        root, guard, "definitely-not-a-binary-xyz -q"
    )
    write.invoke({"path": "tests/test_new.py", "content": "x\n"})
    run_tests.invoke({"paths": []})
    assert guard.state == "red"


def test_a_paused_test_write_cannot_straddle_a_red_verdict(root, monkeypatch):
    """The tool node runs one turn's calls in parallel (ADR 0007), so a
    write's check, mutation and note are one op under the guard's lock: a
    red ``run_tests`` verdict cannot land between check and write."""
    guard = SliceGuard()
    write = make_write_file(root, guard)
    edit = make_edit_file(root, guard)
    run_tests = make_run_tests(root, {"test_cmd": "unused"}, guard)
    write.invoke({"path": "tests/test_app.py", "content": "x = 1\n"})

    checked = threading.Event()
    release = threading.Event()
    verdict = threading.Event()
    monkeypatch.setattr(
        "k0ntrol.tools.run_tests.run_command",
        lambda *args, **kwargs: CommandResult(("unused",), 1, "failed"),
    )

    check = guard.check_write

    def pausing_check(relative):
        check(relative)
        checked.set()
        release.wait(5)

    note = guard.note_test_result

    def noting(ok):
        note(ok)
        verdict.set()

    monkeypatch.setattr(guard, "check_write", pausing_check)
    monkeypatch.setattr(guard, "note_test_result", noting)

    writer = threading.Thread(
        target=lambda: edit.invoke(
            {"path": "tests/test_app.py", "old": "x = 1", "new": "x = 2"}
        ),
        daemon=True,
    )
    writer.start()
    assert checked.wait(5)

    runner = threading.Thread(
        target=lambda: run_tests.invoke({"paths": []}), daemon=True
    )
    runner.start()
    verdict_mid_write = verdict.wait(1)
    release.set()
    writer.join(5)
    runner.join(5)

    assert not verdict_mid_write
    assert (root / "tests" / "test_app.py").read_text() == "x = 2\n"
    assert verdict.wait(5)
    assert guard.state == "red"


def test_a_write_waits_for_an_in_flight_run_tests_and_sees_its_verdict(
    root, monkeypatch
):
    """``run_tests`` holds the lock across the whole command and verdict, so
    a batched write cannot change a test file mid-run: it waits, then meets
    the state the verdict left."""
    guard = SliceGuard()
    write = make_write_file(root, guard)
    run_tests = make_run_tests(root, {"test_cmd": "unused"}, guard)
    write.invoke({"path": "tests/test_app.py", "content": "x\n"})

    running = threading.Event()
    finish = threading.Event()
    write_checked = threading.Event()

    def hanging_command(*args, **kwargs):
        running.set()
        finish.wait(5)
        return CommandResult(("unused",), 1, "failed")

    check = guard.check_write

    def signaling_check(relative):
        check(relative)
        write_checked.set()

    monkeypatch.setattr(
        "k0ntrol.tools.run_tests.run_command", hanging_command
    )
    monkeypatch.setattr(guard, "check_write", signaling_check)

    runner = threading.Thread(
        target=lambda: run_tests.invoke({"paths": []}), daemon=True
    )
    runner.start()
    assert running.wait(5)

    errors = []

    def writing():
        try:
            write.invoke({"path": "tests/test_new.py", "content": "x\n"})
        except ToolError as error:
            errors.append(error)

    writer = threading.Thread(target=writing, daemon=True)
    writer.start()
    checked_mid_run = write_checked.wait(1)
    finish.set()
    writer.join(5)
    runner.join(5)

    assert not checked_mid_run
    assert len(errors) == 1
    assert "red" in str(errors[0])
    assert not (root / "tests" / "test_new.py").exists()
    assert guard.state == "red"


def test_third_denied_test_write_in_red_escalates_for_human_recovery(root):
    """The third refused test write in one red slice ends the agent run for
    a human recovery decision instead of answering the model again (#174)."""
    guard = SliceGuard()
    write, edit, run_tests = _tools(root, guard, RED_CMD)
    write.invoke({"path": "tests/test_new.py", "content": "x\n"})
    run_tests.invoke({"paths": []})
    assert guard.state == "red"

    for _ in range(2):
        with pytest.raises(ToolError):
            edit.invoke({"path": "tests/test_new.py", "old": "x", "new": "y"})
    with pytest.raises(SliceRecoveryNeeded) as error:
        edit.invoke({"path": "tests/test_new.py", "old": "x", "new": "y"})
    assert "recovery" in str(error.value)


def test_code_writes_between_denials_do_not_reset_the_count(root):
    """The bound counts denials in the red slice, not in a row: code may be
    written between them — that is still the same stuck slice (#174)."""
    guard = SliceGuard()
    write, edit, run_tests = _tools(root, guard, RED_CMD)
    write.invoke({"path": "tests/test_new.py", "content": "x\n"})
    run_tests.invoke({"paths": []})

    for index in range(2):
        with pytest.raises(ToolError):
            edit.invoke({"path": "tests/test_new.py", "old": "x", "new": "y"})
        edit.invoke(
            {"path": "src/app.py", "old": f"return {index + 1}", "new": f"return {index + 2}"}
        )
    with pytest.raises(SliceRecoveryNeeded):
        edit.invoke({"path": "tests/test_new.py", "old": "x", "new": "y"})


def test_the_denial_count_starts_fresh_on_a_new_slice(root):
    """A green run_tests ends the slice and its denial count (#174)."""
    guard = SliceGuard()
    write, edit, run_tests = _tools(root, guard, RED_CMD)
    green = make_run_tests(root, {"test_cmd": GREEN_CMD}, guard)
    write.invoke({"path": "tests/test_new.py", "content": "x\n"})
    run_tests.invoke({"paths": []})
    for _ in range(2):
        with pytest.raises(ToolError):
            write.invoke({"path": "tests/test_a.py", "content": "y\n"})

    green.invoke({"paths": []})
    assert guard.state == "green"
    write.invoke({"path": "tests/test_b.py", "content": "z\n"})
    run_tests.invoke({"paths": []})
    assert guard.state == "red"
    for _ in range(2):
        with pytest.raises(ToolError):
            write.invoke({"path": "tests/test_c.py", "content": "w\n"})


def test_denied_code_writes_in_green_do_not_escalate(root):
    """The bound is the red slice's denied test writes: a model retrying a
    code write in green keeps getting ordinary refusals (#174)."""
    guard = SliceGuard()
    write = make_write_file(root, guard)
    for _ in range(4):
        with pytest.raises(ToolError) as error:
            write.invoke({"path": "src/app.py", "content": "X = 2\n"})
        assert error.value.denied == "slice_guard"


def _journaled(root) -> SliceJournal:
    return SliceJournal(root / ".k0-mem" / "snapshots" / "run-1" / "slice")


def test_writes_through_the_tools_fill_the_slice_journal(root):
    """Every write the guard lets through lands in the journal: the test
    write in green and the code write in red, with its pre-slice image."""
    journal = _journaled(root)
    guard = SliceGuard(journal=journal)
    write, edit, run_tests = _tools(root, guard, RED_CMD)

    write.invoke({"path": "tests/test_new.py", "content": "x\n"})
    run_tests.invoke({"paths": []})
    edit.invoke({"path": "src/app.py", "old": "return 1", "new": "return 2"})

    assert journal.paths == ["src/app.py", "tests/test_new.py"]


def test_the_escalation_carries_the_journal_identity(root):
    """The gate restores what the exception names: this journal's slice and
    the paths the slice wrote (#174)."""
    journal = _journaled(root)
    guard = SliceGuard(journal=journal)
    write, _, run_tests = _tools(root, guard, RED_CMD)
    write.invoke({"path": "tests/test_new.py", "content": "x\n"})
    run_tests.invoke({"paths": []})
    for _ in range(2):
        with pytest.raises(ToolError):
            write.invoke({"path": "tests/test_a.py", "content": "y\n"})

    with pytest.raises(SliceRecoveryNeeded) as error:
        write.invoke({"path": "tests/test_a.py", "content": "y\n"})
    assert error.value.slice == journal.slice
    assert error.value.paths == ["tests/test_new.py"]


def test_a_green_verdict_clears_the_journal(root):
    """A slice that ends green is accepted: its journal drops away (#174)."""
    journal = _journaled(root)
    guard = SliceGuard(journal=journal)
    write, _, run_tests = _tools(root, guard, GREEN_CMD)
    write.invoke({"path": "tests/test_new.py", "content": "x\n"})
    run_tests.invoke({"paths": []})

    assert not journal.dir.exists()
