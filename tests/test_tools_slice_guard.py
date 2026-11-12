"""IMPLEMENTER's write tools hold the slice guard's green/red states (ADR 0011).

Green is where every run starts: test files may be written, code writes are
refused. A red ``run_tests`` after a test write moves to red: code may be
written, test writes are refused. A green result moves back. Markdown under
``docs/`` and ``GLOSSARY.md`` pass in either state.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from k0ntrol.tools.errors import ToolError
from k0ntrol.tools.run_tests import make_run_tests
from k0ntrol.tools.slice_guard import SliceGuard
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
