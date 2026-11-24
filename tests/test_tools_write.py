"""Write tools come in a test-path flavour and a non-test flavour."""

import subprocess
from pathlib import Path

import pytest

from k0ntrol.tools.errors import ToolError
from k0ntrol.tools.write import make_edit_file, make_write_file


@pytest.fixture
def root(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    return 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_main():\n    assert True\n")
    return tmp_path


def test_test_flavour_write_file_writes_a_test_path(root):
    tool = make_write_file(root, "test")
    tool.invoke({"path": "tests/test_new.py", "content": "def test_x():\n    pass\n"})
    assert (root / "tests" / "test_new.py").read_text() == "def test_x():\n    pass\n"


def test_write_file_allows_a_new_file_in_new_dirs(root):
    tool = make_write_file(root, "non_test")
    tool.invoke({"path": "pkg/mod.py", "content": "X = 1\n"})
    assert (root / "pkg" / "mod.py").read_text() == "X = 1\n"


def test_write_file_overwrites_an_existing_file(root):
    tool = make_write_file(root, "non_test")
    tool.invoke({"path": "src/app.py", "content": "X = 2\n"})
    assert (root / "src" / "app.py").read_text() == "X = 2\n"


@pytest.mark.parametrize("path", ["src/app.py", "src/new.py"])
def test_test_flavour_refuses_non_test_paths(root, path):
    tool = make_write_file(root, "test")
    with pytest.raises(ToolError) as error:
        tool.invoke({"path": path, "content": "x\n"})
    assert error.value.denied == "path_guard"


@pytest.mark.parametrize("path", ["tests/test_x.py", "conftest.py", "src/conftest.py"])
def test_non_test_flavour_refuses_test_paths(root, path):
    tool = make_write_file(root, "non_test")
    with pytest.raises(ToolError) as error:
        tool.invoke({"path": path, "content": "x\n"})
    assert error.value.denied == "path_guard"


@pytest.mark.parametrize("scope", ["test", "non_test"])
def test_write_file_refuses_protected_and_outside_paths(root, scope):
    tool = make_write_file(root, scope)
    for path in (".k0-mem/note.md", "../escape.py"):
        with pytest.raises(ToolError) as error:
            tool.invoke({"path": path, "content": "x\n"})
        assert error.value.denied == "path_guard"


def test_write_file_refuses_a_symlink_escape(root):
    outside = root.parent / "target_test.py"
    outside.write_text("orig\n")
    (root / "test_link.py").symlink_to(outside)
    tool = make_write_file(root, "test")
    with pytest.raises(ToolError) as error:
        tool.invoke({"path": "test_link.py", "content": "x\n"})
    assert error.value.denied == "path_guard"
    assert outside.read_text() == "orig\n"


def test_edit_file_replaces_unique_text(root):
    tool = make_edit_file(root, "non_test")
    tool.invoke({"path": "src/app.py", "old": "return 1", "new": "return 2"})
    assert "return 2" in (root / "src" / "app.py").read_text()


def test_edit_file_errors_when_old_text_is_absent(root):
    tool = make_edit_file(root, "non_test")
    with pytest.raises(ToolError):
        tool.invoke({"path": "src/app.py", "old": "return 9", "new": "x"})


def test_edit_file_errors_when_old_text_is_ambiguous(root):
    (root / "src" / "app.py").write_text("x = 1\nx = 1\n")
    tool = make_edit_file(root, "non_test")
    with pytest.raises(ToolError):
        tool.invoke({"path": "src/app.py", "old": "x = 1", "new": "y"})


def test_edit_file_errors_on_a_missing_file(root):
    tool = make_edit_file(root, "non_test")
    with pytest.raises(ToolError):
        tool.invoke({"path": "src/missing.py", "old": "a", "new": "b"})


def test_edit_file_enforces_the_scope(root):
    tool = make_edit_file(root, "test")
    with pytest.raises(ToolError) as error:
        tool.invoke({"path": "src/app.py", "old": "return 1", "new": "x"})
    assert error.value.denied == "path_guard"


def test_factories_reject_an_unknown_scope(root):
    with pytest.raises(ValueError):
        make_write_file(root, "everything")


def test_edit_file_reports_an_undecodable_file(root):
    (root / "src" / "blob.py").write_bytes(b"\xff\xfe\x00\x01")
    tool = make_edit_file(root, "non_test")
    with pytest.raises(ToolError):
        tool.invoke({"path": "src/blob.py", "old": "a", "new": "b"})
