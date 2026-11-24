"""Path guard behaviour: symlink resolution, root and protected paths."""

import subprocess
from pathlib import Path

import pytest

from k0ntrol.tools.errors import ToolError
from k0ntrol.tools.path_guard import guard_path, is_test_path


def git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def test_guard_accepts_a_file_inside_the_root(tmp_path):
    root = git_repo(tmp_path)
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("pass\n")
    assert guard_path("src/app.py", root) == (root / "src" / "app.py").resolve()


def test_guard_accepts_a_new_file_that_does_not_exist(tmp_path):
    root = git_repo(tmp_path)
    assert guard_path("new/thing.py", root) == (root / "new" / "thing.py").resolve()


@pytest.mark.parametrize("path", ["../outside.txt", "src/../../outside.txt", "/etc/passwd"])
def test_guard_refuses_paths_outside_the_root(tmp_path, path):
    root = git_repo(tmp_path)
    with pytest.raises(ToolError) as error:
        guard_path(path, root)
    assert error.value.denied == "path_guard"


def test_guard_refuses_a_symlink_that_escapes_the_root(tmp_path):
    root = git_repo(tmp_path)
    outside = tmp_path.parent / "secret-target.txt"
    outside.write_text("secret\n")
    (root / "link").symlink_to(outside)
    with pytest.raises(ToolError) as error:
        guard_path("link", root)
    assert error.value.denied == "path_guard"


def test_guard_refuses_a_symlinked_directory_that_escapes(tmp_path):
    root = git_repo(tmp_path)
    outside_dir = tmp_path.parent / "outside-dir"
    outside_dir.mkdir()
    (outside_dir / "f.txt").write_text("x\n")
    (root / "dirlink").symlink_to(outside_dir)
    with pytest.raises(ToolError) as error:
        guard_path("dirlink/f.txt", root)
    assert error.value.denied == "path_guard"


@pytest.mark.parametrize(
    "path",
    [".git/config", ".git/refs/heads/main", ".k0-mem/events.jsonl", ".k0-mem/checkpoints.sqlite"],
)
def test_guard_refuses_each_protected_path(tmp_path, path):
    root = git_repo(tmp_path)
    (root / ".k0-mem").mkdir()
    (root / ".k0-mem" / "events.jsonl").write_text("{}\n")
    with pytest.raises(ToolError) as error:
        guard_path(path, root)
    assert error.value.denied == "path_guard"


def test_guard_refuses_a_symlink_into_a_protected_dir(tmp_path):
    root = git_repo(tmp_path)
    (root / ".k0-mem").mkdir()
    (root / ".k0-mem" / "events.jsonl").write_text("{}\n")
    (root / "link").symlink_to(root / ".k0-mem" / "events.jsonl")
    with pytest.raises(ToolError) as error:
        guard_path("link", root)
    assert error.value.denied == "path_guard"


def test_tool_error_denied_defaults_to_none():
    assert ToolError("boom").denied is None


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        ("tests/test_app.py", True),
        ("test_app.py", True),
        ("src/test_app.py", True),
        ("src/app_test.py", True),
        ("conftest.py", True),
        ("src/conftest.py", True),
        ("tests/conftest.py", True),
        ("src/app.py", False),
        ("src/testing.py", False),
        ("mytests/app.py", False),
    ],
)
def test_is_test_path(relative, expected):
    assert is_test_path(relative) is expected
