"""Read tools see only files `git ls-files` would see, behind the path guard."""

import subprocess
from pathlib import Path

import pytest

from k0ntrol.tools.errors import ToolError
from k0ntrol.tools.read import make_grep, make_list_files, make_read_file


@pytest.fixture
def root(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text(".env\nignored.txt\nbuild/\n")
    (tmp_path / ".env").write_text("SECRET=hunter2\n")
    (tmp_path / "ignored.txt").write_text("ignored\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    return 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_main():\n    assert True\n")
    (tmp_path / "README.md").write_text("hello main\n")
    return tmp_path


def test_read_file_returns_the_file_content(root):
    tool = make_read_file(root)
    assert "return 1" in tool.invoke({"path": "src/app.py"})


def test_read_file_refuses_an_ignored_env_file(root):
    tool = make_read_file(root)
    with pytest.raises(ToolError) as error:
        tool.invoke({"path": ".env"})
    assert error.value.denied == "path_guard"


def test_read_file_refuses_an_ignored_file_in_a_dir(root):
    (root / "build").mkdir()
    (root / "build" / "out.txt").write_text("artifact\n")
    tool = make_read_file(root)
    with pytest.raises(ToolError) as error:
        tool.invoke({"path": "build/out.txt"})
    assert error.value.denied == "path_guard"


def test_read_file_refuses_a_missing_file_without_leaking(root):
    tool = make_read_file(root)
    with pytest.raises(ToolError) as error:
        tool.invoke({"path": "no/such.py"})
    assert error.value.denied == "path_guard"


def test_read_file_refuses_a_protected_path(root):
    (root / ".k0-mem").mkdir()
    (root / ".k0-mem" / "events.jsonl").write_text("{}\n")
    tool = make_read_file(root)
    with pytest.raises(ToolError) as error:
        tool.invoke({"path": ".k0-mem/events.jsonl"})
    assert error.value.denied == "path_guard"


def test_read_file_refuses_a_directory(root):
    tool = make_read_file(root)
    with pytest.raises(ToolError):
        tool.invoke({"path": "src"})


def test_list_files_returns_only_visible_files(root):
    tool = make_list_files(root)
    listing = tool.invoke({"pattern": "**/*"}).splitlines()
    assert "src/app.py" in listing
    assert "tests/test_app.py" in listing
    assert ".env" not in listing
    assert "ignored.txt" not in listing


def test_list_files_narrows_with_a_glob(root):
    tool = make_list_files(root)
    listing = tool.invoke({"pattern": "tests/**/*.py"}).splitlines()
    assert listing == ["tests/test_app.py"]


def test_list_files_rejects_an_absolute_pattern(root):
    tool = make_list_files(root)
    with pytest.raises(ToolError):
        tool.invoke({"pattern": "/etc/**/*"})


def test_grep_finds_matches_in_visible_files(root):
    tool = make_grep(root)
    result = tool.invoke({"pattern": "main"})
    assert "src/app.py:1:def main():" in result
    assert "README.md:1:hello main" in result


def test_grep_never_searches_ignored_files(root):
    tool = make_grep(root)
    result = tool.invoke({"pattern": "SECRET"})
    assert ".env" not in result


def test_grep_narrows_with_a_path_glob(root):
    tool = make_grep(root)
    result = tool.invoke({"pattern": "main", "path_glob": "tests/**"})
    assert "test_app.py" in result
    assert "src/app.py" not in result


def test_grep_reports_a_bad_regex(root):
    tool = make_grep(root)
    with pytest.raises(ToolError):
        tool.invoke({"pattern": "([unclosed"})


def test_read_tools_fail_cleanly_outside_a_git_repo(tmp_path):
    (tmp_path / "f.txt").write_text("x\n")
    with pytest.raises(ToolError):
        make_read_file(tmp_path).invoke({"path": "f.txt"})


def test_read_file_reports_an_undecodable_file(root):
    (root / "src" / "blob.bin").write_bytes(b"\xff\xfe\x00\x01")
    tool = make_read_file(root)
    with pytest.raises(ToolError):
        tool.invoke({"path": "src/blob.bin"})


def test_read_file_refuses_a_broken_symlink(root):
    (root / "dangling.txt").symlink_to(root / "no-such-target.txt")
    tool = make_read_file(root)
    with pytest.raises(ToolError):
        tool.invoke({"path": "dangling.txt"})


def test_list_files_never_leaves_the_root(root):
    (root.parent / "sibling.txt").write_text("x\n")
    tool = make_list_files(root)
    assert tool.invoke({"pattern": "../*"}) == ""


def test_list_files_truncates_a_long_listing(root, monkeypatch):
    from k0ntrol.tools import read

    monkeypatch.setattr(read, "_LIST_LIMIT", 2)
    tool = make_list_files(root)
    listing = tool.invoke({"pattern": "**/*"}).splitlines()
    assert listing[-1].startswith("... truncated at 2")
    assert len(listing) == 3


def test_grep_skips_an_undecodable_file(root):
    (root / "src" / "blob.bin").write_bytes(b"\xff\xfe\x00\x01main")
    tool = make_grep(root)
    result = tool.invoke({"pattern": "main", "path_glob": "src/**"})
    assert "blob.bin" not in result


def test_grep_truncates_a_long_match_list(root, monkeypatch):
    from k0ntrol.tools import read

    monkeypatch.setattr(read, "_MATCH_LIMIT", 2)
    tool = make_grep(root)
    result = tool.invoke({"pattern": "."})
    assert result.splitlines()[-1] == "... truncated at 2"


def test_grep_never_searches_through_a_symlink_to_an_ignored_file(root):
    (root / "link").symlink_to(".env")
    subprocess.run(["git", "add", "link"], cwd=root, check=True)
    tool = make_grep(root)
    result = tool.invoke({"pattern": "SECRET"})
    assert "hunter2" not in result
    assert "link" not in result


def test_grep_never_searches_through_a_symlink_into_a_protected_dir(root):
    (root / "gitconfig").symlink_to(".git/config")
    subprocess.run(["git", "add", "gitconfig"], cwd=root, check=True)
    tool = make_grep(root)
    result = tool.invoke({"pattern": "repositoryformatversion"})
    assert "gitconfig" not in result
    assert "repositoryformatversion" not in result


def test_grep_never_searches_through_a_symlink_outside_the_root(root):
    (root.parent / "outside.txt").write_text("outside-secret\n")
    (root / "escape.txt").symlink_to("../outside.txt")
    subprocess.run(["git", "add", "escape.txt"], cwd=root, check=True)
    tool = make_grep(root)
    result = tool.invoke({"pattern": "outside-secret"})
    assert "escape.txt" not in result
    assert "outside-secret" not in result


def test_grep_searches_a_symlink_to_a_visible_file(root):
    (root / "alias.py").symlink_to("src/app.py")
    subprocess.run(["git", "add", "alias.py"], cwd=root, check=True)
    tool = make_grep(root)
    result = tool.invoke({"pattern": "def main"})
    assert "alias.py:1:def main():" in result
    assert "src/app.py:1:def main():" in result
