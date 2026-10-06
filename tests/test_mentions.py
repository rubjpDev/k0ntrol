from pathlib import Path

import pytest

from k0ntrol.frontends.mentions import MentionError, mention_paths


def test_two_relative_files_keep_order(tmp_path):
    (tmp_path / "a.py").write_text("a\n")
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "b.py").write_text("b\n")
    found = mention_paths("see @a.py and @dir/b.py", tmp_path)
    assert found == [tmp_path / "a.py", tmp_path / "dir" / "b.py"]


def test_missing_path_is_an_error(tmp_path):
    with pytest.raises(MentionError, match="nope"):
        mention_paths("@nope", tmp_path)


def test_absolute_parent_and_directory_are_errors(tmp_path):
    (tmp_path / "dir").mkdir()
    with pytest.raises(MentionError, match="/etc/passwd"):
        mention_paths("see @/etc/passwd", tmp_path)
    with pytest.raises(MentionError, match="outside"):
        mention_paths("@../outside.txt", tmp_path)
    with pytest.raises(MentionError, match="dir"):
        mention_paths("@dir", tmp_path)
