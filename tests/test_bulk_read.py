import re
from pathlib import Path

import pytest

from k0ntrol.tools.bulk_read import BulkReadError, bulk_read


def test_bulk_read_sends_question_and_xml_in_path_order(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def checkout():\n    return 1\n")
    (tmp_path / "src" / "pay.py").write_text("a < b & c")
    seen = {}

    def fake_invoke(prompt):
        seen["prompt"] = prompt
        return "- checkout (function) line 1: returns 1\n"

    out = bulk_read(
        "what does checkout return?",
        [tmp_path / "src" / "app.py", tmp_path / "src" / "pay.py"],
        root=tmp_path,
        repo="repoFront",
        invoke=fake_invoke,
    )
    assert out == "- checkout (function) line 1: returns 1\n"
    assert seen["prompt"] == (
        "what does checkout return?\n"
        "\n"
        '<file repo="repoFront" path="src/app.py">def checkout():\n'
        "    return 1\n"
        "</file>\n"
        '<file repo="repoFront" path="src/pay.py">a &lt; b &amp; c</file>'
    )


def test_missing_path_names_it_and_does_not_invoke(tmp_path):
    missing = tmp_path / "nope.py"

    def fake_invoke(prompt):
        raise AssertionError("invoke must not run")

    with pytest.raises(BulkReadError, match=re.escape(str(missing))):
        bulk_read(
            "q",
            [missing],
            root=tmp_path,
            repo="repoFront",
            invoke=fake_invoke,
        )


def test_path_outside_root_is_an_error(tmp_path, monkeypatch):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("x\n")

    def fake_invoke(prompt):
        raise AssertionError("invoke must not run")

    with pytest.raises(BulkReadError, match="outside root"):
        bulk_read(
            "q",
            [outside],
            root=tmp_path,
            repo="repoFront",
            invoke=fake_invoke,
        )
