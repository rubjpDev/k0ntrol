import re
from pathlib import Path

import pytest

from k0ntrol.cli import main
from k0ntrol.modespec import ModeSpec
from k0ntrol.stages.bulk_read import BulkReadError, bulk_read


MODE = ModeSpec(
    name="bulk_reader",
    instructions="instructions stay on the mode",
    model="m",
    temperature=0.2,
    one_shot=True,
)


def test_bulk_read_sends_question_and_xml_in_path_order(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def checkout():\n    return 1\n")
    (tmp_path / "src" / "pay.py").write_text("a < b & c")
    seen = {}

    def fake_invoke(mode, prompt):
        seen["mode"] = mode
        seen["prompt"] = prompt
        return "- checkout (function) line 1: returns 1\n"

    out = bulk_read(
        "what does checkout return?",
        [tmp_path / "src" / "app.py", tmp_path / "src" / "pay.py"],
        root=tmp_path,
        repo="repoFront",
        mode=MODE,
        invoke=fake_invoke,
    )
    assert out == "- checkout (function) line 1: returns 1\n"
    assert seen["mode"] is MODE
    assert seen["prompt"] == (
        "what does checkout return?\n"
        "\n"
        '<file repo="repoFront" path="src/app.py">def checkout():\n'
        "    return 1\n"
        "</file>\n"
        '<file repo="repoFront" path="src/pay.py">a &lt; b &amp; c</file>'
    )
    assert "instructions stay on the mode" not in seen["prompt"]


def test_missing_path_names_it_and_does_not_invoke(tmp_path):
    missing = tmp_path / "nope.py"

    def fake_invoke(mode, prompt):
        raise AssertionError("invoke must not run")

    with pytest.raises(BulkReadError, match=re.escape(str(missing))):
        bulk_read(
            "q",
            [missing],
            root=tmp_path,
            repo="repoFront",
            mode=MODE,
            invoke=fake_invoke,
        )


def test_bulk_read_cli_prints_stage_text(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text('say "hi"\n')
    seen = {}

    def fake_invoke(mode, prompt):
        seen["prompt"] = prompt
        seen["name"] = mode.name
        return "- say (name) line 1\n"

    monkeypatch.setattr("k0ntrol.backends.invoke.invoke", fake_invoke)
    assert main(["bulk-read", "--question", "what is say?", "--paths", "a.py"]) == 0
    assert capsys.readouterr().out == "- say (name) line 1\n"
    assert seen["name"] == "bulk_reader"
    assert f'repo="{tmp_path.name}"' in seen["prompt"]
    assert 'path="a.py"' in seen["prompt"]
    assert "say &quot;hi&quot;" in seen["prompt"]


def test_bulk_read_cli_missing_path_exits_1_without_invoke(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    def fake_invoke(mode, prompt):
        raise AssertionError("invoke must not run")

    monkeypatch.setattr("k0ntrol.backends.invoke.invoke", fake_invoke)
    assert main(["bulk-read", "--question", "q", "--paths", "nope.py"]) == 1
    assert "nope.py" in capsys.readouterr().err
