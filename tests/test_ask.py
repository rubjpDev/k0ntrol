from pathlib import Path

import pytest

from k0ntrol.frontends.ask import answer_ask
from k0ntrol.frontends.mentions import MentionError


def config(threshold=350):
    return {
        "backend": "cursor",
        "threshold_lines": threshold,
        "agents": {"bulk_reader": {"model": "m"}},
    }


def test_small_file_is_inlined_and_nothing_is_written(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    target = source / "app.py"
    target.write_text("def checkout():\n    return 1\n")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    seen = {}

    def fake_invoke(mode, prompt):
        seen["mode"] = mode.name
        seen["prompt"] = prompt
        return "- checkout (function) line 1: returns 1\n"

    def fake_bulk(*args, **kwargs):
        raise AssertionError("small file must stay out of bulk_read")

    out = answer_ask(
        "what does checkout return? @src/app.py",
        root=tmp_path,
        config=config(),
        invoke=fake_invoke,
        bulk_read=fake_bulk,
    )
    assert out == "- checkout (function) line 1: returns 1\n"
    assert seen["mode"] == "bulk_reader"
    assert seen["prompt"] == (
        "what does checkout return?\n"
        "\n"
        "src/app.py\n"
        "def checkout():\n"
        "    return 1\n"
    )
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before


def test_three_large_files_go_through_bulk_read(tmp_path):
    paths = []
    for name in ("a.py", "b.py", "c.py"):
        path = tmp_path / name
        path.write_text("x\n" * 351)
        paths.append(path)
    seen = {}

    def fake_bulk(question, bulk_paths, **kwargs):
        seen["question"] = question
        seen["paths"] = list(bulk_paths)
        seen["repo"] = kwargs["repo"]
        return "- a (file) line 1: x\n"

    def fake_invoke(mode, prompt):
        raise AssertionError("delegated files must not be inlined")

    out = answer_ask(
        "what is on line 1? @a.py @b.py @c.py",
        root=tmp_path,
        config=config(),
        invoke=fake_invoke,
        bulk_read=fake_bulk,
    )
    assert out == "- a (file) line 1: x\n"
    assert seen["question"] == "what is on line 1?"
    assert seen["paths"] == paths
    assert seen["repo"] == tmp_path.name


def test_mixed_threshold_splits_the_call(tmp_path):
    (tmp_path / "small.py").write_text("small\n")
    (tmp_path / "big.py").write_text("x\n" * 351)
    seen = {}

    def fake_bulk(question, bulk_paths, **kwargs):
        seen["bulk"] = list(bulk_paths)
        return "BULK\n"

    def fake_invoke(mode, prompt):
        seen["prompt"] = prompt
        return "DIRECT\n"

    out = answer_ask(
        "q @small.py @big.py",
        root=tmp_path,
        config=config(),
        invoke=fake_invoke,
        bulk_read=fake_bulk,
    )
    assert seen["bulk"] == [tmp_path / "big.py"]
    assert "small.py" in seen["prompt"]
    assert "big.py" not in seen["prompt"]
    assert out == "BULK\nDIRECT\n"


def test_missing_file_and_no_mention_do_not_call(tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("called")

    with pytest.raises(MentionError, match="nope"):
        answer_ask("@nope", root=tmp_path, config=config(), invoke=boom, bulk_read=boom)
    with pytest.raises(MentionError, match="@"):
        answer_ask(
            "what does checkout return?",
            root=tmp_path,
            config=config(),
            invoke=boom,
            bulk_read=boom,
        )
