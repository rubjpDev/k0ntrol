import json
import stat
import sys
from pathlib import Path

import pytest

from k0ntrol.backends.invoke import invoke
from k0ntrol.frontends.ask import answer_ask
from k0ntrol.frontends.mentions import MentionError


def config(threshold=350):
    return {
        "backend": "cursor",
        "threshold_lines": threshold,
        "agents": {"bulk_reader": {"model": "m"}},
    }


def memory_config(threshold=350):
    return {
        "backend": "cursor",
        "threshold_lines": threshold,
        "retrieval_top_k": 5,
        "agents": {
            "bulk_reader": {"model": "m"},
            "retrieval_gate": {"model": "m"},
        },
    }


def write_file_memory(root, relative, body):
    path = (
        root
        / ".k0-mem"
        / "Project_context"
        / root.name
        / "files"
        / f"{relative}.md"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return f"{root.name}:file:{relative}"


def write_agent(path: Path, body: str) -> None:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


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


def test_ask_without_file_injects_memory_when_gate_says_yes(tmp_path):
    node_id = write_file_memory(
        tmp_path,
        "src/app.py",
        "# src/app.py\n\n## hace\nreturns 1 from checkout\n\n## gotchas\nnone\n",
    )
    seen = {}

    def fake_invoke(mode, prompt):
        seen.setdefault("modes", []).append(mode.name)
        if mode.name == "retrieval_gate":
            seen["gate_prompt"] = prompt
            return f"retrieve: yes\n- {node_id}\n"
        seen["prompt"] = prompt
        return "- checkout (function): returns 1\n"

    def fake_bulk(*args, **kwargs):
        raise AssertionError("no @file means no bulk_read")

    out = answer_ask(
        "what does checkout return?",
        root=tmp_path,
        config=memory_config(),
        invoke=fake_invoke,
        bulk_read=fake_bulk,
    )
    assert out == "- checkout (function): returns 1\n"
    assert seen["modes"] == ["retrieval_gate", "bulk_reader"]
    assert node_id in seen["gate_prompt"]
    assert "returns 1 from checkout" in seen["prompt"]
    assert "what does checkout return?" in seen["prompt"]
    assert (tmp_path / ".k0-mem" / "index.sqlite").is_file()


def test_ask_gate_no_adds_nothing_to_the_prompt(tmp_path):
    write_file_memory(
        tmp_path,
        "src/app.py",
        "# src/app.py\n\n## hace\nreturns 1 from checkout\n\n## gotchas\nnone\n",
    )
    seen = {}

    def fake_invoke(mode, prompt):
        if mode.name == "retrieval_gate":
            return "retrieve: no\n"
        seen["prompt"] = prompt
        return "ok\n"

    out = answer_ask(
        "what does checkout return?",
        root=tmp_path,
        config=memory_config(),
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bulk")),
    )
    assert out == "ok\n"
    assert seen["prompt"] == "what does checkout return?"
    assert "returns 1 from checkout" not in seen["prompt"]


def test_ask_injects_rules_outside_the_gate(tmp_path):
    write_file_memory(
        tmp_path,
        "src/app.py",
        "# src/app.py\n\n## hace\nreturns 1 from checkout\n\n## gotchas\nnone\n",
    )
    rules = tmp_path / ".k0-mem" / "General_mem" / "Rules.md"
    rules.parent.mkdir(parents=True, exist_ok=True)
    rules.write_text("# Rules\n\nNever log secrets.\n", encoding="utf-8")
    seen = {}

    def fake_invoke(mode, prompt):
        if mode.name == "retrieval_gate":
            return "retrieve: no\n"
        seen["prompt"] = prompt
        return "ok\n"

    answer_ask(
        "what does checkout return?",
        root=tmp_path,
        config=memory_config(),
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bulk")),
    )
    assert seen["prompt"] == (
        "what does checkout return?\n"
        "\n"
        "# Rules\n\nNever log secrets.\n"
    )
    assert "returns 1 from checkout" not in seen["prompt"]


def test_ask_with_file_injects_rules_and_skips_retrieval(tmp_path):
    (tmp_path / "app.py").write_text("def checkout():\n    return 1\n")
    rules = tmp_path / ".k0-mem" / "General_mem" / "Rules.md"
    rules.parent.mkdir(parents=True, exist_ok=True)
    rules.write_text("# Rules\n\nKeep it small.\n", encoding="utf-8")
    seen = {}

    def fake_invoke(mode, prompt):
        seen["mode"] = mode.name
        seen["prompt"] = prompt
        return "ok\n"

    out = answer_ask(
        "what does checkout return? @app.py",
        root=tmp_path,
        config=config(),
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bulk")),
    )
    assert out == "ok\n"
    assert seen["mode"] == "bulk_reader"
    assert "# Rules\n\nKeep it small.\n" in seen["prompt"]
    assert "app.py\ndef checkout():" in seen["prompt"]


def test_ask_gate_emits_one_jsonl_event(tmp_path, monkeypatch):
    write_file_memory(
        tmp_path,
        "src/app.py",
        "# src/app.py\n\n## hace\nreturns 1 from checkout\n\n## gotchas\nnone\n",
    )
    monkeypatch.chdir(tmp_path)
    agent = tmp_path / "cursor-agent"
    write_agent(
        agent,
        "import json, sys\n"
        "prompt = sys.argv[-1]\n"
        "if 'Pick at most' in prompt:\n"
        "    result = 'retrieve: no\\n'\n"
        "else:\n"
        "    result = 'ok\\n'\n"
        "print(json.dumps({'type':'result','subtype':'success','is_error':False,"
        "'result': result}))\n",
    )
    monkeypatch.setenv("K0_CURSOR_AGENT", str(agent))
    out = answer_ask(
        "what does checkout return?",
        root=tmp_path,
        config=memory_config(),
        invoke=invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bulk")),
    )
    assert out == "ok\n"
    events = [
        json.loads(line)
        for line in (tmp_path / ".k0-mem" / "events.jsonl").read_text().splitlines()
    ]
    gate_events = [event for event in events if event["mode"] == "retrieval_gate"]
    assert len(gate_events) == 1
    assert gate_events[0]["ok"] is True
