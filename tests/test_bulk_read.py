import json
import re
import subprocess
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from fakes import FakeChatModel, cfg_with
from k0ntrol.events import EventWriter, K0EventHandler
from k0ntrol.tools.bulk_read import BulkReadError, bulk_read, make_bulk_read
from k0ntrol.tools.errors import ToolError


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


# --- the bulk_read tool ----------------------------------------------------


def _init(model):
    return lambda _model, *, model_provider=None, **kwargs: model


def _cfg():
    return cfg_with(helpers={"bulk_read": {"provider": "anthropic", "model": "m"}})


def _git(root):
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)


def test_bulk_read_tool_asks_the_helper_about_the_files(tmp_path):
    (tmp_path / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    _git(tmp_path)
    model = FakeChatModel(reply=AIMessage(content="the files return 1 and 2"))
    tool = make_bulk_read(tmp_path, _cfg(), init=_init(model))

    assert tool.name == "bulk_read"
    out = tool.invoke({"question": "what do they return?", "paths": ["a.py", "b.py"]})

    assert out == "the files return 1 and 2"
    prompt = model.invocations[0][-1].content
    assert '<file repo="tmp' in prompt or "<file repo=" in prompt
    assert 'path="a.py"' in prompt and 'path="b.py"' in prompt
    assert "what do they return?" in prompt


def test_bulk_read_tool_refuses_paths_outside_the_guard(tmp_path):
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".k0-mem").mkdir()
    (tmp_path / ".k0-mem" / "secret.md").write_text("hidden\n", encoding="utf-8")
    _git(tmp_path)
    tool = make_bulk_read(tmp_path, _cfg(), init=_init(FakeChatModel()))

    with pytest.raises(ToolError) as error:
        tool.invoke({"question": "q", "paths": ["ok.py", "../escape.py"]})
    assert error.value.denied == "path_guard"

    with pytest.raises(ToolError) as error:
        tool.invoke({"question": "q", "paths": [".k0-mem/secret.md"]})
    assert error.value.denied == "path_guard"

    with pytest.raises(ToolError) as error:
        tool.invoke({"question": "q", "paths": ["missing.py"]})
    assert error.value.denied == "path_guard"


def test_bulk_read_tool_call_inside_an_agent_emits_a_helper_event(tmp_path):
    """The helper model call inherits the tool run's callbacks: the model
    event says ``helper`` and lands in the same chat scope."""
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path)
    events_path = tmp_path / "events.jsonl"
    writer = EventWriter(events_path, chat="chat-1", mask=[])
    model = FakeChatModel(reply=AIMessage(content="file contents"))
    tool = make_bulk_read(tmp_path, _cfg(), init=_init(model))

    out = tool.invoke(
        {"question": "q", "paths": ["a.py"]},
        config={
            "callbacks": [K0EventHandler(writer)],
            "metadata": {"k0_agent": "chatty"},
        },
    )

    assert out == "file contents"
    events = [
        json.loads(line) for line in events_path.read_text().splitlines()
    ]
    assert [e["kind"] for e in events] == ["model", "tool"]
    helper_event, tool_event = events
    assert helper_event["helper"] == "bulk_read"
    assert helper_event["chat"] == "chat-1"
    assert "agent" not in helper_event
    assert tool_event["tool"] == "bulk_read"
    assert tool_event["ok"] is True
