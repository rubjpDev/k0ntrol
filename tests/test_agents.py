"""The one agent builder: allowlist, limits, fresh context.

Every agent comes from ``build_agent``: ``create_agent`` with the agent's
allowlist, k0's closed middleware set, a model-call limit and an explicit
derived recursion limit, behind a fresh-context wrapper.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool

from fakes import FakeChatModel, cfg_with
from k0ntrol.agents import ALLOWLISTS, agent_tools, allowlist_for, build_agent
from k0ntrol.tools.errors import ToolError


def _scripted(*messages) -> FakeChatModel:
    return FakeChatModel(messages=list(messages))


def _init(model):
    return lambda _model, *, model_provider=None, **kwargs: model


@tool
def rogue_tool(path: str) -> str:
    """A tool no agent is allowed."""
    return "rogue"


# --- the allowlist ----------------------------------------------------------


def test_agent_tools_equal_the_allowlist(tmp_path):
    """Every agent's tools equal its allowlist — same pattern for each."""
    for name, allowlist in ALLOWLISTS.items():
        tools = agent_tools(
            name,
            root=tmp_path,
            cfg={"test_cmd": "pytest -q"},
            tracker=None if name == "chatty" else _tracker(),
        )
        assert all(isinstance(t, BaseTool) for t in tools)
        assert {t.name for t in tools} == set(allowlist)


def _tracker():
    from fakes import FakeTracker

    return FakeTracker()


def test_chatty_allowlist_is_the_read_tools(tmp_path):
    tools = agent_tools("chatty", root=tmp_path)
    assert [t.name for t in tools] == ["read_file", "list_files", "grep"]


def test_build_agent_accepts_only_basetool(tmp_path):
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 5}})
    tools = [*agent_tools("chatty", root=tmp_path)[:2], rogue_tool]
    with pytest.raises(TypeError):
        build_agent("chatty", cfg, [tools[0], "not-a-tool"], init=_init(_scripted()))


def test_build_agent_refuses_off_allowlist_tools(tmp_path):
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 5}})
    tools = [*agent_tools("chatty", root=tmp_path), rogue_tool]
    with pytest.raises(ToolError) as error:
        build_agent("chatty", cfg, tools, init=_init(_scripted()))
    assert error.value.denied == "allowlist"


# --- /fast drops the tracker reads ---------------------------------------------


def test_allowlist_without_a_tracker_drops_tracker_reads():
    assert "read_work_item" not in allowlist_for("tester", tracker=False)
    assert "read_change_diff" not in allowlist_for("reviewer", tracker=False)
    assert allowlist_for("tester", tracker=True) == ALLOWLISTS["tester"]


def test_agent_tools_without_a_tracker_omits_tracker_reads(tmp_path):
    tools = agent_tools("tester", root=tmp_path, cfg={"test_cmd": "pytest -q"})
    names = {t.name for t in tools}
    assert "read_work_item" not in names
    assert {"read_file", "write_file", "edit_file", "run_tests"} <= names


def test_build_agent_accepts_the_fast_allowlist(tmp_path):
    cfg = cfg_with({"tester": {"provider": "anthropic", "model": "m", "model_calls": 5}})
    tools = agent_tools("tester", root=tmp_path, cfg={"test_cmd": "pytest -q"})
    agent = build_agent(
        "tester",
        cfg,
        tools,
        init=_init(_scripted()),
        allowlist=allowlist_for("tester", tracker=False),
    )
    assert {t.name for t in agent.tools} == set(allowlist_for("tester", tracker=False))


# --- building and answering --------------------------------------------------


def test_chatty_answers_with_its_tools(tmp_path):
    (tmp_path / "note.txt").write_text("the sky is blue\n", encoding="utf-8")
    _git(tmp_path)
    model = _scripted(
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "note.txt"}, "id": "t1"}],
        ),
        AIMessage(content="The note says the sky is blue."),
    )
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 15}})
    agent = build_agent(
        "chatty", cfg, agent_tools("chatty", root=tmp_path), init=_init(model)
    )

    result = agent.invoke([HumanMessage("what does note.txt say?")])

    assert result.text == "The note says the sky is blue."
    assert not result.limit_hit


def test_fresh_context_between_invocations(tmp_path):
    model = _scripted(AIMessage(content="first"), AIMessage(content="second"))
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 5}})
    agent = build_agent(
        "chatty", cfg, agent_tools("chatty", root=tmp_path), init=_init(model)
    )

    agent.invoke([HumanMessage("one")])
    agent.invoke([HumanMessage("two")])

    second_call_messages = model.invocations[1]
    human_texts = [m.content for m in second_call_messages if isinstance(m, HumanMessage)]
    assert human_texts == ["two"]


# --- limits -------------------------------------------------------------------


def _tool_calling_messages(n: int) -> list[AIMessage]:
    """A script that asks for a tool call on every one of ``n`` calls."""
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "x"}, "id": f"t{i}"}],
        )
        for i in range(n)
    ]


def test_model_call_limit_ends_the_run(tmp_path):
    # Three tool-calling replies and nothing else: the agent wants a fourth
    # call, which is exactly when the limit must fire.
    (tmp_path / "x").write_text("x\n", encoding="utf-8")
    _git(tmp_path)
    model = _scripted(*_tool_calling_messages(3))
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 3}})
    agent = build_agent(
        "chatty", cfg, agent_tools("chatty", root=tmp_path), init=_init(model)
    )

    result = agent.invoke([HumanMessage("keep reading")])

    assert result.limit_hit
    assert result.text == ""
    assert len(model.invocations) == 3


def test_recursion_limit_is_derived_and_always_set(tmp_path):
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 15}})
    agent = build_agent(
        "chatty", cfg, agent_tools("chatty", root=tmp_path), init=_init(_scripted())
    )
    # 4 super-steps per model->tools round (before_model, model,
    # after_model, tools) plus bookkeeping slack; always explicit.
    assert agent.recursion_limit == 15 * 4 + 8


def test_limit_hit_run_does_not_hit_the_recursion_limit(tmp_path):
    """Worst case: every call wants another. The derived limit must cover it."""
    (tmp_path / "x").write_text("x\n", encoding="utf-8")
    _git(tmp_path)
    model = _scripted(*_tool_calling_messages(15))
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 15}})
    agent = build_agent(
        "chatty", cfg, agent_tools("chatty", root=tmp_path), init=_init(model)
    )
    result = agent.invoke([HumanMessage("loop")])
    assert result.limit_hit


# --- events ------------------------------------------------------------------


def test_agent_run_emits_chat_scoped_model_and_tool_events(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path)
    model = _scripted(
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "t1"}],
        ),
        AIMessage(content="done"),
    )
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 15}})
    agent = build_agent(
        "chatty", cfg, agent_tools("chatty", root=tmp_path), init=_init(model)
    )

    from k0ntrol.events import EventWriter, K0EventHandler

    events_path = tmp_path / "events.jsonl"
    writer = EventWriter(events_path, chat="chat-1", mask=[])
    agent.invoke(
        [HumanMessage("q")], config={"callbacks": [K0EventHandler(writer)]}
    )

    import json

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [e["kind"] for e in events] == ["model", "tool", "model"]
    assert all(e["chat"] == "chat-1" for e in events)
    tool_event = events[1]
    assert tool_event["agent"] == "chatty"
    assert tool_event["tool"] == "read_file"
    assert tool_event["path"] == "a.py"
    assert tool_event["ok"] is True


def _git(root):
    import subprocess

    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
