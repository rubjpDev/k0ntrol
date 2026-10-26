"""The one agent builder: allowlist, limits, fresh context.

Every agent comes from ``build_agent``: ``create_agent`` with the agent's
allowlist, k0's closed middleware set, a model-call limit and an explicit
derived recursion limit, behind a fresh-context wrapper.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool

from fakes import FakeChatModel, cfg_with
from k0ntrol.agents import ALLOWLISTS, agent_tools, build_agent
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
    assert [t.name for t in tools] == [
        "read_file",
        "list_files",
        "grep",
        "bulk_read",
        "search_memory",
    ]


def test_build_agent_accepts_only_basetool(tmp_path):
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 5}})
    tools = [*agent_tools("chatty", root=tmp_path)[:2], rogue_tool]
    with pytest.raises(TypeError):
        build_agent(
            "chatty", cfg, [tools[0], "not-a-tool"], root=tmp_path, init=_init(_scripted())
        )


def test_build_agent_refuses_off_allowlist_tools(tmp_path):
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 5}})
    tools = [*agent_tools("chatty", root=tmp_path), rogue_tool]
    with pytest.raises(ToolError) as error:
        build_agent("chatty", cfg, tools, root=tmp_path, init=_init(_scripted()))
    assert error.value.denied == "allowlist"


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
        "chatty",
        cfg,
        agent_tools("chatty", root=tmp_path, cfg=cfg, init=_init(model)),
        root=tmp_path,
        init=_init(model),
    )

    result = agent.invoke([HumanMessage("what does note.txt say?")])

    assert result.text == "The note says the sky is blue."
    assert not result.limit_hit


def test_chatty_uses_bulk_read_and_search_memory(tmp_path):
    """End to end: CHATTY calls both memory tools on its allowlist."""
    _git(tmp_path)
    (tmp_path / ".k0-mem" / "Project_context" / tmp_path.name / "files").mkdir(
        parents=True
    )
    (tmp_path / ".k0-mem" / "Project_context" / tmp_path.name / "files" / "a.py.md").write_text(
        "# a\ncheckout code\n", encoding="utf-8"
    )
    (tmp_path / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    _git_add(tmp_path)
    cfg = cfg_with(
        {"chatty": {"provider": "anthropic", "model": "m", "model_calls": 15}},
        helpers={"bulk_read": {"provider": "anthropic", "model": "m"}},
    )
    model = _scripted(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "search_memory", "args": {"query": "checkout"}, "id": "t1"}
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "bulk_read",
                    "args": {"question": "what?", "paths": ["a.py"]},
                    "id": "t2",
                }
            ],
        ),
        AIMessage(content="the helper says hi"),
        AIMessage(content="done"),
    )
    agent = build_agent(
        "chatty",
        cfg,
        agent_tools("chatty", root=tmp_path, cfg=cfg, init=_init(model)),
        root=tmp_path,
        init=_init(model),
    )

    result = agent.invoke([HumanMessage("q")])

    assert result.text == "done"
    # call 2 asks for bulk_read; the helper's one-shot call is call 3
    helper_prompt = model.invocations[2][-1].content
    assert '<file repo=' in helper_prompt and 'path="a.py"' in helper_prompt


# --- system-prompt memory -----------------------------------------------------


def _memory_files(root, rules="# Rules\n\nGlobal rules for every agent.\n", strategy=None):
    mem_dir = root / ".k0-mem"
    (mem_dir / "General_mem").mkdir(parents=True, exist_ok=True)
    (mem_dir / "General_mem" / "Rules.md").write_text(rules, encoding="utf-8")
    if strategy is not None:
        topo = mem_dir / "Project_topology" / root.name
        topo.mkdir(parents=True, exist_ok=True)
        (topo / "strategy.md").write_text(strategy, encoding="utf-8")


def _system_prompt_sent(model) -> str:
    first = model.invocations[0][0]
    return str(first.content)


def test_system_prompt_carries_edited_rules_and_topology(tmp_path):
    _memory_files(tmp_path, rules="always write tests first\n", strategy="pet the cat\n")
    model = _scripted(AIMessage(content="ok"))
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 5}})
    agent = build_agent(
        "chatty",
        cfg,
        agent_tools("chatty", root=tmp_path, cfg=cfg, init=_init(model)),
        root=tmp_path,
        init=_init(model),
    )

    agent.invoke([HumanMessage("hi")])

    prompt = _system_prompt_sent(model)
    assert "always write tests first" in prompt
    assert "pet the cat" in prompt


def test_system_prompt_omits_unedited_memory(tmp_path):
    from k0ntrol.memory.layout import RULES

    _memory_files(tmp_path, rules=RULES)
    model = _scripted(AIMessage(content="ok"))
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 5}})
    agent = build_agent(
        "chatty",
        cfg,
        agent_tools("chatty", root=tmp_path, cfg=cfg, init=_init(model)),
        root=tmp_path,
        init=_init(model),
    )

    agent.invoke([HumanMessage("hi")])

    assert "Global rules for every agent" not in _system_prompt_sent(model)


def test_fresh_context_between_invocations(tmp_path):
    model = _scripted(AIMessage(content="first"), AIMessage(content="second"))
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 5}})
    agent = build_agent(
        "chatty",
        cfg,
        agent_tools("chatty", root=tmp_path, cfg=cfg, init=_init(model)),
        root=tmp_path,
        init=_init(model),
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
        "chatty",
        cfg,
        agent_tools("chatty", root=tmp_path, cfg=cfg, init=_init(model)),
        root=tmp_path,
        init=_init(model),
    )

    result = agent.invoke([HumanMessage("keep reading")])

    assert result.limit_hit
    assert result.text == ""
    assert len(model.invocations) == 3


def test_recursion_limit_is_derived_and_always_set(tmp_path):
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 15}})
    agent = build_agent(
        "chatty",
        cfg,
        agent_tools("chatty", root=tmp_path, cfg=cfg, init=_init(_scripted())),
        root=tmp_path,
        init=_init(_scripted()),
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
        "chatty",
        cfg,
        agent_tools("chatty", root=tmp_path, cfg=cfg, init=_init(model)),
        root=tmp_path,
        init=_init(model),
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
        "chatty",
        cfg,
        agent_tools("chatty", root=tmp_path, cfg=cfg, init=_init(model)),
        root=tmp_path,
        init=_init(model),
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


def _git_add(root):
    import subprocess

    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
