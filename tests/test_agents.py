"""The one agent builder: allowlist, limits, fresh context.

Every agent comes from ``build_agent``: ``create_agent`` with the agent's
allowlist, k0's closed middleware set, a model-call limit and an explicit
derived recursion limit, behind a fresh-context wrapper.
"""

import json
import sys

import httpx
import pytest
from langchain.agents.middleware import model_retry
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from fakes import BadRequestError, FakeChatModel, FakeTracker, cfg_with
from k0ntrol.agents import (
    ALLOWLISTS,
    ANTHROPIC_WEB_SEARCH,
    NATIVE_SEARCH_MAX_CALLS,
    OPENAI_WEB_SEARCH,
    PAUSED_TURN_LIMIT,
    MissingStructuredOutput,
    QuestionRound,
    ResearchFinding,
    Spec,
    TicketList,
    agent_tools,
    allowlist_for,
    build_agent,
    is_transient,
    paused_turn,
)
from k0ntrol.events import EventWriter, K0EventHandler
from k0ntrol.search import uses_native_search
from k0ntrol.tools.errors import ToolError
from k0ntrol.tools.net import HttpResponse
from k0ntrol.tools.tracker import tracker_read_tools
from k0ntrol.tracker import TrackerError


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


def test_planner_and_researcher_tool_allowlists_are_exact(tmp_path):
    tracker = _tracker()
    planner = agent_tools(
        "planner",
        root=tmp_path,
        cfg={"test_cmd": "pytest -q"},
        tracker=tracker,
    )
    researcher = agent_tools(
        "researcher",
        root=tmp_path,
        cfg={"test_cmd": "pytest -q"},
        tracker=tracker,
    )

    assert {t.name for t in planner} == {
        "read_file",
        "list_files",
        "grep",
        "bulk_read",
        "search_memory",
        "read_work_item",
        "read_change_diff",
        "fetch_url",
        "web_search",
    }
    assert {t.name for t in researcher} == {
        "read_file",
        "list_files",
        "grep",
        "fetch_url",
        "web_search",
    }
    for tools in (planner, researcher):
        assert not {t.name for t in tools} & {"write_file", "edit_file", "run_tests"}


@pytest.mark.parametrize("name", ["planner", "researcher"])
def test_network_tools_use_injected_boundaries(tmp_path, name):
    fetches = []

    def fetch(request):
        fetches.append(request)
        return json.dumps(
            {
                "web": {
                    "results": [
                        {
                            "title": "Primary source",
                            "url": "https://example.com/docs",
                            "description": "A snippet.",
                        }
                    ]
                }
            }
        ).encode()

    def transport(target, addresses, *, timeout, max_bytes):
        return HttpResponse(200, {}, b"page text", False)

    tools = agent_tools(
        name,
        root=tmp_path,
        cfg={"search": {"provider": "brave"}},
        tracker=_tracker(),
        resolve=lambda host: ["93.184.216.34"],
        transport=transport,
        environ={"BRAVE_SEARCH_API_KEY": "fake-key"},
        fetch=fetch,
    )
    by_name = {tool.name: tool for tool in tools}

    page = by_name["fetch_url"].invoke({"url": "https://example.com/docs"})
    results = by_name["web_search"].invoke({"query": "primary source"})

    assert "UNTRUSTED" in page
    assert "page text" in page
    assert "UNTRUSTED" in results
    assert "Primary source" in results
    assert fetches


# --- native search: the provider's own web search (#182, ADR 0017) -------------


def _native_cfg(agent="planner", provider="openai", **search):
    cfg = cfg_with(
        {agent: {"provider": provider, "model": "m", "model_calls": 5}}
    )
    cfg["search"] = {"native": True, **search}
    return cfg


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("agent", ["planner", "researcher"])
def test_native_search_keys_on_the_k0_provider(agent, provider):
    cfg = _native_cfg(agent, provider)
    assert uses_native_search(cfg, agent) is True


@pytest.mark.parametrize(
    "provider", ["lmstudio", "openai_compatible", "bedrock", "openrouter"]
)
def test_native_search_follows_the_k0_provider_name(provider):
    # lmstudio and openai_compatible share LangChain's "openai"
    # model_provider — they still use search.provider (ADR 0017).
    cfg = _native_cfg(provider=provider)
    assert uses_native_search(cfg, "planner") is False


@pytest.mark.parametrize("agent", ["implementer", "reviewer", "chatty"])
def test_native_search_is_planner_and_researcher_only(agent):
    cfg = _native_cfg(agent)
    assert uses_native_search(cfg, agent) is False


def test_native_search_is_off_without_the_switch():
    cfg = cfg_with({"planner": {"provider": "openai", "model": "m"}})
    assert uses_native_search(cfg, "planner") is False
    cfg["search"] = {"native": False}
    assert uses_native_search(cfg, "planner") is False
    cfg["search"] = {"provider": "brave"}
    assert uses_native_search(cfg, "planner") is False


def test_agent_tools_swap_the_search_slot_for_the_provider_tool(tmp_path):
    """With native search on, PLANNER's ``web_search`` slot carries
    OpenAI's own tool dict — never k0's ``web_search`` (#182)."""
    cfg = _native_cfg()
    fetches = []
    tools = agent_tools(
        "planner",
        root=tmp_path,
        cfg=cfg,
        tracker=_tracker(),
        fetch=lambda request: fetches.append(request) or b"{}",
    )
    assert OPENAI_WEB_SEARCH in tools
    assert not any(
        isinstance(tool, BaseTool) and tool.name == "web_search" for tool in tools
    )
    assert fetches == []


def test_native_search_binds_the_tool_and_the_call_cap(tmp_path):
    """The mocked OpenAI Responses seam: the request carries
    ``{"type": "web_search"}`` and ``max_tool_calls`` (#182)."""
    model = _scripted(_planner_round())
    cfg = _native_cfg()
    tools = agent_tools(
        "planner", root=tmp_path, cfg=cfg, tracker=_tracker(), init=_init(model)
    )
    agent = build_agent("planner", cfg, tools, root=tmp_path, init=_init(model))

    agent.invoke([HumanMessage("plan")])

    assert OPENAI_WEB_SEARCH in model.bound_tools
    assert model.bind_kwargs["max_tool_calls"] == NATIVE_SEARCH_MAX_CALLS


def test_anthropic_native_search_binds_its_own_capped_tool(tmp_path):
    """On ``anthropic`` the slot carries Anthropic's web search, capped by
    its own ``max_uses``; ``max_tool_calls`` is OpenAI's and never sent (#209)."""
    model = _scripted(_planner_round())
    cfg = _native_cfg(provider="anthropic")
    tools = agent_tools(
        "planner", root=tmp_path, cfg=cfg, tracker=_tracker(), init=_init(model)
    )
    agent = build_agent("planner", cfg, tools, root=tmp_path, init=_init(model))

    agent.invoke([HumanMessage("plan")])

    assert ANTHROPIC_WEB_SEARCH in model.bound_tools
    assert OPENAI_WEB_SEARCH not in model.bound_tools
    assert ANTHROPIC_WEB_SEARCH["max_uses"] == NATIVE_SEARCH_MAX_CALLS
    assert "max_tool_calls" not in model.bind_kwargs


def test_native_search_never_binds_both_search_tools(tmp_path):
    model = _scripted(_planner_round())
    cfg = _native_cfg()
    cfg["search"]["provider"] = "brave"
    tools = agent_tools(
        "planner",
        root=tmp_path,
        cfg=cfg,
        tracker=_tracker(),
        init=_init(model),
        environ={"BRAVE_SEARCH_API_KEY": "x"},
    )
    agent = build_agent("planner", cfg, tools, root=tmp_path, init=_init(model))

    agent.invoke([HumanMessage("plan")])

    bound = [
        tool for tool in model.bound_tools
        if (tool.get("type") if isinstance(tool, dict) else tool.name)
        == "web_search"
    ]
    assert bound == [OPENAI_WEB_SEARCH]


def test_the_allowlist_refuses_both_search_tools(tmp_path):
    """k0's ``web_search`` and the provider's own must never be bound
    together — two tools, one slot (#182)."""
    cfg = _native_cfg()
    cfg["search"]["provider"] = "brave"
    k0_tools = agent_tools(
        "planner",
        root=tmp_path,
        cfg={**cfg, "search": {"provider": "brave"}},
        tracker=_tracker(),
        environ={"BRAVE_SEARCH_API_KEY": "x"},
    )
    with pytest.raises(ToolError) as error:
        build_agent(
            "planner",
            cfg,
            [*k0_tools, OPENAI_WEB_SEARCH],
            root=tmp_path,
            init=_init(_scripted()),
        )
    assert error.value.denied == "allowlist"


def test_the_search_constant_is_refused_off_the_search_agents(tmp_path):
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 5}})
    with pytest.raises(TypeError):
        build_agent(
            "chatty",
            cfg,
            [*agent_tools("chatty", root=tmp_path), OPENAI_WEB_SEARCH],
            root=tmp_path,
            init=_init(_scripted()),
        )


def test_any_other_tool_dict_still_raises(tmp_path):
    cfg = _native_cfg()
    tools = agent_tools("planner", root=tmp_path, cfg=cfg, tracker=_tracker())
    with pytest.raises(TypeError):
        build_agent(
            "planner",
            cfg,
            [*tools, {"type": "computer_use_preview"}],
            root=tmp_path,
            init=_init(_scripted()),
        )


def _planner_round(questions=None):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "QuestionRound",
                "args": {"questions": questions or []},
                "id": "q1",
            }
        ],
    )


def _native_agent(tmp_path, model, provider="openai", **tool_kwargs):
    cfg = _native_cfg(provider=provider)
    tools = agent_tools(
        "planner",
        root=tmp_path,
        cfg=cfg,
        tracker=_tracker(),
        init=_init(model),
        **tool_kwargs,
    )
    agent = build_agent("planner", cfg, tools, root=tmp_path, init=_init(model))
    return agent


def _event_log(root):
    path = root / ".k0-mem" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = EventWriter(path, run="r" * 32, mask=[])
    return K0EventHandler(writer), path


def _read_events(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def test_an_unsupported_model_retries_without_the_tool(tmp_path):
    fetches = []
    model = _scripted(
        BadRequestError("400 Unsupported tool: 'web_search'"),
        _planner_round(),
        _planner_round(),
    )
    agent = _native_agent(
        tmp_path, model, fetch=lambda request: fetches.append(request) or b"{}"
    )
    handler, path = _event_log(tmp_path)

    result = agent.invoke(
        [HumanMessage("plan")], config={"callbacks": [handler]}
    )

    assert len(model.invocations) == 2  # the failed call retried once
    assert OPENAI_WEB_SEARCH not in model.bound_tools
    assert fetches == []  # k0 never falls back to search.provider
    (warning,) = [
        event for event in _read_events(path) if event.get("kind") == "warning"
    ]
    assert warning["agent"] == "planner"
    assert warning["model"] == "m"
    assert warning["reason"] == "web search unsupported"

    # The tool stays off for the rest of the agent run — and no second
    # warning is emitted.
    agent.invoke([HumanMessage("again")], config={"callbacks": [handler]})
    assert OPENAI_WEB_SEARCH not in model.bound_tools
    warnings = [
        event for event in _read_events(path) if event.get("kind") == "warning"
    ]
    assert len(warnings) == 1


@pytest.mark.parametrize(
    "rejection",
    [
        "400 tools.0: 'web_search_20260209' is not supported on this model",
        "400 Web search is not enabled for this organization",
    ],
)
def test_an_anthropic_rejection_drops_the_tool_and_warns(tmp_path, rejection):
    model = _scripted(BadRequestError(rejection), _planner_round())
    agent = _native_agent(tmp_path, model, provider="anthropic")
    handler, path = _event_log(tmp_path)

    agent.invoke([HumanMessage("plan")], config={"callbacks": [handler]})

    assert len(model.invocations) == 2
    assert ANTHROPIC_WEB_SEARCH not in model.bound_tools
    (warning,) = [
        event for event in _read_events(path) if event.get("kind") == "warning"
    ]
    assert warning["reason"] == "web search unsupported"


def _paused(container_id="c1"):
    """Anthropic's ``pause_turn``: an unfinished server-search turn."""
    return AIMessage(
        content=[
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}},
            {"type": "text", "text": "partial"},
        ],
        response_metadata={"stop_reason": "pause_turn", "container": {"id": container_id}},
    )


def test_a_paused_search_turn_continues_with_the_same_tools(tmp_path):
    """``pause_turn`` is unfinished, not final: the paused content goes back
    unchanged, with the same tools and its container, until the model
    answers (#210 review)."""
    model = _scripted(_paused(), _planner_round())
    agent = _native_agent(tmp_path, model, provider="anthropic")

    result = agent.invoke([HumanMessage("plan")])

    assert len(model.invocations) == 2
    sent_back = model.invocations[1][-1]
    assert sent_back.response_metadata["stop_reason"] == "pause_turn"
    assert ANTHROPIC_WEB_SEARCH in model.bound_tools
    # One model node, one assistant message: the paused content is merged
    # ahead of the answer, so the history still counts one call (#210 review).
    (answer,) = [m for m in result.messages if isinstance(m, AIMessage)]
    assert answer.content[0]["type"] == "server_tool_use"
    assert any(block.get("text") == "partial" for block in answer.content)
    assert answer.tool_calls


def test_a_pause_on_the_last_allowed_call_is_not_a_limit_hit(tmp_path):
    """``model_calls: 1``: the turn pauses, then completes — the finding is
    kept, not sent to the limit gate (#210 review)."""
    model = _scripted(_paused(), _planner_round())
    cfg = _native_cfg(provider="anthropic")
    cfg["agents"]["planner"]["model_calls"] = 1
    tools = agent_tools(
        "planner", root=tmp_path, cfg=cfg, tracker=_tracker(), init=_init(model)
    )
    agent = build_agent("planner", cfg, tools, root=tmp_path, init=_init(model))

    result = agent.invoke([HumanMessage("plan")])

    assert result.limit_hit is False
    assert result.structured_response is not None


def test_anthropic_searches_run_directly():
    """The basic web search runs each search directly — no code-execution
    caller and no container — so a paused turn replays intact through
    langchain-anthropic, which drops ``caller`` (#210 review)."""
    assert ANTHROPIC_WEB_SEARCH["type"] == "web_search_20250305"
    assert "allowed_callers" not in ANTHROPIC_WEB_SEARCH


def test_paused_search_turns_are_bounded(tmp_path):
    model = _scripted(*[_paused() for _ in range(PAUSED_TURN_LIMIT + 1)])
    agent = _native_agent(tmp_path, model, provider="anthropic")

    with pytest.raises(RuntimeError, match="paused"):
        agent.invoke([HumanMessage("plan")])

    assert len(model.invocations) == PAUSED_TURN_LIMIT + 1


def test_a_paused_turn_inside_a_structured_output_error_is_found():
    """With provider-native structured output, LangChain parses the paused
    turn and raises; the paused message rides on the error."""
    from langchain.agents.structured_output import StructuredOutputValidationError

    paused = _paused()
    error = StructuredOutputValidationError("ResearchFinding", ValueError("x"), paused)
    assert paused_turn(error) is paused
    assert paused_turn(ValueError("x")) is None


def test_other_bad_request_errors_still_fail_the_run(tmp_path):
    model = _scripted(BadRequestError("400 context length exceeded"))
    agent = _native_agent(tmp_path, model)
    handler, path = _event_log(tmp_path)

    with pytest.raises(BadRequestError):
        agent.invoke([HumanMessage("plan")], config={"callbacks": [handler]})

    assert not [
        event for event in _read_events(path) if event.get("kind") == "warning"
    ]


def test_the_search_rejection_is_not_transient():
    assert not is_transient(BadRequestError("400 Unsupported tool: 'web_search'"))


def _tracker():
    from fakes import FakeTracker

    return FakeTracker()


def _structured_agent(tmp_path, name, model):
    effort = "high" if name == "planner" else "medium"
    cfg = cfg_with(
        {name: {"provider": "anthropic", "model": "m", "model_calls": 5, "effort": effort}}
    )
    tools = agent_tools(
        name,
        root=tmp_path,
        cfg=cfg,
        tracker=_tracker() if name == "planner" else None,
        init=_init(model),
    )
    return build_agent(name, cfg, tools, root=tmp_path, init=_init(model))


def test_researcher_returns_a_research_finding(tmp_path):
    model = _scripted(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ResearchFinding",
                    "args": {
                        "findings": ["The docs recommend X."],
                        "proposed_decision": "Use X",
                        "citations": [
                            {"title": "Docs", "url": "https://example.com/docs"}
                        ],
                    },
                    "id": "r1",
                }
            ],
        )
    )
    agent = _structured_agent(tmp_path, "researcher", model)

    result = agent.invoke([HumanMessage("research X")])

    assert agent.name == "researcher"
    assert agent.model_calls == 5
    assert agent.effort == "medium"
    assert isinstance(result.structured_response, ResearchFinding)
    assert result.structured_response.proposed_decision == "Use X"
    assert result.structured_response.citations[0].url == "https://example.com/docs"


def test_planner_structured_outputs_validate(tmp_path):
    cases = [
        (
            "QuestionRound",
            {
                "questions": [
                    {
                        "number": 1,
                        "question": "Which database?",
                        "recommendation": "Postgres",
                        "rationale": "Already operated here",
                    }
                ]
            },
            QuestionRound,
        ),
        (
            "Spec",
            {
                "title": "Plan workflow",
                "problem_statement": "Planning is one-shot.",
                "solution": "Ask, then publish spec and tickets.",
                "user_stories": ["As a human I want questions first."],
                "implementation_decisions": ["Use tracker work items."],
                "testing_decisions": ["Test workflow entry points."],
                "seams": ["workflow start/resume"],
                "glossary_and_decisions": ["PLANNER asks numbered questions."],
                "not_yet_decided": ["Ticket size default"],
                "out_of_scope": ["Parallel builds"],
            },
            Spec,
        ),
        (
            "TicketList",
            {
                "tickets": [
                    {
                        "title": "Glossary and ADRs",
                        "what_to_build": "Record terms and decisions.",
                        "seams": ["docs"],
                        "acceptance_criteria": ["Glossary updated"],
                        "blocked_by": [],
                    },
                    {
                        "title": "Planner agent",
                        "what_to_build": "Build PLANNER.",
                        "seams": ["agent tools"],
                        "acceptance_criteria": ["Allowlist is exact"],
                        "blocked_by": [0],
                    },
                ]
            },
            TicketList,
        ),
    ]

    for index, (tool_name, args, expected) in enumerate(cases):
        model = _scripted(
            AIMessage(
                content="",
                tool_calls=[{"name": tool_name, "args": args, "id": f"o{index}"}],
            )
        )
        result = _structured_agent(tmp_path, "planner", model).invoke(
            [HumanMessage("plan")]
        )
        assert isinstance(result.structured_response, expected)

    assert result.structured_response.tickets[1].blocked_by == [0]


def test_ticket_list_rejects_blockers_that_are_not_earlier():
    with pytest.raises(ValidationError):
        TicketList.model_validate(
            {
                "tickets": [
                    {
                        "title": "second",
                        "what_to_build": "x",
                        "seams": [],
                        "acceptance_criteria": [],
                        "blocked_by": [1],
                    },
                    {
                        "title": "first",
                        "what_to_build": "x",
                        "seams": [],
                        "acceptance_criteria": [],
                        "blocked_by": [],
                    },
                ]
            }
        )


@pytest.mark.parametrize("field", ["findings", "proposed_decision", "citations"])
def test_research_finding_requires_content(field):
    payload = {
        "findings": ["The docs recommend X."],
        "proposed_decision": "Use X",
        "citations": [{"title": "Docs", "url": "https://example.com/docs"}],
    }
    payload[field] = "" if field == "proposed_decision" else []

    with pytest.raises(ValidationError):
        ResearchFinding.model_validate(payload)


@pytest.mark.parametrize(
    "name,match",
    [
        ("planner", "PLANNER returned no structured output"),
        ("researcher", "RESEARCHER returned no research finding"),
    ],
)
def test_new_agents_missing_structured_output_raises(tmp_path, name, match):
    model = _scripted(AIMessage(content="prose, not a tool call"))

    with pytest.raises(MissingStructuredOutput, match=match):
        _structured_agent(tmp_path, name, model).invoke([HumanMessage("go")])


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


# --- a tracker-less run drops the tracker reads ---------------------------------


def test_allowlist_without_a_tracker_drops_tracker_reads():
    assert "read_work_item" not in allowlist_for("implementer", tracker=False)
    assert "read_change_diff" not in allowlist_for("reviewer", tracker=False)
    assert allowlist_for("implementer", tracker=True) == ALLOWLISTS["implementer"]


def test_agent_tools_without_a_tracker_omits_tracker_reads(tmp_path):
    tools = agent_tools("implementer", root=tmp_path, cfg={"test_cmd": "pytest -q"})
    names = {t.name for t in tools}
    assert "read_work_item" not in names
    assert {"read_file", "write_file", "edit_file", "run_tests"} <= names


def test_build_agent_accepts_the_tracker_less_allowlist(tmp_path):
    cfg = cfg_with({"implementer": {"provider": "anthropic", "model": "m", "model_calls": 5}})
    tools = agent_tools("implementer", root=tmp_path, cfg={"test_cmd": "pytest -q"})
    agent = build_agent(
        "implementer",
        cfg,
        tools,
        root=tmp_path,
        init=_init(_scripted()),
        allowlist=allowlist_for("implementer", tracker=False),
    )
    assert {t.name for t in agent.tools} == set(allowlist_for("implementer", tracker=False))


def test_implementer_tools_equal_its_allowlist(tmp_path):
    tools = agent_tools(
        "implementer",
        root=tmp_path,
        cfg={"test_cmd": "pytest -q"},
        tracker=_tracker(),
        init=_init(_scripted()),
    )
    assert [t.name for t in tools] == [
        "read_file",
        "list_files",
        "grep",
        "bulk_read",
        "search_memory",
        "write_file",
        "edit_file",
        "run_tests",
        "read_work_item",
        "read_change_diff",
    ]


def _implementer_tools(root, test_cmd):
    tools = agent_tools(
        "implementer",
        root=root,
        cfg={"test_cmd": test_cmd},
        tracker=_tracker(),
        init=_init(_scripted()),
    )
    return {tool.name: tool for tool in tools}


def test_implementer_writes_and_run_tests_share_one_slice_guard_per_tool_group(tmp_path):
    red_cmd = f'{sys.executable} -c "import sys; sys.exit(1)"'
    tools = _implementer_tools(tmp_path, red_cmd)

    tools["write_file"].invoke({"path": "tests/test_new.py", "content": "x\n"})
    with pytest.raises(ToolError):
        tools["write_file"].invoke({"path": "src/app.py", "content": "X = 1\n"})
    tools["run_tests"].invoke({"paths": []})
    tools["write_file"].invoke({"path": "src/app.py", "content": "X = 2\n"})
    assert (tmp_path / "src" / "app.py").read_text() == "X = 2\n"
    with pytest.raises(ToolError):
        tools["write_file"].invoke({"path": "tests/test_other.py", "content": "x\n"})

    other = _implementer_tools(tmp_path, red_cmd)
    with pytest.raises(ToolError):
        other["write_file"].invoke({"path": "src/other.py", "content": "X = 3\n"})


def test_build_agent_builds_implementer_from_its_config(tmp_path):
    model = _scripted(AIMessage(content="done"))
    cfg = cfg_with(
        {
            "implementer": {
                "provider": "anthropic",
                "model": "m",
                "effort": "medium",
                "model_calls": 7,
            }
        }
    )
    cfg["test_cmd"] = "pytest -q"
    tools = agent_tools(
        "implementer", root=tmp_path, cfg=cfg, tracker=_tracker(), init=_init(model)
    )

    agent = build_agent(
        "implementer", cfg, tools, root=tmp_path, init=_init(model)
    )

    assert agent.name == "implementer"
    assert agent.model_calls == 7
    assert agent.effort == "medium"
    assert {tool.name for tool in agent.tools} == set(
        allowlist_for("implementer", tracker=True)
    )


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


# --- tool errors reach the agent ---------------------------------------------


@tool
def read_file(path: str) -> str:
    """A read that always refuses."""
    raise ToolError(f"not a readable file: {path}", denied="path_guard")


def _refused_read_agent(tmp_path, model):
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 5}})
    return build_agent(
        "chatty",
        cfg,
        [read_file],
        root=tmp_path,
        init=_init(model),
        allowlist=["read_file"],
    )


def _read_then_finish() -> FakeChatModel:
    return _scripted(
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "x"}, "id": "t1"}],
        ),
        AIMessage(content="recovered"),
    )


def test_tool_error_reaches_the_agent_and_the_run_finishes(tmp_path):
    model = _read_then_finish()

    result = _refused_read_agent(tmp_path, model).invoke([HumanMessage("go")])

    assert result.text == "recovered"
    seen = [m for m in model.invocations[1] if isinstance(m, ToolMessage)]
    assert [(m.status, m.content, m.tool_call_id) for m in seen] == [
        ("error", "not a readable file: x", "t1")
    ]


def test_tracker_error_from_a_read_tool_reaches_the_agent_and_the_run_finishes(tmp_path):
    class FailingTracker(FakeTracker):
        def read_work_item(self, ref):
            raise TrackerError(f"gh api failed: {ref}")

    model = _scripted(
        AIMessage(
            content="",
            tool_calls=[{"name": "read_work_item", "args": {"work_item": "x"}, "id": "t1"}],
        ),
        AIMessage(content="recovered"),
    )
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 5}})
    agent = build_agent(
        "chatty",
        cfg,
        tracker_read_tools(FailingTracker()),
        root=tmp_path,
        init=_init(model),
        allowlist=["read_work_item", "read_change_diff"],
    )

    result = agent.invoke([HumanMessage("go")])

    assert result.text == "recovered"
    seen = [m for m in model.invocations[1] if isinstance(m, ToolMessage)]
    assert [(m.status, m.content) for m in seen] == [("error", "gh api failed: x")]


def test_tool_error_keeps_ok_err_and_denied_in_the_tool_event(tmp_path):
    agent = _refused_read_agent(tmp_path, _read_then_finish())
    events_path = tmp_path / "events.jsonl"
    writer = EventWriter(events_path, chat="chat-1", mask=[])

    agent.invoke([HumanMessage("go")], config={"callbacks": [K0EventHandler(writer)]})

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    tool_event = next(e for e in events if e["kind"] == "tool")
    assert tool_event["ok"] is False
    assert tool_event["err"] == "not a readable file: x"
    assert tool_event["denied"] == "path_guard"


def test_other_tool_exceptions_still_propagate(tmp_path):
    @tool
    def read_file(path: str) -> str:
        """A read that crashes."""
        raise RuntimeError("boom")

    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 5}})
    agent = build_agent(
        "chatty",
        cfg,
        [read_file],
        root=tmp_path,
        init=_init(_read_then_finish()),
        allowlist=["read_file"],
    )

    with pytest.raises(RuntimeError, match="boom"):
        agent.invoke([HumanMessage("go")])


# --- a malformed tool name never reaches the provider ----------------------------


def test_an_invalid_tool_name_is_cleaned_before_the_next_model_call(tmp_path):
    """Bedrock rejects a request whose history holds a tool name outside
    ``[a-zA-Z0-9_-]``; the model still sees the "not a valid tool" error."""
    model = _scripted(
        AIMessage(
            content="",
            tool_calls=[{"name": "functions.read_file", "args": {}, "id": "t1"}],
        ),
        AIMessage(content="recovered"),
    )

    result = _chatty(tmp_path, model).invoke([HumanMessage("go")])

    assert result.text == "recovered"
    sent = model.invocations[1]
    names = [c["name"] for m in sent if isinstance(m, AIMessage) for c in m.tool_calls]
    assert names == ["functions_read_file"]
    errors = [m for m in sent if isinstance(m, ToolMessage) and m.status == "error"]
    assert "functions.read_file is not a valid tool" in errors[0].content


# --- old tool results are cleared once the context grows -----------------------


def test_the_oldest_seen_tool_results_are_cleared_until_the_context_fits(tmp_path):
    """Five sequential reads of ~15k tokens: the oldest seen ones become a
    placeholder until the context fits 32k; the agent's own state keeps them."""

    @tool
    def read_file(path: str) -> str:
        """A read that returns a big file."""
        return f"{path}: " + "x" * 60_000  # ~15k tokens each

    reads = [
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": f"f{i}"}, "id": f"t{i}"}])
        for i in range(5)
    ]
    model = _scripted(*reads, AIMessage(content="done"))
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 10}})
    agent = build_agent(
        "chatty", cfg, [read_file], root=tmp_path, init=_init(model), allowlist=["read_file"]
    )

    result = agent.invoke([HumanMessage("go")])

    seen = [m.content for m in model.invocations[-1] if isinstance(m, ToolMessage)]
    assert [text.startswith("[cleared") for text in seen] == [True, True, True, False, False]
    kept = [m.content for m in result.messages if isinstance(m, ToolMessage)]
    assert kept[0].startswith("f0: ")  # the run's own record is untouched


def test_a_batch_the_model_has_not_seen_yet_is_never_cleared(tmp_path):
    """55 files read in one parallel step all reach the model whole: only
    results from earlier turns, already seen, can be cleared."""

    @tool
    def read_file(path: str) -> str:
        """A read that returns a big file."""
        return f"{path}: " + "x" * 60_000

    batch = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": f"f{i}"}, "id": f"t{i}"} for i in range(5)],
    )
    model = _scripted(batch, AIMessage(content="done"))
    cfg = cfg_with({"chatty": {"provider": "anthropic", "model": "m", "model_calls": 10}})
    agent = build_agent(
        "chatty", cfg, [read_file], root=tmp_path, init=_init(model), allowlist=["read_file"]
    )

    agent.invoke([HumanMessage("go")])

    seen = [m.content for m in model.invocations[-1] if isinstance(m, ToolMessage)]
    assert all(not text.startswith("[cleared") for text in seen)


# --- transient model errors are retried (#85) ----------------------------------


class ProviderError(Exception):
    """A provider HTTP error, shaped like ollama's ``ResponseError``."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"provider said {status_code}")
        self.status_code = status_code


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr(model_retry.time, "sleep", lambda seconds: None)


def _chatty(tmp_path, model, model_calls=5):
    cfg = cfg_with(
        {"chatty": {"provider": "anthropic", "model": "m", "model_calls": model_calls}}
    )
    return build_agent(
        "chatty", cfg, [], root=tmp_path, init=_init(model), allowlist=[]
    )


def test_a_transient_model_error_is_retried_with_one_event_per_try(
    tmp_path, no_backoff
):
    model = _scripted(ProviderError(500), AIMessage(content="recovered"))
    events_path = tmp_path / "events.jsonl"
    handler = K0EventHandler(EventWriter(events_path, chat="chat-1", mask=[]))

    result = _chatty(tmp_path, model).invoke(
        [HumanMessage("go")], config={"callbacks": [handler]}
    )

    assert result.text == "recovered"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [e["ok"] for e in events if e["kind"] == "model"] == [False, True]


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("refused"),
        TimeoutError("slow"),
        ProviderError(429),
        ProviderError(-1),  # ollama: an error inside the stream has no status
        httpx.ConnectError("ollama is not running"),
    ],
)
def test_connection_timeout_and_rate_limit_errors_are_transient(
    tmp_path, no_backoff, error
):
    model = _scripted(error, AIMessage(content="recovered"))

    assert _chatty(tmp_path, model).invoke([HumanMessage("go")]).text == "recovered"


@pytest.mark.parametrize("status", [400, 401, 404])
def test_a_client_error_is_not_retried(tmp_path, no_backoff, status):
    model = _scripted(ProviderError(status), AIMessage(content="never reached"))

    with pytest.raises(ProviderError):
        _chatty(tmp_path, model).invoke([HumanMessage("go")])
    assert len(model.invocations) == 1


def test_retries_give_up_and_the_error_stops_the_run(tmp_path, no_backoff):
    model = _scripted(*[ProviderError(500)] * 3, AIMessage(content="too late"))

    with pytest.raises(ProviderError):
        _chatty(tmp_path, model).invoke([HumanMessage("go")])
    assert len(model.invocations) == 3


def test_retries_do_not_count_toward_the_model_call_limit(tmp_path, no_backoff):
    """One agent turn is one model call, however many tries it took."""
    model = _scripted(ProviderError(503), AIMessage(content="recovered"))

    result = _chatty(tmp_path, model, model_calls=1).invoke([HumanMessage("go")])

    assert result.text == "recovered"
    assert result.limit_hit is False


# --- Bedrock transport failures are transient (#177) ---------------------------


class BotoCoreError(Exception):
    """botocore's base error: an SDK-side failure with no ``status_code``."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(args or (kwargs,))


class HTTPClientError(BotoCoreError):
    """botocore's transport-layer error."""


class ReadTimeoutError(HTTPClientError):
    """botocore's Converse read timeout — also raised mid event stream,
    where the SDK's own retry layer never sees it."""


class ConnectTimeoutError(BotoCoreError):
    """botocore's connect timeout."""


class EndpointConnectionError(BotoCoreError):
    """botocore's connection failure."""


class ClientError(Exception):
    """botocore's API error: the service answered — auth, config, quota."""


@pytest.mark.parametrize(
    "error",
    [
        ReadTimeoutError("read timed out"),
        ConnectTimeoutError(endpoint_url="https://bedrock.example"),
        EndpointConnectionError(endpoint_url="https://bedrock.example"),
    ],
    ids=["read_timeout", "connect_timeout", "endpoint_unreachable"],
)
def test_bedrock_timeouts_and_connection_errors_are_transient(
    tmp_path, no_backoff, error
):
    model = _scripted(error, AIMessage(content="recovered"))

    assert _chatty(tmp_path, model).invoke([HumanMessage("go")]).text == "recovered"


@pytest.mark.parametrize(
    "error",
    [
        ClientError("AccessDeniedException"),  # auth
        ClientError("ValidationException"),  # config
        ClientError("ThrottlingException"),  # quota
    ],
    ids=["auth", "config", "quota"],
)
def test_bedrock_service_errors_are_not_retried(tmp_path, no_backoff, error):
    model = _scripted(error, AIMessage(content="never reached"))

    with pytest.raises(ClientError):
        _chatty(tmp_path, model).invoke([HumanMessage("go")])
    assert len(model.invocations) == 1


def test_bedrock_timeouts_give_up_after_bounded_tries(tmp_path, no_backoff):
    """Repeated Bedrock timeouts end in a clear error after the bounded
    tries — each try emits its own ``model`` event, so every invocation
    attempt is accounted for (#177)."""
    model = _scripted(*[ReadTimeoutError("converse")] * 3)
    events_path = tmp_path / "events.jsonl"
    handler = K0EventHandler(EventWriter(events_path, chat="chat-1", mask=[]))

    with pytest.raises(ReadTimeoutError):
        _chatty(tmp_path, model).invoke(
            [HumanMessage("go")], config={"callbacks": [handler]}
        )

    assert len(model.invocations) == 3
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [e["ok"] for e in events if e["kind"] == "model"] == [False] * 3


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


def test_a_helpers_tokens_never_reach_the_answer():
    """bulk_read runs inside CHATTY's tools: its deltas are not the answer (#205)."""
    from langchain_core.messages import AIMessageChunk

    from k0ntrol.agents import _emit_text
    from k0ntrol.events import META_AGENT, META_HELPER

    shown: list[str] = []
    _emit_text((AIMessageChunk(content="digest"), {META_HELPER: "bulk_read"}), shown.append)
    _emit_text((AIMessageChunk(content="answer"), {META_AGENT: "chatty"}), shown.append)
    assert shown == ["answer"]
