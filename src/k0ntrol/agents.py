"""The one agent builder.

Every agent — CHATTY for ``/ask`` today, the workflow agents as their nodes
land — is built here from ``create_agent``: the agent's allowlist, k0's
closed middleware set (a run-scoped model-call limit with
``exit_behavior="end"``, a retry of transient model errors, old tool
results cleared once the run's context grows, invalid tool names cleaned
from what the provider receives, and tool errors surfaced to the model), an
explicit recursion limit derived from the model-call limit, and a
fresh-context wrapper (``K0Agent``).

The allowlist is enforced in code, twice: ``agent_tools`` builds exactly
the tools on ``ALLOWLISTS[name]``, and ``build_agent`` refuses any tool set
that is not that allowlist — off-allowlist tools get
``ToolError(denied="allowlist")``, the canonical refusal. Only ``BaseTool``
instances are accepted: dict tools (provider built-ins) would bypass the
allowlist check in the tool node — but for the one exception ADR 0017
carves out: a provider's own web search (``NATIVE_SEARCH_TOOLS``) may fill the ``web_search`` slot for
PLANNER and RESEARCHER.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelResponse,
    ModelRetryMiddleware,
    ToolCallRequest,
)
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.types import Command
from pydantic import BaseModel, Field, model_validator

from k0ntrol.events import META_AGENT, META_EFFORT, META_HELPER, WARNING_EVENT
from k0ntrol.memory.index import custom_rules_text, custom_topology_text
from k0ntrol.providers import build_chat_model
from k0ntrol.search import NATIVE_SEARCH_AGENTS, native_search_provider, uses_native_search
from k0ntrol.tools.bulk_read import make_bulk_read
from k0ntrol.tools.errors import ToolError
from k0ntrol.tools.fetch_url import make_fetch_url
from k0ntrol.tools.net import Resolver, Transport
from k0ntrol.tools.read import make_grep, make_list_files, make_read_file
from k0ntrol.tools.run_tests import make_run_tests
from k0ntrol.tools.search_memory import make_search_memory
from k0ntrol.tools.slice_guard import SliceGuard, SliceStart
from k0ntrol.tools.slice_journal import SliceJournal
from k0ntrol.tools.tracker import tracker_read_tools
from k0ntrol.tools.web_search import Fetch, make_web_search
from k0ntrol.tools.write import WritePolicy, make_edit_file, make_write_file

_PRIMARY_READ_TOOLS = ("read_file", "list_files", "grep")
_HELPER_READ_TOOLS = ("bulk_read", "search_memory")
_BASE_READ_TOOLS = _PRIMARY_READ_TOOLS + _HELPER_READ_TOOLS

# Reads that need a tracker; a run without a work item drops them.
_TRACKER_TOOLS = ("read_work_item", "read_change_diff")

_NETWORK_TOOLS = ("fetch_url", "web_search")

# The cap on provider searches per model call — a fixed k0 constant, not
# configurable (ADR 0017): OpenAI's ``max_tool_calls``, Anthropic's
# ``max_uses``.
NATIVE_SEARCH_MAX_CALLS = 5

# The provider built-ins PLANNER and RESEARCHER get instead of k0's
# ``web_search`` when ``uses_native_search`` says so (ADR 0017): constants
# k0 owns, the one exception to "tools are only ``BaseTool`` instances".
# Binding OpenAI's routes ``ChatOpenAI`` through the Responses API (#182);
# Anthropic's carries its own cap (#209).
OPENAI_WEB_SEARCH: dict[str, Any] = {"type": "web_search"}
ANTHROPIC_WEB_SEARCH: dict[str, Any] = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": NATIVE_SEARCH_MAX_CALLS,
}
NATIVE_SEARCH_TOOLS: dict[str, dict[str, Any]] = {
    "openai": OPENAI_WEB_SEARCH,
    "anthropic": ANTHROPIC_WEB_SEARCH,
}

# The roster's tool allowlists (issue #12). `bulk_read` and `search_memory`
# are on the base reads (#40); CHATTY's `read_work_item` joins when the
# tracker lands (#43).
ALLOWLISTS: dict[str, tuple[str, ...]] = {
    "planner": _BASE_READ_TOOLS + _TRACKER_TOOLS + _NETWORK_TOOLS,
    "researcher": _PRIMARY_READ_TOOLS + _NETWORK_TOOLS,
    "chatty": _BASE_READ_TOOLS,
    "implementer": _BASE_READ_TOOLS
    + ("write_file", "edit_file", "run_tests")
    + _TRACKER_TOOLS,
    "reviewer": _BASE_READ_TOOLS + ("read_work_item", "read_change_diff"),
}

# A model→tools round costs four super-steps in the compiled agent graph
# (before_model, model, after_model, tools), plus start/end bookkeeping.
# The recursion limit is a backstop: the model-call limit is the real bound
# and hitting a recursion limit means a k0 bug (#20).
_STEPS_PER_CALL = 4
_STEPS_SLACK = 8

# Tries after a transient provider error before the run stops (#85). The
# retries happen inside one model call, so they never count toward
# ``model_calls``; each try still emits its own ``model`` event.
_MODEL_RETRIES = 2

# Connection-level failures the provider SDKs raise without a status code:
# httpx's (ollama), the OpenAI/Anthropic clients' (timeouts included), and
# botocore's (Bedrock) — timeouts and transport breaks only, never the
# service's own ClientError/EventStreamError, so auth, config and quota
# failures still stop the run. A Converse read timeout raised while the
# event stream is being read escapes botocore's own retry layer (it only
# retries the request exchange), so k0's bounded retry is what covers it
# (#177).
_TRANSIENT_ERROR_TYPES = frozenset(
    {
        "TransportError",
        "APIConnectionError",
        "ReadTimeoutError",
        "ConnectTimeoutError",
        "EndpointConnectionError",
        "ProxyConnectionError",
        "ConnectionClosedError",
        "ResponseStreamingError",
        "IncompleteReadError",
    }
)

# Inside one agent run every model call resends all earlier tool output.
# Past this many tokens, results the model has already read reach it as a
# placeholder (the run's own messages keep them). Half of Ollama's 64k
# window, so a local model never silently drops its prompt.
_CLEAR_TOOL_RESULTS_AT = 32_000
_CLEARED = "[cleared to save context; call the tool again if you need this]"

# The tool names a provider accepts in the history (Bedrock's rule, the
# strictest one k0 talks to).
_TOOL_NAME = re.compile(r"[a-zA-Z0-9_-]{1,64}")


class Review(BaseModel):
    """REVIEWER's structured output: its verdict on the change.

    ``status`` is the verdict — ``completed`` or ``refused`` — tracked per
    axis separately from the text and findings: a refused or missing
    review is not a completed review with zero findings (#175).
    """

    status: Literal["completed", "refused"]
    summary: str
    findings: list[str]
    reason: str = ""

    @model_validator(mode="after")
    def _refusal_has_a_reason(self) -> "Review":
        if self.status == "refused" and not self.reason.strip():
            raise ValueError("a refused review needs a reason")
        return self


class Question(BaseModel):
    """One numbered PLANNER question with its recommended answer."""

    number: int
    question: str
    recommendation: str
    rationale: str = ""


class QuestionRound(BaseModel):
    """PLANNER's structured output for one Q&A round."""

    questions: list[Question]


class Spec(BaseModel):
    """PLANNER's structured output: the spec draft for the spec gate."""

    title: str
    problem_statement: str
    solution: str
    user_stories: list[str]
    implementation_decisions: list[str]
    testing_decisions: list[str]
    seams: list[str]
    glossary_and_decisions: list[str]
    not_yet_decided: list[str]
    out_of_scope: list[str]


class Ticket(BaseModel):
    """One child work item in PLANNER's ticket breakdown."""

    title: str
    what_to_build: str
    seams: list[str]
    acceptance_criteria: list[str]
    blocked_by: list[int] = Field(default_factory=list)


def _earlier_ticket(index: int, blocker: int) -> bool:
    return 0 <= blocker < index


class TicketList(BaseModel):
    """PLANNER's structured output: the spec's tickets, blockers first."""

    tickets: list[Ticket]

    @model_validator(mode="after")
    def _blockers_point_backwards(self) -> "TicketList":
        for index, ticket in enumerate(self.tickets):
            if any(not _earlier_ticket(index, blocker) for blocker in ticket.blocked_by):
                raise ValueError("ticket blockers must be earlier ticket indexes")
        return self


class DecisionTicket(BaseModel):
    """One open question on a Map, typed by who answers it (#124)."""

    question: str
    type: Literal["research", "Q&A", "human"]
    blocked_by: list[int] = Field(default_factory=list)


class ChartedMap(BaseModel):
    """PLANNER's structured output: the map ``/research`` charts."""

    title: str
    destination: str
    notes: str = ""
    decisions: list[DecisionTicket]
    not_yet_specified: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _blockers_point_backwards(self) -> "ChartedMap":
        for index, decision in enumerate(self.decisions):
            if any(
                not _earlier_ticket(index, blocker)
                for blocker in decision.blocked_by
            ):
                raise ValueError(
                    "decision blockers must be earlier decision indexes"
                )
        return self


class DecisionProposal(BaseModel):
    """PLANNER's structured output: the proposed decision on a Q&A
    decision ticket (#125)."""

    proposed_decision: str
    rationale: str = ""


class TriageBrief(BaseModel):
    """The agent brief a ``ready-for-*`` triage outcome posts (#128)."""

    what_to_build: str
    seams: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    parent: str = ""


class TriageOutcome(BaseModel):
    """PLANNER's triage proposal: one category, one state, the comment."""

    category: Literal["bug", "enhancement"]
    state: Literal[
        "needs-triage",
        "needs-info",
        "ready-for-agent",
        "ready-for-human",
        "wontfix",
    ]
    comment: str = ""
    brief: TriageBrief | None = None


class ResearchCitation(BaseModel):
    """A primary source RESEARCHER cites."""

    title: str
    url: str


class ResearchFinding(BaseModel):
    """RESEARCHER's structured output: cited findings and a proposed decision."""

    findings: list[str] = Field(min_length=1)
    proposed_decision: str = Field(min_length=1)
    citations: list[ResearchCitation] = Field(min_length=1)


_PLANNER_OUTPUT = (
    QuestionRound
    | Spec
    | TicketList
    | ChartedMap
    | DecisionProposal
    | TriageOutcome
)

# Agents whose ``create_agent`` gets a structured ``response_format`` (spec
# #33): the schema becomes a tool the model calls; the validated value lands
# on ``AgentResult.structured_response``.
_RESPONSE_FORMATS: dict[str, Any] = {
    "planner": ToolStrategy(_PLANNER_OUTPUT),
    "researcher": ResearchFinding,
    "reviewer": Review,
}


class MissingStructuredOutput(Exception):
    """An agent with a ``response_format`` finished without one and no
    limit hit.

    Raised inside the agent invoke — the same resumable stop path as a
    provider error: the checkpoint stays before the node, the error is
    shown, and the run is never ``stuck`` (spec #33, #58)."""


_MISSING_OUTPUT = {
    "planner": "PLANNER returned no structured output",
    "researcher": "RESEARCHER returned no research finding",
    "reviewer": "REVIEWER returned no review",
}


# The tools a proposal-only run never gets: no writes, no test runs —
# ``/implement``'s seam gate is a human decision, not a build (#116).
_SEAM_DROP = ("write_file", "edit_file", "run_tests")


def allowlist_for(name: str, *, tracker: bool, read_only: bool = False) -> tuple[str, ...]:
    """Agent ``name``'s allowlist for this run.

    Without a tracker (no work item), the tracker
    reads drop off the allowlist; the agent can never see the tool.
    ``read_only`` drops the mutating tools for proposal phases.
    """
    names = ALLOWLISTS[name]
    if read_only:
        names = _without(names, _SEAM_DROP)
    if tracker:
        return names
    return _without(names, _TRACKER_TOOLS)


def _without(names: tuple[str, ...], dropped: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(n for n in names if n not in dropped)


def agent_tools(
    name: str,
    *,
    root: Path,
    cfg: dict | None = None,
    tracker=None,
    init: Callable = init_chat_model,
    resolve: Resolver | None = None,
    transport: Transport | None = None,
    environ: Mapping[str, str] | None = None,
    fetch: Fetch | None = None,
    read_only: bool = False,
    slice_start: SliceStart = "green",
    journal: SliceJournal | None = None,
) -> list[BaseTool | dict[str, Any]]:
    """The tools on agent ``name``'s allowlist, bound to the run context.

    ``cfg`` supplies ``test_cmd``/``test_timeout`` for ``run_tests``,
    ``helpers.*`` for ``bulk_read`` and ``search.provider`` for
    ``web_search`` — or ``search.native``, which puts the provider's own
    search constant in the ``web_search`` slot instead (#182);
    ``tracker`` feeds the tracker reads; ``init`` builds
    helper models; ``resolve``/``transport``/``environ``/``fetch`` are the
    network tools' injected seams. Each is needed only by tools that use
    it, and only when the tool runs. With ``tracker=None`` the tracker
    reads are left off entirely (a ``<task>`` run), matching
    ``allowlist_for(name, tracker=False)``. ``read_only`` matches
    ``allowlist_for(..., read_only=True)`` the same way. ``slice_start``
    is IMPLEMENTER's guard's start state — ``refactor`` for a run started
    from a review redo (#110).
    """
    try:
        names = allowlist_for(
            name, tracker=tracker is not None, read_only=read_only
        )
    except KeyError:
        raise ValueError(f"unknown agent: {name!r}") from None
    factories = _tool_factories(
        name,
        root=root,
        cfg=cfg,
        tracker=tracker,
        init=init,
        resolve=resolve,
        transport=transport,
        environ=environ,
        fetch=fetch,
        slice_start=slice_start,
        journal=journal,
    )
    tools: list[BaseTool | dict[str, Any]] = []
    for tool_name in names:
        if tool_name == "web_search" and uses_native_search(cfg or {}, name):
            tools.append(NATIVE_SEARCH_TOOLS[native_search_provider(cfg or {}, name)])
        else:
            tools.append(factories[tool_name]())
    return tools


def build_agent(
    name: str,
    cfg: dict,
    tools: Sequence[BaseTool | dict[str, Any]],
    *,
    root: Path,
    init: Callable = init_chat_model,
    allowlist: Sequence[str] | None = None,
) -> "K0Agent":
    """Build agent ``name`` behind the fresh-context wrapper.

    ``tools`` must be exactly the agent's allowlist — ``allowlist``, or
    ``ALLOWLISTS[name]`` when it is ``None`` (a ``<task>`` run passes the
    tracker-less list). The model comes from the ``agents.<name>`` config
    entry, and the prompt from ``prompts/<name>.md`` plus the project's
    `Rules.md` and topology when someone has edited them.
    """
    entry = cfg["agents"][name]
    checked = _check_allowlist(name, tools, allowlist)
    model = build_chat_model(cfg, entry, init=init)
    middleware: list[AgentMiddleware] = [
        ModelCallLimitMiddleware(  # type: ignore[list-item]
            run_limit=entry["model_calls"], exit_behavior="end"
        ),
        ModelRetryMiddleware(
            max_retries=_MODEL_RETRIES,
            retry_on=is_transient,
            on_failure="error",
        ),
        ContextEditingMiddleware(
            edits=[ClearSeenToolResults(exclude_tools=("read_work_item",))],  # type: ignore[list-item]
        ),
        ToolNameMiddleware(),
        ToolErrorMiddleware(),
    ]
    if uses_native_search(cfg, name):
        middleware.append(
            NativeSearchMiddleware(
                agent=name, model=entry["model"], provider=entry["provider"]
            )
        )
    graph = create_agent(
        model,
        tools=checked,
        system_prompt=_system_prompt(name, root),
        response_format=_RESPONSE_FORMATS.get(name),
        middleware=middleware,
        name=name,
    )
    return K0Agent(
        name,
        graph,
        checked,
        model_calls=entry["model_calls"],
        effort=entry.get("effort"),
    )


def is_transient(error: Exception) -> bool:
    """A model error worth another try: anything but a 4xx client error
    (408 and 429 excepted), or a connection or timeout failure. Ollama
    reports an error inside its response stream with status ``-1``."""
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status in (408, 429) or not 400 <= status < 500
    return any(
        cls.__name__ in _TRANSIENT_ERROR_TYPES for cls in type(error).__mro__
    )


@dataclass
class ClearSeenToolResults:
    """A context edit: clear tool results the model has already read.

    Only results from before the model's latest message qualify: it saw
    them and answered. The results it is about to read, a whole parallel
    batch included, always reach it whole; clearing those made the model
    invent what the files said. The oldest go first, and only until the
    context is back under the trigger.
    """

    trigger: int = _CLEAR_TOOL_RESULTS_AT
    placeholder: str = _CLEARED
    exclude_tools: tuple[str, ...] = ()

    def apply(self, messages: list[BaseMessage], *, count_tokens: Callable) -> None:
        for index in _seen_tool_results(messages, self.exclude_tools):
            if count_tokens(messages) <= self.trigger:
                return
            messages[index] = messages[index].model_copy(update={"content": self.placeholder})


def _seen_tool_results(messages: list[BaseMessage], excluded: tuple[str, ...]) -> list[int]:
    """Indexes of tool results before the latest AI message, oldest first."""
    latest = max((i for i, m in enumerate(messages) if isinstance(m, AIMessage)), default=0)
    return [
        index
        for index, message in enumerate(messages[:latest])
        if isinstance(message, ToolMessage) and message.name not in excluded
    ]


class ToolNameMiddleware(AgentMiddleware):
    """Send the provider only valid tool names.

    A model can call a tool named ``functions.edit_file``. The tool node
    answers with a "not a valid tool" error the model sees, but the bad
    name stays in the history, and Bedrock rejects every later request.
    Only the request copy is cleaned: the agent's messages keep what the
    model said.
    """

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        messages = [_clean_tool_names(message) for message in request.messages]
        return handler(request.override(messages=messages))  # type: ignore[arg-type]


def _clean_tool_names(message: BaseMessage) -> BaseMessage:
    if not isinstance(message, AIMessage) or all(
        _TOOL_NAME.fullmatch(call["name"]) for call in message.tool_calls
    ):
        return message
    calls = [{**call, "name": _clean_name(call["name"])} for call in message.tool_calls]
    return message.model_copy(update={"tool_calls": calls})


def _clean_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64] or "invalid_tool"


class ToolErrorMiddleware(AgentMiddleware):
    """Turn a ``ToolError`` into an error ``ToolMessage`` the model sees.

    The tool node only handles argument errors by default and re-raises any
    other ``ToolException``, which would end the run. The error has already
    reached the event handler by the time it gets here, so the ``tool``
    event keeps ``err`` and ``denied``. Any other exception propagates.
    """

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        try:
            return handler(request)
        except ToolError as error:
            return ToolMessage(
                content=str(error),
                name=request.tool_call["name"],
                tool_call_id=request.tool_call["id"],
                status="error",
            )


class NativeSearchMiddleware(AgentMiddleware):
    """Bind the provider search's call cap; drop the tool if the model
    cannot use it (ADR 0017).

    On OpenAI every model call carries ``max_tool_calls`` so the
    provider's searches stay capped; Anthropic's tool carries its own
    ``max_uses`` and gets no extra setting. When the provider rejects the
    tool — the configured model does not support web search, or the
    organization has it switched off — the call is retried
    once without it and the tool stays off for the rest of the agent
    run. One ``warning`` event names the agent and model; k0 never falls
    back to ``search.provider`` (Brave): a silent switch would hide a
    misconfiguration and could bill a second account.
    """

    def __init__(self, *, agent: str, model: str, provider: str = "openai") -> None:
        super().__init__()
        self._agent = agent
        self._model = model
        self._cap = (
            {"max_tool_calls": NATIVE_SEARCH_MAX_CALLS} if provider == "openai" else {}
        )
        self._search_dropped = False

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        request = request.override(
            model_settings={**request.model_settings, **self._cap}
        )
        if self._search_dropped:
            request = request.override(tools=_without_search(request.tools))
        try:
            return _resume_paused(request, handler)
        except Exception as error:
            if not _unsupported_search(error):
                raise
        self._search_dropped = True
        dispatch_custom_event(
            WARNING_EVENT,
            {
                "agent": self._agent,
                "model": self._model,
                "reason": "web search unsupported",
            },
        )
        return handler(request.override(tools=_without_search(request.tools)))


def _without_search(tools: Sequence[Any]) -> list[Any]:
    return [tool for tool in tools if tool not in NATIVE_SEARCH_TOOLS.values()]


# Anthropic may pause a long server-search turn (``pause_turn``): the turn
# is unfinished, not final. k0 sends the paused content back unchanged,
# with the same tools and container, at most this many times per model
# call; one more pause fails the call (#210 review). Every continuation is
# its own model invocation, so each still emits one ``model`` event.
PAUSED_TURN_LIMIT = 3


def _resume_paused(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    paused: list[AIMessage] = []
    for _ in range(PAUSED_TURN_LIMIT + 1):
        outcome = _call_once(request, handler)
        if isinstance(outcome, ModelResponse):
            return _with_paused(outcome, paused)
        paused.append(outcome)
        request = _continuation(request, outcome)
    raise RuntimeError(
        f"the provider paused the search turn {PAUSED_TURN_LIMIT + 1} times; "
        "stopping this model call"
    )


def _call_once(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse | AIMessage:
    """The response, or the paused message when the turn paused — also
    when LangChain's provider structured output failed to parse it."""
    try:
        response = handler(request)
    except Exception as error:
        message = paused_turn(error)
        if message is None:
            raise
        return message
    return paused_turn(response) or response


def paused_turn(source: Any) -> AIMessage | None:
    """The paused assistant message in a model response, or in the
    structured-output error LangChain raises when it parses one."""
    message = getattr(source, "ai_message", None)
    if isinstance(source, ModelResponse):
        message = source.result[-1] if source.result else None
    return message if _is_paused(message) else None


def _is_paused(message: Any) -> bool:
    return (
        isinstance(message, AIMessage)
        and message.response_metadata.get("stop_reason") == "pause_turn"
    )


def _continuation(request: ModelRequest, message: AIMessage) -> ModelRequest:
    """The same request with the paused turn appended unchanged, in the
    container Anthropic ran it in."""
    settings = dict(request.model_settings)
    container = message.response_metadata.get("container")
    if isinstance(container, dict) and container.get("id"):
        settings["container"] = container["id"]
    return request.override(
        messages=[*request.messages, message], model_settings=settings
    )


def _with_paused(response: ModelResponse, paused: list[AIMessage]) -> ModelResponse:
    """Keep the paused turns in the agent's history before the answer."""
    if not paused:
        return response
    return ModelResponse(
        result=[*paused, *response.result],
        structured_response=response.structured_response,
    )


_WEB_SEARCH = re.compile(r"web[ _]search", re.IGNORECASE)


def _unsupported_search(error: Exception) -> bool:
    """A provider SDK's ``BadRequestError`` naming web search: the model
    cannot use the tool (``web_search``, ``web_search_20260209``), or the
    organization has it off ("Web search is not enabled"). Matched on the
    class name — the provider extras may not be installed — like
    ``_TRANSIENT_ERROR_TYPES``."""
    return bool(_WEB_SEARCH.search(str(error))) and any(
        cls.__name__ == "BadRequestError" for cls in type(error).__mro__
    )


def _check_allowlist(
    name: str,
    tools: Sequence[BaseTool | dict[str, Any]],
    allowlist: Sequence[str] | None = None,
) -> list[BaseTool | dict[str, Any]]:
    slots = [_tool_slot(name, item) for item in tools]
    expected = set(allowlist if allowlist is not None else ALLOWLISTS[name])
    if len(set(slots)) != len(slots) or set(slots) != expected:
        raise ToolError(
            f"agent {name!r} tools must be its allowlist {sorted(expected)}; "
            f"got {sorted(slots)}",
            denied="allowlist",
        )
    return list(tools)


def _tool_slot(agent: str, item: BaseTool | dict[str, Any]) -> str:
    """The allowlist slot a tool occupies; dict tools are refused but for
    the one provider built-in PLANNER and RESEARCHER may carry (ADR 0017)."""
    if isinstance(item, BaseTool):
        return item.name
    if agent in NATIVE_SEARCH_AGENTS and item in NATIVE_SEARCH_TOOLS.values():
        return "web_search"
    raise TypeError(f"tools must be BaseTool instances, got {item!r}")


def _prompt(name: str) -> str:
    return (files("k0ntrol") / "prompts" / f"{name}.md").read_text(
        encoding="utf-8"
    )


def _system_prompt(name: str, root: Path) -> str:
    """The agent's prompt file plus edited memory: `Rules.md` and topology
    go in when someone has changed them from the init template."""
    parts = [_prompt(name)]
    rules = custom_rules_text(root)
    if rules:
        parts.append(rules)
    topology = custom_topology_text(root)
    if topology:
        parts.append(topology)
    return "\n\n".join(parts)


def _tool_factories(
    agent: str,
    *,
    root: Path,
    cfg: dict | None,
    tracker,
    init: Callable,
    resolve: Resolver | None,
    transport: Transport | None,
    environ: Mapping[str, str] | None,
    fetch: Fetch | None,
    slice_start: SliceStart = "green",
    journal: SliceJournal | None = None,
) -> dict[str, Callable[[], BaseTool]]:
    guard = (
        SliceGuard(start=slice_start, journal=journal)
        if agent == "implementer"
        else None
    )
    factories: dict[str, Callable[[], BaseTool]] = {
        "read_file": lambda: make_read_file(root),
        "list_files": lambda: make_list_files(root),
        "grep": lambda: make_grep(root),
        "bulk_read": lambda: make_bulk_read(root, cfg, init=init),
        "search_memory": lambda: make_search_memory(root),
        "fetch_url": lambda: make_fetch_url(resolve=resolve, transport=transport),
        "web_search": lambda: make_web_search(
            cfg or {}, environ=environ, fetch=fetch
        ),
        "run_tests": lambda: make_run_tests(root, _need_cfg(cfg), guard),
        "read_work_item": lambda: _tracker_tool("read_work_item", tracker),
        "read_change_diff": lambda: _tracker_tool("read_change_diff", tracker),
    }
    policy: WritePolicy | None = guard
    if policy is not None:
        factories["write_file"] = lambda: make_write_file(root, policy)
        factories["edit_file"] = lambda: make_edit_file(root, policy)
    return factories


def _need_cfg(cfg: dict | None) -> dict:
    if cfg is None:
        raise ValueError("run_tests needs the project config")
    return cfg


def _tracker_tool(name: str, tracker) -> BaseTool:
    if tracker is None:
        raise ValueError(f"the {name} tool needs a tracker")
    tools = {tool.name: tool for tool in tracker_read_tools(tracker)}
    return tools[name]


def _recursion_limit(model_calls: int) -> int:
    return model_calls * _STEPS_PER_CALL + _STEPS_SLACK


@dataclass
class AgentResult:
    """What one agent run leaves: the messages and whether the model-call
    limit ended it."""

    messages: list[BaseMessage]
    limit_hit: bool
    structured_response: Any = None

    @property
    def text(self) -> str:
        """The last real AI text; the injected limit message never counts."""
        return _answer_text(self.messages)


def _answer_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if _is_answer(message):
            return _content_text(message.content)
    return ""


def _is_answer(message: BaseMessage) -> bool:
    if not isinstance(message, AIMessage):
        return False
    text = _content_text(message.content)
    return bool(text) and not _is_limit_marker(text)


def _content_text(content) -> str:
    """A message's text, whether it is a plain string or content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _is_limit_marker(text: str) -> bool:
    return text.startswith("Model call limits exceeded")


def _limit_hit(messages: list[BaseMessage], model_calls: int) -> bool:
    # With exit_behavior="end" the middleware appends one extra AI message
    # when the agent wants a call past the limit, so a limit hit shows as
    # more AI messages than calls allowed.
    calls = sum(1 for m in messages if isinstance(m, AIMessage))
    return calls > model_calls


class K0Agent:
    """An agent behind the fresh-context wrapper.

    Every ``invoke``/``stream`` sends a brand-new ``{"messages": ...}`` into
    the compiled ``create_agent`` graph — no checkpointer, no thread state —
    so each run starts fresh (ADR 0002). The wrapper always sets the
    derived ``recursion_limit`` and tags callback metadata with the agent
    name and effort so the event handler can fill event fields.
    """

    def __init__(
        self,
        name: str,
        graph,
        tools: Sequence[BaseTool | dict[str, Any]],
        *,
        model_calls: int,
        effort: str | None = None,
    ) -> None:
        self.name = name
        self.tools = tuple(tools)
        self.model_calls = model_calls
        self.effort = effort
        self.recursion_limit = _recursion_limit(model_calls)
        self._graph = graph

    def invoke(
        self, messages: Sequence[BaseMessage], *, config: dict | None = None
    ) -> AgentResult:
        """Run the agent once on fresh context; return its messages."""
        output = self._graph.invoke(
            {"messages": list(messages)}, config=self._config(config)
        )
        return self._result(output)

    def stream(
        self,
        messages: Sequence[BaseMessage],
        *,
        config: dict | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> AgentResult:
        """Run the agent once, streaming answer text through ``on_text``."""
        output = None
        for mode, data in self._graph.stream(
            {"messages": list(messages)},
            config=self._config(config),
            stream_mode=["messages", "values"],
        ):
            if mode == "values":
                output = data
            elif on_text is not None:
                _emit_text(data, on_text)
        return self._result(output)

    def _config(self, config: dict | None) -> dict:
        merged = dict(config or {})
        merged["recursion_limit"] = self.recursion_limit
        metadata = dict(merged.get("metadata") or {})
        metadata[META_AGENT] = self.name
        if self.effort:
            metadata[META_EFFORT] = self.effort
        merged["metadata"] = metadata
        # The run graph's checkpointer leaks into this invoke through
        # runnable config context: without a fresh namespace the inner run
        # checkpoints under the node's task namespace and a retried node
        # replays its stale output instead of re-running on fresh context
        # (ADR 0002, #58).
        configurable = dict(merged.get("configurable") or {})
        configurable["checkpoint_ns"] = f"{self.name}-{uuid4().hex}"
        merged["configurable"] = configurable
        return merged

    def _result(self, output: dict | None) -> AgentResult:
        messages = list((output or {}).get("messages") or [])
        limited = _limit_hit(messages, self.model_calls)
        structured = (output or {}).get("structured_response")
        if self.name in _MISSING_OUTPUT and not limited and structured is None:
            raise MissingStructuredOutput(_MISSING_OUTPUT[self.name])
        return AgentResult(
            messages=messages,
            limit_hit=limited,
            structured_response=structured,
        )


def _emit_text(data, on_text: Callable[[str], None]) -> None:
    chunk, metadata = data
    if not isinstance(chunk, AIMessageChunk) or metadata.get(META_HELPER):
        return
    text = _content_text(chunk.content)
    if text:
        on_text(text)
