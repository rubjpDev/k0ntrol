"""The one agent builder.

Every agent — CHATTY for ``/ask`` today, the workflow agents as their nodes
land — is built here from ``create_agent``: the agent's allowlist, k0's
closed middleware set (a run-scoped model-call limit with
``exit_behavior="end"``), an explicit recursion limit derived from the
model-call limit, and a fresh-context wrapper (``K0Agent``).

The allowlist is enforced in code, twice: ``agent_tools`` builds exactly
the tools on ``ALLOWLISTS[name]``, and ``build_agent`` refuses any tool set
that is not that allowlist — off-allowlist tools get
``ToolError(denied="allowlist")``, the canonical refusal. Only ``BaseTool``
instances are accepted: dict tools (provider built-ins) would bypass the
allowlist check in the tool node.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from k0ntrol.events import META_AGENT, META_EFFORT
from k0ntrol.memory.index import custom_rules_text, custom_topology_text
from k0ntrol.providers import build_chat_model
from k0ntrol.tools.bulk_read import make_bulk_read
from k0ntrol.tools.errors import ToolError
from k0ntrol.tools.read import make_grep, make_list_files, make_read_file
from k0ntrol.tools.run_tests import make_run_tests
from k0ntrol.tools.search_memory import make_search_memory
from k0ntrol.tools.tracker import tracker_read_tools
from k0ntrol.tools.write import make_edit_file, make_write_file

_BASE_READ_TOOLS = (
    "read_file",
    "list_files",
    "grep",
    "bulk_read",
    "search_memory",
)

# Reads that need a tracker; ``/fast`` has no work item so they drop off.
_TRACKER_TOOLS = ("read_work_item", "read_change_diff")

# The roster's tool allowlists (issue #12). `bulk_read` and `search_memory`
# are on the base reads (#40); CHATTY's `read_work_item` joins when the
# tracker lands (#43).
ALLOWLISTS: dict[str, tuple[str, ...]] = {
    "chatty": _BASE_READ_TOOLS,
    "spec": _BASE_READ_TOOLS + ("read_work_item",),
    "tester": _BASE_READ_TOOLS
    + ("read_work_item", "write_file", "edit_file", "run_tests"),
    "coder": _BASE_READ_TOOLS
    + ("read_work_item", "write_file", "edit_file", "run_tests"),
    "reviewer": _BASE_READ_TOOLS + ("read_work_item", "read_change_diff"),
}

# TESTER writes test paths only, CODER non-test paths only (spec story 9).
_WRITE_SCOPES = {"tester": "test", "coder": "non_test"}

# A model→tools round costs four super-steps in the compiled agent graph
# (before_model, model, after_model, tools), plus start/end bookkeeping.
# The recursion limit is a backstop: the model-call limit is the real bound
# and hitting a recursion limit means a k0 bug (#20).
_STEPS_PER_CALL = 4
_STEPS_SLACK = 8


class AgentBrief(BaseModel):
    """SPEC's structured output: the contract the rest of the run works from."""

    title: str
    category: str
    summary: str
    current_behavior: str
    desired_behavior: str
    key_interfaces: str
    acceptance_criteria: list[str]
    out_of_scope: str


class Review(BaseModel):
    """REVIEWER's structured output: the review of the change."""

    summary: str
    findings: list[str]


# Agents whose ``create_agent`` gets a structured ``response_format`` (spec
# #33): the schema becomes a tool the model calls; the validated value lands
# on ``AgentResult.structured_response``.
_RESPONSE_FORMATS = {"spec": AgentBrief, "reviewer": Review}


class MissingStructuredOutput(Exception):
    """An agent with a ``response_format`` finished without one and no
    limit hit.

    Raised inside the agent invoke — the same resumable stop path as a
    provider error: the checkpoint stays before the node, the error is
    shown, and the run is never ``stuck`` (spec #33, #58)."""


_MISSING_OUTPUT = {
    "spec": "SPEC returned no agent brief",
    "reviewer": "REVIEWER returned no review",
}


def allowlist_for(name: str, *, tracker: bool) -> tuple[str, ...]:
    """Agent ``name``'s allowlist for this run.

    Without a tracker (``/fast`` — there is no work item), the tracker
    reads drop off the allowlist; the agent can never see the tool.
    """
    names = ALLOWLISTS[name]
    if tracker:
        return names
    return tuple(n for n in names if n not in _TRACKER_TOOLS)


def agent_tools(
    name: str,
    *,
    root: Path,
    cfg: dict | None = None,
    tracker=None,
    init: Callable = init_chat_model,
) -> list[BaseTool]:
    """The tools on agent ``name``'s allowlist, bound to the run context.

    ``cfg`` supplies ``test_cmd``/``test_timeout`` for ``run_tests`` and
    ``helpers.*`` for ``bulk_read``; ``tracker`` feeds the tracker reads;
    ``init`` builds helper models. Each is needed only by tools that use
    it, and only when the tool runs. With ``tracker=None`` the tracker
    reads are left off entirely (``/fast``), matching
    ``allowlist_for(name, tracker=False)``.
    """
    try:
        names = allowlist_for(name, tracker=tracker is not None)
    except KeyError:
        raise ValueError(f"unknown agent: {name!r}") from None
    factories = _tool_factories(
        name, root=root, cfg=cfg, tracker=tracker, init=init
    )
    return [factories[tool_name]() for tool_name in names]


def build_agent(
    name: str,
    cfg: dict,
    tools: Sequence[BaseTool],
    *,
    root: Path,
    init: Callable = init_chat_model,
    allowlist: Sequence[str] | None = None,
) -> "K0Agent":
    """Build agent ``name`` behind the fresh-context wrapper.

    ``tools`` must be exactly the agent's allowlist — ``allowlist``, or
    ``ALLOWLISTS[name]`` when it is ``None`` (``/fast`` passes the
    tracker-less list). The model comes from the ``agents.<name>`` config
    entry, and the prompt from ``prompts/<name>.md`` plus the project's
    `Rules.md` and topology when someone has edited them.
    """
    entry = cfg["agents"][name]
    checked = _check_allowlist(name, tools, allowlist)
    model = build_chat_model(cfg, entry, init=init)
    graph = create_agent(
        model,
        tools=checked,
        system_prompt=_system_prompt(name, root),
        response_format=_RESPONSE_FORMATS.get(name),
        middleware=[
            ModelCallLimitMiddleware(
                run_limit=entry["model_calls"], exit_behavior="end"
            )
        ],
        name=name,
    )
    return K0Agent(
        name,
        graph,
        checked,
        model_calls=entry["model_calls"],
        effort=entry.get("effort"),
    )


def _check_allowlist(
    name: str, tools: Sequence[BaseTool], allowlist: Sequence[str] | None = None
) -> list[BaseTool]:
    for item in tools:
        if not isinstance(item, BaseTool):
            raise TypeError(
                f"tools must be BaseTool instances, got {item!r}"
            )
    expected = set(allowlist if allowlist is not None else ALLOWLISTS[name])
    got = {tool.name for tool in tools}
    if got != expected:
        raise ToolError(
            f"agent {name!r} tools must be its allowlist {sorted(expected)}; "
            f"got {sorted(got)}",
            denied="allowlist",
        )
    return list(tools)


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
    agent: str, *, root: Path, cfg: dict | None, tracker, init: Callable
) -> dict[str, Callable[[], BaseTool]]:
    scope = _WRITE_SCOPES.get(agent)
    return {
        "read_file": lambda: make_read_file(root),
        "list_files": lambda: make_list_files(root),
        "grep": lambda: make_grep(root),
        "bulk_read": lambda: make_bulk_read(root, cfg, init=init),
        "search_memory": lambda: make_search_memory(root),
        "write_file": lambda: make_write_file(root, scope),
        "edit_file": lambda: make_edit_file(root, scope),
        "run_tests": lambda: make_run_tests(root, _need_cfg(cfg)),
        "read_work_item": lambda: _tracker_tool("read_work_item", tracker),
        "read_change_diff": lambda: _tracker_tool("read_change_diff", tracker),
    }


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
        tools: Sequence[BaseTool],
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
    chunk, _metadata = data
    if not isinstance(chunk, AIMessageChunk):
        return
    text = _content_text(chunk.content)
    if text:
        on_text(text)
