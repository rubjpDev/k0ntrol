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

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.tools import BaseTool

from k0ntrol.events import META_AGENT, META_EFFORT
from k0ntrol.providers import build_chat_model
from k0ntrol.tools.errors import ToolError
from k0ntrol.tools.read import make_grep, make_list_files, make_read_file
from k0ntrol.tools.run_tests import make_run_tests
from k0ntrol.tools.tracker import tracker_read_tools
from k0ntrol.tools.write import make_edit_file, make_write_file

_BASE_READ_TOOLS = ("read_file", "list_files", "grep")

# The roster's tool allowlists (issue #12). `bulk_read` and `search_memory`
# join every list in #40 once those tools exist; CHATTY's `read_work_item`
# too.
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


def agent_tools(
    name: str,
    *,
    root: Path,
    cfg: dict | None = None,
    tracker=None,
) -> list[BaseTool]:
    """The tools on agent ``name``'s allowlist, bound to the run context.

    ``cfg`` supplies ``test_cmd``/``test_timeout`` for ``run_tests`` and
    ``tracker`` the tracker reads; both are needed only by agents whose
    allowlist contains those tools.
    """
    try:
        names = ALLOWLISTS[name]
    except KeyError:
        raise ValueError(f"unknown agent: {name!r}") from None
    factories = _tool_factories(name, root=root, cfg=cfg, tracker=tracker)
    return [factories[tool_name]() for tool_name in names]


def build_agent(
    name: str,
    cfg: dict,
    tools: Sequence[BaseTool],
    *,
    init: Callable = init_chat_model,
) -> "K0Agent":
    """Build agent ``name`` behind the fresh-context wrapper.

    ``tools`` must be exactly the agent's allowlist; the model comes from
    the ``agents.<name>`` config entry, and the prompt from
    ``prompts/<name>.md``.
    """
    entry = cfg["agents"][name]
    checked = _check_allowlist(name, tools)
    model = build_chat_model(cfg, entry, init=init)
    graph = create_agent(
        model,
        tools=checked,
        system_prompt=_prompt(name),
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


def _check_allowlist(name: str, tools: Sequence[BaseTool]) -> list[BaseTool]:
    for item in tools:
        if not isinstance(item, BaseTool):
            raise TypeError(
                f"tools must be BaseTool instances, got {item!r}"
            )
    expected = set(ALLOWLISTS[name])
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


def _tool_factories(
    agent: str, *, root: Path, cfg: dict | None, tracker
) -> dict[str, Callable[[], BaseTool]]:
    scope = _WRITE_SCOPES.get(agent)
    return {
        "read_file": lambda: make_read_file(root),
        "list_files": lambda: make_list_files(root),
        "grep": lambda: make_grep(root),
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
        return merged

    def _result(self, output: dict | None) -> AgentResult:
        messages = list((output or {}).get("messages") or [])
        return AgentResult(
            messages=messages,
            limit_hit=_limit_hit(messages, self.model_calls),
        )


def _emit_text(data, on_text: Callable[[str], None]) -> None:
    chunk, _metadata = data
    if not isinstance(chunk, AIMessageChunk):
        return
    text = _content_text(chunk.content)
    if text:
        on_text(text)
