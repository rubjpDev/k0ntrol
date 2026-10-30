"""``/ask <question>``: CHATTY answers with its read-only tools.

``/ask`` runs no graph: the free doctor checks for CHATTY and the
``bulk_read`` helper gate the call, project memory is refreshed once
(rescan + one index rebuild), the answer streams to the REPL, and every
model and tool call writes a ``chat``-scoped event (never ``run``) —
including the rescan's helper calls. ``@path`` mentions pass relative
paths only — CHATTY reads the files itself; file contents are never
inlined into the prompt.
"""

from collections.abc import Callable
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from k0ntrol import __version__
from k0ntrol.agents import AgentResult, agent_tools, build_agent
from k0ntrol.events import (
    EventWriter,
    K0EventHandler,
    ProgressRenderer,
    events_path,
    new_chat_id,
)
from k0ntrol.frontends.mentions import MentionError, mention_paths
from k0ntrol.helpers import helper_invoker
from k0ntrol.langfuse import build_trace
from k0ntrol.memory.refresh import refresh_memory
from k0ntrol.providers import free_checks, has_errors, report_lines
from k0ntrol.workflow.change import current_branch


def answer_ask(
    question: str,
    *,
    root: Path,
    config: dict,
    write: Callable[[str], None],
    on_text: Callable[[str], None] | None = None,
    init: Callable = init_chat_model,
    check: Callable = free_checks,
    refresh: Callable = refresh_memory,
    tracing: Callable = build_trace,
) -> AgentResult | None:
    """Answer ``question`` with CHATTY.

    ``write`` takes progress lines and errors; ``on_text`` takes the
    streamed answer deltas (defaults to ``write``). Returns the agent
    result, or ``None`` when the doctor checks, a bad ``@mention`` or the
    memory refresh refuse the run.
    """
    prompt = _prepare(question, root, config, write, check)
    if prompt is None:
        return None
    writer, handler, chat_id = _chat_events(root, write)
    trace = tracing(chat_id)  # each /ask is its own trace (#49)
    if not _refresh(root, config, init, handler, write, refresh, trace):
        trace.finish(kind="ask", outcome="failed", metadata=_ask_metadata(root))
        return None
    agent = build_agent(
        "chatty",
        config,
        agent_tools("chatty", root=root, cfg=config, init=init),
        root=root,
        init=init,
    )
    try:
        result = _run_chatty(agent, prompt, handler, on_text or write, trace)
    except Exception as error:
        write(f"CHATTY failed: {error}")
        trace.finish(kind="ask", outcome="failed", metadata=_ask_metadata(root))
        return None
    if result.limit_hit:
        write(
            f"CHATTY hit its {agent.model_calls} model-call limit; "
            "ask again or narrow the question."
        )
    trace.finish(kind="ask", outcome="done", metadata=_ask_metadata(root))
    return result


def _prepare(
    question: str,
    root: Path,
    config: dict,
    write: Callable[[str], None],
    check: Callable,
) -> str | None:
    """Usage check, the free doctor gate, then the prompt; None = refuse."""
    if not question.strip():
        write("usage: /ask <question>")
        return None
    checks = check(
        config, names=("agents.chatty", "helpers.bulk_read"), root=root
    )
    if has_errors(checks):
        for line in report_lines(checks):
            write(line)
        return None
    try:
        return _build_prompt(question, root)
    except MentionError as error:
        write(str(error))
        return None


def _build_prompt(question: str, root: Path) -> str:
    """The question plus the mentioned files as root-relative paths."""
    mentions = mention_paths(question, root)
    if not mentions:
        return question
    relative = [
        path.relative_to(root.resolve()).as_posix() for path in mentions
    ]
    return (
        question
        + "\n\nMentioned files (read them with your tools):\n"
        + "\n".join(relative)
    )


def _chat_events(root: Path, write: Callable[[str], None]):
    """One chat-scoped writer + handler shared by the refresh and CHATTY."""
    path = events_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    chat_id = new_chat_id()
    writer = EventWriter(
        path, chat=chat_id, progress=ProgressRenderer(write)
    )
    return writer, K0EventHandler(writer), chat_id


def _callbacks(handler: K0EventHandler, trace) -> list:
    """The events handler plus the optional Langfuse one (#49)."""
    callbacks = [handler]
    if trace.handler is not None:
        callbacks.append(trace.handler)
    return callbacks


def _ask_metadata(root: Path) -> dict:
    """The ask trace's metadata (#49): branch and k0 version."""
    metadata = {"k0_version": __version__}
    branch = current_branch(root)
    if branch:
        metadata["branch"] = branch
    return metadata


def _refresh(
    root: Path,
    config: dict,
    init: Callable,
    handler: K0EventHandler,
    write: Callable[[str], None],
    refresh: Callable,
    trace,
) -> bool:
    """Rescan + one index rebuild; the rescan's helper calls are
    chat-scoped ``model`` events. Returns False when refresh fails."""
    invoke = helper_invoker(
        config, "bulk_read", init=init, config={"callbacks": _callbacks(handler, trace)}
    )
    try:
        refresh(root, config=config, invoke=invoke)
    except Exception as error:
        write(f"memory refresh failed: {error}")
        return False
    return True


def _run_chatty(agent, prompt, handler, on_text, trace) -> AgentResult:
    return agent.stream(
        [HumanMessage(prompt)],
        config={
            "callbacks": _callbacks(handler, trace),
            "metadata": {"langfuse_tags": ["ask"]},
        },
        on_text=on_text,
    )
