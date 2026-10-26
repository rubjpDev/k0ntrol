"""``/ask <question>``: CHATTY answers with its read-only tools.

``/ask`` runs no graph: the free doctor checks for CHATTY gate the call,
the answer streams to the REPL, and every model and tool call writes a
``chat``-scoped event (never ``run``). ``@path`` mentions pass relative
paths only — CHATTY reads the files itself; file contents are never
inlined into the prompt.
"""

from collections.abc import Callable
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from k0ntrol.agents import AgentResult, agent_tools, build_agent
from k0ntrol.events import (
    EventWriter,
    K0EventHandler,
    ProgressRenderer,
    events_path,
    new_chat_id,
)
from k0ntrol.frontends.mentions import MentionError, mention_paths
from k0ntrol.providers import free_checks, has_errors, report_lines


def answer_ask(
    question: str,
    *,
    root: Path,
    config: dict,
    write: Callable[[str], None],
    on_text: Callable[[str], None] | None = None,
    init: Callable = init_chat_model,
    check: Callable = free_checks,
) -> AgentResult | None:
    """Answer ``question`` with CHATTY.

    ``write`` takes progress lines and errors; ``on_text`` takes the
    streamed answer deltas (defaults to ``write``). Returns the agent
    result, or ``None`` when the doctor checks or a bad ``@mention``
    refuse the run.
    """
    prompt = _prepare(question, root, config, write, check)
    if prompt is None:
        return None
    agent = build_agent(
        "chatty", config, agent_tools("chatty", root=root), init=init
    )
    try:
        result = _run_chatty(agent, prompt, root, write, on_text or write)
    except Exception as error:
        write(f"CHATTY failed: {error}")
        return None
    if result.limit_hit:
        write(
            f"CHATTY hit its {agent.model_calls} model-call limit; "
            "ask again or narrow the question."
        )
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
    checks = check(config, names=("agents.chatty",), root=root)
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


def _run_chatty(agent, prompt, root, write, on_text) -> AgentResult:
    path = events_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = EventWriter(
        path, chat=new_chat_id(), progress=ProgressRenderer(write)
    )
    return agent.stream(
        [HumanMessage(prompt)],
        config={"callbacks": [K0EventHandler(writer)]},
        on_text=on_text,
    )
