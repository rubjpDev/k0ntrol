"""The ``/implement <task>`` working-tree graph (#116).

``seam`` → seam gate → ``implementer`` → ``validator`` → diff gate, on
the working tree: no tracker, branch or pull request. IMPLEMENTER first
proposes the seams in a fresh run — its own agent instance, so a stray
proposal write can never leak slice-guard state into the build — and the
human answers ``ok`` or ``redo <feedback>``. IMPLEMENTER then builds
test-first under the slice guard; a red VALIDATOR sends it back with the
raw output, within ``max_attempts``. The diff gate's ``n`` restores the
working tree to the snapshot ``start`` took before the run.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from k0ntrol.workflow.change import restore_worktree, worktree_snapshot_dir
from k0ntrol.workflow.context import RunContext
from k0ntrol.workflow.nodes import (
    _brief_with_retry,
    _call_limit,
    make_agent_node,
    make_diff_gate,
    make_limit_hit,
    make_validator,
)
from k0ntrol.workflow.routers import (
    route_after_implementer,
    route_after_seam,
    route_implement_validator,
    route_limit_hit,
    route_seam_gate,
)
from k0ntrol.workflow.state import RunState


def build_implement_graph(
    ctx: RunContext, checkpointer: BaseCheckpointSaver | None
) -> CompiledStateGraph:
    """Compile the ``/implement <task>`` graph for one run context."""
    graph = StateGraph(RunState)
    graph.add_node("seam", make_seam_node(ctx))
    graph.add_node("seam_gate", make_seam_gate(ctx))
    graph.add_node(
        "implementer",
        make_agent_node(ctx, "implementer", "files", implementer_brief),
    )
    graph.add_node("validator", make_validator(ctx, retry_node="implementer"))
    graph.add_node("diff_gate", make_diff_gate(ctx, on_reject=_revert(ctx)))
    graph.add_node("limit_hit", make_limit_hit(ctx))
    graph.add_edge(START, "seam")
    graph.add_conditional_edges("seam", route_after_seam)
    graph.add_conditional_edges("seam_gate", route_seam_gate)
    graph.add_conditional_edges("implementer", route_after_implementer)
    graph.add_conditional_edges("validator", route_implement_validator)
    graph.add_conditional_edges("limit_hit", route_limit_hit)
    graph.add_edge("diff_gate", END)
    return graph.compile(checkpointer=checkpointer)


# --- the seam gate -------------------------------------------------------------


def make_seam_node(ctx: RunContext) -> Callable[[dict], dict]:
    """IMPLEMENTER proposes the seams; its answer is the gate's detail.

    The proposal runs on a second IMPLEMENTER instance (``slot="seam"``,
    ``read_only``): it can read the repo but has no write or test tools —
    a proposal that disobeys cannot touch the tree before the gate, and
    its guard state can never leak into the build phase.
    """

    def node(state: dict) -> dict:
        attempt = state.get("seam_attempts", 0) + 1
        result = ctx.agent(
            "implementer", slot="seam", read_only=True
        ).invoke(
            [HumanMessage(_brief_with_retry(seam_brief, state))],
            config=ctx.agent_config(attempt),
        )
        limited = result.limit_hit
        updates: dict[str, Any] = {
            "seam_attempts": state.get("seam_attempts", 0) + (0 if limited else 1),
            "limit": None,
            "retry_node": None,
            "retry_context": None,
        }
        if limited:
            updates["limit"] = _call_limit(ctx, "implementer", result, node="seam")
        else:
            updates["seam_proposal"] = result.text
        return updates

    return node


def make_seam_gate(ctx: RunContext) -> Callable[[dict], dict]:
    """The seam gate: ``ok`` builds, ``redo [feedback]`` proposes again."""

    def node(state: dict) -> dict:
        decision = interrupt(
            {
                "gate": "seam",
                "question": "approve these seams?",
                "answers": ["ok", "redo"],
                "detail": state.get("seam_proposal")
                or "(IMPLEMENTER proposed no seams)",
            }
        )
        word, _, feedback = str(decision).strip().partition(" ")
        if word.lower() == "ok":
            return {"seam_approved": True, "seam_feedback": None}
        return {
            "seam_approved": False,
            "seam_feedback": feedback.strip(),
        }

    return node


def _revert(ctx: RunContext) -> Callable[[], None]:
    """The diff gate's ``n``: put the working tree back as it was."""

    def revert() -> None:
        restore_worktree(ctx.root, worktree_snapshot_dir(ctx.root, ctx.run_id))

    return revert


# --- briefs --------------------------------------------------------------------


def seam_brief(state: dict) -> str:
    """The proposal's fresh context: the task, plus redo feedback."""
    text = (
        f"Task: {state['task']}\n\n"
        "Propose the seams to test this task at — the public boundaries "
        "tests may exercise. Read the repo first, then answer with the "
        "seams. Do not write any files."
    )
    if state.get("seam_feedback"):
        text += (
            "\n\nYour last proposal was rejected with this feedback: "
            f"{state['seam_feedback']}"
        )
    return text


def implementer_brief(state: dict) -> str:
    """The build's fresh context: task, approved seams, VALIDATOR output."""
    seams = state.get("seam_proposal") or "(none approved)"
    text = (
        f"Task: {state['task']}\n\n"
        f"Approved seams:\n{seams}\n\n"
        "Build the task test-first, one red-to-green slice at a time: "
        "write a failing test at a seam, run it red, then write only the "
        "code that makes it pass."
    )
    if state.get("implementer_attempts"):
        text += (
            "\n\nVALIDATOR is still red. Its raw output:\n"
            f"{state.get('validator_output', '')}\nFix it."
        )
    return text
