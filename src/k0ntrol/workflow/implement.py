"""The ``/implement`` graph (#116 working tree, #117 tracker path).

``/implement <task>`` runs the working-tree path: ``seam`` → seam gate →
``implementer`` → ``validator`` → diff gate — no tracker, branch or pull
request. IMPLEMENTER first proposes the seams in a fresh run — its own
agent instance, so a stray proposal write can never leak slice-guard
state into the build — and the human answers ``ok`` or
``redo <feedback>``. The diff gate's ``n`` restores the working tree to
the snapshot ``start`` took before the run.

``/implement #<n>`` runs the tracker path on a ready work item:
``branch_setup`` cuts ``k0/<n>-<slug>``, IMPLEMENTER builds test-first on
the item's latest agent brief (or its body), a red VALIDATOR sends it
back with the raw output within ``max_attempts``, and on green the
harness commits, passes the CI gate when ``.github/`` changes, pushes and
opens the draft change. Reviews are the next ticket.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from k0ntrol.workflow.change import restore_worktree, worktree_snapshot_dir
from k0ntrol.workflow.ci_gate import gated_push, make_ci_gate
from k0ntrol.workflow.context import RunContext
from k0ntrol.workflow.full import (
    make_branch_setup,
    make_ci_comment,
    make_ci_reject_comment,
    make_publish,
)
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
    route_implement_ci_gate,
    route_implement_start,
    route_implement_validator,
    route_limit_hit,
    route_seam_gate,
)
from k0ntrol.workflow.state import RunState


def build_implement_graph(
    ctx: RunContext, checkpointer: BaseCheckpointSaver | None
) -> CompiledStateGraph:
    """Compile the ``/implement`` graph for one run context."""
    graph = StateGraph(RunState)
    graph.add_node("seam", make_seam_node(ctx))
    graph.add_node("seam_gate", make_seam_gate(ctx))
    graph.add_node("branch_setup", make_branch_setup(ctx))
    graph.add_node(
        "implementer",
        make_agent_node(ctx, "implementer", "files", implementer_brief),
    )
    graph.add_node("validator", make_validator(ctx, retry_node="implementer"))
    graph.add_node("publish", make_publish(ctx))
    graph.add_node(
        "ci_gate", make_ci_gate(ctx, attempts_key="implementer_attempts")
    )
    graph.add_node("ci_comment", make_ci_comment(ctx))
    graph.add_node("ci_reject_comment", make_ci_reject_comment(ctx))
    graph.add_node("push", make_push(ctx))
    graph.add_node("diff_gate", make_diff_gate(ctx, on_reject=_revert(ctx)))
    graph.add_node("limit_hit", make_limit_hit(ctx))
    graph.add_conditional_edges(START, route_implement_start)
    graph.add_conditional_edges("seam", route_after_seam)
    graph.add_conditional_edges("seam_gate", route_seam_gate)
    graph.add_edge("branch_setup", "implementer")
    graph.add_conditional_edges("implementer", route_after_implementer)
    graph.add_conditional_edges("validator", route_implement_validator)
    graph.add_edge("publish", "ci_gate")
    graph.add_conditional_edges("ci_gate", route_implement_ci_gate)
    graph.add_edge("ci_comment", "push")
    graph.add_edge("ci_reject_comment", "implementer")
    graph.add_edge("diff_gate", END)
    graph.add_edge("push", END)
    graph.add_conditional_edges("limit_hit", route_limit_hit)
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


# --- the tracker path (#117) ----------------------------------------------------


def make_push(ctx: RunContext) -> Callable[[dict], dict]:
    """Push through the CI gate's clearance; open the draft change once.

    The tracker path's last node: the run ends ``done`` with the branch
    pushed and the draft change open, linked to the work item.
    """

    def node(state: dict) -> dict:
        updates = gated_push(ctx, state)
        if not state.get("change_ref"):
            change = ctx.tracker.open_change(
                state["branch"],
                ctx.base_branch,
                state["item_title"],
                _change_body(ctx, state),
            )
            updates["change_ref"] = change.ref
            updates["change_url"] = change.url
        return {**updates, "outcome": "done"}

    return node


def _change_body(ctx: RunContext, state: dict) -> str:
    """The draft change's body: ``Closes #<n>``, IMPLEMENTER's files, the run."""
    lines = [f"Closes #{state['item_ref']}", "", "## Files changed", ""]
    lines += [f"- {path}" for path in state.get("files", [])]
    lines += ["", f"Run: {ctx.run_id}"]
    return "\n".join(lines)


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
    """The build's fresh context: the contract, then VALIDATOR's red output."""
    text = _task_brief(state)
    if state.get("ci_feedback"):
        text += (
            "\n\nThe human rejected your .github/ changes at the CI gate:\n"
            f"{state['ci_feedback']}"
        )
    if state.get("implementer_attempts"):
        text += (
            "\n\nVALIDATOR is still red. Its raw output:\n"
            f"{state.get('validator_output', '')}\nFix it."
        )
    return text


def _task_brief(state: dict) -> str:
    """The work to build: the item's contract, or the task and its seams."""
    if state.get("item_ref"):
        return (
            f"Work item #{state['item_ref']}: {state['item_title']} — its "
            f"contract:\n\n{state['task']}\n\n"
            "Build it test-first, one red-to-green slice at a time: write a "
            "failing test at a seam, run it red, then write only the code "
            "that makes it pass."
        )
    seams = state.get("seam_proposal") or "(none approved)"
    return (
        f"Task: {state['task']}\n\n"
        f"Approved seams:\n{seams}\n\n"
        "Build the task test-first, one red-to-green slice at a time: "
        "write a failing test at a seam, run it red, then write only the "
        "code that makes it pass."
    )
