"""The stuck path of ``/full`` (#46): side effects after ``stop``.

``stop`` at ``limit_hit``, or a failed event-log write, ends the run
``stuck`` and leaves the work where a human can find it: a
``#<n>: wip (stuck)`` commit if the tree is dirty, the push (through the
CI gate, like every ``/full`` push), the ``k0:stuck`` label and one
comment on the issue — and on the PR if one exists. It never opens a PR:
a PR means green. Every effect is its own marker-idempotent node, so a
failed write stops the run resumable and the retry never duplicates.
"""

from collections.abc import Callable

from k0ntrol.workflow.change import head_sha
from k0ntrol.workflow.ci_gate import gated_push
from k0ntrol.workflow.context import RunContext
from k0ntrol.workflow.nodes import _limit_desc

STUCK_LABEL = "k0:stuck"
STUCK_NODE = "stuck"


def stuck_update(node: str | None) -> dict:
    """The state of a run stuck by its event log, as if ``limit_hit`` stopped."""
    return {
        "outcome": "stuck",
        "reason": "event_log",
        "node": node,
        "limit": None,
        "retry_node": None,
    }


def make_stuck_commit(ctx: RunContext) -> Callable[[dict], dict]:
    """``#<n>: wip (stuck)`` when the tree is dirty; the CI gate follows.

    Only runs once the branch exists; a run stuck earlier (SPEC) has
    nothing to commit or push and goes straight to the label.
    """

    def node(state: dict) -> dict:
        ctx.tracker.commit(f"#{state['item_ref']}: wip (stuck)")
        return {}

    return node


def make_stuck_push(ctx: RunContext) -> Callable[[dict], dict]:
    """Push the branch through the CI gate's clearance; no PR is opened."""

    def node(state: dict) -> dict:
        return gated_push(ctx, state)

    return node


def make_stuck_label(ctx: RunContext) -> Callable[[dict], dict]:
    """``k0:stuck`` on the issue, created if the tracker lacks it."""

    def node(state: dict) -> dict:
        if state.get("item_ref"):
            ctx.tracker.ensure_label(STUCK_LABEL)
            ctx.tracker.add_label(state["item_ref"], STUCK_LABEL)
        return {}

    return node


def make_stuck_comment(ctx: RunContext) -> Callable[[dict], dict]:
    """One stuck comment on the issue."""

    def node(state: dict) -> dict:
        if state.get("item_ref"):
            ctx.tracker.comment_work_item(
                state["item_ref"],
                stuck_text(ctx, state),
                run=ctx.run_id,
                node=STUCK_NODE,
            )
        return {}

    return node


def make_stuck_pr_comment(ctx: RunContext) -> Callable[[dict], dict]:
    """The same comment on the PR, only if one is open for the branch."""

    def node(state: dict) -> dict:
        change = _open_change(ctx, state)
        if change is not None:
            ctx.tracker.comment_change(
                change.ref, stuck_text(ctx, state), run=ctx.run_id, node=STUCK_NODE
            )
        return {}

    return node


def make_unstick(ctx: RunContext) -> Callable[[dict], dict]:
    """``/full #<n>`` removes ``k0:stuck`` once SPEC is approved."""

    def node(state: dict) -> dict:
        if state.get("refining"):
            ctx.tracker.remove_label(state["item_ref"], STUCK_LABEL)
        return {}

    return node


def _open_change(ctx: RunContext, state: dict):
    if not state.get("branch"):
        return None
    change = ctx.tracker.read_change(state["branch"])
    return change if change is not None and change.state == "open" else None


def stuck_text(ctx: RunContext, state: dict) -> str:
    """The node, the limit, retries granted, the output tail and the branch."""
    limit = state.get("limit") or {}
    lines = [
        f"k0 run {ctx.run_id} is stuck.",
        "",
        f"- node: {state.get('node') or limit.get('node') or '?'}",
        f"- {_reason_line(state, limit)}",
        f"- retries granted: {state.get('retries_granted', 0)}",
    ]
    lines += _branch_lines(ctx, state)
    output = str(limit.get("output") or "").rstrip()
    if output:
        lines += ["", "Last output:", "", "```", output, "```"]
    return "\n".join(lines)


def _reason_line(state: dict, limit: dict) -> str:
    if state.get("reason") == "event_log":
        return "reason: the event log could not be written"
    return f"limit: {_limit_desc(limit)}"


def _branch_lines(ctx: RunContext, state: dict) -> list[str]:
    branch = state.get("branch")
    if not branch:
        return []
    lines = [f"- branch: {branch}"]
    if state.get("ci_approved_sha") != head_sha(ctx.root):
        lines.append("- HEAD was not pushed (the .github/ changes were not approved)")
    return lines
