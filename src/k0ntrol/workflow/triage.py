"""The ``/triage`` graph: a short Q&A on a raw work item, then the
outcome gate (#128).

``/triage #<n>`` reads the work item and runs the same Q&A rounds as
``/plan`` — one fresh PLANNER run per round, the human's one-line answer
carried forward — until a round holds no questions. A fresh PLANNER run
then proposes the triage outcome: one category (``bug`` or
``enhancement``), one state (``needs-triage``, ``needs-info``,
``ready-for-agent``, ``ready-for-human`` or ``wontfix``) and the comment
to post. The outcome gate shows all three before anything is written;
``ok`` applies exactly one category and one state label and posts the
comment — the agent brief for ``ready-for-agent``, the triage notes for
``needs-info``, a closing reason for ``wontfix`` — and ``redo`` sends the
feedback back for a new proposal. The REPL command is #129.
"""

from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from k0ntrol.agents import TriageOutcome
from k0ntrol.workflow.context import RunContext
from k0ntrol.workflow.full import _BULLET
from k0ntrol.workflow.nodes import (
    _attempt_limit,
    _invoke_counted,
    _last_output,
    make_limit_hit,
)
from k0ntrol.workflow.plan import (
    _answered_lines,
    _cap,
    make_planner_node,
    make_round_gate,
)
from k0ntrol.workflow.research import _BLANK, _bullets, _prose
from k0ntrol.workflow.routers import (
    _exhausted,
    route_after_outcome,
    route_after_triage_planner,
    route_limit_hit,
    route_outcome_gate,
)
from k0ntrol.workflow.state import RunState

CATEGORIES = ("bug", "enhancement")
STATES = (
    "needs-triage",
    "needs-info",
    "ready-for-agent",
    "ready-for-human",
    "wontfix",
)
_TRIAGE_LABELS = frozenset(CATEGORIES + STATES)


def build_triage_graph(
    ctx: RunContext, checkpointer: BaseCheckpointSaver | None
) -> CompiledStateGraph:
    """Compile the ``/triage`` graph for one run context."""
    graph = StateGraph(RunState)
    graph.add_node("planner", make_planner_node(ctx, triage_brief))
    graph.add_node("round_gate", make_round_gate(ctx))
    graph.add_node("outcome", make_outcome_node(ctx))
    graph.add_node("outcome_gate", make_outcome_gate(ctx))
    graph.add_node("apply", make_apply(ctx))
    graph.add_node("limit_hit", make_limit_hit(ctx))
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_after_triage_planner)
    graph.add_edge("round_gate", "planner")
    graph.add_conditional_edges("outcome", route_after_outcome)
    graph.add_conditional_edges("outcome_gate", route_outcome_gate)
    graph.add_edge("apply", END)
    graph.add_conditional_edges("limit_hit", route_limit_hit)
    return graph.compile(checkpointer=checkpointer)


def make_outcome_node(ctx: RunContext) -> Callable[[dict], dict]:
    """The outcome draft: a fresh PLANNER run given the item and every
    answered round — ``outcome_brief``, never the round's brief.

    A run answering with another structured output — or proposing
    ``ready-for-agent`` without the agent brief it must post — drafts
    again while attempts remain, then the limit pause.
    """

    def node(state: dict) -> dict:
        result, updates = _invoke_counted(
            ctx,
            "planner",
            state,
            outcome_brief,
            attempts_key="outcome_attempts",
            limit_node="outcome",
        )
        if result.limit_hit:
            return updates
        draft = _proposal(result.structured_response)
        if draft is not None:
            updates["triage_outcome"] = draft.model_dump()
            return updates
        merged = {**state, **updates}
        if _exhausted(merged, "outcome_attempts"):
            updates["limit"] = _attempt_limit(
                merged, "outcome", "outcome", _last_output(result)
            )
        return updates

    return node


def _proposal(response) -> TriageOutcome | None:
    """The draft as a proposal: a TriageOutcome, and a ``ready-for-agent``
    one carries the agent brief the apply node must post."""
    if not isinstance(response, TriageOutcome):
        return None
    if response.state == "ready-for-agent" and response.brief is None:
        return None
    return response


def make_outcome_gate(ctx: RunContext) -> Callable[[dict], dict]:
    """The outcome gate: the proposed category, state and comment, shown
    before anything is written. ``ok`` applies the outcome; ``redo``
    rejects it — the next draft gets the feedback."""

    def node(state: dict) -> dict:
        decision = interrupt(
            {
                "gate": "outcome",
                "question": "apply this triage outcome?",
                "answers": ["ok", "redo"],
                "detail": render_outcome(state.get("triage_outcome")),
            }
        )
        word, _, feedback = str(decision).strip().partition(" ")
        if word.lower() == "ok":
            return {"triage_approved": True, "triage_feedback": None}
        return {
            "triage_outcome": None,
            "triage_approved": False,
            "outcome_attempts": 0,
            "triage_feedback": feedback.strip(),
            "triage_rejections": [
                *state.get("triage_rejections", []), feedback.strip()
            ],
        }

    return node


def make_apply(ctx: RunContext) -> Callable[[dict], dict]:
    """Apply the approved outcome: exactly one category and one state
    label — any earlier triage labels drop off — then the comment,
    marker-idempotent like every tracker write. ``wontfix`` closes the
    item on its reason."""

    def node(state: dict) -> dict:
        ref = state["item_ref"]
        outcome = state["triage_outcome"]
        chosen = {outcome["category"], outcome["state"]}
        for label in sorted(chosen):
            ctx.tracker.ensure_label(label)
        stale = (set(ctx.tracker.read_labels(ref)) & _TRIAGE_LABELS) - chosen
        for label in sorted(stale):
            ctx.tracker.remove_label(ref, label)
        ctx.tracker.add_label(ref, outcome["category"])
        ctx.tracker.add_label(ref, outcome["state"])
        _post(ctx, ref, outcome)
        return {"outcome": "done"}

    return node


def _post(ctx: RunContext, ref: str, outcome: dict) -> None:
    """The outcome's comment: a ``wontfix`` closes on its reason; any
    other state posts the body when there is one to post."""
    body = outcome_comment(outcome)
    if outcome["state"] == "wontfix":
        ctx.tracker.close_work_item(
            ref, body or "wontfix", run=ctx.run_id, node="apply"
        )
    elif body:
        ctx.tracker.comment_work_item(ref, body, run=ctx.run_id, node="apply")


def render_outcome(outcome: dict | None) -> str:
    """What the gate shows: the proposed category, state and the comment
    as it will be posted."""
    if not outcome:
        return "(PLANNER proposed no outcome)"
    return "\n\n".join(
        [
            f"**Category:** {outcome.get('category', '')}",
            f"**State:** {outcome.get('state', '')}",
            outcome_comment(outcome) or _BLANK,
        ]
    )


def outcome_comment(outcome: dict) -> str:
    """The comment the apply node posts: the rendered agent brief when
    the proposal carries one, then the notes or reason text."""
    parts = []
    if outcome.get("brief"):
        parts.append(render_agent_brief(outcome["brief"]))
    if outcome.get("comment"):
        parts.append(outcome["comment"])
    return "\n\n".join(parts)


def render_agent_brief(brief: dict) -> str:
    """The ``## Agent Brief`` comment in the work-item format: what to
    build, seams, acceptance criteria, blocked by and parent (#102)."""
    criteria = "\n".join(
        f"- [ ] {_BULLET.sub('', item)}"
        for item in brief.get("acceptance_criteria", [])
    )
    return "\n\n".join(
        [
            "## Agent Brief",
            _prose("What to build", brief.get("what_to_build", "")),
            _bullets("Seams", brief.get("seams", [])),
            f"## Acceptance criteria\n\n{criteria or _BLANK}",
            _bullets("Blocked by", brief.get("blocked_by", [])),
            _prose("Parent", brief.get("parent", "")),
        ]
    )


def triage_brief(state: dict) -> str:
    """PLANNER's fresh context for one triage round: the work item, the
    round's ask, then every earlier round's questions and the line the
    human answered."""
    lines = [
        _item(state),
        "",
        "Triaging this work item — ask the next round of questions, at "
        f"most {_cap(state)}, each numbered with your recommended answer "
        "— or an empty QuestionRound when nothing is left to decide.",
    ]
    return "\n".join([*lines, *_rounds(state)])


def outcome_brief(state: dict) -> str:
    """PLANNER's fresh context for the outcome draft: the work item and
    every answered round, and on a gate ``redo`` the human's feedback."""
    lines = [
        _item(state),
        "",
        "The Q&A is over — propose the triage outcome: answer with "
        "TriageOutcome — one category (bug or enhancement), one state "
        "(needs-triage, needs-info, ready-for-agent, ready-for-human or "
        "wontfix), the comment to post (triage notes for needs-info, the "
        "reason for wontfix), and for a ready state the brief — what to "
        "build, seams, acceptance criteria, blocked by and parent.",
    ]
    lines.extend(_rounds(state))
    if state.get("triage_feedback"):
        lines.append(
            "\nYour last proposal was rejected with this feedback: "
            f"{state['triage_feedback']}"
        )
    return "\n".join(lines)


def _item(state: dict) -> str:
    return (
        f"Work item #{state['item_ref']}: {state.get('item_title', '')}"
        f"\n\n{state['task']}"
    )


def _rounds(state: dict) -> list[str]:
    lines = []
    for index, round_ in enumerate(state.get("rounds", []), 1):
        lines.append(f"\nRound {index}:")
        lines.extend(_answered_lines(round_))
    return lines
