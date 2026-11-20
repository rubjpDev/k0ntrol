"""The ``/implement`` graph (#116 working tree, #117 tracker path).

``/implement <task>`` runs the working-tree path: ``seam`` → seam gate →
``implementer`` → ``validator`` → diff gate — no tracker, branch or pull
request. IMPLEMENTER first proposes the seams in a fresh run — its own
agent instance, so a stray proposal write can never leak slice-guard
state into the build — and lists the design questions the task leaves
open; the human answers ``ok [answers]`` or ``redo <feedback>``. The diff gate lists
the pre-run files the run modified or deleted, apart from the files it
added, all against the snapshot ``start`` took before the run (#149);
its ``n`` restores the working tree to that snapshot.

``/implement #<n>`` runs the tracker path on a ready work item:
``branch_setup`` cuts ``k0/<n>-<slug>``, IMPLEMENTER builds test-first on
the item's latest agent brief (or its body), a red VALIDATOR sends it
back with the raw output within ``max_attempts``, and on green the
harness commits, passes the CI gate when ``.github/`` changes, pushes and
opens the draft change. REVIEWER then runs twice with fresh context — a
Conventions review, then a Contract review (#118) — each posted on the
change as its own labelled review, marker-idempotent on its pass's round.

The review gate (#119) shows both reviews unmerged: ``ok`` posts the gate
comment and marks the change ready, ``redo <feedback>`` reruns IMPLEMENTER
with the reviews and the feedback — its guard starting in refactor
(#110) — through VALIDATOR, the push and both reviews again, and
``plan <feedback>`` comments the feedback on the work item, lifts
``ready-for-agent`` and ends the run.
"""

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from k0ntrol.tools.errors import SliceRecoveryNeeded
from k0ntrol.tools.slice_guard import SliceStart
from k0ntrol.tools.slice_journal import restore_slice
from k0ntrol.tools.tracker import render_work_item
from k0ntrol.tracker import TRUSTED_ASSOCIATIONS, TrackerError
from k0ntrol.workflow.change import (
    decider,
    existing_branch,
    head_sha,
    remote_tip,
    reset_scan_output,
    restore_worktree,
    slugify,
    worktree_snapshot_dir,
)
from k0ntrol.workflow.ci_gate import gated_push, make_ci_gate
from k0ntrol.workflow.context import RunContext
from k0ntrol.workflow.nodes import (
    _invoke_counted,
    _merge,
    _written,
    make_diff_gate,
    make_limit_hit,
    make_validator,
)
from k0ntrol.workflow.routers import (
    route_after_contract_review,
    route_after_conventions_review,
    route_after_implementer,
    route_after_seam,
    route_implement_ci_gate,
    route_implement_start,
    route_implement_validator,
    route_limit_hit,
    route_mark_ready,
    route_recovery_gate,
    route_review_gate,
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
    graph.add_node("implementer", make_implementer_node(ctx))
    graph.add_node("validator", make_validator(ctx, retry_node="implementer"))
    graph.add_node("publish", make_publish(ctx))
    graph.add_node(
        "ci_gate", make_ci_gate(ctx, attempts_key="implementer_attempts")
    )
    graph.add_node("ci_comment", make_ci_comment(ctx))
    graph.add_node("ci_reject_comment", make_ci_reject_comment(ctx))
    graph.add_node("push", make_push(ctx))
    graph.add_node(
        "conventions_review",
        make_axis_review_node(
            ctx,
            "conventions",
            with_work_item(ctx, with_change(ctx, conventions_brief(ctx))),
        ),
    )
    graph.add_node(
        "post_conventions", make_post_axis_review(ctx, "conventions")
    )
    graph.add_node(
        "contract_review",
        make_axis_review_node(
            ctx,
            "contract",
            with_work_item(ctx, with_change(ctx, contract_brief)),
        ),
    )
    graph.add_node("post_contract", make_post_axis_review(ctx, "contract"))
    graph.add_node("review_gate", make_review_gate(ctx))
    graph.add_node("review_comment", make_review_comment(ctx))
    graph.add_node("mark_ready", make_mark_ready(ctx))
    graph.add_node("plan_feedback", make_plan_feedback(ctx))
    graph.add_node("diff_gate", make_diff_gate(ctx, on_reject=_revert(ctx)))
    graph.add_node("recovery_gate", make_recovery_gate(ctx))
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
    graph.add_edge("push", "conventions_review")
    graph.add_conditional_edges(
        "conventions_review", route_after_conventions_review
    )
    graph.add_edge("post_conventions", "contract_review")
    graph.add_conditional_edges("contract_review", route_after_contract_review)
    graph.add_edge("post_contract", "review_gate")
    graph.add_conditional_edges("review_gate", route_review_gate)
    graph.add_conditional_edges("recovery_gate", route_recovery_gate)
    graph.add_edge("review_comment", "mark_ready")
    graph.add_conditional_edges("mark_ready", route_mark_ready)
    graph.add_edge("plan_feedback", END)
    graph.add_edge("diff_gate", END)
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
        result, updates = _invoke_counted(
            ctx,
            "implementer",
            state,
            seam_brief,
            attempts_key="seam_attempts",
            slot="seam",
            read_only=True,
            limit_node="seam",
        )
        if not result.limit_hit:
            seams, questions = split_proposal(result.text)
            updates["seam_proposal"] = seams
            updates["seam_questions"] = questions
        return updates

    return node


def make_seam_gate(ctx: RunContext) -> Callable[[dict], dict]:
    """The seam gate: ``ok [answers]`` builds — the answers to the open
    design questions reach the build brief — ``redo [feedback]`` proposes
    again."""

    def node(state: dict) -> dict:
        decision = interrupt(
            {
                "gate": "seam",
                "question": "approve these seams?",
                "answers": ["ok", "redo"],
                "detail": seam_gate_detail(state),
            }
        )
        word, _, feedback = str(decision).strip().partition(" ")
        if word.lower() == "ok":
            return {
                "seam_approved": True,
                "seam_feedback": None,
                "seam_answers": feedback.strip() or None,
            }
        return {
            "seam_approved": False,
            "seam_feedback": feedback.strip(),
            "seam_answers": None,
        }

    return node


QUESTIONS_HEADING = "Open design questions:"
_QUESTIONS_LINE = re.compile(
    r"^[#*_\s]*open design questions[*_:\s]*", re.IGNORECASE | re.MULTILINE
)
_NO_QUESTIONS = {"", "none", "(none)"}


def split_proposal(text: str) -> tuple[str, str]:
    """IMPLEMENTER's proposal as its seams and its open design questions
    — ``""`` when it lists none (#148)."""
    match = _QUESTIONS_LINE.search(text)
    if match is None:
        return text.strip(), ""
    questions = text[match.end():].strip()
    if questions.lower().rstrip(".") in _NO_QUESTIONS:
        questions = ""
    return text[: match.start()].strip(), questions


def seam_gate_detail(state: dict) -> str:
    """The seams, then the open design questions under their own heading."""
    seams = state.get("seam_proposal") or "(IMPLEMENTER proposed no seams)"
    questions = state.get("seam_questions") or "(none)"
    return f"{seams}\n\n{QUESTIONS_HEADING}\n{questions}"


def _revert(ctx: RunContext) -> Callable[[], None]:
    """The diff gate's ``n``: put the working tree back as it was."""

    def revert() -> None:
        restore_worktree(ctx.root, worktree_snapshot_dir(ctx.root, ctx.run_id))

    return revert


# --- the tracker path (#117) ----------------------------------------------------


def make_implementer_node(ctx: RunContext) -> Callable[[dict], dict]:
    """IMPLEMENTER's build node; a review redo starts the guard in refactor.

    The redo runs on a second IMPLEMENTER instance (``slot="refactor"``):
    its slice guard starts in refactor — code may change before any test
    write, and the first test write attempted returns the guard to the
    green/red cycle (#110). A slice-guard escalation (#174) ends the run
    for the recovery gate; a ``recover`` answer retries on a fresh instance
    (``slot="<base>_recovery<n>"``) so its guard and journal start clean in
    green.
    """

    def node(state: dict) -> dict:
        try:
            result, updates = _invoke_counted(
                ctx,
                "implementer",
                state,
                implementer_brief,
                slot=_implementer_slot(state),
                slice_start=_slice_start(state),
            )
        except SliceRecoveryNeeded as error:
            return _recovery_escalation(error)
        updates["files"] = _merge(state.get("files"), _written(result))
        return updates

    return node


def _implementer_slot(state: dict) -> str | None:
    """The instance the run builds on. Each recovery is a fresh slot, so
    its guard and journal start clean in green (#174)."""
    recoveries = state.get("recoveries", 0)
    if not recoveries:
        return "refactor" if state.get("refactor") else None
    return f"{'refactor' if state.get('refactor') else 'build'}_recovery{recoveries}"


def _slice_start(state: dict) -> SliceStart:
    """refactor only until a recovery; a recovered run always starts green."""
    if state.get("refactor") and not state.get("recoveries"):
        return "refactor"
    return "green"


def _recovery_escalation(error: SliceRecoveryNeeded) -> dict:
    """The state update the escalation leaves for the recovery gate."""
    return {
        "recovery": {
            "slice": error.slice,
            "paths": error.paths,
            "refusal": None,
        },
        "limit": None,
        "retry_node": None,
        "retry_context": None,
    }


# --- the recovery gate (#174) ----------------------------------------------------


def make_recovery_gate(ctx: RunContext) -> Callable[[dict], dict]:
    """The recovery gate: the slice guard's denial bound ran out.

    ``recover`` restores the journaled paths to their pre-slice state and
    retries IMPLEMENTER fresh — a missing, stale or conflicting journal
    refuses safely and asks again; ``stop`` ends the run ``stuck`` with the
    tree untouched.
    """

    def node(state: dict) -> dict:
        recovery = state.get("recovery") or {}
        decision = interrupt(
            {
                "gate": "recovery",
                "question": "the slice guard refused test writes — recover the slice?",
                "answers": ["recover", "stop"],
                "detail": _recovery_detail(recovery),
            }
        )
        word, _, feedback = str(decision).strip().partition(" ")
        if word.lower() != "recover":
            return {
                "outcome": "stuck",
                "reason": "slice_recovery",
                "node": "implementer",
                "recovery_route": None,
            }
        journal_dir = worktree_snapshot_dir(ctx.root, ctx.run_id) / "slice"
        result = restore_slice(
            ctx.root,
            journal_dir,
            expected_slice=str(recovery.get("slice") or ""),
        )
        if not result.applied:
            return {
                "recovery": {**recovery, "refusal": result.detail},
                "recovery_route": "recovery_gate",
            }
        return {
            "recovery": None,
            "recoveries": state.get("recoveries", 0) + 1,
            "implementer_attempts": 0,
            "retry_context": _recovery_context(recovery, feedback.strip()),
            "recovery_route": "implementer",
        }

    return node


def _recovery_detail(recovery: dict) -> str:
    """What the gate shows: the slice's journaled paths, the last refusal."""
    paths = recovery.get("paths") or []
    lines = [
        "The slice guard refused 3 test writes in the active red slice —",
        "the test file itself may be broken.",
        "",
        "recover restores what the slice wrote and reruns IMPLEMENTER:",
        *([f"  {path}" for path in paths] or ["  (journal lists no paths)"]),
    ]
    if recovery.get("refusal"):
        lines += ["", f"recovery refused: {recovery['refusal']}"]
    lines += ["", "stop ends the run; the working tree stays as it is."]
    return "\n".join(lines)


def _recovery_context(recovery: dict, feedback: str) -> str:
    """The retry context for the recovered run: what was rolled back."""
    paths = ", ".join(recovery.get("paths") or []) or "(none)"
    parts = [
        "The slice guard refused test edits until the run paused; the human "
        f"chose to recover the slice. These paths were restored to their "
        f"pre-slice state: {paths}. Write the failing test again, then the "
        "code that makes it pass.",
    ]
    if feedback:
        parts.append(f"Human feedback: {feedback}")
    return "\n\n".join(parts)


def make_push(ctx: RunContext) -> Callable[[dict], dict]:
    """Push through the CI gate's clearance; open the draft change once.

    The run's reviews follow: REVIEWER's Conventions and Contract passes
    post on the change before the run ends ``done`` (#118).
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
        return updates

    return node


def _change_body(ctx: RunContext, state: dict) -> str:
    """The draft change's body: ``Closes #<n>``, IMPLEMENTER's files, the run."""
    lines = [f"Closes #{state['item_ref']}", "", "## Files changed", ""]
    lines += [f"- {path}" for path in state.get("files", [])]
    lines += ["", f"Run: {ctx.run_id}"]
    return "\n".join(lines)


# --- the reviews (#118) ---------------------------------------------------------


def make_axis_review_node(
    ctx: RunContext,
    axis: Literal["conventions", "contract"],
    brief: Callable[[dict], str],
) -> Callable[[dict], dict]:
    """REVIEWER's fresh-context run for one review axis.

    ``axis`` names the graph node (``<axis>_review``), the state field the
    structured Review lands on and the round the post marker suffixes: a
    retry re-reviews, a re-run of the post node never double-posts.
    """

    def node(state: dict) -> dict:
        name = f"{axis}_review"
        refusal = _axis_refusal(ctx, state, name)
        if refusal is not None:
            return refusal
        head = _reviewed_head(ctx, state, axis)
        result, updates = _invoke_counted(
            ctx,
            "reviewer",
            state,
            brief,
            attempts_key=f"{name}_attempts",
            limit_node=name,
        )
        if not result.limit_hit:
            updates.update(_axis_verdict(ctx, state, name, axis, head, result))
        return updates

    return node


def _axis_verdict(
    ctx: RunContext, state: dict, name: str, axis: str, head: str, result
) -> dict:
    """The axis's recorded verdict after a pass: the published head
    re-checked, then the Review dict bound to it (#175)."""
    _confirm_head(ctx, state, head, axis)
    review = brief_dict(result.structured_response)
    if review is not None:
        review["head_sha"] = head
    return {
        name: review,
        f"{name}_round": state.get(f"{name}_round", 0) + 1,
    }


def _axis_refusal(ctx: RunContext, state: dict, name: str) -> dict | None:
    """A refusal recorded in code when the change's author is untrusted —
    REVIEWER is never asked, and its verdict could never pass anyway
    (#175). ``None`` on a trusted change."""
    item = ctx.tracker.read_work_item(state["change_ref"])
    if item.author_association in TRUSTED_ASSOCIATIONS:
        return None
    return {
        name: {
            "status": "refused",
            "reason": (
                f"the change's author {item.author} "
                f"({item.author_association}) is not a trusted author"
            ),
            "summary": "",
            "findings": [],
            "head_sha": _published_head(ctx, state),
        },
        f"{name}_round": state.get(f"{name}_round", 0) + 1,
    }


def _published_head(ctx: RunContext, state: dict) -> str | None:
    """The change's published head — what ``read_change_diff`` shows —
    or ``None`` when the tracker cannot name it (#175)."""
    change = ctx.tracker.read_change(state["branch"])
    return change.head_sha if change is not None else None


def _reviewed_head(ctx: RunContext, state: dict, axis: str) -> str:
    """The published head an axis review binds its verdict to — the run
    stops resumable when the tracker cannot name it (#175)."""
    head = _published_head(ctx, state)
    if head is None:
        raise TrackerError(
            f"the change's head is unknown — the {axis} review cannot "
            "bind its verdict to the published revision"
        )
    return head


def _confirm_head(ctx: RunContext, state: dict, head: str, axis: str) -> None:
    """The published head must not have moved while the pass collected
    its inputs — a moved head is a resumable stop, never a verdict on a
    mixed revision (#175)."""
    if _published_head(ctx, state) != head:
        raise TrackerError(
            f"the change's head moved during the {axis} review — resume "
            "to re-review the published revision"
        )


def make_post_axis_review(
    ctx: RunContext,
    axis: Literal["conventions", "contract"],
) -> Callable[[dict], dict]:
    """Post the ``axis`` review on the change, labelled — its own COMMENT
    review, marker-idempotent on the pass's round, so re-running the node
    never posts it twice."""

    def node(state: dict) -> dict:
        ctx.tracker.post_review(
            state["change_ref"],
            render_review(
                state.get(f"{axis}_review"), f"{axis.capitalize()} review"
            ),
            run=ctx.run_id,
            node=f"post_{axis}{round_suffix(state, f'{axis}_review_round')}",
        )
        return {}

    return node


# --- the review gate (#119, #175) -------------------------------------------------

#: The axes the change must be reviewed on before ``ok`` can mark it ready.
_AXES = ("conventions", "contract")


def review_status(
    review: dict | None, published: str | None, local: str
) -> str:
    """One axis's availability for the change under review.

    ``completed`` only when REVIEWER recorded its ``completed`` verdict
    against a head that is still both the change's published head and
    the local HEAD — a remote move or an unpushed local commit makes it
    ``stale``; ``refused`` when a refusal was recorded; ``missing`` when
    there is no verdict at all — including pre-verdict review dicts from
    before #175, which carry no ``status`` and read as missing.
    """
    if not isinstance(review, dict):
        return "missing"
    status = review.get("status")
    if status == "refused":
        return "refused"
    if status != "completed":
        return "missing"
    judged = review.get("head_sha")
    if published is None or not (judged == published == local):
        return "stale"
    return "completed"


def reviews_completed(ctx: RunContext, state: dict) -> bool:
    """Whether every required axis holds a completed verdict on the
    change's current published head — the only evidence ``ok`` may act
    on (#175)."""
    published = _published_head(ctx, state)
    local = head_sha(ctx.root)
    return all(
        review_status(state.get(f"{axis}_review"), published, local)
        == "completed"
        for axis in _AXES
    )


def make_review_gate(ctx: RunContext) -> Callable[[dict], dict]:
    """The review gate: both posted reviews, unmerged; ``ok`` marks the
    change ready, ``redo`` reruns IMPLEMENTER in refactor, ``plan`` sends
    the work item back to planning.

    ``ok`` exists only while both axes hold a completed verdict on the
    change's published head — a missing, refused or stale axis is named
    in the detail and never offers ``ok`` (#175)."""

    def node(state: dict) -> dict:
        ready = reviews_completed(ctx, state)
        decision = interrupt(
            {
                "gate": "review",
                "question": "approve the change?",
                "answers": ["ok", "redo", "plan"] if ready else ["redo", "plan"],
                "detail": _reviews_detail(ctx, state),
                "diff": code_diff(
                    ctx.tracker.read_change_diff(state["change_ref"])
                ),
            }
        )
        return _gate_route(state, str(decision), ready)

    return node


def _gate_route(state: dict, decision: str, ready: bool) -> dict:
    """The review gate's decision as a state update."""
    word, _, feedback = decision.strip().partition(" ")
    word = word.lower()
    if word == "ok" and not ready:
        # A stored ``ok`` from before the verdicts is no approval:
        # re-show the gate with the unavailable axes named (#175).
        return {"review_route": "review_gate"}
    if word == "ok":
        return {"review_approved": True, "review_route": "review_comment"}
    rejections = [*state.get("review_rejections", []), feedback.strip()]
    if word == "redo":
        return {
            "review_approved": False,
            "review_route": "implementer",
            "implementer_attempts": 0,
            "refactor": True,
            "review_feedback": _reviews_feedback(state, feedback),
            "review_rejections": rejections,
        }
    return {
        "review_approved": False,
        "review_route": "plan_feedback",
        "plan_feedback": feedback.strip(),
        "review_rejections": rejections,
    }


def _reviews_detail(ctx: RunContext, state: dict) -> str:
    """The gate's detail: each axis's availability, then both reviews,
    unmerged — a refused or missing axis says why (#175)."""
    published = _published_head(ctx, state)
    local = head_sha(ctx.root)
    lines = [f"PR {state.get('change_ref')}", "", "Reviews:"]
    for axis in _AXES:
        lines.append(
            f"- {axis}: "
            + _availability(state.get(f"{axis}_review"), published, local)
        )
    return (
        "\n".join(lines)
        + "\n\n"
        + render_review(state.get("conventions_review"), "Conventions review")
        + "\n\n"
        + render_review(state.get("contract_review"), "Contract review")
    )


def _availability(
    review: dict | None, published: str | None, local: str
) -> str:
    status = review_status(review, published, local)
    if status == "refused":
        reason = (review or {}).get("reason") or "(no reason given)"
        return f"refused — {reason}"
    if status == "stale":
        return "stale — the change's head moved since this review"
    return status


def _reviews_feedback(state: dict, extra: str) -> str:
    """Both reviews sent back to IMPLEMENTER, plus the human's words."""
    text = (
        render_review(state.get("conventions_review"), "Conventions review")
        + "\n\n"
        + render_review(state.get("contract_review"), "Contract review")
    )
    if extra.strip():
        text += f"\nHuman feedback: {extra.strip()}"
    return text


def make_review_comment(ctx: RunContext) -> Callable[[dict], dict]:
    """The review gate's pass record on the change, marker-idempotent on
    the round: after a redo the pass comments again."""

    def node(state: dict) -> dict:
        if not reviews_completed(ctx, state):
            # never record a pass the verdicts do not back (#175)
            return {}
        ctx.tracker.comment_change(
            state["change_ref"],
            gate_comment(
                "review", decider(ctx.root), state.get("review_rejections", [])
            ),
            run=ctx.run_id,
            node=f"review_gate{round_suffix(state, 'contract_review_round')}",
        )
        return {}

    return node


def make_plan_feedback(ctx: RunContext) -> Callable[[dict], dict]:
    """``plan``'s effects: the feedback on the work item, the
    ``ready-for-agent`` label off — the run ends with the change unmerged."""

    def node(state: dict) -> dict:
        ctx.tracker.comment_work_item(
            state["item_ref"],
            reject_comment(
                "review", decider(ctx.root), state.get("plan_feedback") or ""
            ),
            run=ctx.run_id,
            node="plan_feedback",
        )
        ctx.tracker.remove_label(state["item_ref"], "ready-for-agent")
        return {"outcome": "rejected"}

    return node


# The Conventions review's fixed list of code smells (Fowler, _Refactoring_
# ch.3) — the same baseline the repo's own /code-review skill carries.
_CODE_SMELLS = """\
- **Mysterious Name**: a function, variable, or type whose name doesn't reveal what it does or holds. → rename it; if no honest name comes, the design's murky.
- **Duplicated Code**: the same logic shape appears in more than one hunk or file in the change. → extract the shared shape, call it from both.
- **Feature Envy**: a method that reaches into another object's data more than its own. → move the method onto the data it envies.
- **Data Clumps**: the same few fields or params keep travelling together (a type wanting to be born). → bundle them into one type, pass that.
- **Primitive Obsession**: a primitive or string standing in for a domain concept that deserves its own type. → give the concept its own small type.
- **Repeated Switches**: the same `switch`/`if`-cascade on the same type recurs across the change. → replace with polymorphism, or one map both sites share.
- **Shotgun Surgery**: one logical change forces scattered edits across many files in the diff. → gather what changes together into one module.
- **Divergent Change**: one file or module is edited for several unrelated reasons. → split so each module changes for one reason.
- **Speculative Generality**: abstraction, parameters, or hooks added for needs the spec doesn't have. → delete it; inline back until a real need shows.
- **Message Chains**: long `a.b().c().d()` navigation the caller shouldn't depend on. → hide the walk behind one method on the first object.
- **Middle Man**: a class or function that mostly just delegates onward. → cut it, call the real target direct.
- **Refused Bequest**: a subclass or implementer that ignores or overrides most of what it inherits. → drop the inheritance, use composition."""


def conventions_brief(ctx: RunContext) -> Callable[[dict], str]:
    """The Conventions review's brief builder, bound to the run's gates:
    judge the diff against the repo's documented rules and the smell
    list — never against what the run's checks already enforce."""

    def brief(state: dict) -> str:
        return (
            f"Review pull request {state['change_ref']} — this is a "
            "Conventions review. Its diff is below, without the .k0-mem/ "
            "memory files.\n\n"
            "Judge the change against the repo's documented coding rules "
            "— find and read them with the read tools (AGENTS.md, "
            "CONTRIBUTING.md, docs/) — and the fixed list of code smells "
            "below. A documented rule always wins over the list, and "
            "every smell is a judgement call labelled with its name. "
            "Skip anything the run's checks already enforce:\n"
            f"{_gate_lines(ctx)}\n\n"
            f"## Code smells\n\n{_CODE_SMELLS}\n\n"
            "Read the files the diff touches with the read tools. Call "
            "the Review tool exactly once with your verdict — completed, "
            "or refused with a reason — plus your summary and findings."
        )

    return brief


def _gate_lines(ctx: RunContext) -> str:
    """What VALIDATOR already enforces, for the Conventions brief."""
    lines = [f"- the test command `{ctx.cfg['test_cmd']}`"]
    lines += [
        f"- {gate['name']} (`{gate['cmd']}`)"
        for gate in ctx.cfg.get("gates") or []
    ]
    return "\n".join(lines)


# --- briefs --------------------------------------------------------------------


def seam_brief(state: dict) -> str:
    """The proposal's fresh context: the task, plus redo feedback."""
    text = (
        f"Task: {state['task']}\n\n"
        "Propose the seams to test this task at — the public boundaries "
        "tests may exercise. Read the repo first, then answer with the "
        "seams. Do not write any files.\n\n"
        f"After the seams, add a line `{QUESTIONS_HEADING}` and list each "
        "design question the task leaves open — a fork you would otherwise "
        "settle alone — or write `none`. The human answers them at the gate."
    )
    if state.get("seam_feedback"):
        text += (
            "\n\nYour last proposal was rejected with this feedback: "
            f"{state['seam_feedback']}"
        )
    return text


def implementer_brief(state: dict) -> str:
    """The build's fresh context: the contract, the reviews on a redo,
    then VALIDATOR's red output."""
    text = _task_brief(state)
    if state.get("review_feedback"):
        text += (
            "\n\nThe human sent the change back after review:\n"
            f"{state['review_feedback']}"
        )
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
        f"{_seam_answers(state)}"
        "Build the task test-first, one red-to-green slice at a time: "
        "write a failing test at a seam, run it red, then write only the "
        "code that makes it pass."
    )


def _seam_answers(state: dict) -> str:
    """The open design questions and the human's ``ok`` answers to them —
    nothing on a bare ``ok``."""
    if not state.get("seam_answers"):
        return ""
    questions = state.get("seam_questions") or "(none)"
    return (
        f"{QUESTIONS_HEADING}\n{questions}\n\n"
        "The human's answers to them — decisions, not suggestions:\n"
        f"{state['seam_answers']}\n\n"
    )

# --- tracker side effects -------------------------------------------------------


def make_branch_setup(ctx: RunContext) -> Callable[[dict], dict]:
    """Cut ``k0/<n>-<slug>`` from the base, rescan, commit ``.k0-mem/``.

    A reused branch carries what an earlier run pushed: its remote tip is
    the first push's CI-gate baseline, so ``.github/`` changes already
    there are not asked about again.
    """

    def node(state: dict) -> dict:
        ref = state["item_ref"]
        branch = (
            state.get("branch")
            or existing_branch(ctx.root, ref)
            or f"k0/{ref}-{slugify(state['item_title'])}"
        )
        reset_scan_output(ctx.root)
        ctx.tracker.create_branch(branch, ctx.base_branch)
        ctx.scan(ctx.root)
        ctx.tracker.commit(f"#{ref}: update memory")
        tip = None if state.get("ci_approved_sha") else remote_tip(ctx.root, branch)
        return {"branch": branch, **({"ci_approved_sha": tip} if tip else {})}

    return node


def make_publish(ctx: RunContext) -> Callable[[dict], dict]:
    """On green: commit, rescan + memory commit; the CI gate and push follow.

    ``commit`` is whole-tree, so the title commit runs first — a memory
    commit first would sweep the change into "update memory" and leave
    ``#<n>: <title>`` empty (carrier contract, #13).
    """

    def node(state: dict) -> dict:
        ref = state["item_ref"]
        message = (
            f"#{ref}: {state['item_title']}"
            if not state.get("change_ref")
            else f"#{ref}: address review"
        )
        ctx.tracker.commit(message)
        ctx.scan(ctx.root)
        ctx.tracker.commit(f"#{ref}: update memory")
        return {}

    return node


def make_ci_comment(ctx: RunContext) -> Callable[[dict], dict]:
    """The CI gate's pass record: on the PR once it exists, else the issue."""

    def node(state: dict) -> dict:
        body = gate_comment("ci", decider(ctx.root), state.get("ci_rejections", []))
        _post_ci_comment(ctx, state, body, f"ci_gate_{state.get('ci_round', 1)}")
        return {}

    return node


def make_ci_reject_comment(ctx: RunContext) -> Callable[[dict], dict]:
    """The CI gate's reject record, with the human's feedback."""

    def node(state: dict) -> dict:
        rejections = state.get("ci_rejections", [])
        body = reject_comment("ci", decider(ctx.root), rejections[-1])
        _post_ci_comment(ctx, state, body, f"ci_gate_reject_{len(rejections)}")
        return {}

    return node


def _post_ci_comment(
    ctx: RunContext, state: dict, body: str, marker_node: str
) -> None:
    """On the PR once it exists, else on the issue (marker-idempotent)."""
    if state.get("change_ref"):
        ctx.tracker.comment_change(
            state["change_ref"], body, run=ctx.run_id, node=marker_node
        )
    else:
        ctx.tracker.comment_work_item(
            state["item_ref"], body, run=ctx.run_id, node=marker_node
        )


def make_mark_ready(ctx: RunContext) -> Callable[[dict], dict]:
    """``gh pr ready``: the draft leaves draft state; the run is done.

    The readiness check runs again here, so a checkpoint that resumes
    straight into this node (paused by older code) can never un-draft a
    change without both completed verdicts (#175).
    """

    def node(state: dict) -> dict:
        if not reviews_completed(ctx, state):
            return {}
        ctx.tracker.mark_change_ready(state["change_ref"])
        return {"outcome": "done"}

    return node


# --- comments, briefs and rendering ----------------------------------------------


def brief_dict(response: object) -> dict | None:
    if response is None:
        return None
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return dict(response)


def with_change(
    ctx: RunContext, brief: Callable[[dict], str]
) -> Callable[[dict], str]:
    """``brief`` plus the change as an agent sees it: the PR rendered like
    a work item, and its code diff — so REVIEWER never judges without
    reading the change (#94)."""

    def build(state: dict) -> str:
        ref = state["change_ref"]
        change = render_work_item(ctx.tracker.read_work_item(ref))
        diff = code_diff(ctx.tracker.read_change_diff(ref))
        return f"{brief(state)}\n\n{change}\n\n## Diff\n\n{diff}"

    return build


def code_diff(diff: str) -> str:
    """``diff`` without its ``.k0-mem/`` files: the harness's memory commits
    are derived data, not part of the change to judge (#84)."""
    files = re.split(r"(?m)^(?=diff --git )", diff)
    return "".join(f for f in files if not f.startswith("diff --git a/.k0-mem/"))


def with_work_item(
    ctx: RunContext, brief: Callable[[dict], str]
) -> Callable[[dict], str]:
    """``brief`` plus the run's work item, rendered as an agent sees it.

    Agents get the request in their first message instead of depending on
    the model to call ``read_work_item`` (#86); ``render_work_item`` keeps
    the trusted-author filter and the untrusted-body refusal.
    """

    def build(state: dict) -> str:
        text = brief(state)
        if not state.get("item_ref"):
            return text
        item = ctx.tracker.read_work_item(state["item_ref"])
        return f"{text}\n\n{render_work_item(item)}"

    return build


def contract_brief(state: dict) -> str:
    """The Contract review's fresh context: the work item is the contract.

    ``/implement #<n>``'s second review pass judges (#118): what the
    contract asked that is missing, what the diff adds unasked, what
    looks done but wrong.
    """
    return (
        f"Work item #{state['item_ref']}: {state['item_title']} — shown "
        "below; the latest ## Agent Brief comment is the contract.\n\n"
        f"Review pull request {state['change_ref']} — this is a Contract "
        "review: judge the change against the work item and quote it for "
        "each finding — work the contract asks that is missing or "
        "partial, work the diff adds that was never asked for, and work "
        "that looks done but wrong. Its diff is below, without the "
        ".k0-mem/ memory files. Read the files it touches with the read "
        "tools. Call the Review tool exactly once with your verdict — "
        "completed, or refused with a reason — plus your summary and "
        "findings."
    )


def render_review(review: dict | None, label: str = "Review") -> str:
    """The ``## <label>`` markdown: PR review body and gate detail.

    A refused verdict renders its reason; a pre-verdict dict still
    renders its text — its ``missing`` status is the gate's call (#175).
    """
    if not review:
        return "(REVIEWER produced no review)"
    if review.get("status") == "refused":
        reason = review.get("reason") or "(no reason given)"
        return f"## {label}\n\nRefused: {reason}\n"
    findings = "\n".join(f"- {item}" for item in review.get("findings", []))
    return (
        f"## {label}\n\n"
        f"{review.get('summary', '')}\n\n"
        f"### Findings\n{findings or '- none'}\n"
    )


def gate_comment(gate: str, who: str, rejections: list[str]) -> str:
    """One comment per passed gate: gate, who, when, earlier rejections."""
    lines = [
        f"Gate passed: {gate}",
        f"Decided by: {who}",
        f"At: {datetime.now(UTC).isoformat(timespec='seconds')}",
    ]
    if rejections:
        lines += ["", "Earlier rejection feedback:"] + [
            f"- {item}" for item in rejections
        ]
    return "\n".join(lines)


def reject_comment(gate: str, who: str, feedback: str) -> str:
    """One comment per rejected gate: gate, who, when and the feedback."""
    return "\n".join(
        [
            f"Gate rejected: {gate}",
            f"Decided by: {who}",
            f"At: {datetime.now(UTC).isoformat(timespec='seconds')}",
            "",
            f"Feedback: {feedback or '(none)'}",
        ]
    )


def round_suffix(state: dict, key: str) -> str:
    """``_<round>`` for a marker-idempotent node's second pass, "" first."""
    round_ = state.get(key, 1)
    return f"_{round_}" if round_ > 1 else ""
