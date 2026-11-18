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
opens the draft change. REVIEWER then runs twice with fresh context — a
Conventions review, then a Contract review (#118) — each posted on the
change as its own labelled review, marker-idempotent on its pass's round.
"""

from collections.abc import Callable
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from k0ntrol.workflow.change import restore_worktree, worktree_snapshot_dir
from k0ntrol.workflow.ci_gate import gated_push, make_ci_gate
from k0ntrol.workflow.context import RunContext
from k0ntrol.workflow.full import (
    brief_dict,
    contract_brief,
    make_branch_setup,
    make_ci_comment,
    make_ci_reject_comment,
    make_publish,
    render_review,
    round_suffix,
    with_change,
    with_work_item,
)
from k0ntrol.workflow.nodes import (
    _invoke_counted,
    make_agent_node,
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
    graph.add_node(
        "post_contract", make_post_axis_review(ctx, "contract", done=True)
    )
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
    graph.add_edge("push", "conventions_review")
    graph.add_conditional_edges(
        "conventions_review", route_after_conventions_review
    )
    graph.add_edge("post_conventions", "contract_review")
    graph.add_conditional_edges("contract_review", route_after_contract_review)
    graph.add_edge("post_contract", END)
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
        result, updates = _invoke_counted(
            ctx,
            "reviewer",
            state,
            brief,
            attempts_key=f"{name}_attempts",
            limit_node=name,
        )
        if not result.limit_hit:
            updates[name] = brief_dict(result.structured_response)
            updates[f"{name}_round"] = state.get(f"{name}_round", 0) + 1
        return updates

    return node


def make_post_axis_review(
    ctx: RunContext,
    axis: Literal["conventions", "contract"],
    *,
    done: bool = False,
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
        return {"outcome": "done"} if done else {}

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
            "the Review tool exactly once with your summary and findings."
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
