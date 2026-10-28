"""The ``/full`` workflow graph: request → approved tests (#44's slice).

SPEC drafts an agent brief on fresh context and the spec gate decides:
``ok`` publishes — the issue for ``/full <text>`` (body = the task text,
the brief as first comment), the refined brief for ``/full #<n>`` — and
``redo [feedback]`` loops back to SPEC with the feedback. Only then does
the harness cut ``k0/<n>-<slug>`` from the base branch, rescan memory and
commit ``.k0-mem/``.

TESTER then writes red tests, the deterministic red check runs
``test_cmd`` on just the test files, and the human test gate shows the
diff; ``redo`` sends TESTER back with the counter reset, ``ok`` commits
``#<n>: tests for <title>`` and records the gate comment. Every tracker
side effect is its own node, marker-idempotent (carrier contract, #13).
Everything after — CODER, VALIDATOR, the draft PR, REVIEWER — is #45.
"""

from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from k0ntrol.workflow.change import decider, slugify, worktree_summary
from k0ntrol.workflow.context import RunContext
from k0ntrol.workflow.nodes import (
    _brief_with_retry,
    _call_limit,
    make_agent_node,
    make_limit_hit,
    make_red_check,
)
from k0ntrol.workflow.routers import (
    route_after_spec,
    route_after_tester,
    route_full_red_check,
    route_limit_hit,
    route_spec_gate,
    route_test_gate,
)
from k0ntrol.workflow.state import RunState


def build_full_graph(ctx: RunContext, checkpointer):
    """Compile the ``/full`` graph for one run context."""
    graph = StateGraph(RunState)
    graph.add_node("spec", make_spec_node(ctx))
    graph.add_node("spec_gate", make_spec_gate(ctx))
    graph.add_node("create_issue", make_create_issue(ctx))
    graph.add_node("post_brief", make_post_brief(ctx))
    graph.add_node("spec_comment", make_spec_comment(ctx))
    graph.add_node("branch_setup", make_branch_setup(ctx))
    graph.add_node(
        "tester", make_agent_node(ctx, "tester", "test_files", tester_brief)
    )
    graph.add_node("red_check", make_red_check(ctx))
    graph.add_node("test_gate", make_test_gate(ctx))
    graph.add_node("commit_tests", make_commit_tests(ctx))
    graph.add_node("test_comment", make_test_comment(ctx))
    graph.add_node("limit_hit", make_limit_hit(ctx))
    graph.add_edge(START, "spec")
    graph.add_conditional_edges("spec", route_after_spec)
    graph.add_conditional_edges("spec_gate", route_spec_gate)
    graph.add_edge("create_issue", "post_brief")
    graph.add_edge("post_brief", "spec_comment")
    graph.add_edge("spec_comment", "branch_setup")
    graph.add_edge("branch_setup", "tester")
    graph.add_conditional_edges("tester", route_after_tester)
    graph.add_conditional_edges("red_check", route_full_red_check)
    graph.add_conditional_edges("test_gate", route_test_gate)
    graph.add_edge("commit_tests", "test_comment")
    graph.add_edge("test_comment", END)
    graph.add_conditional_edges("limit_hit", route_limit_hit)
    return graph.compile(checkpointer=checkpointer)


# --- spec side -----------------------------------------------------------------


def make_spec_node(ctx: RunContext):
    """SPEC drafts the brief; the structured AgentBrief lands on ``brief``."""

    def node(state: dict) -> dict:
        attempt = state.get("spec_attempts", 0) + 1
        result = ctx.agent("spec").invoke(
            [HumanMessage(_brief_with_retry(spec_brief, state))],
            config=ctx.agent_config(attempt),
        )
        updates: dict[str, Any] = {
            "spec_attempts": attempt,
            "limit": None,
            "retry_node": None,
            "retry_context": None,
        }
        if result.limit_hit:
            updates["spec_attempts"] = state.get("spec_attempts", 0)
            updates["limit"] = _call_limit(ctx, "spec", result)
            return updates
        updates["brief"] = _brief_dict(result.structured_response)
        return updates

    return node


def _brief_dict(response) -> dict | None:
    if response is None:
        return None
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return dict(response)


def make_spec_gate(ctx: RunContext):
    """The spec gate: ``ok`` publishes, ``redo [feedback]`` loops back."""

    def node(state: dict) -> dict:
        decision = interrupt(
            {
                "gate": "spec",
                "question": "approve this agent brief?",
                "answers": ["ok", "redo"],
                "detail": render_brief(state.get("brief")),
            }
        )
        word, _, feedback = str(decision).strip().partition(" ")
        if word.lower() == "ok":
            return {"spec_approved": True, "spec_feedback": None}
        return {
            "spec_approved": False,
            "spec_feedback": feedback.strip(),
            "spec_rejections": [
                *state.get("spec_rejections", []), feedback.strip()
            ],
        }

    return node


def make_create_issue(ctx: RunContext):
    """``/full <text>``: the issue exists only after the spec gate approves."""

    def node(state: dict) -> dict:
        item = ctx.tracker.ensure_work_item(
            state["brief"]["title"], state["task"],
            run=ctx.run_id, node="create_issue",
        )
        return {"item_ref": item.ref, "item_title": item.title}

    return node


def make_post_brief(ctx: RunContext):
    """The brief as an ``## Agent Brief`` comment; the body is never edited."""

    def node(state: dict) -> dict:
        ctx.tracker.comment_work_item(
            state["item_ref"],
            render_brief(state["brief"]),
            run=ctx.run_id,
            node="post_brief",
        )
        return {}

    return node


def make_spec_comment(ctx: RunContext):
    """The spec gate's pass record on the issue."""

    def node(state: dict) -> dict:
        ctx.tracker.comment_work_item(
            state["item_ref"],
            gate_comment(
                "spec", decider(ctx.root), state.get("spec_rejections", [])
            ),
            run=ctx.run_id,
            node="spec_gate",
        )
        return {}

    return node


def make_branch_setup(ctx: RunContext):
    """Cut ``k0/<n>-<slug>`` from the base, rescan, commit ``.k0-mem/``."""

    def node(state: dict) -> dict:
        ref = state["item_ref"]
        branch = state.get("branch") or f"k0/{ref}-{slugify(state['item_title'])}"
        ctx.tracker.create_branch(branch, ctx.base_branch)
        ctx.scan(ctx.root)
        ctx.tracker.commit(f"#{ref}: update memory")
        return {"branch": branch}

    return node


# --- test side -----------------------------------------------------------------


def make_test_gate(ctx: RunContext):
    """The test gate: show the diff; ``redo`` resets TESTER's counter."""

    def node(state: dict) -> dict:
        decision = interrupt(
            {
                "gate": "test",
                "question": "approve these tests?",
                "answers": ["ok", "redo"],
                "detail": worktree_summary(ctx.root),
            }
        )
        word, _, feedback = str(decision).strip().partition(" ")
        if word.lower() == "ok":
            return {"test_approved": True, "test_feedback": None}
        return {
            "test_approved": False,
            "tester_attempts": 0,
            "tests_red": False,
            "test_feedback": feedback.strip(),
            "test_rejections": [
                *state.get("test_rejections", []), feedback.strip()
            ],
        }

    return node


def make_commit_tests(ctx: RunContext):
    """``#<n>: tests for <title>`` — the whole tree at test-gate approval."""

    def node(state: dict) -> dict:
        ctx.tracker.commit(
            f"#{state['item_ref']}: tests for {state['item_title']}"
        )
        return {}

    return node


def make_test_comment(ctx: RunContext):
    """The test gate's pass record; then #44's slice is done."""

    def node(state: dict) -> dict:
        ctx.tracker.comment_work_item(
            state["item_ref"],
            gate_comment(
                "test", decider(ctx.root), state.get("test_rejections", [])
            ),
            run=ctx.run_id,
            node="test_gate",
        )
        return {"outcome": "done"}

    return node


# --- briefs and rendering --------------------------------------------------------


def spec_brief(state: dict) -> str:
    """SPEC's fresh context: the request, plus rejection feedback on redo."""
    if state.get("item_ref"):
        text = (
            f"Refine work item #{state['item_ref']}: read it with "
            "read_work_item — its body is the human's request, the latest "
            "## Agent Brief comment the contract to revise — and call the "
            "AgentBrief tool with the updated brief."
        )
    else:
        text = (
            f"Task: {state['task']}\n\n"
            "Turn this request into an agent brief and call the AgentBrief "
            "tool with it."
        )
    if state.get("spec_feedback"):
        text += (
            "\n\nYour last draft was rejected with this feedback: "
            f"{state['spec_feedback']}"
        )
    return text


def tester_brief(state: dict) -> str:
    """TESTER's fresh context: the work item, gate feedback, red-check output."""
    text = (
        f"Work item #{state['item_ref']}: {state['item_title']} — read it "
        "with read_work_item; the latest ## Agent Brief comment is the "
        "contract.\n\nWrite tests that prove the contract — and that FAIL "
        "right now, before the change exists. Write test files only."
    )
    if state.get("test_feedback"):
        text += f"\n\nThe human rejected your tests: {state['test_feedback']}"
    if state.get("tester_attempts"):
        text += (
            "\n\nThe red check ran the suite on your tests and it PASSED — "
            "tests that pass without the change prove nothing. "
            f"Output:\n{state.get('red_check_output', '')}\n"
            "Revise the tests so they fail."
        )
    return text


def render_brief(brief: dict | None) -> str:
    """The ``## Agent Brief`` markdown: gate detail and issue comment."""
    if not brief:
        return "(SPEC produced no brief)"
    criteria = "\n".join(
        f"- [ ] {item}" for item in brief.get("acceptance_criteria", [])
    )
    return (
        "## Agent Brief\n\n"
        f"**Title:** {brief.get('title', '')}\n"
        f"**Category:** {brief.get('category', '')}\n\n"
        f"### Summary\n{brief.get('summary', '')}\n\n"
        f"### Current behavior\n{brief.get('current_behavior', '')}\n\n"
        f"### Desired behavior\n{brief.get('desired_behavior', '')}\n\n"
        f"### Key interfaces\n{brief.get('key_interfaces', '')}\n\n"
        f"### Acceptance criteria\n{criteria}\n\n"
        f"### Out of scope\n{brief.get('out_of_scope', '')}\n"
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
