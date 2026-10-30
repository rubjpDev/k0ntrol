"""The ``/full`` workflow graph: request → ready PR (#44 + #45).

SPEC drafts an agent brief on fresh context and the spec gate decides:
``ok`` publishes — the issue for ``/full <text>`` (body = the task text,
the brief as first comment), the refined brief for ``/full #<n>`` — and
``redo [feedback]`` loops back to SPEC with the feedback. Only then does
the harness cut ``k0/<n>-<slug>`` from the base branch, rescan memory and
commit ``.k0-mem/``.

TESTER then writes red tests, the deterministic red check runs
``test_cmd`` on just the test files, and the human test gate shows the
diff; ``redo`` sends TESTER back with the counter reset, ``ok`` commits
``#<n>: tests for <title>`` and records the gate comment.

CODER implements against the tests with /fast's retry rule; on green the
harness commits ``#<n>: <title>`` (``#<n>: address review`` after a
redo), rescans and commits memory, pushes, opens the draft PR once, then
REVIEWER posts a structured COMMENT review. The review gate: ``ok``
posts the gate comment on the PR, marks it ready and ends ``done``;
``redo`` sends the review back to CODER (counter reset); ``spec`` sends
it to SPEC. Every tracker side effect is its own node, marker-idempotent
with a round suffix on repasses (carrier contract, #13).
"""

import ast
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from k0ntrol.tools.tracker import render_work_item
from k0ntrol.workflow.change import decider, slugify, worktree_summary
from k0ntrol.workflow.context import RunContext
from k0ntrol.workflow.nodes import (
    _brief_with_retry,
    _call_limit,
    make_agent_node,
    make_limit_hit,
    make_red_check,
    make_validator,
)
from k0ntrol.workflow.routers import (
    route_after_coder,
    route_after_reviewer,
    route_after_spec,
    route_after_tester,
    route_full_red_check,
    route_full_validator,
    route_limit_hit,
    route_review_gate,
    route_spec_gate,
    route_test_gate,
)
from k0ntrol.workflow.state import RunState


def build_full_graph(
    ctx: RunContext, checkpointer: BaseCheckpointSaver | None,
) -> CompiledStateGraph:
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
    graph.add_node("coder", make_agent_node(ctx, "coder", "code_files", coder_brief))
    graph.add_node("validator", make_validator(ctx))
    graph.add_node("publish", make_publish(ctx))
    graph.add_node("reviewer", make_reviewer_node(ctx))
    graph.add_node("post_review", make_post_review(ctx))
    graph.add_node("review_gate", make_review_gate(ctx))
    graph.add_node("review_comment", make_review_comment(ctx))
    graph.add_node("mark_ready", make_mark_ready(ctx))
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
    graph.add_edge("test_comment", "coder")
    graph.add_conditional_edges("coder", route_after_coder)
    graph.add_conditional_edges("validator", route_full_validator)
    graph.add_edge("publish", "reviewer")
    graph.add_conditional_edges("reviewer", route_after_reviewer)
    graph.add_edge("post_review", "review_gate")
    graph.add_conditional_edges("review_gate", route_review_gate)
    graph.add_edge("review_comment", "mark_ready")
    graph.add_edge("mark_ready", END)
    graph.add_conditional_edges("limit_hit", route_limit_hit)
    return graph.compile(checkpointer=checkpointer)


# --- spec side -----------------------------------------------------------------


def make_spec_node(ctx: RunContext) -> Callable[[dict], dict]:
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


def make_spec_gate(ctx: RunContext) -> Callable[[dict], dict]:
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
            return {
                "spec_approved": True,
                "spec_feedback": None,
                "spec_round": state.get("spec_round", 0) + 1,
            }
        return {
            "spec_approved": False,
            "spec_feedback": feedback.strip(),
            "spec_rejections": [
                *state.get("spec_rejections", []), feedback.strip()
            ],
        }

    return node


def make_create_issue(ctx: RunContext) -> Callable[[dict], dict]:
    """``/full <text>``: the issue exists only after the spec gate approves."""

    def node(state: dict) -> dict:
        item = ctx.tracker.ensure_work_item(
            state["brief"]["title"], state["task"],
            run=ctx.run_id, node="create_issue",
        )
        return {"item_ref": item.ref, "item_title": item.title}

    return node


def make_post_brief(ctx: RunContext) -> Callable[[dict], dict]:
    """The brief as an ``## Agent Brief`` comment; the body is never edited."""

    def node(state: dict) -> dict:
        ctx.tracker.comment_work_item(
            state["item_ref"],
            render_brief(state["brief"]),
            run=ctx.run_id,
            node=f"post_brief{_round_suffix(state, 'spec_round')}",
        )
        return {}

    return node


def make_spec_comment(ctx: RunContext) -> Callable[[dict], dict]:
    """The spec gate's pass record on the issue."""

    def node(state: dict) -> dict:
        ctx.tracker.comment_work_item(
            state["item_ref"],
            gate_comment(
                "spec", decider(ctx.root), state.get("spec_rejections", [])
            ),
            run=ctx.run_id,
            node=f"spec_gate{_round_suffix(state, 'spec_round')}",
        )
        return {}

    return node


def make_branch_setup(ctx: RunContext) -> Callable[[dict], dict]:
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


def make_test_gate(ctx: RunContext) -> Callable[[dict], dict]:
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


def make_commit_tests(ctx: RunContext) -> Callable[[dict], dict]:
    """``#<n>: tests for <title>`` — the whole tree at test-gate approval."""

    def node(state: dict) -> dict:
        ctx.tracker.commit(
            f"#{state['item_ref']}: tests for {state['item_title']}"
        )
        return {}

    return node


def make_test_comment(ctx: RunContext) -> Callable[[dict], dict]:
    """The test gate's pass record; then CODER implements (#45)."""

    def node(state: dict) -> dict:
        ctx.tracker.comment_work_item(
            state["item_ref"],
            gate_comment(
                "test", decider(ctx.root), state.get("test_rejections", [])
            ),
            run=ctx.run_id,
            node=f"test_gate{_round_suffix(state, 'spec_round')}",
        )
        return {}

    return node


# --- code to review (#45) ------------------------------------------------------


def make_publish(ctx: RunContext) -> Callable[[dict], dict]:
    """On green: commit, rescan + memory commit, push, open the draft once.

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
        ctx.tracker.push(state["branch"])
        if state.get("change_ref"):
            return {}
        change = ctx.tracker.open_change(
            state["branch"],
            ctx.base_branch,
            state["item_title"],
            change_body(ctx, state),
        )
        return {"change_ref": change.ref, "change_url": change.url}

    return node


def make_reviewer_node(ctx: RunContext) -> Callable[[dict], dict]:
    """REVIEWER reads the item and the diff, returns a structured Review."""

    def node(state: dict) -> dict:
        attempt = state.get("reviewer_attempts", 0) + 1
        result = ctx.agent("reviewer").invoke(
            [HumanMessage(_brief_with_retry(_review_brief(ctx), state))],
            config=ctx.agent_config(attempt),
        )
        updates: dict[str, Any] = {
            "reviewer_attempts": attempt,
            "limit": None,
            "retry_node": None,
            "retry_context": None,
        }
        if result.limit_hit:
            updates["reviewer_attempts"] = state.get("reviewer_attempts", 0)
            updates["limit"] = _call_limit(ctx, "reviewer", result)
            return updates
        updates["review"] = _brief_dict(result.structured_response)
        updates["review_round"] = state.get("review_round", 0) + 1
        return updates

    return node


def _review_brief(ctx: RunContext) -> Callable[[dict], str]:
    """The reviewer brief builder bound to the run's tracker reads."""

    def brief(state: dict) -> str:
        change = render_work_item(ctx.tracker.read_work_item(state["change_ref"]))
        return f"{reviewer_brief(state)}\n\n{change}"

    return brief


def make_post_review(ctx: RunContext) -> Callable[[dict], dict]:
    """REVIEWER's review lands on the PR as a COMMENT review."""

    def node(state: dict) -> dict:
        ctx.tracker.post_review(
            state["change_ref"],
            render_review(state.get("review")),
            run=ctx.run_id,
            node=f"post_review{_round_suffix(state, 'review_round')}",
        )
        return {}

    return node


def make_review_gate(ctx: RunContext) -> Callable[[dict], dict]:
    """The review gate: ``ok`` finishes, ``redo`` → CODER, ``spec`` → SPEC."""

    def node(state: dict) -> dict:
        decision = interrupt(
            {
                "gate": "review",
                "question": "approve the change?",
                "answers": ["ok", "redo", "spec"],
                "detail": f"PR {state.get('change_ref')}\n\n"
                + render_review(state.get("review")),
            }
        )
        word, _, feedback = str(decision).strip().partition(" ")
        word = word.lower()
        if word == "ok":
            return {"review_approved": True, "review_route": "review_comment"}
        rejections = [*state.get("review_rejections", []), feedback.strip()]
        if word == "redo":
            return {
                "review_approved": False,
                "review_route": "coder",
                "coder_attempts": 0,
                "review_feedback": _review_feedback(state, feedback),
                "review_rejections": rejections,
            }
        return {
            "review_approved": False,
            "review_route": "spec",
            "spec_approved": False,
            "coder_attempts": 0,
            "review_feedback": None,
            "spec_feedback": _review_feedback(state, feedback),
            "review_rejections": rejections,
        }

    return node


def make_review_comment(ctx: RunContext) -> Callable[[dict], dict]:
    """The review gate's pass record on the PR."""

    def node(state: dict) -> dict:
        ctx.tracker.comment_change(
            state["change_ref"],
            gate_comment(
                "review", decider(ctx.root), state.get("review_rejections", [])
            ),
            run=ctx.run_id,
            node="review_gate",
        )
        return {}

    return node


def make_mark_ready(ctx: RunContext) -> Callable[[dict], dict]:
    """``gh pr ready``: the draft leaves draft state; the run is done."""

    def node(state: dict) -> dict:
        ctx.tracker.mark_change_ready(state["change_ref"])
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


def coder_brief(state: dict) -> str:
    """CODER's fresh context: the item, the tests, review or VALIDATOR output."""
    files = ", ".join(state.get("test_files", [])) or "(none)"
    text = (
        f"Work item #{state['item_ref']}: {state['item_title']} — read it "
        "with read_work_item; the latest ## Agent Brief comment is the "
        "contract.\n\n"
        f"The tests defining the contract live in: {files}. "
        "Implement the task so they pass. Write non-test files only."
    )
    if state.get("review_feedback"):
        text += (
            "\n\nThe human sent the change back after review:\n"
            f"{state['review_feedback']}"
        )
    if state.get("coder_attempts"):
        text += (
            "\n\nVALIDATOR is still red. Its raw output:\n"
            f"{state.get('validator_output', '')}\nFix it."
        )
    return text


def reviewer_brief(state: dict) -> str:
    """REVIEWER's fresh context: the work item and the change to judge."""
    return (
        f"Work item #{state['item_ref']}: {state['item_title']} — read it "
        "with read_work_item; the latest ## Agent Brief comment is the "
        "contract.\n\n"
        f"Review pull request {state['change_ref']}: read the diff with "
        "read_change_diff and the files it touches with the read tools. "
        "Call the Review tool exactly once with your summary and findings."
    )


def render_review(review: dict | None) -> str:
    """The ``## Review`` markdown: PR review body and gate detail."""
    if not review:
        return "(REVIEWER produced no review)"
    findings = "\n".join(f"- {item}" for item in review.get("findings", []))
    return (
        "## Review\n\n"
        f"{review.get('summary', '')}\n\n"
        f"### Findings\n{findings or '- none'}\n"
    )


def _review_feedback(state: dict, extra: str) -> str:
    """The review sent back to CODER or SPEC, plus the human's words."""
    text = render_review(state.get("review"))
    if extra.strip():
        text += f"\nHuman feedback: {extra.strip()}"
    return text


def change_body(ctx: RunContext, state: dict) -> str:
    """The draft PR body: Closes #<n>, files, criteria → tests, run id."""
    files = sorted(set(state.get("test_files", [])) | set(state.get("code_files", [])))
    tests = _test_names(ctx.root, state.get("test_files", []))
    lines = [f"Closes #{state['item_ref']}", "", "## Files changed", ""]
    lines += [f"- {path}" for path in files]
    lines += ["", "## Acceptance criteria", ""]
    for criterion in (state.get("brief") or {}).get("acceptance_criteria", []):
        lines.append(f"- {criterion} — {_tests_line(criterion, tests)}")
    lines += ["", f"Run: {ctx.run_id}"]
    return "\n".join(lines)


def _test_names(root: Path, files: list[str]) -> list[str]:
    """The test functions TESTER wrote, as ``path::test_name``."""
    names = []
    for path in files:
        file = root / path
        if file.is_file():
            tree = ast.parse(file.read_text(encoding="utf-8"))
            names += _test_nodes(tree.body, path)
    return names


def _test_nodes(nodes: list[ast.stmt], prefix: str) -> list[str]:
    names = []
    for node in nodes:
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            names += _test_nodes(node.body, f"{prefix}::{node.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                names.append(f"{prefix}::{node.name}")
    return names


def _tests_line(criterion: str, names: list[str]) -> str:
    """A criterion's tests: keyword overlap, all of them as fallback."""
    if not names:
        return "(no tests recorded)"
    words = _significant(criterion)
    matched = [name for name in names if words & _significant(name)]
    return ", ".join(matched or names)


_STOPWORDS = frozenset(
    {"with", "that", "this", "from", "should", "must", "when", "then", "test", "tests"}
)


def _significant(text: str) -> set[str]:
    """The content words of a criterion or a ``path::test_name``."""
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) >= 4 and word not in _STOPWORDS
    }


def _round_suffix(state: dict, key: str) -> str:
    """``_<round>`` for a marker-idempotent node's second pass, "" first."""
    round_ = state.get(key, 1)
    return f"_{round_}" if round_ > 1 else ""
