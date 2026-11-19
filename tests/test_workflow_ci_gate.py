"""The CI gate's incremental diff and push-side helpers.

The gate flow itself — holding the push, ``ok`` pushing and opening the
draft, ``redo`` feeding IMPLEMENTER and gating again — is covered by
test_workflow_implement. This module keeps the incremental-diff cases
(a later push only shows what changed since the last approved head) and
the error paths of the helpers it relies on.
"""

import subprocess
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from fakes import FakeChatModel, FakeTracker
from test_workflow_implement import (
    CALC_SRC,
    events,
    git_repo,
    git_runner_for,
    item_cfg,
    item_repo,
    models_init,
    passing_reviewer,
    ready_item,
    run_item,
    workflow_builder,
    write_call,
)
from k0ntrol.workflow import Pending, resume

WORKFLOW = ".github/workflows/ci.yml"
WORKFLOW_SRC = "name: ci\non: push\njobs: {}\n"


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def to_ci_gate(tmp_path):
    """``/implement #<n>`` parked at the ci gate: IMPLEMENTER touched ``.github/``."""
    item_repo(tmp_path)
    tracker = FakeTracker(git_runner=git_runner_for(tmp_path))
    ready_item(tracker)
    result = run_item(tmp_path, item_cfg(), tracker, workflow_builder())
    assert isinstance(result, Pending) and result.gate == "ci"
    return tracker, result


def pushes(tracker):
    return [c for c in tracker.calls if c[0] == "push"]


def answer(tmp_path, tracker, run, decision, *models):
    return resume(
        run, decision, root=tmp_path, cfg=item_cfg(),
        init=models_init(*models), tracker=tracker,
    )


def test_approving_pushes_opens_the_draft_and_records_the_decision(tmp_path):
    tracker, gate = to_ci_gate(tmp_path)

    result = answer(tmp_path, tracker, gate.run, "ok", passing_reviewer())

    assert isinstance(result, Pending) and result.gate == "review"
    assert pushes(tracker) == [("push", "k0/1-calc-answers-42")]
    assert tracker.read_change("k0/1-calc-answers-42").is_draft
    # First push: the PR does not exist yet, so the record is on the issue.
    comments = [c.body for c in tracker.read_work_item("1").comments]
    ci = [c for c in comments if "Gate passed: ci" in c]
    assert len(ci) == 1 and "Decided by: t" in ci[0]
    gates = [e for e in events(tmp_path) if e["kind"] == "gate" and e["gate"] == "ci"]
    assert [e["result"] for e in gates] == ["waiting", "approve"]


def test_rejecting_sends_the_feedback_back_and_gates_again(tmp_path):
    tracker, gate = to_ci_gate(tmp_path)
    second = FakeChatModel(
        messages=[
            write_call(WORKFLOW, "name: ci\non: pull_request\njobs: {}\n", "c2"),
            AIMessage(content="narrowed"),
        ]
    )

    again = answer(
        tmp_path, tracker, gate.run, "redo only run on pull_request", second
    )

    assert "only run on pull_request" in second.invocations[0][-1].content
    assert isinstance(again, Pending) and again.gate == "ci"
    assert "pull_request" in again.detail
    assert pushes(tracker) == []
    rejected = [
        c.body for c in tracker.read_work_item("1").comments
        if "Gate rejected: ci" in c.body
    ]
    assert len(rejected) == 1 and "only run on pull_request" in rejected[0]

    done = answer(tmp_path, tracker, again.run, "ok", passing_reviewer())
    assert isinstance(done, Pending) and done.gate == "review"
    assert len(pushes(tracker)) == 1
    comment = [
        c.body for c in tracker.read_work_item("1").comments
        if "Gate passed: ci" in c.body
    ][0]
    assert "- only run on pull_request" in comment
    results = [
        e["result"] for e in events(tmp_path)
        if e["kind"] == "gate" and e["gate"] == "ci"
    ]
    assert results == ["waiting", "reject", "waiting", "approve"]


def test_a_later_push_that_changes_github_gates_again(tmp_path):
    tracker, gate = to_ci_gate(tmp_path)
    review = answer(tmp_path, tracker, gate.run, "ok", passing_reviewer())
    second = FakeChatModel(
        messages=[
            write_call(
                WORKFLOW, "name: ci\non: [push, pull_request]\njobs: {}\n", "c2"
            ),
            AIMessage(content="more triggers"),
        ]
    )

    again = answer(tmp_path, tracker, review.run, "redo more triggers", second)

    assert isinstance(again, Pending) and again.gate == "ci"
    assert "pull_request" in again.detail
    assert "node=ci_gate_1" not in again.detail  # only the new change is shown
    assert "+name: ci" not in again.detail  # not the already-approved lines
    assert len(pushes(tracker)) == 1

    done = answer(tmp_path, tracker, again.run, "ok", passing_reviewer())
    assert isinstance(done, Pending) and done.gate == "review"
    assert len(pushes(tracker)) == 2
    change = tracker.read_change("k0/1-calc-answers-42")
    # Later push: the record is on the PR, not the issue.
    assert any(
        "Gate passed: ci" in c for c in tracker.change_comments_on(change.ref)
    )


def test_a_later_push_without_github_changes_skips_the_gate(tmp_path):
    tracker, gate = to_ci_gate(tmp_path)
    review = answer(tmp_path, tracker, gate.run, "ok", passing_reviewer())
    second = FakeChatModel(
        messages=[
            write_call("calc.py", '"""Calc."""\n' + CALC_SRC, "c2"),
            AIMessage(content="docstring"),
        ]
    )

    again = answer(
        tmp_path, tracker, review.run, "redo add a docstring", second,
        passing_reviewer(),
    )

    assert isinstance(again, Pending) and again.gate == "review"
    assert len(pushes(tracker)) == 2
    waiting = [
        e for e in events(tmp_path)
        if e["kind"] == "gate" and e["gate"] == "ci" and e["result"] == "waiting"
    ]
    assert len(waiting) == 1


# --- the helpers behind the gate -----------------------------------------------


def test_push_refuses_a_head_the_gate_has_not_cleared(tmp_path):
    from k0ntrol.workflow.ci_gate import CiGateError, gated_push
    from k0ntrol.workflow.context import RunContext

    git_repo(tmp_path)
    tracker = FakeTracker()
    ctx = RunContext(root=tmp_path, cfg={}, kind="implement", tracker=tracker)

    with pytest.raises(CiGateError, match="not cleared"):
        gated_push(ctx, {"branch": "k0/1-x", "ci_cleared_sha": "deadbeef"})
    assert pushes(tracker) == []


def test_without_a_base_branch_the_first_gate_shows_all_of_github(tmp_path):
    from k0ntrol.workflow.change import diff_since, merge_base

    git_repo(tmp_path)
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "ci.yml").write_text("name: ci\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "ci")

    since = merge_base(tmp_path, "no-such-branch")

    assert "+name: ci" in diff_since(tmp_path, since, ".github/")


def test_the_gate_detail_is_capped(tmp_path):
    from k0ntrol.workflow.change import diff_since, head_sha

    git_repo(tmp_path)
    base = head_sha(tmp_path)
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "big.yml").write_text("x: 1\n" * 5000)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "big")

    assert diff_since(tmp_path, base, ".github/").endswith("... diff truncated ...")


def test_a_diff_git_cannot_compute_is_an_error_not_an_empty_diff(tmp_path):
    from k0ntrol.workflow.change import DiffError, diff_since

    git_repo(tmp_path)
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "x.yml").write_text("name: x\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "ci")

    with pytest.raises(DiffError, match="0000000"):
        diff_since(tmp_path, "0" * 40, ".github/")


def test_a_baseline_missing_from_the_repo_never_clears_head_for_a_push(tmp_path):
    from k0ntrol.workflow.change import DiffError
    from k0ntrol.workflow.ci_gate import CiGateError, gated_push, run_ci_gate
    from k0ntrol.workflow.context import RunContext

    git_repo(tmp_path)
    tracker = FakeTracker()
    ctx = RunContext(root=tmp_path, cfg={}, kind="implement", tracker=tracker)
    state = {"branch": "k0/1-x", "ci_approved_sha": "0" * 40}

    with pytest.raises(DiffError, match="0000000"):
        run_ci_gate(ctx, state)
    with pytest.raises(CiGateError, match="not cleared"):
        gated_push(ctx, state)
    assert pushes(tracker) == []
