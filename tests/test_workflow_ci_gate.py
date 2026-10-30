"""The CI gate: a push that changes ``.github/`` waits for the human (#47).

Same seams as the /full suite: real temp git repo, scripted fake models,
``FakeTracker``. CODER writes a workflow file; the gate must interrupt
before the harness pushes.
"""

import subprocess

from langchain_core.messages import AIMessage

from fakes import FakeTracker
from test_workflow_full import (
    CALC_SRC,
    events,
    fake_tester,
    full_cfg,
    full_to_test_gate,
    models_init,
    reviewer_model,
    write_call,
    writing_scan,
)
from k0ntrol.workflow import Outcome, Pending, resume

WORKFLOW = ".github/workflows/ci.yml"
WORKFLOW_SRC = "name: ci\non: push\njobs: {}\n"


def coder_writing_workflow(call_id="c1"):
    """A CODER fake that writes the code and a workflow file."""
    return fake_tester(
        write_call("calc.py", CALC_SRC, call_id),
        write_call(WORKFLOW, WORKFLOW_SRC, f"{call_id}w"),
        AIMessage(content="implemented"),
    )


def to_ci_gate(tmp_path, coder=None):
    """``/full <text>`` past the test gate; CODER touches ``.github/``."""
    tracker, _, _, tests = full_to_test_gate(tmp_path)
    coder = coder or coder_writing_workflow()
    result = resume(
        tests.run, "ok", root=tmp_path, cfg=full_cfg(), tracker=tracker,
        init=models_init(coder, reviewer_model()), scan=writing_scan,
    )
    return tracker, result


def pushes(tracker):
    return [c for c in tracker.calls if c[0] == "push"]


def test_a_workflow_change_holds_the_push_behind_the_ci_gate(tmp_path):
    tracker, result = to_ci_gate(tmp_path)

    assert isinstance(result, Pending) and result.gate == "ci"
    assert result.answers == ("ok", "redo")
    assert WORKFLOW in result.detail and "name: ci" in result.detail
    assert "calc.py" not in result.detail  # only the .github/ diff is shown
    assert pushes(tracker) == []
    assert not [c for c in tracker.calls if c[0] == "open_change"]


def answer(tmp_path, tracker, run, decision, *models):
    return resume(
        run, decision, root=tmp_path, cfg=full_cfg(), tracker=tracker,
        init=models_init(*models), scan=writing_scan,
    )


def test_approving_pushes_opens_the_draft_and_records_the_decision(tmp_path):
    tracker, gate = to_ci_gate(tmp_path)

    result = answer(tmp_path, tracker, gate.run, "ok", reviewer_model())

    assert isinstance(result, Pending) and result.gate == "review"
    assert pushes(tracker) == [("push", "k0/1-calc-answers-42")]
    assert tracker.read_change("k0/1-calc-answers-42").is_draft
    # First push: the PR does not exist yet, so the record is on the issue.
    comments = [c.body for c in tracker.read_work_item("1").comments]
    ci = [c for c in comments if "Gate passed: ci" in c]
    assert len(ci) == 1 and "Decided by: t" in ci[0]
    gates = [e for e in events(tmp_path) if e["kind"] == "gate" and e["gate"] == "ci"]
    assert [e["result"] for e in gates] == ["waiting", "approve"]


def test_rejecting_sends_the_feedback_to_coder_and_gates_again(tmp_path):
    tracker, gate = to_ci_gate(tmp_path)
    coder2 = fake_tester(
        write_call(WORKFLOW, "name: ci\non: pull_request\njobs: {}\n", "c2"),
        AIMessage(content="narrowed"),
    )

    again = answer(tmp_path, tracker, gate.run, "redo only run on pull_request", coder2)

    brief = coder2.invocations[0][-1].content
    assert "only run on pull_request" in brief
    assert isinstance(again, Pending) and again.gate == "ci"
    assert "pull_request" in again.detail
    assert pushes(tracker) == []

    done = answer(tmp_path, tracker, again.run, "ok", reviewer_model())
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
    review = answer(tmp_path, tracker, gate.run, "ok", reviewer_model())
    coder2 = fake_tester(
        write_call(WORKFLOW, "name: ci\non: [push, pull_request]\njobs: {}\n", "c2"),
        AIMessage(content="more triggers"),
    )

    again = answer(tmp_path, tracker, review.run, "redo more triggers", coder2)

    assert isinstance(again, Pending) and again.gate == "ci"
    assert "pull_request" in again.detail
    assert "node=ci_gate_1" not in again.detail  # only the new change is shown
    assert "+name: ci" not in again.detail  # not the already-approved lines
    assert len(pushes(tracker)) == 1

    done = answer(tmp_path, tracker, again.run, "ok", reviewer_model())
    assert isinstance(done, Pending) and done.gate == "review"
    assert len(pushes(tracker)) == 2
    change = tracker.read_change("k0/1-calc-answers-42")
    # Later push: the record is on the PR, not the issue.
    assert any("Gate passed: ci" in c for c in tracker.change_comments_on(change.ref))


def test_a_later_push_without_github_changes_skips_the_gate(tmp_path):
    tracker, gate = to_ci_gate(tmp_path)
    review = answer(tmp_path, tracker, gate.run, "ok", reviewer_model())
    coder2 = fake_tester(
        write_call("calc.py", '"""Calc."""\n' + CALC_SRC, "c2"),
        AIMessage(content="docstring"),
    )

    again = answer(tmp_path, tracker, review.run, "redo add a docstring", coder2,
                   reviewer_model())

    assert isinstance(again, Pending) and again.gate == "review"
    assert len(pushes(tracker)) == 2
    waiting = [
        e for e in events(tmp_path)
        if e["kind"] == "gate" and e["gate"] == "ci" and e["result"] == "waiting"
    ]
    assert len(waiting) == 1


def test_push_refuses_a_head_the_gate_has_not_cleared(tmp_path):
    import pytest
    from k0ntrol.workflow.ci_gate import CiGateError, gated_push
    from k0ntrol.workflow.context import RunContext
    from test_workflow_full import git_repo

    git_repo(tmp_path)
    tracker = FakeTracker()
    ctx = RunContext(root=tmp_path, cfg={}, kind="full", tracker=tracker)

    with pytest.raises(CiGateError, match="not cleared"):
        gated_push(ctx, {"branch": "k0/1-x", "ci_cleared_sha": "deadbeef"})
    assert pushes(tracker) == []


def test_without_a_base_branch_the_first_gate_shows_all_of_github(tmp_path):
    from k0ntrol.workflow.change import diff_since, merge_base
    from test_workflow_full import git, git_repo

    git_repo(tmp_path)
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "ci.yml").write_text("name: ci\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "ci")

    since = merge_base(tmp_path, "no-such-branch")

    assert "+name: ci" in diff_since(tmp_path, since, ".github/")


def test_the_gate_detail_is_capped(tmp_path):
    from k0ntrol.workflow.change import diff_since, head_sha
    from test_workflow_full import git, git_repo

    git_repo(tmp_path)
    base = head_sha(tmp_path)
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "big.yml").write_text("x: 1\n" * 5000)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "big")

    assert diff_since(tmp_path, base, ".github/").endswith("... diff truncated ...")
