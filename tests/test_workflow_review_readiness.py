"""Review readiness (#175): only two completed review verdicts on the
pushed diff let the human's ``ok`` take the change out of draft.

Each required axis — Conventions and Contract — carries REVIEWER's
structured verdict, ``completed`` or ``refused`` with a reason,
persisted against the commit it judged. A missing, refused, stale or
pre-verdict (legacy) axis is never completion evidence: the gate hides
``ok``, a stored ``ok`` from before the verdicts fails closed, and
``mark_ready`` re-checks before calling the tracker. A posted comment
is never evidence.

Same seams as the other workflow suites (the issue's pre-agreed seams):
``start``/``resume``/``abandon`` with a real temp git repo, scripted
``FakeChatModel``s and a ``FakeTracker``.
"""

import subprocess

import pytest
from langchain_core.messages import AIMessage

from fakes import FakeChatModel, FakeTracker
from k0ntrol.agents import MissingStructuredOutput
from k0ntrol.tracker import TrackerError
from k0ntrol.workflow import Outcome, Pending, abandon, resume
from k0ntrol.workflow.context import RunContext
from k0ntrol.workflow.implement import build_implement_graph
from k0ntrol.workflow.run import _open_checkpoints

from test_workflow_implement import (
    builder_model,
    events,
    git_runner_for,
    item_cfg,
    item_repo,
    models_init,
    read_call,
    ready_item,
    run_item,
    run_tests_call,
    write_call,
)


def completed_call(summary, findings, call_id):
    """A scripted Review tool call judging the change: verdict completed."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "Review",
                "args": {
                    "status": "completed",
                    "summary": summary,
                    "findings": findings,
                    "reason": "",
                },
                "id": call_id,
            }
        ],
    )


def refusal_call(reason, call_id):
    """A scripted Review tool call refusing to judge the change."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "Review",
                "args": {
                    "status": "refused",
                    "summary": "",
                    "findings": [],
                    "reason": reason,
                },
                "id": call_id,
            }
        ],
    )


def completed_reviewer():
    """Both axes completed, no findings."""
    return FakeChatModel(
        messages=[
            completed_call("conventions pass", [], "c1"),
            completed_call("contract pass", [], "c2"),
        ]
    )


def resume_item(run_id, decision, root, tracker, *models):
    """``resume`` on the item run's cfg with scripted follow-on models."""
    return resume(
        run_id,
        decision,
        root=root,
        cfg=item_cfg(),
        init=models_init(*models),
        tracker=tracker,
    )


def seed_checkpoint(root, run_id, values, as_node=None):
    """Rewrite a paused run's checkpointed values — how a run paused by
    code from before the verdicts (pre-#175) looks on resume."""
    conn, saver = _open_checkpoints(root)
    try:
        ctx = RunContext(root=root, cfg=item_cfg(), kind="implement")
        graph = build_implement_graph(ctx, saver)
        config = {"configurable": {"thread_id": run_id}}
        graph.update_state(config, values, as_node=as_node)
    finally:
        conn.close()


def legacy_reviews():
    """Pre-verdict review dicts: ``{summary, findings}``, no status, no head."""
    return {
        "conventions_review": {"summary": "old conventions", "findings": []},
        "contract_review": {"summary": "old contract", "findings": []},
    }


def commit_more(root):
    """A commit on the run's branch after the reviews: the diff they
    judged is no longer HEAD."""
    (root / "later.txt").write_text("moved on\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "later"],
        cwd=root,
        check=True,
    )


def redo_models():
    """The refactor IMPLEMENTER and the second pair of reviews a ``redo``
    at the gate runs."""
    refactor = FakeChatModel(
        messages=[
            write_call("calc.py", "def answer():\n    return 42  # fixed\n", "w9"),
            run_tests_call("r9"),
            AIMessage(content="addressed the reviews"),
        ]
    )
    reviewer = FakeChatModel(
        messages=[
            completed_call("conventions again", [], "c3"),
            completed_call("contract again", [], "c4"),
        ]
    )
    return refactor, reviewer


def pass_comments(tracker, ref):
    return [
        comment
        for comment in tracker.change_comments_on(ref)
        if "Gate passed: review" in comment
    ]


# --- start: a refused or missing axis on either side cannot mark ready ---------


def test_a_refused_conventions_review_never_marks_ready(tmp_path):
    """REVIEWER refused the Conventions axis: the verdict posts labelled,
    the gate shows the refusal and offers no ``ok`` (#175)."""
    item_repo(tmp_path)
    tracker = FakeTracker(git_runner=git_runner_for(tmp_path))
    ready_item(tracker)
    reviewer = FakeChatModel(
        messages=[
            refusal_call("the change is authored by an untrusted app", "c1"),
            completed_call("does what was asked", [], "c2"),
        ]
    )

    pending = run_item(tmp_path, item_cfg(), tracker, builder_model(), reviewer)

    assert isinstance(pending, Pending) and pending.gate == "review"
    assert pending.answers == ("redo", "plan")
    assert "refused" in pending.detail
    assert "untrusted app" in pending.detail
    change = tracker.read_change("k0/1-calc-answers-42")
    reviews = tracker.reviews_on(change.ref)
    assert len(reviews) == 2
    assert "Refused" in reviews[0] and "untrusted app" in reviews[0]
    assert change.is_draft
    assert pass_comments(tracker, change.ref) == []
    assert ("mark_change_ready", change.ref) not in tracker.calls


def test_a_refused_contract_review_never_marks_ready(tmp_path):
    """Same on the Contract axis: a refusal is not a review with zero
    findings — ``ok`` is not offered."""
    item_repo(tmp_path)
    tracker = FakeTracker(git_runner=git_runner_for(tmp_path))
    ready_item(tracker)
    reviewer = FakeChatModel(
        messages=[
            completed_call("fits the rules", ["minor smell"], "c1"),
            refusal_call("the work item is unreadable", "c2"),
        ]
    )

    pending = run_item(tmp_path, item_cfg(), tracker, builder_model(), reviewer)

    assert isinstance(pending, Pending) and pending.gate == "review"
    assert pending.answers == ("redo", "plan")
    assert "contract" in pending.detail.lower()
    assert "unreadable" in pending.detail
    change = tracker.read_change("k0/1-calc-answers-42")
    assert change.is_draft
    assert pass_comments(tracker, change.ref) == []


def test_an_untrusted_change_gets_a_refusal_not_a_review(tmp_path):
    """An App-authored change can never carry a completed verdict: the
    axis is refused in code before REVIEWER runs — a model that would
    have answered ``completed`` changes nothing (#175)."""
    item_repo(tmp_path)
    tracker = _AppAuthoredTracker(git_runner=git_runner_for(tmp_path))
    ready_item(tracker)
    reviewer = completed_reviewer()  # would report "completed" on both axes

    pending = run_item(tmp_path, item_cfg(), tracker, builder_model(), reviewer)

    assert reviewer.invocations == []  # the model is never asked
    assert isinstance(pending, Pending) and pending.answers == ("redo", "plan")
    assert "k0-app[bot]" in pending.detail
    change = tracker.read_change("k0/1-calc-answers-42")
    reviews = tracker.reviews_on(change.ref)
    assert len(reviews) == 2
    assert all("Refused" in review for review in reviews)
    assert "k0-app[bot]" in reviews[0]
    assert change.is_draft
    assert pass_comments(tracker, change.ref) == []


class _AppAuthoredTracker(FakeTracker):
    """The draft change is authored by a GitHub App (``NONE``)."""

    def open_change(self, branch, base, title, body):
        change = super().open_change(branch, base, title, body)
        self._changes[change.ref]["author"] = "k0-app[bot]"
        self._changes[change.ref]["author_association"] = "NONE"
        return change


def test_a_completed_review_with_no_findings_still_marks_ready(tmp_path):
    """Two completed verdicts — even with zero findings — keep the normal
    ``ok``: completion is the verdict, not the findings (#175)."""
    item_repo(tmp_path)
    tracker = FakeTracker(git_runner=git_runner_for(tmp_path))
    ready_item(tracker)

    pending = run_item(
        tmp_path, item_cfg(), tracker, builder_model(), completed_reviewer()
    )

    assert isinstance(pending, Pending) and pending.answers == ("ok", "redo", "plan")
    outcome = resume_item(pending.run, "ok", tmp_path, tracker)

    assert isinstance(outcome, Outcome) and outcome.outcome == "done"
    assert not tracker.read_change("k0/1-calc-answers-42").is_draft


def test_plan_on_unavailable_reviews_still_never_marks_ready(tmp_path):
    """``plan`` on a refused axis keeps its meaning: the item goes back
    to planning and the draft is never marked ready (#175)."""
    item_repo(tmp_path)
    tracker = FakeTracker(git_runner=git_runner_for(tmp_path))
    ready_item(tracker)
    reviewer = FakeChatModel(
        messages=[
            refusal_call("untrusted change", "c1"),
            completed_call("contract pass", [], "c2"),
        ]
    )
    pending = run_item(tmp_path, item_cfg(), tracker, builder_model(), reviewer)

    outcome = resume_item(
        pending.run, "plan the app cannot author this", tmp_path, tracker
    )

    assert isinstance(outcome, Outcome) and outcome.outcome == "rejected"
    assert "ready-for-agent" not in tracker.labels_on("1")
    change = tracker.read_change("k0/1-calc-answers-42")
    assert change.is_draft
    assert pass_comments(tracker, change.ref) == []


# --- resume: stored decisions and old checkpoints fail closed ------------------


def test_a_stored_ok_on_stale_reviews_fails_closed(tmp_path):
    """The stored gate answers still list ``ok`` but the diff moved since
    the reviews: ``ok`` is refused in the handler and the gate re-shows
    without it — only an explicit ``redo`` gets fresh verdicts (#175)."""
    item_repo(tmp_path)
    tracker = FakeTracker(git_runner=git_runner_for(tmp_path))
    ready_item(tracker)
    pending = run_item(
        tmp_path, item_cfg(), tracker, builder_model(), completed_reviewer()
    )
    assert pending.answers == ("ok", "redo", "plan")
    commit_more(tmp_path)  # the diff the reviews judged is no longer HEAD

    again = resume_item(pending.run, "ok", tmp_path, tracker)

    assert isinstance(again, Pending) and again.gate == "review"
    assert "ok" not in again.answers
    assert "stale" in again.detail
    change = tracker.read_change("k0/1-calc-answers-42")
    assert change.is_draft
    assert pass_comments(tracker, change.ref) == []
    assert ("mark_change_ready", change.ref) not in tracker.calls

    refactor, reviewer = redo_models()
    gate = resume_item(again.run, "redo address them anyway", tmp_path, tracker, refactor, reviewer)

    assert isinstance(gate, Pending) and gate.answers == ("ok", "redo", "plan")
    outcome = resume_item(gate.run, "ok", tmp_path, tracker)
    assert isinstance(outcome, Outcome) and outcome.outcome == "done"
    assert not tracker.read_change("k0/1-calc-answers-42").is_draft


def test_a_legacy_checkpoint_at_the_gate_reads_reviews_as_missing(tmp_path):
    """A run paused by pre-verdict code resumes at the gate with legacy
    review dicts — they count as missing, ``ok`` is not offered, and a
    ``redo`` re-reviews to recover (#175)."""
    item_repo(tmp_path)
    tracker = FakeTracker(git_runner=git_runner_for(tmp_path))
    ready_item(tracker)
    pending = run_item(
        tmp_path, item_cfg(), tracker, builder_model(), completed_reviewer()
    )
    seed_checkpoint(tmp_path, pending.run, legacy_reviews(), as_node="post_contract")

    gate = resume_item(pending.run, None, tmp_path, tracker)

    assert isinstance(gate, Pending) and gate.gate == "review"
    assert "ok" not in gate.answers
    assert "missing" in gate.detail
    assert tracker.read_change("k0/1-calc-answers-42").is_draft

    refactor, reviewer = redo_models()
    gate = resume_item(gate.run, "redo re-review", tmp_path, tracker, refactor, reviewer)
    assert gate.answers == ("ok", "redo", "plan")


def test_a_missing_axis_is_not_re_reviewed_behind_the_human(tmp_path):
    """An interrupted axis stays missing — no auto re-review: resuming
    runs only the axis it stopped on, and the gate still refuses ``ok``
    for the axis that never produced a verdict (#175)."""
    item_repo(tmp_path)
    tracker = FakeTracker(git_runner=git_runner_for(tmp_path))
    ready_item(tracker)
    pending = run_item(
        tmp_path, item_cfg(), tracker, builder_model(), completed_reviewer()
    )
    # paused between the axes; the Conventions dict is legacy on resume
    seed_checkpoint(
        tmp_path,
        pending.run,
        {"conventions_review": {"summary": "old", "findings": []}},
        as_node="post_conventions",
    )
    contract_only = FakeChatModel(
        messages=[completed_call("contract pass", [], "c9")]
    )

    gate = resume_item(pending.run, None, tmp_path, tracker, contract_only)

    assert isinstance(gate, Pending) and gate.gate == "review"
    assert "ok" not in gate.answers
    assert "missing" in gate.detail
    change = tracker.read_change("k0/1-calc-answers-42")
    assert change.is_draft
    reviews = tracker.reviews_on(change.ref)
    # neither axis reposts on the unchanged head — posts key to the
    # reviewed revision, not the run's round (#178)
    assert sum("post_conventions" in review for review in reviews) == 1
    assert sum("post_contract" in review for review in reviews) == 1


def test_a_legacy_checkpoint_at_review_comment_posts_no_pass(tmp_path):
    """Interrupted after the gate's ``ok`` but before its pass comment:
    a resume with legacy reviews posts no pass and never marks ready —
    the run re-shows the gate instead (#175)."""
    item_repo(tmp_path)
    tracker = _FlakyGateComment(git_runner=git_runner_for(tmp_path))
    ready_item(tracker)
    pending = run_item(
        tmp_path, item_cfg(), tracker, builder_model(), completed_reviewer()
    )
    with pytest.raises(TrackerError):
        resume_item(pending.run, "ok", tmp_path, tracker)
    seed_checkpoint(tmp_path, pending.run, legacy_reviews())

    gate = resume_item(pending.run, None, tmp_path, tracker)

    assert isinstance(gate, Pending) and gate.gate == "review"
    assert "ok" not in gate.answers
    change = tracker.read_change("k0/1-calc-answers-42")
    assert change.is_draft
    assert pass_comments(tracker, change.ref) == []
    assert ("mark_change_ready", change.ref) not in tracker.calls


class _FlakyGateComment(FakeTracker):
    """The first review-gate pass comment dies inside ``comment_change``,
    leaving the run paused at ``review_comment``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flaked = False

    def comment_change(self, ref, body, *, run, node):
        if node.startswith("review_gate") and not self._flaked:
            self._flaked = True
            raise TrackerError("the comment's answer was lost")
        super().comment_change(ref, body, run=run, node=node)


def test_a_legacy_checkpoint_at_mark_ready_never_marks_ready(tmp_path):
    """Interrupted one node later, at ``mark_ready``: the readiness check
    runs again before the tracker call — legacy reviews send the run
    back to the gate instead of un-drafting the change (#175)."""
    item_repo(tmp_path)
    tracker = FakeTracker(git_runner=git_runner_for(tmp_path))
    ready_item(tracker)
    pending = run_item(
        tmp_path, item_cfg(), tracker, builder_model(), completed_reviewer()
    )
    seed_checkpoint(tmp_path, pending.run, legacy_reviews(), as_node="review_comment")

    gate = resume_item(pending.run, None, tmp_path, tracker)

    assert isinstance(gate, Pending) and gate.gate == "review"
    assert "ok" not in gate.answers
    change = tracker.read_change("k0/1-calc-answers-42")
    assert change.is_draft
    assert pass_comments(tracker, change.ref) == []
    assert ("mark_change_ready", change.ref) not in tracker.calls


# --- abandon -------------------------------------------------------------------


def test_abandon_leaves_an_unreviewed_draft_draft(tmp_path):
    """A draft whose reviews are unavailable stays a draft when the run
    is abandoned, and no passing gate decision was ever posted (#175)."""
    item_repo(tmp_path)
    tracker = FakeTracker(git_runner=git_runner_for(tmp_path))
    ready_item(tracker)
    reviewer = FakeChatModel(
        messages=[
            refusal_call("untrusted change", "c1"),
            completed_call("contract pass", [], "c2"),
        ]
    )
    pending = run_item(tmp_path, item_cfg(), tracker, builder_model(), reviewer)

    abandon(pending.run, root=tmp_path)

    change = tracker.read_change("k0/1-calc-answers-42")
    assert change.is_draft
    assert pass_comments(tracker, change.ref) == []
    assert ("mark_change_ready", change.ref) not in tracker.calls


# --- the published revision: verdicts bind to the PR's head, not HEAD ---------
#
# ``head_sha`` on a verdict is the pull request's published head — what
# ``gh pr diff`` showed — re-read through ``read_change`` at the gate and
# before ``mark_ready``. ``completed`` requires it to match both the
# published head and local HEAD, so a remote move OR an unpushed local
# commit makes the verdict stale.


def test_a_published_head_move_blocks_a_stored_ok(tmp_path):
    """Someone else pushed to the branch between the reviews and the
    human's ``ok``: the published head moved while local HEAD did not —
    the verdicts judged another revision, ``ok`` fails closed and only
    an explicit ``redo`` re-reviews (#175)."""
    item_repo(tmp_path)
    tracker = FakeTracker(git_runner=git_runner_for(tmp_path))
    ready_item(tracker)
    pending = run_item(
        tmp_path, item_cfg(), tracker, builder_model(), completed_reviewer()
    )
    assert pending.answers == ("ok", "redo", "plan")
    change = tracker.read_change("k0/1-calc-answers-42")
    # the remote moved without k0 pushing: the local checkout is untouched
    tracker._changes[change.ref]["head_sha"] = "deadbeef" * 5

    again = resume_item(pending.run, "ok", tmp_path, tracker)

    assert isinstance(again, Pending) and again.gate == "review"
    assert "ok" not in again.answers
    assert "stale" in again.detail
    assert tracker.read_change("k0/1-calc-answers-42").is_draft
    assert pass_comments(tracker, change.ref) == []
    assert ("mark_change_ready", change.ref) not in tracker.calls

    refactor, reviewer = redo_models()
    gate = resume_item(
        again.run, "redo reconcile the heads", tmp_path, tracker, refactor, reviewer
    )
    assert gate.answers == ("ok", "redo", "plan")
    outcome = resume_item(gate.run, "ok", tmp_path, tracker)
    assert isinstance(outcome, Outcome) and outcome.outcome == "done"
    assert not tracker.read_change("k0/1-calc-answers-42").is_draft


class _MovingHeadTracker(FakeTracker):
    """The published head moves once — while the Contract pass collects
    its diff (the second ``read_change_diff`` call)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._diff_reads = 0

    def read_change_diff(self, ref):
        self._diff_reads += 1
        if self._diff_reads == 2:
            self._changes[ref]["head_sha"] = "deadbeef" * 5
        return super().read_change_diff(ref)


def test_a_head_move_mid_review_stops_resumable(tmp_path):
    """The head moved while an axis collected its inputs: the pass stops
    resumable instead of recording a verdict on a mixed revision; resume
    re-reviews and ``redo`` gets both axes back onto one head (#175)."""
    item_repo(tmp_path)
    tracker = _MovingHeadTracker(git_runner=git_runner_for(tmp_path))
    ready_item(tracker)

    with pytest.raises(TrackerError, match="moved"):
        run_item(
            tmp_path, item_cfg(), tracker, builder_model(), completed_reviewer()
        )

    change = tracker.read_change("k0/1-calc-answers-42")
    assert len(tracker.reviews_on(change.ref)) == 1  # only conventions posted
    assert change.is_draft

    reviewer = completed_reviewer()
    gate = resume_item(events(tmp_path)[0]["run"], None, tmp_path, tracker, reviewer)

    # contract re-reviewed on the moved head; conventions judged the old
    # one and the local checkout never moved — both read stale, no ok
    assert isinstance(gate, Pending) and gate.gate == "review"
    assert "ok" not in gate.answers
    assert "stale" in gate.detail
    assert tracker.read_change("k0/1-calc-answers-42").is_draft
    assert pass_comments(tracker, change.ref) == []

    refactor, reviewer2 = redo_models()
    gate = resume_item(gate.run, "redo reconcile", tmp_path, tracker, refactor, reviewer2)
    assert gate.answers == ("ok", "redo", "plan")
    outcome = resume_item(gate.run, "ok", tmp_path, tracker)
    assert isinstance(outcome, Outcome) and outcome.outcome == "done"
    assert not tracker.read_change("k0/1-calc-answers-42").is_draft


# --- a missing or failed axis surfaces through stop/limit, never a pass -------


def test_a_reviewer_with_no_review_call_stops_resumable(tmp_path):
    """REVIEWER answered without calling the Review tool: the axis stops
    resumable on the missing verdict — nothing posts, the draft stays
    draft — and a resume re-runs the axis (#175)."""
    item_repo(tmp_path)
    tracker = FakeTracker(git_runner=git_runner_for(tmp_path))
    ready_item(tracker)
    reviewer = FakeChatModel(
        messages=[
            AIMessage(content="I could not reach a verdict"),
            completed_call("contract pass", [], "c2"),
        ]
    )

    with pytest.raises(MissingStructuredOutput):
        run_item(tmp_path, item_cfg(), tracker, builder_model(), reviewer)

    change = tracker.read_change("k0/1-calc-answers-42")
    assert change.is_draft
    assert pass_comments(tracker, change.ref) == []
    assert tracker.reviews_on(change.ref) == []
    assert ("mark_change_ready", change.ref) not in tracker.calls

    gate = resume_item(
        events(tmp_path)[0]["run"], None, tmp_path, tracker, completed_reviewer()
    )
    assert isinstance(gate, Pending) and gate.answers == ("ok", "redo", "plan")


def test_a_failed_axis_names_the_node_and_posts_no_pass(tmp_path):
    """A limited Contract pass pauses at ``limit_hit`` naming its node —
    retry completes it, and before any human answer there is still no
    pass comment and no ready call (#175)."""
    item_repo(tmp_path)
    tracker = FakeTracker(git_runner=git_runner_for(tmp_path))
    ready_item(tracker)
    cfg = item_cfg()
    cfg["agents"]["reviewer"]["model_calls"] = 3
    reviewer = FakeChatModel(
        messages=[
            completed_call("conventions pass", [], "c1"),
            *[read_call("README.md", f"r{i}") for i in range(6)],
        ]
    )

    pending = run_item(tmp_path, cfg, tracker, builder_model(), reviewer)

    assert isinstance(pending, Pending) and pending.gate == "limit_hit"
    assert "contract_review" in pending.question
    change = tracker.read_change("k0/1-calc-answers-42")
    assert len(tracker.reviews_on(change.ref)) == 1
    assert change.is_draft
    assert pass_comments(tracker, change.ref) == []

    gate = resume_item(
        pending.run, "retry", tmp_path, tracker, completed_reviewer()
    )

    assert isinstance(gate, Pending) and gate.gate == "review"
    assert gate.answers == ("ok", "redo", "plan")
    assert len(tracker.reviews_on(change.ref)) == 2
    assert tracker.read_change("k0/1-calc-answers-42").is_draft
    assert pass_comments(tracker, change.ref) == []


def test_an_incomplete_resume_runs_to_the_gate_without_a_pass(tmp_path):
    """Resuming mid-reviews can only reach the gate again — no pass
    comment, no ready call — before the human answers (#175)."""
    item_repo(tmp_path)
    tracker = FakeTracker(git_runner=git_runner_for(tmp_path))
    ready_item(tracker)
    pending = run_item(
        tmp_path, item_cfg(), tracker, builder_model(), completed_reviewer()
    )
    # paused between the axes, with the verdicts already recorded
    seed_checkpoint(tmp_path, pending.run, {}, as_node="post_conventions")
    contract_again = FakeChatModel(
        messages=[completed_call("contract pass", [], "c9")]
    )

    gate = resume_item(pending.run, None, tmp_path, tracker, contract_again)

    change = tracker.read_change("k0/1-calc-answers-42")
    assert isinstance(gate, Pending) and gate.gate == "review"
    assert gate.answers == ("ok", "redo", "plan")
    assert change.is_draft
    assert pass_comments(tracker, change.ref) == []
    assert ("mark_change_ready", change.ref) not in tracker.calls


def test_redo_on_an_unchanged_diff_re_reviews_and_marks_ready(tmp_path):
    """A ``redo`` whose rebuild commits nothing re-reviews on the same
    head — fresh verdicts, new round markers, no duplicate posts — and
    ``ok`` can mark ready (#175)."""
    item_repo(tmp_path)
    tracker = FakeTracker(git_runner=git_runner_for(tmp_path))
    ready_item(tracker)
    pending = run_item(
        tmp_path, item_cfg(), tracker, builder_model(), completed_reviewer()
    )
    refactor = FakeChatModel(
        messages=[
            run_tests_call("r9"),
            AIMessage(content="the change already matches the reviews"),
        ]
    )
    reviewer = completed_reviewer()

    gate = resume_item(
        pending.run, "redo check it again", tmp_path, tracker, refactor, reviewer
    )

    assert isinstance(gate, Pending) and gate.answers == ("ok", "redo", "plan")
    change = tracker.read_change("k0/1-calc-answers-42")
    reviews = tracker.reviews_on(change.ref)
    assert len(reviews) == 4  # both axes, both rounds
    assert len({r.splitlines()[0] for r in reviews}) == 4  # distinct markers
    outcome = resume_item(gate.run, "ok", tmp_path, tracker)
    assert isinstance(outcome, Outcome) and outcome.outcome == "done"
    assert not tracker.read_change("k0/1-calc-answers-42").is_draft
