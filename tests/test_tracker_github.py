"""Contract tests for the gh adapter: recorded --json/gh api output, fake runner."""

import json
import sys

import pytest

from fakes import FakeRunner
from k0ntrol.tracker import TrackerError, comment_marker
from k0ntrol.tracker.github import GhTracker, subprocess_runner

ISSUE = {
    "number": 42,
    "title": "Add login",
    "body": "Users must log in.",
    "state": "open",
    "html_url": "https://github.com/o/r/issues/42",
    "user": {"login": "rubenjp"},
    "author_association": "OWNER",
    "labels": [],
}

COMMENTS = [
    {
        "body": "first take",
        "user": {"login": "octo"},
        "author_association": "MEMBER",
    }
]

ISSUE_ENDPOINT = ["gh", "api", "repos/{owner}/{repo}/issues/42"]
COMMENTS_ENDPOINT = ["gh", "api", "--paginate", "repos/{owner}/{repo}/issues/42/comments"]


def _issue(overrides=None):
    return {**ISSUE, **(overrides or {})}


def _tracker(issue=ISSUE, comments=COMMENTS):
    runner = FakeRunner()
    runner.script(ISSUE_ENDPOINT, stdout=json.dumps(issue))
    runner.script(COMMENTS_ENDPOINT, stdout=json.dumps(comments))
    return runner, GhTracker(runner)


def test_read_work_item_parses_issue_and_comments():
    _, tracker = _tracker()

    item = tracker.read_work_item("42")

    assert item.ref == "42"
    assert item.title == "Add login"
    assert item.state == "open"
    assert item.author_association == "OWNER"
    assert item.qualified_ref == "o/r#42"
    assert [c.author for c in item.comments] == ["octo"]


@pytest.mark.parametrize("existing", [False, True])
def test_ensure_work_item_uses_creation_marker(existing):
    marker = comment_marker("r1", "create_issue")
    runner, tracker = _tracker(issue=_issue({"body": f"request\n\n{marker}"}))
    runner.script(
        ["gh", "api", "--paginate", "--slurp", "repos/{owner}/{repo}/issues?state=all&per_page=100"],
        stdout=json.dumps([[ISSUE], [_issue({"body": marker})]] if existing else [[]]),
    )
    runner.script(["gh", "api", "user"], stdout=json.dumps({"login": "rubenjp"}))
    runner.script(["gh", "issue", "create"], stdout="https://github.com/o/r/issues/42")
    item = tracker.ensure_work_item("Task", "request", run="r1", node="create_issue")
    assert item.ref == "42"
    creates = [c for c in runner.calls if c[:3] == ["gh", "issue", "create"]]
    assert len(creates) == (0 if existing else 1)
    if creates:
        assert creates[0][-1] == f"request\n\n{marker}"


def test_ensure_work_item_ignores_a_spoofed_marker():
    """A third party can paste the marker into a body, not fake its author."""
    marker = comment_marker("r1", "create_issue")
    runner, tracker = _tracker(issue=_issue({"body": f"request\n\n{marker}"}))
    spoofed = _issue({"number": 99, "body": marker, "user": {"login": "mallory"}})
    runner.script(
        ["gh", "api", "--paginate", "--slurp", "repos/{owner}/{repo}/issues?state=all&per_page=100"],
        stdout=json.dumps([[spoofed]]),
    )
    runner.script(["gh", "api", "user"], stdout=json.dumps({"login": "rubenjp"}))
    runner.script(
        ["gh", "api", "repos/{owner}/{repo}/issues/99"],
        stdout=json.dumps(spoofed),
    )
    runner.script(
        ["gh", "api", "--paginate", "repos/{owner}/{repo}/issues/99/comments"],
        stdout="[]",
    )
    runner.script(["gh", "issue", "create"], stdout="https://github.com/o/r/issues/42")

    item = tracker.ensure_work_item("Task", "request", run="r1", node="create_issue")

    assert item.ref == "42"
    creates = [c for c in runner.calls if c[:3] == ["gh", "issue", "create"]]
    assert len(creates) == 1


def test_create_work_item_posts_and_reads_back():
    runner = FakeRunner()
    runner.script(["gh", "issue", "create"], stdout="https://github.com/o/r/issues/7\n")
    runner.script(
        ["gh", "api", "repos/{owner}/{repo}/issues/7"],
        stdout=json.dumps(_issue({"number": 7})),
    )
    runner.script(
        ["gh", "api", "--paginate", "repos/{owner}/{repo}/issues/7/comments"],
        stdout="[]",
    )
    tracker = GhTracker(runner)

    item = tracker.create_work_item("Add login", "Users must log in.")

    assert item.ref == "7"
    assert [
        "gh",
        "issue",
        "create",
        "--title",
        "Add login",
        "--body",
        "Users must log in.",
    ] in runner.calls


def test_comment_work_item_marks_and_skips_when_marker_present():
    runner, tracker = _tracker()
    runner.script(["gh", "issue", "comment"])
    marker = comment_marker("r1", "spec_gate")

    tracker.comment_work_item("42", "gate passed", run="r1", node="spec_gate")
    posted = [c for c in runner.calls if c[:3] == ["gh", "issue", "comment"]]
    assert posted == [
        ["gh", "issue", "comment", "42", "--body", f"{marker}\n\ngate passed"]
    ]

    runner.script(
        COMMENTS_ENDPOINT,
        stdout=json.dumps(COMMENTS + [{"body": f"{marker}\n\ngate passed", "user": {"login": "k0"}, "author_association": "OWNER"}]),
    )
    tracker.comment_work_item("42", "gate passed", run="r1", node="spec_gate")
    posted = [c for c in runner.calls if c[:3] == ["gh", "issue", "comment"]]
    assert len(posted) == 1  # second call was a no-op


def test_find_comment_returns_none_without_marker():
    _, tracker = _tracker()
    assert tracker.find_comment("42", comment_marker("r1", "n")) is None
    assert (
        tracker.find_comment("42", "first take") is not None
    )


def test_label_operations_use_gh_issue_edit_and_gh_label():
    runner = FakeRunner()
    runner.script(ISSUE_ENDPOINT, stdout=json.dumps(_issue({"labels": [{"name": "k0:stuck"}]})))
    runner.script(["gh", "issue", "edit"])
    runner.script(["gh", "label", "list", "--json", "name"], stdout='[{"name":"bug"}]')
    runner.script(["gh", "label", "create"])
    tracker = GhTracker(runner)

    tracker.add_label("42", "k0:stuck")
    assert ["gh", "issue", "edit", "42", "--add-label", "k0:stuck"] in runner.calls

    tracker.remove_label("42", "k0:stuck")
    assert ["gh", "issue", "edit", "42", "--remove-label", "k0:stuck"] in runner.calls

    runner.script(ISSUE_ENDPOINT, stdout=json.dumps(_issue()))
    tracker.remove_label("42", "k0:stuck")  # absent: no edit call this time

    tracker.ensure_label("k0:stuck")
    creates = [c for c in runner.calls if c[:3] == ["gh", "label", "create"]]
    assert len(creates) == 1 and "k0:stuck" in creates[0]


def test_create_branch_checks_out_a_remote_only_branch_from_its_tip():
    runner = FakeRunner()
    runner.script(["git", "fetch", "origin", "main"])
    runner.script(["git", "rev-parse", "--verify", "--quiet", "refs/heads/k0/1-x"], returncode=1)
    runner.script(["git", "rev-parse", "--verify", "--quiet", "refs/remotes/origin/k0/1-x"])
    runner.script(["git", "checkout"])
    tracker = GhTracker(runner)

    tracker.create_branch("k0/1-x", "main")

    assert ["git", "checkout", "-b", "k0/1-x", "origin/k0/1-x"] in runner.calls
    assert ["git", "checkout", "-b", "k0/1-x", "origin/main"] not in runner.calls


def test_create_branch_fetches_and_checks_out_new_branch():
    runner = FakeRunner()
    runner.script(["git", "fetch", "origin", "main"])
    runner.script(["git", "rev-parse", "--verify", "--quiet", "refs/heads/k0/1-x"], returncode=1)
    runner.script(
        ["git", "rev-parse", "--verify", "--quiet", "refs/remotes/origin/k0/1-x"],
        returncode=1,
    )
    runner.script(["git", "checkout"])
    tracker = GhTracker(runner)

    tracker.create_branch("k0/1-x", "main")

    assert ["git", "fetch", "origin", "main"] in runner.calls
    assert ["git", "checkout", "-b", "k0/1-x", "origin/main"] in runner.calls


def test_create_branch_checks_out_existing_branch():
    runner = FakeRunner()
    runner.script(["git", "fetch", "origin", "main"])
    runner.script(["git", "rev-parse", "--verify", "--quiet", "refs/heads/k0/1-x"])
    runner.script(["git", "checkout", "k0/1-x"])
    tracker = GhTracker(runner)

    tracker.create_branch("k0/1-x", "main")

    assert ["git", "checkout", "k0/1-x"] in runner.calls
    assert not any("-b" in c for c in runner.calls)


def test_commit_stages_all_and_returns_whether_it_committed():
    runner = FakeRunner()
    runner.script(["git", "status", "--porcelain"], stdout=" M x.py\n")
    runner.script(["git", "add", "-A"])
    runner.script(["git", "commit", "-m", "#42: add login"])
    tracker = GhTracker(runner)

    assert tracker.commit("#42: add login") is True

    runner.script(["git", "status", "--porcelain"], stdout="")
    assert tracker.commit("#42: again") is False
    assert not any(c[:3] == ["git", "commit", "-m"] and "again" in c[-1] for c in runner.calls)


def test_push_uses_upstream_tracking():
    runner = FakeRunner()
    runner.script(["git", "push"])
    tracker = GhTracker(runner)

    tracker.push("k0/1-x")

    assert ["git", "push", "-u", "origin", "k0/1-x"] in runner.calls


def test_read_change_returns_none_or_the_change():
    runner = FakeRunner()
    runner.script(
        ["gh", "pr", "list", "--head", "b", "--state", "all", "--json"],
        stdout="[]",
    )
    tracker = GhTracker(runner)
    assert tracker.read_change("b") is None

    runner.script(
        ["gh", "pr", "list", "--head", "b", "--state", "all", "--json"],
        stdout=json.dumps(
            [
                {
                    "number": 7,
                    "title": "Add login",
                    "state": "MERGED",
                    "isDraft": False,
                    "url": "https://github.com/o/r/pull/7",
                }
            ]
        ),
    )
    change = tracker.read_change("b")
    assert change.ref == "7" and change.state == "merged"


def test_open_change_reuses_open_pr_or_creates_a_draft():
    runner = FakeRunner()
    runner.script(
        ["gh", "pr", "list", "--head", "b", "--state", "all", "--json"],
        stdout="[]",
    )
    runner.script(
        ["gh", "pr", "create"], stdout="https://github.com/o/r/pull/7\n"
    )
    tracker = GhTracker(runner)

    change = tracker.open_change("b", "main", "Add login", "Closes #42")
    assert change.ref == "7" and change.is_draft and change.state == "open"
    create_call = [c for c in runner.calls if c[:3] == ["gh", "pr", "create"]][0]
    assert "--draft" in create_call and "--base" in create_call

    runner.script(
        ["gh", "pr", "list", "--head", "b", "--state", "all", "--json"],
        stdout=json.dumps(
            [
                {
                    "number": 7,
                    "title": "Add login",
                    "state": "OPEN",
                    "isDraft": True,
                    "url": "https://github.com/o/r/pull/7",
                }
            ]
        ),
    )
    again = tracker.open_change("b", "main", "Add login", "Closes #42")
    assert again.ref == "7"
    creates = [c for c in runner.calls if c[:3] == ["gh", "pr", "create"]]
    assert len(creates) == 1


def test_mark_change_ready_posts_comment_review_and_reads_diff():
    runner = FakeRunner()
    runner.script(["gh", "pr", "ready"])
    runner.script(
        ["gh", "api", "repos/{owner}/{repo}/pulls/7/reviews"], stdout="[]"
    )
    runner.script(["gh", "pr", "review"])
    runner.script(
        ["gh", "api", "--paginate", "repos/{owner}/{repo}/issues/7/comments"],
        stdout="[]",
    )
    runner.script(["gh", "pr", "comment"])
    runner.script(["gh", "pr", "diff"], stdout="diff --git a/x b/x\n+1\n")
    tracker = GhTracker(runner)
    marker = comment_marker("r1", "review_gate")

    tracker.mark_change_ready("7")
    assert ["gh", "pr", "ready", "7"] in runner.calls

    tracker.post_review("7", "looks good", run="r1", node="reviewer")
    review = [c for c in runner.calls if c[:3] == ["gh", "pr", "review"]][0]
    assert "--comment" in review
    assert comment_marker("r1", "reviewer") in review[review.index("--body") + 1]

    tracker.comment_change("7", "gate ok", run="r1", node="review_gate")
    comment = [c for c in runner.calls if c[:3] == ["gh", "pr", "comment"]][0]
    assert marker in comment[comment.index("--body") + 1]

    assert tracker.read_change_diff("7") == "diff --git a/x b/x\n+1\n"


def test_post_review_is_noop_when_marker_already_reviewed():
    runner = FakeRunner()
    runner.script(
        ["gh", "api", "repos/{owner}/{repo}/pulls/7/reviews"],
        stdout=json.dumps(
            [{"body": f"{comment_marker('r1', 'reviewer')}\n\nlooks good"}]
        ),
    )
    tracker = GhTracker(runner)

    tracker.post_review("7", "looks good", run="r1", node="reviewer")

    assert not any(c[:3] == ["gh", "pr", "review"] for c in runner.calls)


def test_gh_failure_raises_tracker_error_without_body_in_message():
    runner = FakeRunner()
    runner.script(
        ["gh", "issue", "comment"], returncode=4, stderr="not logged in"
    )
    runner.script(COMMENTS_ENDPOINT, stdout="[]")
    tracker = GhTracker(runner)

    with pytest.raises(TrackerError, match="not logged in"):
        tracker.comment_work_item("42", "gate", run="r", node="n")


def test_subprocess_runner_runs_real_commands(tmp_path):
    run = subprocess_runner(tmp_path)
    result = run([sys.executable, "-c", "print('hi')"])
    assert result.stdout.strip() == "hi"
    assert result.returncode == 0


def test_ensure_label_is_noop_when_label_exists():
    runner = FakeRunner()
    runner.script(
        ["gh", "label", "list", "--json", "name"], stdout='[{"name":"k0:stuck"}]'
    )
    tracker = GhTracker(runner)

    tracker.ensure_label("k0:stuck")

    assert not any(c[:3] == ["gh", "label", "create"] for c in runner.calls)


def test_comment_change_is_noop_when_marker_present():
    runner = FakeRunner()
    marker = comment_marker("r1", "review_gate")
    runner.script(
        ["gh", "api", "--paginate", "repos/{owner}/{repo}/issues/7/comments"],
        stdout=json.dumps(
            [
                {
                    "body": f"{marker}\n\ngate ok",
                    "user": {"login": "k0"},
                    "author_association": "OWNER",
                }
            ]
        ),
    )
    tracker = GhTracker(runner)

    tracker.comment_change("7", "gate ok", run="r1", node="review_gate")

    assert not any(c[:3] == ["gh", "pr", "comment"] for c in runner.calls)
