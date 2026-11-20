"""#173: private run files stay out of k0 commits and pushes.

Acceptance at the run-module seam (``start``/``resume``/``abandon``):
real temp git repos behind a local bare origin, ``.k0-mem/`` without
the ``.gitignore`` ``k0 init`` writes — the checkout shape that leaked
session transcripts, event logs and checkpoint databases into a pushed
PR in the November E2E. The fake tracker runs real git, so the commit
and push boundaries are exercised for real; the ``gh`` side stays a
recorded fake.
"""

import subprocess
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from fakes import STUB_TEST_CMD, FakeChatModel, FakeTracker, cfg_with
from k0ntrol.tracker import TrackerError
from k0ntrol.tracker.local_files import commit as local_commit
from k0ntrol.tracker.local_files import is_local_only
from k0ntrol.workflow import Outcome, Pending, RunRefused, abandon, resume, start

SECRET = "PRIVATE-RUN-BYTES"

#: The local-only set a live checkout carries under ``.k0-mem/``.
SEEDED_PRIVATE = (
    ".k0-mem/sessions/run-9.txt",
    ".k0-mem/index.sqlite",
    ".k0-mem/index.sqlite-wal",
    ".k0-mem/snapshots/run-9/tree/calc.py",
)
RUN_PRIVATE = (".k0-mem/events.jsonl", ".k0-mem/checkpoints.sqlite")
PRIVATE_PATHS = SEEDED_PRIVATE + RUN_PRIVATE

TEST_SRC = "from calc import answer\n\n\ndef test_answer():\n    assert answer() == 42\n"
CALC_SRC = "def answer():\n    return 42\n"
WORKFLOW_SRC = "name: ci\non: push\njobs: {}\n"
BRANCH = "k0/1-calc-answers-42"


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def runner_for(root: Path):
    def run(argv: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            argv, cwd=root, capture_output=True, text=True, check=False
        )

    return run


def repo_with_origin(root: Path, origin: Path) -> None:
    """A repo on ``main`` pushed to a local bare ``origin`` — no
    ``.k0-mem/.gitignore``, the November-E2E checkout shape (#173)."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    git(root, "config", "user.email", "t@t")
    git(root, "config", "user.name", "t")
    (root / "README.md").write_text("tmp repo\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "init")
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    git(root, "remote", "add", "origin", str(origin))
    git(root, "push", "-q", "-u", "origin", "main")


def seed_private_files(root: Path) -> None:
    """The run-private files of a live checkout, each carrying a marker
    that pushed commits and refusals must never show."""
    for relative in SEEDED_PRIVATE:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{SECRET} {relative}\n", encoding="utf-8")


def seed_shared_memory(root: Path) -> None:
    """``.k0-mem/`` project memory and config: shared on purpose (#173)."""
    (root / ".k0-mem" / "General_mem").mkdir(parents=True, exist_ok=True)
    (root / ".k0-mem" / "config.yaml").write_text(
        "tracker: github\n", encoding="utf-8"
    )
    (root / ".k0-mem" / "General_mem" / "Mem.md").write_text(
        "# Mem\n\n- a fact\n", encoding="utf-8"
    )


def item_cfg(**overrides):
    cfg = cfg_with(
        {
            "implementer": {
                "provider": "anthropic",
                "model": "m",
                "model_calls": 20,
            },
            "reviewer": {
                "provider": "anthropic",
                "model": "m",
                "model_calls": 25,
            },
        }
    )
    cfg.update(
        {
            "test_cmd": STUB_TEST_CMD,
            "gates": [],
            "max_attempts": 2,
            "test_timeout": 60,
            "tracker": "github",
        }
    )
    cfg.update(overrides)
    return cfg


def models_init(*models):
    queue = list(models)

    def init(model, *, model_provider=None, **kwargs):
        assert queue, "init_chat_model called more times than scripted"
        return queue.pop(0)

    return init


def ready_item(tracker, title="Calc answers 42", body="build the calc module"):
    tracker.ensure_label("ready-for-agent")
    item = tracker.create_work_item(title, body)
    tracker.add_label(item.ref, "ready-for-agent")
    tracker.calls.clear()
    return item


def run_item(root, cfg, tracker, *models, input="#1", **kwargs):
    kwargs.setdefault("check", lambda *a, **k: [])
    kwargs.setdefault("scan", lambda root: "scanned")
    return start(
        "implement",
        input,
        root=root,
        cfg=cfg,
        init=models_init(*models),
        tracker=tracker,
        **kwargs,
    )


def write_call(path, content, call_id="w1"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"path": path, "content": content},
                "id": call_id,
            }
        ],
    )


def run_tests_call(call_id="r1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": "run_tests", "args": {"paths": []}, "id": call_id}],
    )


def builder_model(*extra):
    return FakeChatModel(
        messages=[
            write_call("tests/test_calc.py", TEST_SRC),
            run_tests_call(),
            write_call("calc.py", CALC_SRC, "w2"),
            run_tests_call("r2"),
            AIMessage(content="built calc"),
            *extra,
        ]
    )


def workflow_builder():
    """An IMPLEMENTER that also writes a ``.github/`` file: the run
    pauses at the CI gate, where a test can act on the branch."""
    return FakeChatModel(
        messages=[
            write_call("tests/test_calc.py", TEST_SRC),
            run_tests_call(),
            write_call("calc.py", CALC_SRC, "w2"),
            write_call(".github/workflows/ci.yml", WORKFLOW_SRC, "w3"),
            run_tests_call("r2"),
            AIMessage(content="built calc and the workflow"),
        ]
    )


def review_call(summary, findings, call_id):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "Review",
                "args": {"summary": summary, "findings": findings},
                "id": call_id,
            }
        ],
    )


def passing_reviewer():
    return FakeChatModel(
        messages=[
            review_call("conventions pass", [], "c1"),
            review_call("contract pass", [], "c2"),
        ]
    )


def outgoing_shas(root: Path, branch: str = BRANCH) -> list[str]:
    """The commits a push of ``branch`` would send to origin."""
    return git(
        root, "rev-list", branch, "--not", "--remotes=origin"
    ).stdout.split()


def outgoing_trees(root: Path, branch: str = BRANCH) -> list[list[str]]:
    """Every file in each outgoing commit's tree."""
    return [
        git(root, "ls-tree", "-r", sha, "--name-only").stdout.splitlines()
        for sha in outgoing_shas(root, branch)
    ]


def outgoing_touched(root: Path, branch: str = BRANCH) -> set[str]:
    """Every path any outgoing commit touched — the history view."""
    touched: set[str] = set()
    for sha in outgoing_shas(root, branch):
        names = git(
            root,
            "diff-tree",
            "--root",
            "--no-commit-id",
            "-r",
            "--name-only",
            sha,
        ).stdout
        touched.update(names.splitlines())
    return touched


@pytest.mark.parametrize(
    ("path", "local_only"),
    [
        (".k0-mem/sessions/run-9.txt", True),
        (".k0-mem/snapshots/r9/tree/calc.py", True),
        (".k0-mem/nested/sessions/x", True),
        (".k0-mem/events.jsonl", True),
        (".k0-mem/events.archive.jsonl", True),
        (".k0-mem/nested/events.jsonl", True),
        (".k0-mem/checkpoints.sqlite", True),
        (".k0-mem/checkpoints.sqlite-wal", True),
        (".k0-mem/index.sqlite", True),
        (".k0-mem/config.yaml", False),
        (".k0-mem/General_mem/Mem.md", False),
        (".k0-mem/Project_context/repo/graph.json", False),
        (".k0-mem/.gitignore", False),
        ("calc.py", False),
        (".k0-mem-events.jsonl", False),
        ("src/.k0-mem/events.jsonl", False),
    ],
)
def test_the_local_only_set(path, local_only):
    """The private set mirrors the gitignore `k0 init` writes — plus
    worktree snapshots — and never the shared memory or config (#173)."""
    assert is_local_only(path) is local_only


def test_start_ships_the_change_without_the_private_run_files(tmp_path):
    """Start: with ``.k0-mem/.gitignore`` absent, transcripts, event
    logs, databases and worktree snapshots never enter an outgoing
    commit — while the intentional change and the shared ``.k0-mem/``
    memory still ship (#173)."""
    root, origin = tmp_path / "repo", tmp_path / "origin.git"
    repo_with_origin(root, origin)
    seed_shared_memory(root)
    seed_private_files(root)
    tracker = FakeTracker(git_runner=runner_for(root))
    ready_item(tracker)

    result = run_item(
        root, item_cfg(), tracker, builder_model(), passing_reviewer()
    )

    assert isinstance(result, Pending) and result.gate == "review"
    trees = outgoing_trees(root)
    assert trees, "the run committed nothing"
    for tree in trees:
        for private in PRIVATE_PATHS:
            assert private not in tree
    touched = outgoing_touched(root)
    for private in PRIVATE_PATHS:
        assert private not in touched
    # the intentional change and shared project memory did ship
    assert "calc.py" in touched
    assert "tests/test_calc.py" in touched
    assert ".k0-mem/config.yaml" in touched
    assert ".k0-mem/General_mem/Mem.md" in touched
    # the private bytes stayed on disk, untracked as the run found them
    status = git(root, "status", "--porcelain", "-uall").stdout
    for private in PRIVATE_PATHS:
        assert (root / private).exists()
        assert f"?? {private}" in status or f" M {private}" in status


def test_start_refuses_a_run_with_a_staged_private_file(tmp_path):
    """Start: a local-only path the developer staged makes the run's
    first commit refuse — paths named, index and bytes untouched, and
    nothing committed (#173)."""
    root, origin = tmp_path / "repo", tmp_path / "origin.git"
    repo_with_origin(root, origin)
    seed_shared_memory(root)
    seed_private_files(root)
    staged_path = ".k0-mem/sessions/run-9.txt"
    git(root, "add", staged_path)
    tracker = FakeTracker(git_runner=runner_for(root))
    ready_item(tracker)

    with pytest.raises(TrackerError) as error:
        run_item(root, item_cfg(), tracker, builder_model())

    assert staged_path in str(error.value)
    assert SECRET not in str(error.value)
    # index and bytes are exactly as found: still staged, still local
    status = git(root, "status", "--porcelain", "-uall").stdout
    assert f"A  {staged_path}" in status
    assert (root / staged_path).read_text(encoding="utf-8") == (
        f"{SECRET} {staged_path}\n"
    )
    # nothing was committed — HEAD still sits on the pushed base
    assert not git(root, "log", "--format=%H", "origin/main..HEAD").stdout.strip()


def test_start_refuses_a_run_whose_head_carries_a_private_path(tmp_path):
    """Start: a private path already in HEAD's tree would ride into
    every new commit's tree — the run refuses, naming paths, and the
    index (a staged untrack) is left untouched (#173)."""
    root, origin = tmp_path / "repo", tmp_path / "origin.git"
    repo_with_origin(root, origin)
    seed_shared_memory(root)
    leaked = ".k0-mem/events.jsonl"
    (root / leaked).write_text(f"{SECRET} old\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "an old leak already on origin")
    git(root, "push", "-q", "origin", "main")
    git(root, "rm", "--cached", "-q", leaked)
    tracker = FakeTracker(git_runner=runner_for(root))
    ready_item(tracker)

    with pytest.raises(TrackerError) as error:
        run_item(root, item_cfg(), tracker, builder_model())

    assert leaked in str(error.value)
    assert SECRET not in str(error.value)
    # the staged untrack and the log are exactly as found
    staged = git(root, "diff", "--cached", "--name-status").stdout
    assert f"D\t{leaked}" in staged
    assert not git(root, "log", "--format=%H", "origin/main..HEAD").stdout.strip()
    assert (root / leaked).exists()


def test_a_quoted_private_path_never_enters_a_commit(tmp_path):
    """Non-ASCII and space-carrying private names — the ones porcelain
    quotes and escapes — never enter an outgoing commit (#173)."""
    root, origin = tmp_path / "repo", tmp_path / "origin.git"
    repo_with_origin(root, origin)
    seed_shared_memory(root)
    quoted = (".k0-mem/events-ñ run.jsonl", ".k0-mem/sessions/a b -> c.txt")
    for relative in quoted:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{SECRET} {relative}\n", encoding="utf-8")
    tracker = FakeTracker(git_runner=runner_for(root))
    ready_item(tracker)

    result = run_item(
        root, item_cfg(), tracker, builder_model(), passing_reviewer()
    )

    assert isinstance(result, Pending) and result.gate == "review"
    for tree in outgoing_trees(root):
        for private in quoted:
            assert private not in tree
    assert "calc.py" in outgoing_touched(root)
    for private in quoted:
        assert (root / private).exists()


def test_a_conflicted_private_path_refuses_the_commit(tmp_path):
    """A local-only path with unresolved index entries cannot be put
    back as found — the commit refuses, naming paths, and stages or
    stashes nothing (#173)."""
    root = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    git(root, "config", "user.email", "t@t")
    git(root, "config", "user.name", "t")
    path = ".k0-mem/events.jsonl"
    (root / ".k0-mem").mkdir()
    (root / path).write_text("ours\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "base")
    git(root, "checkout", "-qb", "side")
    (root / path).write_text("theirs\n", encoding="utf-8")
    git(root, "commit", "-qam", "theirs")
    git(root, "checkout", "-q", "main")
    (root / path).write_text("ours-edited\n", encoding="utf-8")
    git(root, "commit", "-qam", "ours")
    assert git(root, "merge", "side").returncode != 0
    (root / "feature.py").write_text("x = 1\n", encoding="utf-8")
    before = git(root, "log", "--format=%H").stdout

    with pytest.raises(TrackerError) as error:
        local_commit(runner_for(root), "#1: work")

    assert path in str(error.value)
    assert "theirs" not in str(error.value)
    # nothing committed, and the conflicted index is exactly as found
    assert git(root, "log", "--format=%H").stdout == before
    assert git(root, "diff", "--name-only", "--diff-filter=U").stdout.strip() == path


def refused_push(tmp_path):
    """A run paused at the CI gate whose branch was then poisoned —
    a private file committed then removed, so the net diff looks clean.
    Returns (root, tracker, pending, outgoing-log-before)."""
    root, origin = tmp_path / "repo", tmp_path / "origin.git"
    repo_with_origin(root, origin)
    seed_shared_memory(root)
    seed_private_files(root)
    tracker = FakeTracker(git_runner=runner_for(root))
    ready_item(tracker)
    pending = run_item(root, item_cfg(), tracker, workflow_builder())
    assert isinstance(pending, Pending) and pending.gate == "ci"

    git(root, "add", ".k0-mem/events.jsonl")
    git(root, "commit", "-qm", "leak the event log")
    git(root, "rm", "-q", ".k0-mem/events.jsonl")
    git(root, "commit", "-qm", "and remove it again")
    before = git(root, "log", "--format=%H", "origin/main..HEAD").stdout
    return root, tracker, pending, before


def test_resume_refuses_the_push_when_outgoing_history_leaks(tmp_path):
    """Resume: outgoing commits that add then remove a private file
    still expose it in history — the push refuses, naming paths and
    never contents, and nothing is pushed or opened (#173)."""
    root, tracker, pending, before = refused_push(tmp_path)

    with pytest.raises(TrackerError) as error:
        resume(
            pending.run,
            "ok",
            root=root,
            cfg=item_cfg(),
            init=models_init(),
            tracker=tracker,
        )

    assert ".k0-mem/events.jsonl" in str(error.value)
    assert SECRET not in str(error.value)
    assert not [c for c in tracker.calls if c[0] in ("push", "open_change")]
    assert (
        git(
            root,
            "rev-parse",
            "--verify",
            "-q",
            f"refs/remotes/origin/{BRANCH}",
        ).returncode
        != 0
    )
    # the poisoned commits are untouched — k0 never rewrites history
    assert git(root, "log", "--format=%H", "origin/main..HEAD").stdout == before


def test_resume_refuses_the_push_over_a_merge_inherited_private_path(tmp_path):
    """A private path inherited through a merge: no outgoing commit
    touched it — the introducing commit is already on origin — but the
    merge's tree carries it. The push still refuses (#173)."""
    root, origin = tmp_path / "repo", tmp_path / "origin.git"
    repo_with_origin(root, origin)
    seed_shared_memory(root)
    seed_private_files(root)
    tracker = FakeTracker(git_runner=runner_for(root))
    ready_item(tracker)
    pending = run_item(root, item_cfg(), tracker, workflow_builder())
    assert isinstance(pending, Pending) and pending.gate == "ci"

    # a private file lands on a side branch that is already on origin,
    # then merges in — the leak enters the tree without an outgoing touch
    inherited = ".k0-mem/sessions/leaked.txt"
    git(root, "checkout", "-qb", "leaked", "main")
    (root / inherited).write_text(f"{SECRET} merge\n", encoding="utf-8")
    git(root, "add", inherited)
    git(root, "commit", "-qm", "leak on a side branch")
    git(root, "push", "-q", "origin", "leaked")
    git(root, "checkout", "-q", BRANCH)
    git(root, "merge", "-q", "--no-edit", "leaked")
    before = git(root, "log", "--format=%H", "origin/main..HEAD").stdout

    with pytest.raises(TrackerError) as error:
        resume(
            pending.run,
            "ok",
            root=root,
            cfg=item_cfg(),
            init=models_init(),
            tracker=tracker,
        )

    assert inherited in str(error.value)
    assert SECRET not in str(error.value)
    assert not [c for c in tracker.calls if c[0] in ("push", "open_change")]
    assert git(root, "log", "--format=%H", "origin/main..HEAD").stdout == before


def test_resume_refuses_the_push_despite_a_forged_origin_ref(tmp_path):
    """A local ``refs/remotes/origin/`` ref pretending the remote already
    has the poisoned commits must not shrink the check: the outgoing
    range is compared against the live origin refs (#173)."""
    root, tracker, pending, before = refused_push(tmp_path)
    head = git(root, "rev-parse", "HEAD").stdout.strip()
    git(root, "update-ref", f"refs/remotes/origin/{BRANCH}", head)

    with pytest.raises(TrackerError) as error:
        resume(
            pending.run,
            "ok",
            root=root,
            cfg=item_cfg(),
            init=models_init(),
            tracker=tracker,
        )

    assert ".k0-mem/events.jsonl" in str(error.value)
    assert not [c for c in tracker.calls if c[0] in ("push", "open_change")]
    # the forged local ref is left alone — k0 never rewrites refs
    assert (
        git(root, "rev-parse", f"refs/remotes/origin/{BRANCH}").stdout.strip()
        == head
    )


def test_abandon_after_the_refusal_leaves_files_and_commits(tmp_path):
    """Abandon: dropping the refused run leaves every file and every
    outgoing commit exactly where the refusal left them (#173)."""
    root, tracker, pending, before = refused_push(tmp_path)
    with pytest.raises(TrackerError):
        resume(
            pending.run,
            "ok",
            root=root,
            cfg=item_cfg(),
            init=models_init(),
            tracker=tracker,
        )

    abandon(pending.run, root=root)

    assert git(root, "log", "--format=%H", "origin/main..HEAD").stdout == before
    assert (root / ".k0-mem/index.sqlite").read_text(
        encoding="utf-8"
    ).startswith(SECRET)
    assert (root / ".k0-mem/snapshots/run-9/tree/calc.py").exists()
    assert not [c for c in tracker.calls if c[0] in ("push", "open_change")]
    with pytest.raises(RunRefused, match="no run"):
        resume(
            pending.run,
            None,
            root=root,
            cfg=item_cfg(),
            init=models_init(),
            tracker=tracker,
        )


def test_a_dirty_task_run_stays_local_and_keeps_the_files(tmp_path):
    """``/implement <task>`` on a dirty working tree is non-publishing
    by design: no commit, no push, and the pre-existing private files
    and edits survive the run (#173)."""
    root, origin = tmp_path / "repo", tmp_path / "origin.git"
    repo_with_origin(root, origin)
    seed_private_files(root)
    (root / "scratch.txt").write_text("keep me\n", encoding="utf-8")
    (root / "README.md").write_text("HUMAN EDIT\n", encoding="utf-8")
    tracker = FakeTracker(git_runner=runner_for(root))

    pending = start(
        "implement",
        "add a calc module that answers 42",
        root=root,
        cfg=item_cfg(),
        check=lambda *a, **k: [],
        scan=lambda root: "scanned",
        init=models_init(
            FakeChatModel(
                messages=[
                    AIMessage(
                        content="seam: calc.answer() -> int\n"
                        "seam: tests/test_calc.py"
                    )
                ]
            )
        ),
        tracker=tracker,
    )
    assert isinstance(pending, Pending) and pending.gate == "seam"

    pending = resume(
        pending.run,
        "ok",
        root=root,
        cfg=item_cfg(),
        init=models_init(builder_model()),
        tracker=tracker,
    )
    assert isinstance(pending, Pending) and pending.gate == "diff"

    outcome = resume(
        pending.run,
        "y",
        root=root,
        cfg=item_cfg(),
        init=models_init(),
        tracker=tracker,
    )
    assert isinstance(outcome, Outcome) and outcome.outcome == "done"

    # non-publishing: the tracker was never touched and nothing committed
    assert tracker.calls == []
    assert not git(
        root, "log", "--format=%H", "origin/main..HEAD"
    ).stdout.strip()
    # pre-existing files and edits preserved; the change stays local
    assert (root / "scratch.txt").read_text() == "keep me\n"
    assert (root / "README.md").read_text() == "HUMAN EDIT\n"
    assert (root / "calc.py").exists()
    for private in SEEDED_PRIVATE:
        assert (root / private).exists()
