"""The /implement <task> working-tree graph through the run module seam.

Same seams as the /fast and /full suites: a real temp git repo, a real
``test_cmd``, scripted ``FakeChatModel``s behind the ``init`` seam and a
``FakeTracker`` the run must never touch — the working-tree path has no
tracker, branch or pull request (spec #102, ticket #116).
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from fakes import STUB_TEST_CMD, FakeChatModel, FakeTracker, cfg_with
from k0ntrol.workflow import (
    Outcome,
    Pending,
    RunRefused,
    Unfinished,
    resume,
    start,
)
from k0ntrol.workflow.change import worktree_snapshot_dir

SEAMS = "seam: calc.answer() -> int\nseam: tests/test_calc.py"
TEST_SRC = "from calc import answer\n\n\ndef test_answer():\n    assert answer() == 42\n"
CALC_SRC = "def answer():\n    return 42\n"


def implement_cfg(**overrides):
    cfg = cfg_with(
        {
            "implementer": {
                "provider": "anthropic",
                "model": "m",
                "model_calls": 20,
            }
        }
    )
    cfg.update(
        {
            "test_cmd": STUB_TEST_CMD,
            "gates": [],
            "max_attempts": 2,
            "test_timeout": 60,
        }
    )
    cfg.update(overrides)
    return cfg


def git_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "README.md").write_text("tmp repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=root,
        check=True,
    )


def models_init(*models):
    """An ``init_chat_model`` seam handing out scripted fakes in call order."""
    queue = list(models)

    def init(model, *, model_provider=None, **kwargs):
        assert queue, "init_chat_model called more times than scripted"
        return queue.pop(0)

    return init


def run_implement(root, cfg, *models, **kwargs):
    """``start('implement', ...)`` with the doctor and rescan seams stubbed."""
    kwargs.setdefault("check", lambda *a, **k: [])
    kwargs.setdefault("scan", lambda root: "scanned")
    kwargs.setdefault("tracker", FakeTracker())
    return start(
        "implement",
        "add a calc module that answers 42",
        root=root,
        cfg=cfg,
        init=models_init(*models),
        **kwargs,
    )


def seams_model(*extra):
    """An IMPLEMENTER fake whose first call proposes the seams as text."""
    return FakeChatModel(
        messages=[AIMessage(content=SEAMS), *extra]
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


def events(root):
    path = root / ".k0-mem" / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- refusals ----------------------------------------------------------------


def test_implement_without_a_task_is_refused(tmp_path):
    with pytest.raises(RunRefused, match="usage"):
        start("implement", "   ", root=tmp_path, cfg=implement_cfg())


def test_implement_hash_input_is_refused(tmp_path):
    """``/implement #N`` is the tracker path's form, not this ticket's."""
    with pytest.raises(RunRefused, match="usage"):
        start("implement", "#12", root=tmp_path, cfg=implement_cfg())


def test_bulk_read_doctor_error_refuses_the_run(tmp_path):
    """The run roster gates on ``helpers.bulk_read`` like /fast does (#61)."""
    git_repo(tmp_path)
    captured = {}

    def bulk_read_fails(cfg, *, names=None, **kwargs):
        captured["names"] = tuple(names or ())
        return [
            type(
                "C",
                (),
                {
                    "status": "error",
                    "name": "helpers.bulk_read",
                    "detail": "unreachable",
                },
            )()
        ]

    with pytest.raises(RunRefused, match="unreachable"):
        start(
            "implement",
            "add a calc module",
            root=tmp_path,
            cfg=implement_cfg(),
            check=bulk_read_fails,
            scan=lambda root: "scanned",
            init=models_init(seams_model()),
        )
    assert "helpers.bulk_read" in captured["names"]
    assert "agents.implementer" in captured["names"]
    assert not (tmp_path / ".k0-mem" / "events.jsonl").exists()


# --- the seam gate -------------------------------------------------------------


def test_implement_stops_at_the_seam_gate_before_any_test(tmp_path):
    """IMPLEMENTER proposes the seams; the run pauses before any test is
    written, and the tracker is never touched (ticket #116)."""
    git_repo(tmp_path)
    tracker = FakeTracker()
    seams = seams_model()

    result = run_implement(tmp_path, implement_cfg(), seams, tracker=tracker)

    assert isinstance(result, Pending)
    assert result.gate == "seam"
    assert result.answers == ("ok", "redo")
    assert SEAMS in result.detail
    # the proposal is IMPLEMENTER's answer, shown to the human
    assert seams.invocations, "the seam proposal never ran"
    # no test file exists yet — the gate comes before the first test
    assert not list(tmp_path.rglob("test_*.py"))
    assert tracker.calls == []


def test_the_seam_gate_shows_the_task_context_to_the_proposal(tmp_path):
    git_repo(tmp_path)
    seams = seams_model()

    run_implement(tmp_path, implement_cfg(), seams)

    brief = seams.invocations[0][-1].content
    assert "add a calc module that answers 42" in brief


def test_implement_outside_a_git_repo_is_refused(tmp_path):
    """The ``n`` restore is built on ``git ls-files`` — no repo, no
    snapshot, so the run refuses instead of silently keeping changes."""
    with pytest.raises(RunRefused, match="git"):
        run_implement(tmp_path, implement_cfg(), seams_model())


def test_the_seam_proposal_cannot_write_files(tmp_path):
    """The proposal phase is read-only in code, not just in the prompt:
    a disobedient IMPLEMENTER has no write tools to call (#116)."""
    git_repo(tmp_path)
    disobedient = FakeChatModel(
        messages=[
            # the green guard allows test writes — only a missing tool stops it
            write_call("tests/sneaky_test.py", "must never exist"),
            AIMessage(content=SEAMS),
        ]
    )

    pending = run_implement(tmp_path, implement_cfg(), disobedient)

    assert isinstance(pending, Pending) and pending.gate == "seam"
    assert not (tmp_path / "tests" / "sneaky_test.py").exists()


def test_seam_gate_redo_reruns_the_proposal_with_the_feedback(tmp_path):
    git_repo(tmp_path)
    first = seams_model()
    pending = run_implement(tmp_path, implement_cfg(), first)
    assert isinstance(pending, Pending) and pending.gate == "seam"

    second = FakeChatModel(messages=[AIMessage(content="seam: calc.py only")])
    redo = resume(
        pending.run,
        "redo name the module boundary too",
        root=tmp_path,
        cfg=implement_cfg(),
        init=models_init(second),
    )

    assert isinstance(redo, Pending)
    assert redo.gate == "seam"
    assert "seam: calc.py only" in redo.detail
    brief = second.invocations[0][-1].content
    assert "name the module boundary too" in brief
    # still no test written — every seam-gate round precedes the build
    assert not list(tmp_path.rglob("test_*.py"))


# --- build → VALIDATOR → diff gate ----------------------------------------------


def builder_model(*extra):
    """An IMPLEMENTER fake running one red-to-green slice to green."""
    return FakeChatModel(
        messages=[
            write_call("tests/test_calc.py", TEST_SRC),
            run_tests_call(),
            write_call("calc.py", CALC_SRC),
            run_tests_call("r2"),
            AIMessage(content="built calc"),
            *extra,
        ]
    )


def to_diff_gate(tmp_path, cfg=None, builder=None):
    """``/implement <task>`` past the seam gate to the diff gate."""
    cfg = cfg or implement_cfg()
    tracker = FakeTracker()
    pending = run_implement(tmp_path, cfg, seams_model(), tracker=tracker)
    assert isinstance(pending, Pending) and pending.gate == "seam"
    builder = builder or builder_model()
    result = resume(
        pending.run,
        "ok",
        root=tmp_path,
        cfg=cfg,
        init=models_init(builder),
        tracker=tracker,
    )
    return tracker, builder, result


def test_implement_runs_to_the_diff_gate_and_done(tmp_path):
    git_repo(tmp_path)
    tracker, builder, result = to_diff_gate(tmp_path)

    assert isinstance(result, Pending)
    assert result.gate == "diff"
    assert result.answers == ("y", "n")
    assert (tmp_path / "tests" / "test_calc.py").read_text() == TEST_SRC
    assert (tmp_path / "calc.py").read_text() == CALC_SRC
    assert tracker.calls == []  # the working-tree path never touches it

    outcome = resume(result.run, "y", root=tmp_path, cfg=implement_cfg(), init=models_init())

    assert isinstance(outcome, Outcome)
    assert outcome.outcome == "done"


def test_diff_gate_n_rejects_and_restores_the_working_tree(tmp_path):
    """``n`` leaves the tree as it was before the run: agent files gone,
    the human's own uncommitted edits kept (#116)."""
    git_repo(tmp_path)
    (tmp_path / "existing.py").write_text("ORIGINAL\n", encoding="utf-8")
    (tmp_path / "realdir").mkdir()
    (tmp_path / "realdir" / "inner.txt").write_text("inner\n", encoding="utf-8")
    (tmp_path / "dirlink").symlink_to("realdir", target_is_directory=True)
    subprocess.run(
        ["git", "add", "existing.py", "realdir", "dirlink"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "add"],
        cwd=tmp_path,
        check=True,
    )
    # the human's uncommitted state before the run: a dirty edit + untracked
    (tmp_path / "existing.py").write_text("HUMAN EDIT\n", encoding="utf-8")
    (tmp_path / "scratch.txt").write_text("keep me\n", encoding="utf-8")

    # IMPLEMENTER also edits the tracked file — a code write while red
    builder = FakeChatModel(
        messages=[
            write_call("tests/test_calc.py", TEST_SRC),
            run_tests_call(),
            write_call("calc.py", CALC_SRC, "w2"),
            write_call("existing.py", "AGENT OVERWROTE\n", "w3"),
            run_tests_call("r2"),
            AIMessage(content="built calc"),
        ]
    )
    tracker, builder, result = to_diff_gate(tmp_path, builder=builder)
    assert isinstance(result, Pending) and result.gate == "diff"
    assert builder.invocations, "the build never ran"
    assert (tmp_path / "existing.py").read_text() == "AGENT OVERWROTE\n"

    outcome = resume(result.run, "n", root=tmp_path, cfg=implement_cfg(), init=models_init())

    assert isinstance(outcome, Outcome)
    assert outcome.outcome == "rejected"
    assert not (tmp_path / "tests" / "test_calc.py").exists()
    assert not (tmp_path / "tests").exists()  # the emptied dir is pruned
    assert not (tmp_path / "calc.py").exists()
    assert (tmp_path / "existing.py").read_text() == "HUMAN EDIT\n"
    assert (tmp_path / "scratch.txt").read_text() == "keep me\n"
    assert (tmp_path / "README.md").read_text() == "tmp repo\n"
    # a tracked dir symlink the run never touched survives the restore
    assert (tmp_path / "dirlink").is_symlink()
    assert os.readlink(tmp_path / "dirlink") == "realdir"
    # a finished run drops its snapshot
    assert not worktree_snapshot_dir(tmp_path, outcome.run).exists()


def test_diff_gate_n_restores_a_root_manifest_json(tmp_path):
    """A root ``manifest.json`` shares its name with the snapshot's own
    metadata: ``n`` must return the human's bytes, not the snapshot
    manifest — backed-up files live in a payload dir, not beside it."""
    git_repo(tmp_path)
    manifest = tmp_path / "manifest.json"  # the human's uncommitted settings
    manifest.write_text('{"settings": "human"}\n', encoding="utf-8")

    builder = FakeChatModel(
        messages=[
            write_call("tests/test_calc.py", TEST_SRC),
            run_tests_call(),
            write_call("calc.py", CALC_SRC, "w2"),
            write_call("manifest.json", '{"files": "agent"}\n', "w3"),
            run_tests_call("r2"),
            AIMessage(content="built calc"),
        ]
    )
    _, _, result = to_diff_gate(tmp_path, builder=builder)
    assert isinstance(result, Pending) and result.gate == "diff"
    assert manifest.read_text() == '{"files": "agent"}\n'

    outcome = resume(result.run, "n", root=tmp_path, cfg=implement_cfg(), init=models_init())

    assert outcome.outcome == "rejected"
    assert manifest.read_text() == '{"settings": "human"}\n'


def test_a_broken_event_log_keeps_the_resumable_runs_snapshot(tmp_path):
    """An event-write failure ends the call stuck but leaves the
    checkpoint resumable — the snapshot must stay so a later ``n`` can
    still roll the run's changes back."""
    git_repo(tmp_path)
    (tmp_path / "existing.py").write_text("HUMAN EDIT\n", encoding="utf-8")
    _, _, result = to_diff_gate(tmp_path)
    assert isinstance(result, Pending) and result.gate == "diff"

    # events.jsonl as a directory makes every write fail (EventWriteError)
    log = tmp_path / ".k0-mem" / "events.jsonl"
    log.unlink()
    log.mkdir()
    stuck = resume(result.run, "n", root=tmp_path, cfg=implement_cfg(), init=models_init())

    assert isinstance(stuck, Outcome) and stuck.outcome == "stuck"
    assert worktree_snapshot_dir(tmp_path, result.run).exists()

    log.rmdir()  # log repaired: the next write recreates the file
    outcome = resume(result.run, "n", root=tmp_path, cfg=implement_cfg(), init=models_init())

    assert isinstance(outcome, Outcome)
    assert outcome.outcome == "rejected"
    assert (tmp_path / "existing.py").read_text() == "HUMAN EDIT\n"
    assert not (tmp_path / "calc.py").exists()
    assert not worktree_snapshot_dir(tmp_path, outcome.run).exists()


def test_diff_gate_n_keeps_a_submodule_checkout(tmp_path):
    """A gitlink is neither file nor symlink: the snapshot must record it
    as untouchable or ``n`` would ``rmtree`` the submodule's checkout."""
    git_repo(tmp_path)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "inner.txt").write_text("sub\n", encoding="utf-8")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(
        ["git", "update-index", "--add", "--cacheinfo", f"160000,{head},sub"],
        cwd=tmp_path,
        check=True,
    )

    _, _, result = to_diff_gate(tmp_path)
    assert isinstance(result, Pending) and result.gate == "diff"

    outcome = resume(result.run, "n", root=tmp_path, cfg=implement_cfg(), init=models_init())

    assert outcome.outcome == "rejected"
    assert (tmp_path / "sub" / "inner.txt").read_text() == "sub\n"


def test_diff_gate_y_keeps_the_change_and_drops_the_snapshot(tmp_path):
    git_repo(tmp_path)
    _, _, result = to_diff_gate(tmp_path)
    assert isinstance(result, Pending)

    outcome = resume(result.run, "y", root=tmp_path, cfg=implement_cfg(), init=models_init())

    assert outcome.outcome == "done"
    assert (tmp_path / "calc.py").read_text() == CALC_SRC
    assert not worktree_snapshot_dir(tmp_path, outcome.run).exists()


def test_validator_red_sends_implementer_back_with_the_output(tmp_path):
    """A red VALIDATOR reruns IMPLEMENTER with its raw output (#116)."""
    git_repo(tmp_path)
    cfg = implement_cfg()
    pending = run_implement(tmp_path, cfg, seams_model())
    builder = FakeChatModel(
        messages=[
            # attempt 1: test red, then code that still fails VALIDATOR
            write_call("tests/test_calc.py", TEST_SRC),
            run_tests_call(),
            write_call("calc.py", "def answer():\n    return 41\n", "w2"),
            run_tests_call("r2"),
            AIMessage(content="first pass"),
            # attempt 2: the fix
            write_call("calc.py", CALC_SRC, "w3"),
            run_tests_call("r3"),
            AIMessage(content="fixed"),
        ]
    )
    result = resume(
        pending.run, "ok", root=tmp_path, cfg=cfg, init=models_init(builder)
    )

    assert isinstance(result, Pending)
    assert result.gate == "diff"
    second_brief = builder.invocations[5][-1].content
    assert "VALIDATOR is still red" in second_brief
    assert "FAILED" in second_brief or "failed" in second_brief
    assert (tmp_path / "calc.py").read_text() == CALC_SRC


def test_validator_red_exhausts_attempts_to_limit_hit(tmp_path):
    git_repo(tmp_path)
    cfg = implement_cfg(max_attempts=2)
    pending = run_implement(tmp_path, cfg, seams_model())
    slice_ = [
        write_call("tests/test_calc.py", TEST_SRC),
        run_tests_call(),
        write_call("calc.py", "def answer():\n    return 41\n", "w2"),
        run_tests_call("r2"),
        AIMessage(content="pass"),
    ]
    builder = FakeChatModel(messages=[*slice_, *slice_])
    result = resume(
        pending.run, "ok", root=tmp_path, cfg=cfg, init=models_init(builder)
    )

    assert isinstance(result, Pending)
    assert result.gate == "limit_hit"

    outcome = resume(result.run, "stop", root=tmp_path, cfg=cfg, init=models_init())

    assert outcome.outcome == "stuck"
    assert outcome.reason == "max_attempts"
    assert outcome.node == "validator"


def test_validator_limit_retry_reruns_implementer(tmp_path):
    """``retry`` at the exhausted-VALIDATOR pause routes back to
    IMPLEMENTER — not to ``coder``, which this graph does not have."""
    git_repo(tmp_path)
    cfg = implement_cfg(max_attempts=2)
    pending = run_implement(tmp_path, cfg, seams_model())
    slice_ = [
        write_call("tests/test_calc.py", TEST_SRC),
        run_tests_call(),
        write_call("calc.py", "def answer():\n    return 41\n", "w2"),
        run_tests_call("r2"),
        AIMessage(content="pass"),
    ]
    builder = FakeChatModel(messages=[*slice_, *slice_])
    result = resume(
        pending.run, "ok", root=tmp_path, cfg=cfg, init=models_init(builder)
    )
    assert isinstance(result, Pending) and result.gate == "limit_hit"

    # a fresh RunContext means a fresh guard in green: re-prove red first
    fixed = FakeChatModel(
        messages=[
            write_call("tests/test_calc.py", TEST_SRC, "w8"),
            run_tests_call("r8"),
            write_call("calc.py", CALC_SRC, "w9"),
            run_tests_call("r9"),
            AIMessage(content="fixed for real"),
        ]
    )
    retried = resume(
        result.run, "retry", root=tmp_path, cfg=cfg, init=models_init(fixed)
    )

    assert isinstance(retried, Pending)
    assert retried.gate == "diff"


# --- events and limits -----------------------------------------------------------


def test_events_record_the_implement_run(tmp_path):
    git_repo(tmp_path)
    _, _, result = to_diff_gate(tmp_path)
    assert isinstance(result, Pending)
    resume(result.run, "y", root=tmp_path, cfg=implement_cfg(), init=models_init())

    log = events(tmp_path)
    kinds = [(e["kind"], e.get("gate"), e.get("result")) for e in log]
    assert log[0]["kind"] == "run_start"
    assert log[0]["run_kind"] == "implement"
    assert ("gate", "seam", "waiting") in kinds
    assert ("gate", "seam", "approve") in kinds
    assert ("gate", "validator", "pass") in kinds
    assert ("gate", "diff", "waiting") in kinds
    assert ("gate", "diff", "approve") in kinds
    assert kinds[-1][0] == "run_end"
    assert log[-1]["outcome"] == "done"


def test_a_second_implement_run_gets_the_unfinished_run_back(tmp_path):
    git_repo(tmp_path)
    pending = run_implement(tmp_path, implement_cfg(), seams_model())
    assert isinstance(pending, Pending)

    again = run_implement(tmp_path, implement_cfg(), seams_model())

    assert again == Unfinished(run=pending.run, node="seam_gate")


def test_seam_proposal_hitting_its_call_limit_pauses(tmp_path):
    git_repo(tmp_path)
    cfg = implement_cfg()
    cfg["agents"]["implementer"]["model_calls"] = 2
    looper = FakeChatModel(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"path": "README.md"},
                        "id": f"r{i}",
                    }
                ],
            )
            for i in range(4)
        ]
    )

    result = run_implement(tmp_path, cfg, looper)

    assert isinstance(result, Pending)
    assert result.gate == "limit_hit"
    assert result.answers == ("retry", "stop")
    assert "seam" in result.question

    outcome = resume(result.run, "stop", root=tmp_path, cfg=cfg, init=models_init())

    assert outcome.outcome == "stuck"
    assert outcome.reason == "model_call_limit"
    assert outcome.node == "seam"


def test_implement_public_functions_have_complete_annotations():
    import inspect

    from k0ntrol.workflow import implement

    for name, function in inspect.getmembers(implement, inspect.isfunction):
        if name.startswith("_") or function.__module__ != implement.__name__:
            continue
        signature = inspect.signature(function)
        assert signature.return_annotation is not inspect.Signature.empty, name
        assert all(
            parameter.annotation is not inspect.Parameter.empty
            for parameter in signature.parameters.values()
        ), name
