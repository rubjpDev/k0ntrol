"""The slice-recovery gate through the run module seam (#174).

A malformed test puts IMPLEMENTER in a red slice whose only repair the
guard refuses. Three denied test writes end the agent run and pause the
graph at the ``recovery`` gate: ``recover`` restores what the slice wrote
from the journal and retries IMPLEMENTER fresh, ``stop`` ends the run
``stuck`` with the working tree untouched, and a journal that is missing,
stale or conflicting refuses the restore before mutating anything.

Same seams as the other workflow suites: a real temp git repo, the stdlib
``stub_runner`` ``test_cmd``, scripted ``FakeChatModel``s behind the
``init`` seam.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from fakes import STUB_TEST_CMD, FakeChatModel, FakeTracker, cfg_with
from k0ntrol.workflow import Outcome, Pending, resume, start
from k0ntrol.workflow.change import worktree_snapshot_dir

SEAMS = "seam: calc.answer() -> int\nseam: tests/test_calc.py"
TEST_SRC = "from calc import answer\n\n\ndef test_answer():\n    assert answer() == 42\n"
CALC_SRC = "def answer():\n    return 42\n"
BAD_TEST = "import missing_module_xyz\n\n\ndef test_x():\n    pass\n"
GOOD_TEST = "def test_ok():\n    assert True\n"


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


def write_call(path, content, call_id=None):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"path": path, "content": content},
                "id": call_id or f"w_{path}",
            }
        ],
    )


def run_tests_call(call_id="r1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": "run_tests", "args": {"paths": []}, "id": call_id}],
    )


def seams_model():
    return FakeChatModel(messages=[AIMessage(content=SEAMS)])


def bad_builder():
    """IMPLEMENTER writes a malformed test, runs it red, then keeps trying
    test writes the guard keeps refusing — the November E2E loop."""
    return FakeChatModel(
        messages=[
            write_call("tests/test_calc.py", BAD_TEST),
            run_tests_call(),
            write_call("tests/test_calc.py", "fix one\n"),
            write_call("tests/test_calc.py", "fix two\n"),
            write_call("tests/test_calc.py", "fix three\n"),
        ]
    )


def green_builder():
    """A retried IMPLEMENTER writing a valid red assertion, then the code."""
    return FakeChatModel(
        messages=[
            write_call("tests/test_calc.py", TEST_SRC),
            run_tests_call(),
            write_call("calc.py", CALC_SRC),
            run_tests_call(),
            AIMessage(content="done"),
        ]
    )


def to_recovery_gate(tmp_path):
    """Run to the recovery gate; returns the ``Pending`` it pauses at."""
    git_repo(tmp_path)
    cfg = implement_cfg()
    pending = run_implement(tmp_path, cfg, seams_model())
    assert isinstance(pending, Pending) and pending.gate == "seam"
    result = resume(
        pending.run, "ok", root=tmp_path, cfg=cfg, init=models_init(bad_builder())
    )
    assert isinstance(result, Pending) and result.gate == "recovery"
    return result


def test_a_malformed_test_reaches_the_recovery_gate_bounded(tmp_path):
    """A bad import reaches recovery within a bounded number of calls —
    the gate, not forty refused edits (#174)."""
    pending = to_recovery_gate(tmp_path)
    assert pending.answers == ("recover", "stop")
    assert "tests/test_calc.py" in pending.detail


def test_recover_restores_the_slice_and_retries_to_green(tmp_path):
    """``recover`` rolls back only the active slice: the malformed test file
    goes away, the retry writes a valid red assertion then green (#174)."""
    pending = to_recovery_gate(tmp_path)
    result = resume(
        pending.run,
        "recover",
        root=tmp_path,
        cfg=implement_cfg(),
        init=models_init(green_builder()),
    )
    assert isinstance(result, Pending) and result.gate == "diff"
    assert (tmp_path / "calc.py").read_text() == CALC_SRC
    assert (tmp_path / "tests" / "test_calc.py").read_text() == TEST_SRC


def test_stop_leaves_the_tree_untouched(tmp_path):
    """Declined recovery: the run ends stuck, the slice's files stay (#174)."""
    pending = to_recovery_gate(tmp_path)
    outcome = resume(
        pending.run, "stop", root=tmp_path, cfg=implement_cfg(), init=models_init()
    )
    assert isinstance(outcome, Outcome) and outcome.outcome == "stuck"
    assert (tmp_path / "tests" / "test_calc.py").read_text() == BAD_TEST


def test_recover_twice_does_not_replay_the_restore(tmp_path):
    """A second malformed slice escalates again; the earlier journal was
    consumed, the new one restores only the new slice (#174)."""
    pending = to_recovery_gate(tmp_path)
    # First recovery: builder malforms the test a second time.
    second = resume(
        pending.run,
        "recover",
        root=tmp_path,
        cfg=implement_cfg(),
        init=models_init(bad_builder()),
    )
    assert isinstance(second, Pending) and second.gate == "recovery"
    final = resume(
        second.run,
        "recover",
        root=tmp_path,
        cfg=implement_cfg(),
        init=models_init(green_builder()),
    )
    assert isinstance(final, Pending) and final.gate == "diff"
    assert (tmp_path / "tests" / "test_calc.py").read_text() == TEST_SRC


def test_a_conflicting_journal_refuses_and_reasks(tmp_path):
    """A developer edit to a journaled path refuses the restore before any
    mutation and offers the gate again (#174)."""
    pending = to_recovery_gate(tmp_path)
    (tmp_path / "tests" / "test_calc.py").write_text("developer edit\n")
    refused = resume(
        pending.run, "recover", root=tmp_path, cfg=implement_cfg(), init=models_init()
    )
    assert isinstance(refused, Pending) and refused.gate == "recovery"
    assert "tests/test_calc.py" in refused.detail
    # The conflict changed nothing.
    assert (tmp_path / "tests" / "test_calc.py").read_text() == "developer edit\n"


def test_a_missing_journal_refuses_safely(tmp_path):
    """Journal deleted under the paused run: recover cannot restore (#174)."""
    pending = to_recovery_gate(tmp_path)
    journal = worktree_snapshot_dir(tmp_path, pending.run) / "slice"
    shutil.rmtree(journal)
    refused = resume(
        pending.run, "recover", root=tmp_path, cfg=implement_cfg(), init=models_init()
    )
    assert isinstance(refused, Pending) and refused.gate == "recovery"
    assert "missing" in refused.detail
    assert (tmp_path / "tests" / "test_calc.py").read_text() == BAD_TEST


def test_preexisting_files_survive_recovery(tmp_path):
    """Files the run did not write are not the slice's business (#174)."""
    git_repo(tmp_path)
    (tmp_path / "notes.txt").write_text("developer's notes\n")
    cfg = implement_cfg()
    pending = run_implement(tmp_path, cfg, seams_model())
    assert isinstance(pending, Pending) and pending.gate == "seam"
    result = resume(
        pending.run, "ok", root=tmp_path, cfg=cfg, init=models_init(bad_builder())
    )
    assert isinstance(result, Pending) and result.gate == "recovery"
    outcome = resume(
        result.run, "stop", root=tmp_path, cfg=implement_cfg(), init=models_init()
    )
    assert isinstance(outcome, Outcome) and outcome.outcome == "stuck"
    assert (tmp_path / "notes.txt").read_text() == "developer's notes\n"
    assert (tmp_path / "tests" / "test_calc.py").read_text() == BAD_TEST


def test_a_legitimately_red_slice_still_denies_test_writes(tmp_path):
    """A normal failing assertion: the guard's contract is unchanged —
    code writes pass, test writes are refused, no escalation (#174)."""
    git_repo(tmp_path)
    cfg = implement_cfg()
    builder = FakeChatModel(
        messages=[
            write_call("tests/test_calc.py", TEST_SRC),
            run_tests_call(),
            write_call("calc.py", CALC_SRC),
            run_tests_call(),
            AIMessage(content="done"),
        ]
    )
    pending = run_implement(tmp_path, cfg, seams_model())
    assert isinstance(pending, Pending) and pending.gate == "seam"
    result = resume(
        pending.run, "ok", root=tmp_path, cfg=cfg, init=models_init(builder)
    )
    assert isinstance(result, Pending) and result.gate == "diff"


def test_the_gate_event_names_the_recovery(tmp_path):
    pending = to_recovery_gate(tmp_path)
    log = tmp_path / ".k0-mem" / "events.jsonl"
    kinds = [json.loads(line).get("gate") for line in log.read_text().splitlines()]
    assert "recovery" in kinds


def test_the_gate_survives_a_resume_without_an_answer(tmp_path):
    """A pending recovery is checkpointed: resuming with no answer re-asks
    the gate rather than silently restoring (#174)."""
    pending = to_recovery_gate(tmp_path)
    again = resume(pending.run, None, root=tmp_path, cfg=implement_cfg())
    assert isinstance(again, Pending) and again.gate == "recovery"
    assert (tmp_path / "tests" / "test_calc.py").read_text() == BAD_TEST


# --- the tracker path: /implement #<n> (#174) -------------------------------------


def item_cfg():
    cfg = implement_cfg(tracker="github")
    cfg["agents"]["reviewer"] = {
        "provider": "anthropic",
        "model": "m",
        "model_calls": 25,
    }
    return cfg


def item_repo(root: Path) -> None:
    """The tracker-path fixture: a ``main`` branch, ``.k0-mem/`` ignored."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    (root / "README.md").write_text("tmp repo\n", encoding="utf-8")
    mem = root / ".k0-mem"
    mem.mkdir()
    (mem / ".gitignore").write_text(
        "sessions/\nevents.jsonl\n*.sqlite\n*.sqlite-*\nsnapshots/\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True)


def git_runner_for(root: Path):
    def run(argv):
        return subprocess.run(
            argv, cwd=root, capture_output=True, text=True, check=False
        )

    return run


def review_call(summary, call_id):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "Review", "args": {"summary": summary, "findings": []},
             "id": call_id}
        ],
    )


def test_recovery_works_on_the_tracker_path(tmp_path):
    """``/implement #<n>`` — the path the November E2E took — escalates to
    the same gate, restores from the journal, and retries to green (#174)."""
    item_repo(tmp_path)
    tracker = FakeTracker(git_runner=git_runner_for(tmp_path))
    tracker.ensure_label("ready-for-agent")
    item = tracker.create_work_item("Calc answers 42", "build the calc module")
    tracker.add_label(item.ref, "ready-for-agent")
    tracker.calls.clear()
    cfg = item_cfg()
    result = start(
        "implement",
        f"#{item.ref}",
        root=tmp_path,
        cfg=cfg,
        init=models_init(bad_builder()),
        tracker=tracker,
        check=lambda *a, **k: [],
        scan=lambda root: "scanned",
    )
    assert isinstance(result, Pending) and result.gate == "recovery"
    reviewer = FakeChatModel(
        messages=[review_call("conventions pass", "c1"),
                  review_call("contract pass", "c2")]
    )
    recovered = resume(
        result.run,
        "recover",
        root=tmp_path,
        cfg=cfg,
        init=models_init(green_builder(), reviewer),
        tracker=tracker,
    )
    assert isinstance(recovered, Pending) and recovered.gate == "review"
    assert (tmp_path / "calc.py").read_text() == CALC_SRC
