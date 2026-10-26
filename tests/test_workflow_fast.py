"""The /fast workflow graph through the run module seam.

Tests drive ``start``/``resume`` with a real temp git repo, a real
``test_cmd`` and scripted ``FakeChatModel``s behind the ``init`` seam —
the only injected seams besides the tracker (spec #33, Testing Decisions).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from fakes import FakeChatModel, cfg_with
from k0ntrol.workflow import Outcome, Pending, RunRefused, resume, start

TEST_SRC = "from calc import answer\n\n\ndef test_answer():\n    assert answer() == 42\n"
CALC_SRC = "def answer():\n    return 42\n"
MATHX_TEST = "import mathx\n\n\ndef test_pi():\n    assert mathx.PI > 3\n"
MATHX_SRC = "PI = 3.14159\n"


def fast_cfg(**overrides):
    cfg = cfg_with(
        {
            "tester": {"provider": "anthropic", "model": "m", "model_calls": 40},
            "coder": {"provider": "anthropic", "model": "m", "model_calls": 60},
        }
    )
    cfg.update(
        {
            "test_cmd": f"{sys.executable} -m pytest -q",
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


def run_fast(root, cfg, *models, **kwargs):
    """``start('fast', ...)`` with the doctor and rescan seams stubbed."""
    kwargs.setdefault("check", lambda *a, **k: [])
    kwargs.setdefault("scan", lambda root: "scanned")
    return start(
        "fast",
        "add a calc module that answers 42",
        root=root,
        cfg=cfg,
        init=models_init(*models),
        **kwargs,
    )


def scripted_tester(*messages):
    return FakeChatModel(messages=list(messages))


def write_call(path, content, call_id="t1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": "write_file", "args": {"path": path, "content": content}, "id": call_id}],
    )


def events(root):
    path = root / ".k0-mem" / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- refusals ----------------------------------------------------------------


def test_start_without_a_task_is_refused(tmp_path):
    with pytest.raises(RunRefused, match="usage"):
        start("fast", "   ", root=tmp_path, cfg=fast_cfg())


def test_unknown_run_kind_is_refused(tmp_path):
    with pytest.raises(RunRefused, match="unknown"):
        start("full", "add a feature", root=tmp_path, cfg=fast_cfg())


def test_doctor_errors_refuse_the_run(tmp_path):
    git_repo(tmp_path)
    checks = [
        type(
            "C",
            (),
            {"status": "error", "name": "provider anthropic", "detail": "missing package"},
        )()
    ]

    def failing_check(cfg, **kwargs):
        return checks

    with pytest.raises(RunRefused) as error:
        start(
            "fast",
            "add a calc module",
            root=tmp_path,
            cfg=fast_cfg(),
            check=failing_check,
            scan=lambda root: "scanned",
            init=models_init(scripted_tester()),
        )
    assert "missing package" in str(error.value)
    assert not (tmp_path / ".k0-mem" / "events.jsonl").exists()


# --- the happy path -----------------------------------------------------------


def test_fast_runs_to_the_diff_gate_and_done(tmp_path):
    git_repo(tmp_path)
    scanned = []
    tester = scripted_tester(
        write_call("tests/test_calc.py", TEST_SRC),
        AIMessage(content="wrote a failing test"),
    )
    coder = scripted_tester(
        write_call("calc.py", CALC_SRC),
        AIMessage(content="implemented calc"),
    )

    result = run_fast(tmp_path, fast_cfg(), tester, coder, scan=lambda r: scanned.append(r))

    assert isinstance(result, Pending)
    assert result.gate == "diff"
    assert result.answers == ("y", "n")
    assert scanned == [tmp_path]
    # the agents really wrote the files
    assert (tmp_path / "tests" / "test_calc.py").read_text() == TEST_SRC
    assert (tmp_path / "calc.py").read_text() == CALC_SRC

    outcome = resume(result.run, "y", root=tmp_path, cfg=fast_cfg(), init=models_init())

    assert isinstance(outcome, Outcome)
    assert outcome.outcome == "done"


def test_diff_gate_n_rejects_the_run(tmp_path):
    git_repo(tmp_path)
    tester = scripted_tester(
        write_call("tests/test_calc.py", TEST_SRC), AIMessage(content="tests")
    )
    coder = scripted_tester(write_call("calc.py", CALC_SRC), AIMessage(content="code"))
    result = run_fast(tmp_path, fast_cfg(), tester, coder)

    outcome = resume(result.run, "n", root=tmp_path, cfg=fast_cfg(), init=models_init())

    assert outcome.outcome == "rejected"


def test_events_record_the_run(tmp_path):
    git_repo(tmp_path)
    tester = scripted_tester(
        write_call("tests/test_calc.py", TEST_SRC), AIMessage(content="tests")
    )
    coder = scripted_tester(write_call("calc.py", CALC_SRC), AIMessage(content="code"))
    result = run_fast(tmp_path, fast_cfg(), tester, coder)
    resume(result.run, "y", root=tmp_path, cfg=fast_cfg(), init=models_init())

    log = events(tmp_path)
    kinds = [(e["kind"], e.get("gate"), e.get("result")) for e in log]
    assert kinds[0][0] == "run_start"
    assert log[0]["run_kind"] == "fast"
    assert ("gate", "red_check", "pass") in kinds
    assert ("gate", "validator", "pass") in kinds
    assert ("gate", "diff", "waiting") in kinds
    assert ("gate", "diff", "approve") in kinds
    assert kinds[-1][0] == "run_end"
    assert log[-1]["outcome"] == "done"
    assert all(e.get("run") == result.run for e in log)


# --- loops and limits ----------------------------------------------------------


def test_tests_that_pass_send_tester_back_until_stuck(tmp_path):
    """Tests that never go red exhaust max_attempts; the run ends stuck."""
    git_repo(tmp_path)
    always_green_tests = "def test_ok():\n    assert True\n"
    tester = scripted_tester(
        write_call("tests/test_ok.py", always_green_tests),
        AIMessage(content="tests"),
        write_call("tests/test_ok.py", always_green_tests, "t2"),
        AIMessage(content="tests again"),
    )
    coder = scripted_tester()

    outcome = run_fast(tmp_path, fast_cfg(max_attempts=2), tester, coder)

    assert isinstance(outcome, Outcome)
    assert outcome.outcome == "stuck"
    assert outcome.reason == "max_attempts"
    assert outcome.node == "red_check"
    # TESTER ran twice (two model calls each), CODER never ran
    assert len(tester.invocations) == 4
    assert coder.invocations == []


def test_validator_red_sends_coder_back_with_raw_output(tmp_path):
    git_repo(tmp_path)
    tester = scripted_tester(
        write_call("tests/test_calc.py", TEST_SRC), AIMessage(content="tests")
    )
    coder = scripted_tester(
        AIMessage(content="forgot the file"),
        write_call("calc.py", CALC_SRC, "c2"),
        AIMessage(content="implemented"),
    )
    cfg = fast_cfg()
    result = run_fast(tmp_path, cfg, tester, coder)

    assert isinstance(result, Pending)
    # CODER's second run saw VALIDATOR's raw output
    second_brief = coder.invocations[1][-1].content
    assert "FAILED" in second_brief or "error" in second_brief.lower()
    # two CODER runs: one model call on the first, two on the retry
    assert len(coder.invocations) == 3


def test_validator_red_exhausts_attempts_to_stuck(tmp_path):
    git_repo(tmp_path)
    tester = scripted_tester(
        write_call("tests/test_calc.py", TEST_SRC), AIMessage(content="tests")
    )
    coder = scripted_tester(
        AIMessage(content="nothing"), AIMessage(content="still nothing")
    )

    outcome = run_fast(tmp_path, fast_cfg(max_attempts=2), tester, coder)

    assert outcome.outcome == "stuck"
    assert outcome.reason == "max_attempts"
    assert outcome.node == "validator"


def test_model_call_limit_ends_the_run_stuck(tmp_path):
    git_repo(tmp_path)
    # Every reply wants another tool call: the model-call limit must fire.
    tester = scripted_tester(
        *[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_file", "args": {"path": "README.md"}, "id": f"t{i}"}
                ],
            )
            for i in range(5)
        ]
    )
    cfg = fast_cfg()
    cfg["agents"]["tester"]["model_calls"] = 3

    outcome = run_fast(tmp_path, cfg, tester)

    assert outcome.outcome == "stuck"
    assert outcome.reason == "model_call_limit"
    assert outcome.node == "tester"


# --- one run per checkout, kill + resume ---------------------------------------


def test_a_second_run_in_the_same_checkout_is_refused(tmp_path):
    git_repo(tmp_path)
    tester = scripted_tester(
        write_call("tests/test_calc.py", TEST_SRC), AIMessage(content="tests")
    )
    coder = scripted_tester(write_call("calc.py", CALC_SRC), AIMessage(content="code"))
    result = run_fast(tmp_path, fast_cfg(), tester, coder)
    assert isinstance(result, Pending)

    with pytest.raises(RunRefused, match="worktree"):
        run_fast(tmp_path, fast_cfg(), scripted_tester())


def test_kill_at_the_diff_gate_resumes_from_a_fresh_process(tmp_path):
    """A run killed at the diff gate resumes to done from a new graph on the
    same checkpoint DB — exactly what a fresh process does."""
    git_repo(tmp_path)
    tester = scripted_tester(
        write_call("tests/test_calc.py", TEST_SRC), AIMessage(content="tests")
    )
    coder = scripted_tester(write_call("calc.py", CALC_SRC), AIMessage(content="code"))
    result = run_fast(tmp_path, fast_cfg(), tester, coder)
    assert isinstance(result, Pending)
    assert (tmp_path / ".k0-mem" / "checkpoints.sqlite").exists()

    # "fresh process": resume builds a new connection and compiles a new graph
    outcome = resume(result.run, "y", root=tmp_path, cfg=fast_cfg(), init=models_init())

    assert outcome.outcome == "done"
    # and the run is finished: a new run may start (on fresh files, since
    # the tree keeps the first run's green state)
    tester2 = scripted_tester(
        write_call("tests/test_mathx.py", MATHX_TEST), AIMessage(content="tests")
    )
    coder2 = scripted_tester(write_call("mathx.py", MATHX_SRC), AIMessage(content="code"))
    again = run_fast(tmp_path, fast_cfg(), tester2, coder2)
    assert isinstance(again, Pending)


def test_resume_with_an_invalid_answer_is_refused(tmp_path):
    git_repo(tmp_path)
    tester = scripted_tester(
        write_call("tests/test_calc.py", TEST_SRC), AIMessage(content="tests")
    )
    coder = scripted_tester(write_call("calc.py", CALC_SRC), AIMessage(content="code"))
    result = run_fast(tmp_path, fast_cfg(), tester, coder)

    with pytest.raises(RunRefused, match="one of"):
        resume(result.run, "maybe", root=tmp_path, cfg=fast_cfg(), init=models_init())


def test_resume_an_unknown_run_is_refused(tmp_path):
    git_repo(tmp_path)
    with pytest.raises(RunRefused, match="no run"):
        resume("deadbeef" * 4, "y", root=tmp_path, cfg=fast_cfg(), init=models_init())


def test_resume_a_finished_run_is_refused(tmp_path):
    git_repo(tmp_path)
    tester = scripted_tester(
        write_call("tests/test_calc.py", TEST_SRC), AIMessage(content="tests")
    )
    coder = scripted_tester(write_call("calc.py", CALC_SRC), AIMessage(content="code"))
    result = run_fast(tmp_path, fast_cfg(), tester, coder)
    resume(result.run, "n", root=tmp_path, cfg=fast_cfg(), init=models_init())

    with pytest.raises(RunRefused, match="already ended"):
        resume(result.run, "y", root=tmp_path, cfg=fast_cfg(), init=models_init())


def test_crash_between_steps_resumes_with_invoke_none(tmp_path):
    """A node that dies mid-run leaves a resumable checkpoint, never stuck:
    resume (fresh graph, same DB) re-runs the failed node."""
    git_repo(tmp_path)
    dead_tester = scripted_tester()  # empty script raises on the model call
    with pytest.raises(AssertionError, match="script exhausted"):
        run_fast(tmp_path, fast_cfg(), dead_tester)
    run_id = events(tmp_path)[0]["run"]

    tester = scripted_tester(
        write_call("tests/test_calc.py", TEST_SRC), AIMessage(content="tests")
    )
    coder = scripted_tester(write_call("calc.py", CALC_SRC), AIMessage(content="code"))
    result = resume(run_id, "", root=tmp_path, cfg=fast_cfg(), init=models_init(tester, coder))

    assert isinstance(result, Pending)
    assert result.gate == "diff"


def test_a_broken_event_log_ends_the_run_stuck(tmp_path):
    git_repo(tmp_path)
    # events.jsonl as a directory makes every write fail (EventWriteError).
    (tmp_path / ".k0-mem" / "events.jsonl").mkdir(parents=True)
    tester = scripted_tester(
        write_call("tests/test_calc.py", TEST_SRC), AIMessage(content="tests")
    )
    coder = scripted_tester(write_call("calc.py", CALC_SRC), AIMessage(content="code"))

    outcome = run_fast(tmp_path, fast_cfg(), tester, coder)

    assert outcome.outcome == "stuck"
