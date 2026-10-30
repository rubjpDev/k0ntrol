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
        start("wat", "add a feature", root=tmp_path, cfg=fast_cfg())


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


def test_bulk_read_doctor_error_refuses_the_run(tmp_path):
    """The run roster gates on ``helpers.bulk_read`` like /ask does (#61)."""
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
            "fast",
            "add a calc module",
            root=tmp_path,
            cfg=fast_cfg(),
            check=bulk_read_fails,
            scan=lambda root: "scanned",
            init=models_init(scripted_tester()),
        )
    assert "helpers.bulk_read" in captured["names"]
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


def test_tests_that_pass_send_tester_back_until_limit_hit(tmp_path):
    """Tests that never go red exhaust max_attempts; the run pauses at
    limit_hit, and `stop` ends it stuck (issue #20, ticket #42)."""
    git_repo(tmp_path)
    always_green_tests = "def test_ok():\n    assert True\n"
    tester = scripted_tester(
        write_call("tests/test_ok.py", always_green_tests),
        AIMessage(content="tests"),
        write_call("tests/test_ok.py", always_green_tests, "t2"),
        AIMessage(content="tests again"),
    )
    coder = scripted_tester()

    result = run_fast(tmp_path, fast_cfg(max_attempts=2), tester, coder)

    assert isinstance(result, Pending)
    assert result.gate == "limit_hit"
    # TESTER ran twice (two model calls each), CODER never ran
    assert len(tester.invocations) == 4
    assert coder.invocations == []

    outcome = resume(result.run, "stop", root=tmp_path, cfg=fast_cfg(), init=models_init())

    assert isinstance(outcome, Outcome)
    assert outcome.outcome == "stuck"
    assert outcome.reason == "max_attempts"
    assert outcome.node == "red_check"


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


def test_validator_red_exhausts_attempts_to_limit_hit(tmp_path):
    git_repo(tmp_path)
    tester = scripted_tester(
        write_call("tests/test_calc.py", TEST_SRC), AIMessage(content="tests")
    )
    coder = scripted_tester(
        AIMessage(content="nothing"), AIMessage(content="still nothing")
    )

    result = run_fast(tmp_path, fast_cfg(max_attempts=2), tester, coder)

    assert isinstance(result, Pending)
    assert result.gate == "limit_hit"

    outcome = resume(result.run, "stop", root=tmp_path, cfg=fast_cfg(), init=models_init())

    assert outcome.outcome == "stuck"
    assert outcome.reason == "max_attempts"
    assert outcome.node == "validator"


def read_call(path="README.md", call_id="r1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": path}, "id": call_id}],
    )


def looping_model(calls=8):
    """A model that always wants another tool call: hits any call limit."""
    return scripted_tester(*[read_call(call_id=f"r{i}") for i in range(calls)])


def test_model_call_limit_pauses_at_limit_hit(tmp_path):
    git_repo(tmp_path)
    tester = looping_model()
    cfg = fast_cfg()
    cfg["agents"]["tester"]["model_calls"] = 3

    result = run_fast(tmp_path, cfg, tester)

    assert isinstance(result, Pending)
    assert result.gate == "limit_hit"
    assert result.answers == ("retry", "stop")
    assert "model calls 3/3" in result.question + result.detail

    outcome = resume(result.run, "stop", root=tmp_path, cfg=cfg, init=models_init())

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


# --- the limit_hit pause (#42) ---------------------------------------------------


def test_limit_hit_pending_shows_node_limit_output_and_diff_stat(tmp_path):
    """The pause shows the node, the limit, the tail of the last failing
    output and `git diff --stat` (issue #20)."""
    git_repo(tmp_path)
    green = "def test_ok():\n    assert True\n"
    tester = scripted_tester(
        write_call("tests/test_ok.py", green),
        AIMessage(content="tests"),
        write_call("tests/test_ok.py", green, "t2"),
        AIMessage(content="tests again"),
    )

    result = run_fast(tmp_path, fast_cfg(max_attempts=2), tester)

    assert isinstance(result, Pending)
    assert result.gate == "limit_hit"
    assert "red_check" in result.question
    assert "attempts 2/2" in result.question + result.detail
    # tail of the last failing output: the green pytest run
    assert "passed" in result.detail
    # git diff --stat: the untracked test file the agent wrote is listed
    assert "test_ok.py" in result.detail


def test_limit_hit_stop_leaves_the_fast_tree_and_writes_events(tmp_path):
    git_repo(tmp_path)
    green = "def test_ok():\n    assert True\n"
    tester = scripted_tester(
        write_call("tests/test_ok.py", green), AIMessage(content="tests"),
        write_call("tests/test_ok.py", green, "t2"), AIMessage(content="more"),
    )

    result = run_fast(tmp_path, fast_cfg(max_attempts=2), tester)
    assert isinstance(result, Pending)

    outcome = resume(result.run, "stop", root=tmp_path, cfg=fast_cfg(), init=models_init())

    assert outcome.outcome == "stuck"
    # /fast leaves the working tree exactly as it is: nothing reverted
    assert (tmp_path / "tests" / "test_ok.py").read_text() == green
    # one gate event per decision, and run_end carries reason + node
    gates = [(e.get("gate"), e.get("result")) for e in events(tmp_path) if e["kind"] == "gate"]
    assert ("limit_hit", "waiting") in gates
    assert ("limit_hit", "stop") in gates
    end = events(tmp_path)[-1]
    assert end["kind"] == "run_end"
    assert end["outcome"] == "stuck"
    assert end["reason"] == "max_attempts"
    assert end["node"] == "red_check"


def test_limit_hit_retry_reruns_the_node_with_note_tail_and_feedback(tmp_path):
    git_repo(tmp_path)
    green = "def test_ok():\n    assert True\n"
    tester = scripted_tester(
        write_call("tests/test_ok.py", green), AIMessage(content="tests"),
        write_call("tests/test_ok.py", green, "t2"), AIMessage(content="more"),
    )

    result = run_fast(tmp_path, fast_cfg(max_attempts=2), tester)
    assert isinstance(result, Pending)

    # retry [feedback]: the node reruns on a fresh context — a new model
    # behind the init seam, the same limit, the counter reset.
    tester2 = scripted_tester(
        write_call("tests/test_calc.py", TEST_SRC, "r1"), AIMessage(content="red now")
    )
    coder = scripted_tester(write_call("calc.py", CALC_SRC), AIMessage(content="code"))
    again = resume(
        result.run,
        "retry make it fail for real",
        root=tmp_path,
        cfg=fast_cfg(),
        init=models_init(tester2, coder),
    )

    assert isinstance(again, Pending)
    assert again.gate == "diff"
    brief = tester2.invocations[0][-1].content
    assert "limit" in brief.lower()          # the note that it hit its limit
    assert "attempts 2/2" in brief           # which limit
    assert "passed" in brief                 # the output tail
    assert "make it fail for real" in brief  # the human's feedback
    # files are kept: the earlier green test file was never reverted
    assert (tmp_path / "tests" / "test_ok.py").read_text() == green
    gates = [(e.get("gate"), e.get("result")) for e in events(tmp_path) if e["kind"] == "gate"]
    assert ("limit_hit", "retry") in gates


def test_limit_hit_retries_are_unlimited(tmp_path):
    git_repo(tmp_path)
    green = "def test_ok():\n    assert True\n"
    tester = scripted_tester(
        write_call("tests/test_ok.py", green), AIMessage(content="tests"),
        write_call("tests/test_ok.py", green, "t2"), AIMessage(content="more"),
    )

    first = run_fast(tmp_path, fast_cfg(max_attempts=2), tester)
    assert isinstance(first, Pending) and first.gate == "limit_hit"

    # a retry that still does not go red pauses again after a fresh
    # max_attempts, not after fewer (the counter reset)
    tester2 = scripted_tester(
        write_call("tests/test_ok.py", green, "r1"), AIMessage(content="one"),
        write_call("tests/test_ok.py", green, "r2"), AIMessage(content="two"),
    )
    second = resume(
        first.run, "retry", root=tmp_path, cfg=fast_cfg(), init=models_init(tester2)
    )
    assert isinstance(second, Pending) and second.gate == "limit_hit"
    assert len(tester2.invocations) == 4  # two full attempts again


def test_model_call_limit_never_runs_the_check_nor_spends_an_attempt(tmp_path):
    """A model-call limit goes straight to the pause: it never falls
    through to the red check or VALIDATOR, and it never uses up an
    attempt (issue #20)."""
    git_repo(tmp_path)
    tester = looping_model()
    cfg = fast_cfg(max_attempts=2)
    cfg["agents"]["tester"]["model_calls"] = 3

    result = run_fast(tmp_path, cfg, tester)
    assert isinstance(result, Pending) and result.gate == "limit_hit"

    gates = [(e.get("gate"), e.get("result")) for e in events(tmp_path) if e["kind"] == "gate"]
    assert not any(gate == "red_check" for gate, _ in gates)

    # retry: TESTER now writes green tests, and it still gets the full
    # max_attempts loop — the limited run did not spend an attempt
    green = "def test_ok():\n    assert True\n"
    tester2 = scripted_tester(
        write_call("tests/test_ok.py", green, "r1"), AIMessage(content="one"),
        write_call("tests/test_ok.py", green, "r2"), AIMessage(content="two"),
    )
    second = resume(
        result.run, "retry", root=tmp_path, cfg=cfg, init=models_init(tester2)
    )
    assert isinstance(second, Pending) and second.gate == "limit_hit"
    assert "attempts 2/2" in second.question + second.detail
    assert len(tester2.invocations) == 4


def test_coder_model_call_limit_pauses_at_limit_hit(tmp_path):
    git_repo(tmp_path)
    tester = scripted_tester(
        write_call("tests/test_calc.py", TEST_SRC), AIMessage(content="tests")
    )
    coder = looping_model()
    cfg = fast_cfg()
    cfg["agents"]["coder"]["model_calls"] = 3

    result = run_fast(tmp_path, cfg, tester, coder)

    assert isinstance(result, Pending)
    assert result.gate == "limit_hit"
    assert "coder" in result.question
    assert "model calls 3/3" in result.question + result.detail
    # VALIDATOR never ran for the limited CODER
    gates = [(e.get("gate"), e.get("result")) for e in events(tmp_path) if e["kind"] == "gate"]
    assert not any(gate == "validator" for gate, _ in gates)


def test_limit_hit_quit_resumes_from_a_fresh_process(tmp_path):
    """Quitting while paused leaves the run resumable, like any gate."""
    git_repo(tmp_path)
    green = "def test_ok():\n    assert True\n"
    tester = scripted_tester(
        write_call("tests/test_ok.py", green), AIMessage(content="tests"),
        write_call("tests/test_ok.py", green, "t2"), AIMessage(content="more"),
    )

    result = run_fast(tmp_path, fast_cfg(max_attempts=2), tester)
    assert isinstance(result, Pending) and result.gate == "limit_hit"

    # "fresh process": resume builds a new connection and a new graph
    outcome = resume(result.run, "stop", root=tmp_path, cfg=fast_cfg(), init=models_init())
    assert outcome.outcome == "stuck"


def test_an_unhandled_failure_stops_resumably_and_never_stuck(tmp_path):
    """Other failures — a provider error here — stop the run so it can
    resume; they are never `stuck` (no run_end, checkpoint kept)."""
    git_repo(tmp_path)
    broken_tester = scripted_tester()  # empty script raises on the call

    with pytest.raises(AssertionError, match="script exhausted"):
        run_fast(tmp_path, fast_cfg(), broken_tester)

    assert not any(e["kind"] == "run_end" for e in events(tmp_path))

    tester = scripted_tester(
        write_call("tests/test_calc.py", TEST_SRC), AIMessage(content="tests")
    )
    coder = scripted_tester(write_call("calc.py", CALC_SRC), AIMessage(content="code"))
    run_id = events(tmp_path)[0]["run"]
    result = resume(run_id, "", root=tmp_path, cfg=fast_cfg(), init=models_init(tester, coder))
    assert isinstance(result, Pending)
    assert result.gate == "diff"
