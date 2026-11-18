"""The /research graph through the run module seam.

Same seams as the /fast, /full, /implement and /plan suites: a real temp
git repo, scripted ``FakeChatModel``s behind the ``init`` seam and a
``FakeTracker``. ``/research <question>`` runs one PLANNER pass that
charts a Map — destination, notes, one decision ticket per open question
typed research, Q&A or human — and the harness publishes the Map work
item, the decision tickets as its children and the blocker links (spec
#102, ticket #124).
"""

import json
import subprocess
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from fakes import FakeChatModel, FakeTracker, cfg_with
from k0ntrol.workflow import (
    Outcome,
    Pending,
    RunRefused,
    Unfinished,
    abandon,
    resume,
    start,
)

QUESTION = "which auth library should the project use?"


def research_cfg(**overrides):
    cfg = cfg_with(
        {"planner": {"provider": "anthropic", "model": "m", "model_calls": 25}}
    )
    cfg.update({"tracker": "github"})
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


def run_research(root, cfg, *models, input=QUESTION, **kwargs):
    """``start('research', ...)`` with the doctor and rescan seams stubbed."""
    kwargs.setdefault("check", lambda *a, **k: [])
    kwargs.setdefault("scan", lambda root: "scanned")
    return start(
        "research",
        input,
        root=root,
        cfg=cfg,
        init=models_init(*models),
        **kwargs,
    )


def planner_model(*messages):
    return FakeChatModel(messages=list(messages))


def chart_call(decisions=None, call_id="c1", **overrides):
    """A PLANNER answer: the ChartedMap structured output."""
    args = {
        "title": "Auth library map",
        "destination": "decide the auth library",
        "notes": "session tokens already exist",
        "decisions": decisions if decisions is not None else [DECISION_1],
        "not_yet_specified": [],
        "out_of_scope": ["user migration"],
    }
    args.update(overrides)
    return AIMessage(
        content="",
        tool_calls=[{"name": "ChartedMap", "args": args, "id": call_id}],
    )


DECISION_1 = {"question": "library or hand-rolled?", "type": "research", "blocked_by": []}
DECISION_2 = {
    "question": "session store?",
    "type": "Q&A",
    "blocked_by": [0],
}
DECISION_3 = {"question": "budget cap?", "type": "human", "blocked_by": []}


def read_call(call_id):
    """A PLANNER tool call that burns a model call (for the limit tests)."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "read_file", "args": {"path": "README.md"}, "id": call_id}
        ],
    )


def section(body: str, title: str) -> str:
    """The text between ``## <title>`` and the next ``##`` heading."""
    lines = body.splitlines()
    collected = []
    for line in lines[lines.index(f"## {title}") + 1 :]:
        if line.startswith("## "):
            break
        collected.append(line)
    return "\n".join(collected).strip()


def events(root):
    path = root / ".k0-mem" / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- usage and start-time refusals ---------------------------------------------


def test_research_without_input_is_refused(tmp_path):
    with pytest.raises(RunRefused, match="usage"):
        start("research", "   ", root=tmp_path, cfg=research_cfg())


def test_research_hash_form_is_refused(tmp_path):
    """``/research #<n>`` is the map-working command (#125), not charting."""
    with pytest.raises(RunRefused, match="usage: /research"):
        start("research", "#5", root=tmp_path, cfg=research_cfg())


def test_research_without_a_tracker_is_refused(tmp_path):
    """Charting publishes work items: the run needs a tracker."""
    git_repo(tmp_path)
    with pytest.raises(RunRefused, match="tracker"):
        run_research(
            tmp_path, cfg_with(
                {"planner": {"provider": "anthropic", "model": "m", "model_calls": 25}}
            ), planner_model(chart_call())
        )


def test_the_planner_and_bulk_read_roster_gates_the_run(tmp_path):
    """The doctor check runs on the research roster like the other kinds."""
    git_repo(tmp_path)
    captured = {}

    def doctor(cfg, *, names=None, **kwargs):
        captured["names"] = tuple(names or ())
        return []

    run_research(
        tmp_path,
        research_cfg(),
        planner_model(chart_call()),
        tracker=FakeTracker(),
        check=doctor,
    )

    assert "agents.planner" in captured["names"]
    assert "helpers.bulk_read" in captured["names"]


# --- charting and publishing -----------------------------------------------------


def test_the_published_map_has_a_destination_and_empty_decisions(tmp_path):
    git_repo(tmp_path)
    tracker = FakeTracker()

    outcome = run_research(
        tmp_path, research_cfg(), planner_model(chart_call()), tracker=tracker
    )

    assert isinstance(outcome, Outcome) and outcome.outcome == "done"
    map_item = tracker.read_work_item("1")
    assert map_item.title == "Auth library map"
    assert "decide the auth library" in section(map_item.body, "Destination")
    assert section(map_item.body, "Decisions so far") == ""


def test_each_decision_is_a_child_with_its_question_and_type(tmp_path):
    git_repo(tmp_path)
    tracker = FakeTracker()

    run_research(
        tmp_path,
        research_cfg(),
        planner_model(chart_call(decisions=[DECISION_1, DECISION_2, DECISION_3])),
        tracker=tracker,
    )

    children = tracker.list_children("1")
    assert [child.title for child in children] == [
        decision["question"] for decision in (DECISION_1, DECISION_2, DECISION_3)
    ]
    for child, decision in zip(children, (DECISION_1, DECISION_2, DECISION_3)):
        body = tracker.read_work_item(child.ref).body
        assert decision["question"] in body
        assert decision["type"] in body


def test_decision_tickets_carry_their_blocker_links(tmp_path):
    git_repo(tmp_path)
    tracker = FakeTracker()

    run_research(
        tmp_path,
        research_cfg(),
        planner_model(chart_call(decisions=[DECISION_1, DECISION_2, DECISION_3])),
        tracker=tracker,
    )

    children = tracker.list_children("1")
    blocker_links = [
        call for call in tracker.calls if call[0] == "link_blocked_by"
    ]
    assert blocker_links == [("link_blocked_by", children[1].ref, children[0].ref)]


def test_a_map_with_no_decisions_publishes_no_children(tmp_path):
    git_repo(tmp_path)
    tracker = FakeTracker()

    outcome = run_research(
        tmp_path,
        research_cfg(),
        planner_model(chart_call(decisions=[])),
        tracker=tracker,
    )

    assert isinstance(outcome, Outcome) and outcome.outcome == "done"
    assert tracker.list_children("1") == ()


def test_the_chart_brief_carries_the_question(tmp_path):
    git_repo(tmp_path)
    planner = planner_model(chart_call())

    run_research(tmp_path, research_cfg(), planner, tracker=FakeTracker())

    assert QUESTION in planner.invocations[0][-1].content


# --- limits and retries -----------------------------------------------------------


def test_a_planner_run_at_its_call_limit_pauses(tmp_path):
    git_repo(tmp_path)
    cfg = research_cfg()
    cfg["agents"]["planner"]["model_calls"] = 1
    looper = planner_model(read_call("r1"), read_call("r2"))

    result = run_research(tmp_path, cfg, looper, tracker=FakeTracker())

    assert isinstance(result, Pending)
    assert result.gate == "limit_hit"
    assert result.answers == ("retry", "stop")
    assert "planner" in result.question

    outcome = resume(
        result.run, "stop", root=tmp_path, cfg=cfg, init=models_init(),
        tracker=FakeTracker(),
    )

    assert outcome.outcome == "stuck"
    assert outcome.reason == "model_call_limit"
    assert outcome.node == "planner"


def test_limit_retry_reruns_the_chart(tmp_path):
    git_repo(tmp_path)
    cfg = research_cfg()
    cfg["agents"]["planner"]["model_calls"] = 1
    looper = planner_model(read_call("r1"), read_call("r2"))
    tracker = FakeTracker()
    result = run_research(tmp_path, cfg, looper, tracker=tracker)
    assert isinstance(result, Pending) and result.gate == "limit_hit"

    outcome = resume(
        result.run,
        "retry",
        root=tmp_path,
        cfg=cfg,
        init=models_init(planner_model(chart_call())),
        tracker=tracker,
    )

    assert isinstance(outcome, Outcome) and outcome.outcome == "done"
    assert tracker.list_children("1")


def test_a_non_map_answer_sends_planner_back(tmp_path):
    """PLANNER answering with another output charts again until its
    attempts run out; then the limit pause decides."""
    git_repo(tmp_path)
    round_call = AIMessage(
        content="",
        tool_calls=[
            {"name": "QuestionRound", "args": {"questions": []}, "id": "q1"}
        ],
    )
    planner = planner_model(round_call, chart_call())
    tracker = FakeTracker()

    outcome = run_research(tmp_path, research_cfg(), planner, tracker=tracker)

    assert isinstance(outcome, Outcome) and outcome.outcome == "done"
    assert len(planner.invocations) == 2
    assert tracker.list_children("1")


def test_repeated_non_map_answers_pause_at_the_attempt_limit(tmp_path):
    git_repo(tmp_path)
    round_call = AIMessage(
        content="",
        tool_calls=[
            {"name": "QuestionRound", "args": {"questions": []}, "id": "q1"}
        ],
    )
    planner = planner_model(round_call, round_call)

    result = run_research(tmp_path, research_cfg(), planner, tracker=FakeTracker())

    assert isinstance(result, Pending) and result.gate == "limit_hit"


# --- one run per checkout, abandon --------------------------------------------------


def test_resuming_a_paused_research_run_without_a_tracker_is_refused(tmp_path):
    """A paused run still has publishing ahead of it: ``resume`` needs the
    tracker like ``start`` does, not a crash at the publish node."""
    git_repo(tmp_path)
    cfg = research_cfg()
    cfg["agents"]["planner"]["model_calls"] = 1
    looper = planner_model(read_call("r1"), read_call("r2"))
    result = run_research(tmp_path, cfg, looper, tracker=FakeTracker())
    assert isinstance(result, Pending) and result.gate == "limit_hit"

    bare_cfg = cfg_with(
        {"planner": {"provider": "anthropic", "model": "m", "model_calls": 25}}
    )
    with pytest.raises(RunRefused, match="tracker"):
        resume(
            result.run,
            "retry",
            root=tmp_path,
            cfg=bare_cfg,
            init=models_init(planner_model(chart_call())),
        )


def test_a_second_research_run_gets_the_unfinished_run_back(tmp_path):
    git_repo(tmp_path)
    cfg = research_cfg()
    cfg["agents"]["planner"]["model_calls"] = 1
    looper = planner_model(read_call("r1"), read_call("r2"))
    tracker = FakeTracker()
    result = run_research(tmp_path, cfg, looper, tracker=tracker)
    assert isinstance(result, Pending)

    again = run_research(
        tmp_path, cfg, planner_model(chart_call()), tracker=tracker
    )

    assert again == Unfinished(run=result.run, node="limit_hit")


def test_abandoning_a_research_run_needs_no_clean_tree(tmp_path):
    """A research checkpoint holds no uncommitted work."""
    git_repo(tmp_path)
    cfg = research_cfg()
    cfg["agents"]["planner"]["model_calls"] = 1
    looper = planner_model(read_call("r1"), read_call("r2"))
    result = run_research(tmp_path, cfg, looper, tracker=FakeTracker())
    assert isinstance(result, Pending)
    (tmp_path / "scratch.py").write_text("x = 1\n", encoding="utf-8")

    abandon(result.run, root=tmp_path)

    with pytest.raises(RunRefused, match="no run"):
        resume(
            result.run, "stop", root=tmp_path, cfg=cfg, init=models_init(),
            tracker=FakeTracker(),
        )


# --- events and annotations ----------------------------------------------------------


def test_events_record_the_research_run(tmp_path):
    git_repo(tmp_path)
    outcome = run_research(
        tmp_path, research_cfg(), planner_model(chart_call()), tracker=FakeTracker()
    )
    assert outcome.outcome == "done"

    log = events(tmp_path)
    assert log[0]["kind"] == "run_start"
    assert log[0]["run_kind"] == "research"
    assert log[-1]["kind"] == "run_end"
    assert log[-1]["outcome"] == "done"


def test_research_public_functions_have_complete_annotations():
    import inspect

    from k0ntrol.workflow import research

    for name, function in inspect.getmembers(research, inspect.isfunction):
        if name.startswith("_") or function.__module__ != research.__name__:
            continue
        signature = inspect.signature(function)
        assert signature.return_annotation is not inspect.Signature.empty, name
        assert all(
            parameter.annotation is not inspect.Parameter.empty
            for parameter in signature.parameters.values()
        ), name
