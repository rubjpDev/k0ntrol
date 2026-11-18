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
from k0ntrol.agents import MissingStructuredOutput
from k0ntrol.workflow import (
    Outcome,
    Pending,
    RunRefused,
    Unfinished,
    abandon,
    resume,
    start,
)
from k0ntrol.workflow.research import decision_body

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


def researcher_model(*messages):
    return FakeChatModel(messages=list(messages))


def finding_call(call_id="f1", **overrides):
    """A RESEARCHER answer: the ResearchFinding structured output."""
    args = {
        "findings": ["Authlib covers OAuth + JWT", "docs list every grant"],
        "proposed_decision": "use Authlib",
        "citations": [{"title": "Authlib docs", "url": "https://docs.authlib.org"}],
    }
    args.update(overrides)
    return AIMessage(
        content="",
        tool_calls=[{"name": "ResearchFinding", "args": args, "id": call_id}],
    )


def map_cfg(**overrides):
    """``/research #<n>``'s config: RESEARCHER may run on a decision ticket."""
    cfg = research_cfg(tracker="github")
    cfg["agents"]["researcher"] = {
        "provider": "anthropic",
        "model": "m",
        "model_calls": 40,
    }
    cfg.update(overrides)
    return cfg


def round_call(questions, call_id="q1"):
    """A PLANNER answer: the QuestionRound structured output."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "QuestionRound", "args": {"questions": questions}, "id": call_id}
        ],
    )


def proposal_call(call_id="p1", **overrides):
    """A PLANNER answer: the DecisionProposal structured output."""
    args = {
        "proposed_decision": "keep sessions in Redis",
        "rationale": "shared across workers",
    }
    args.update(overrides)
    return AIMessage(
        content="",
        tool_calls=[{"name": "DecisionProposal", "args": args, "id": call_id}],
    )


Q1 = {
    "number": 1,
    "question": "Which store?",
    "recommendation": "Redis",
    "rationale": "shared across workers",
}


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

# A published Map body: ``map_body`` leaves Decisions so far empty (#124).
MAP_BODY = (
    "## Destination\n\ndecide the auth library\n\n## Decisions so far\n\n"
    "## Not yet specified\n\n—\n\n## Out of scope\n\n—"
)


def seed_map(tracker):
    """A Map work item as ``/research <question>`` published it."""
    return tracker.create_work_item("Auth library map", MAP_BODY)


def seed_decision(tracker, parent, question, type_="human", index=0):
    """A decision-ticket child of ``parent`` in the published body format."""
    return tracker.create_child_work_item(
        parent,
        question,
        decision_body({"question": question, "type": type_}),
        run="r0",
        node=f"publish_decision_{index}",
    )


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


def test_research_hash_with_a_non_digit_is_refused(tmp_path):
    """``#`` plus a work-item number is the only numbered form (#125)."""
    with pytest.raises(RunRefused, match="usage: /research"):
        start("research", "#abc", root=tmp_path, cfg=research_cfg())


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


# --- /research #<n>: the map-working run (#125) ---------------------------------------


def test_a_missing_map_item_is_refused(tmp_path):
    git_repo(tmp_path)
    with pytest.raises(RunRefused, match="work item"):
        run_research(
            tmp_path,
            research_cfg(tracker="github"),
            tracker=FakeTracker(),
            input="#9",
        )


def test_an_untrusted_map_item_is_refused(tmp_path):
    """A map filed by an untrusted author never reaches an agent (ADR 0006)."""
    git_repo(tmp_path)
    tracker = FakeTracker()
    item = tracker.create_work_item(
        "Map", "the body", author="mallory", author_association="NONE"
    )
    with pytest.raises(RunRefused, match="restate"):
        run_research(
            tmp_path,
            research_cfg(tracker="github"),
            tracker=tracker,
            input=f"#{item.ref}",
        )


def test_a_closed_map_item_is_refused(tmp_path):
    git_repo(tmp_path)
    tracker = FakeTracker()
    map_item = seed_map(tracker)
    tracker.set_work_item_closed(map_item.ref)
    with pytest.raises(RunRefused, match="closed"):
        run_research(
            tmp_path,
            research_cfg(tracker="github"),
            tracker=tracker,
            input=f"#{map_item.ref}",
        )


def test_the_run_picks_the_first_open_unblocked_decision(tmp_path):
    """Children come in tracker order: closed ones are done, an open one
    whose blockers are all closed is the next decision to work — a human
    ticket goes straight to its gate."""
    git_repo(tmp_path)
    tracker = FakeTracker()
    map_item = seed_map(tracker)
    done = seed_decision(tracker, map_item.ref, "settled already", index=0)
    tracker.set_work_item_closed(done.ref)
    waiting = seed_decision(tracker, map_item.ref, "waits on the pick", index=1)
    pick = seed_decision(tracker, map_item.ref, "budget cap?", index=2)
    tracker.link_blocked_by(waiting.ref, pick.ref)

    result = run_research(
        tmp_path, research_cfg(tracker="github"), tracker=tracker, input="#1"
    )

    assert isinstance(result, Pending)
    assert result.gate == "decision"
    assert "budget cap?" in result.question


def test_a_child_blocked_by_an_open_blocker_is_not_picked(tmp_path):
    """The blocker lives outside the map here, so nothing is workable."""
    git_repo(tmp_path)
    tracker = FakeTracker()
    map_item = seed_map(tracker)
    external = tracker.create_work_item("External blocker", "not a child")
    child = seed_decision(tracker, map_item.ref, "still waiting", index=0)
    tracker.link_blocked_by(child.ref, external.ref)

    outcome = run_research(
        tmp_path, research_cfg(tracker="github"), tracker=tracker, input="#1"
    )

    assert isinstance(outcome, Outcome) and outcome.outcome == "done"
    assert tracker.read_work_item(child.ref).state == "open"


def test_a_map_with_no_open_decision_ends_done(tmp_path):
    git_repo(tmp_path)
    tracker = FakeTracker()
    map_item = seed_map(tracker)
    child = seed_decision(tracker, map_item.ref, "settled", index=0)
    tracker.set_work_item_closed(child.ref)

    outcome = run_research(
        tmp_path, research_cfg(tracker="github"), tracker=tracker, input="#1"
    )

    assert isinstance(outcome, Outcome) and outcome.outcome == "done"


def test_a_child_without_a_decision_type_asks_the_human(tmp_path):
    """A child that is no k0 decision ticket — a note the human filed on
    the map — is worked as a human decision: only a gate reads it."""
    git_repo(tmp_path)
    tracker = FakeTracker()
    map_item = seed_map(tracker)
    tracker.create_child_work_item(
        map_item.ref, "a plain note", "no type section", run="r0", node="note"
    )

    result = run_research(
        tmp_path, research_cfg(tracker="github"), tracker=tracker, input="#1"
    )

    assert isinstance(result, Pending) and result.gate == "decision"


def test_the_humans_own_answer_closes_the_ticket_and_indexes_the_map(tmp_path):
    """Whatever the human types is the decision: the ticket closes with it
    and the map's Decisions so far gains its one-line gist and link."""
    git_repo(tmp_path)
    tracker = FakeTracker()
    map_item = seed_map(tracker)
    pick = seed_decision(tracker, map_item.ref, "budget cap?", index=0)
    cfg = research_cfg(tracker="github")
    pending = run_research(tmp_path, cfg, tracker=tracker, input="#1")
    assert isinstance(pending, Pending) and pending.gate == "decision"

    outcome = resume(
        pending.run, "cap it at 40 calls", root=tmp_path, cfg=cfg, tracker=tracker
    )

    assert isinstance(outcome, Outcome) and outcome.outcome == "done"
    closed = tracker.read_work_item(pick.ref)
    assert closed.state == "closed"
    assert "cap it at 40 calls" in closed.comments[-1].body
    line = section(tracker.read_work_item("1").body, "Decisions so far")
    assert "cap it at 40 calls" in line
    assert f"issues/{pick.ref}" in line


# --- research tickets: RESEARCHER answers, the gate confirms --------------------------


def test_a_research_ticket_gets_cited_findings_commented_before_the_gate(tmp_path):
    """RESEARCHER's findings and citations are a comment on the ticket
    before the decision gate shows the proposal."""
    git_repo(tmp_path)
    tracker = FakeTracker()
    map_item = seed_map(tracker)
    pick = seed_decision(
        tracker, map_item.ref, "library or hand-rolled?", "research", index=0
    )

    result = run_research(
        tmp_path, map_cfg(), researcher_model(finding_call()), tracker=tracker,
        input="#1",
    )

    assert isinstance(result, Pending) and result.gate == "decision"
    assert "use Authlib" in result.detail
    comments = tracker.read_work_item(pick.ref).comments
    assert comments, "the findings were never commented"
    assert "Authlib covers OAuth + JWT" in comments[0].body
    assert "docs.authlib.org" in comments[0].body


def test_the_researcher_brief_carries_the_decision_question(tmp_path):
    git_repo(tmp_path)
    tracker = FakeTracker()
    map_item = seed_map(tracker)
    seed_decision(
        tracker, map_item.ref, "library or hand-rolled?", "research", index=0
    )
    researcher = researcher_model(finding_call())

    run_research(tmp_path, map_cfg(), researcher, tracker=tracker, input="#1")

    brief = researcher.invocations[0][-1].content
    assert "library or hand-rolled?" in brief


def test_ok_at_the_gate_closes_with_the_proposed_decision(tmp_path):
    """``ok`` takes RESEARCHER's proposal: the ticket closes with it and
    the map indexes its one-line gist and link."""
    git_repo(tmp_path)
    tracker = FakeTracker()
    map_item = seed_map(tracker)
    pick = seed_decision(
        tracker, map_item.ref, "library or hand-rolled?", "research", index=0
    )
    cfg = map_cfg()
    pending = run_research(
        tmp_path, cfg, researcher_model(finding_call()), tracker=tracker,
        input="#1",
    )
    assert isinstance(pending, Pending) and pending.gate == "decision"

    outcome = resume(pending.run, "ok", root=tmp_path, cfg=cfg, tracker=tracker)

    assert isinstance(outcome, Outcome) and outcome.outcome == "done"
    closed = tracker.read_work_item(pick.ref)
    assert closed.state == "closed"
    assert "use Authlib" in closed.comments[-1].body
    line = section(tracker.read_work_item("1").body, "Decisions so far")
    assert "use Authlib" in line
    assert f"issues/{pick.ref}" in line


def test_a_researcher_run_at_its_call_limit_pauses(tmp_path):
    git_repo(tmp_path)
    tracker = FakeTracker()
    map_item = seed_map(tracker)
    seed_decision(
        tracker, map_item.ref, "library or hand-rolled?", "research", index=0
    )
    cfg = map_cfg()
    cfg["agents"]["researcher"]["model_calls"] = 1
    looper = researcher_model(read_call("r1"), read_call("r2"))

    result = run_research(tmp_path, cfg, looper, tracker=tracker, input="#1")

    assert isinstance(result, Pending) and result.gate == "limit_hit"
    assert "researcher" in result.question

    outcome = resume(
        result.run, "stop", root=tmp_path, cfg=cfg, tracker=tracker
    )

    assert outcome.outcome == "stuck"
    assert outcome.node == "researcher"


def test_researcher_without_a_finding_stops_resumably(tmp_path):
    """RESEARCHER answering in prose stops the run resumably, like every
    structured-output agent (#58): no findings comment, no ``stuck``
    run_end — and resume re-runs RESEARCHER fresh."""
    git_repo(tmp_path)
    tracker = FakeTracker()
    map_item = seed_map(tracker)
    pick = seed_decision(tracker, map_item.ref, "oauth lib?", "research", index=0)
    cfg = map_cfg()
    prose = researcher_model(AIMessage(content="my finding, in prose"))

    with pytest.raises(
        MissingStructuredOutput, match="RESEARCHER returned no research finding"
    ):
        run_research(tmp_path, cfg, prose, tracker=tracker, input="#1")

    assert not tracker.read_work_item(pick.ref).comments
    assert not any(e["kind"] == "run_end" for e in events(tmp_path))

    researcher = researcher_model(finding_call())
    run_id = events(tmp_path)[0]["run"]
    result = resume(
        run_id, "", root=tmp_path, cfg=cfg,
        init=models_init(researcher), tracker=tracker,
    )
    assert isinstance(result, Pending) and result.gate == "decision"


# --- Q&A tickets: PLANNER's short Q&A, then the gate ----------------------------------


def test_a_qa_ticket_runs_a_short_qa_before_the_gate(tmp_path):
    """PLANNER asks numbered questions in rounds; once it can propose the
    decision the decision gate shows the proposal."""
    git_repo(tmp_path)
    tracker = FakeTracker()
    map_item = seed_map(tracker)
    seed_decision(tracker, map_item.ref, "session store?", "Q&A", index=0)
    cfg = map_cfg()
    pending = run_research(
        tmp_path, cfg, planner_model(round_call([Q1])), tracker=tracker,
        input="#1",
    )
    assert isinstance(pending, Pending) and pending.gate == "round"
    assert "1. Which store?" in pending.detail
    assert "Redis" in pending.detail

    proposer = planner_model(proposal_call())
    gate = resume(
        pending.run, "1 yes", root=tmp_path, cfg=cfg,
        init=models_init(proposer), tracker=tracker,
    )

    assert isinstance(gate, Pending) and gate.gate == "decision"
    assert "keep sessions in Redis" in gate.detail


def test_the_decision_qa_brief_carries_the_answered_rounds(tmp_path):
    """The next PLANNER run's fresh context sees every earlier question
    and the line the human answered it with."""
    git_repo(tmp_path)
    tracker = FakeTracker()
    map_item = seed_map(tracker)
    seed_decision(tracker, map_item.ref, "session store?", "Q&A", index=0)
    cfg = map_cfg()
    pending = run_research(
        tmp_path, cfg, planner_model(round_call([Q1])), tracker=tracker,
        input="#1",
    )
    proposer = planner_model(proposal_call())

    resume(
        pending.run, "1 yes", root=tmp_path, cfg=cfg,
        init=models_init(proposer), tracker=tracker,
    )

    brief = proposer.invocations[0][-1].content
    assert "session store?" in brief
    assert "Which store?" in brief
    assert "1 yes" in brief


def test_ok_closes_a_qa_ticket_with_the_proposal(tmp_path):
    git_repo(tmp_path)
    tracker = FakeTracker()
    map_item = seed_map(tracker)
    pick = seed_decision(
        tracker, map_item.ref, "session store?", "Q&A", index=0
    )
    cfg = map_cfg()
    pending = run_research(
        tmp_path, cfg, planner_model(proposal_call()), tracker=tracker,
        input="#1",
    )
    assert isinstance(pending, Pending) and pending.gate == "decision"

    outcome = resume(pending.run, "ok", root=tmp_path, cfg=cfg, tracker=tracker)

    assert isinstance(outcome, Outcome) and outcome.outcome == "done"
    closed = tracker.read_work_item(pick.ref)
    assert closed.state == "closed"
    assert "keep sessions in Redis" in closed.comments[-1].body
    line = section(tracker.read_work_item("1").body, "Decisions so far")
    assert "keep sessions in Redis" in line
    assert f"issues/{pick.ref}" in line


def test_a_qa_round_with_no_questions_goes_to_the_gate(tmp_path):
    """An empty round ends the Q&A like /plan's does; with no proposal
    the gate falls back to asking the human."""
    git_repo(tmp_path)
    tracker = FakeTracker()
    map_item = seed_map(tracker)
    seed_decision(tracker, map_item.ref, "session store?", "Q&A", index=0)

    pending = run_research(
        tmp_path, map_cfg(), planner_model(round_call([])), tracker=tracker,
        input="#1",
    )

    assert isinstance(pending, Pending) and pending.gate == "decision"
    assert "session store?" in pending.question


def test_a_decision_planner_run_at_its_call_limit_pauses(tmp_path):
    """A PLANNER round that hits its model-call limit pauses like every
    agent run; ``stop`` ends the run stuck at the decision planner."""
    git_repo(tmp_path)
    tracker = FakeTracker()
    map_item = seed_map(tracker)
    seed_decision(tracker, map_item.ref, "session store?", "Q&A", index=0)
    cfg = map_cfg()
    cfg["agents"]["planner"]["model_calls"] = 1
    looper = planner_model(read_call("r1"), read_call("r2"))

    result = run_research(tmp_path, cfg, looper, tracker=tracker, input="#1")

    assert isinstance(result, Pending) and result.gate == "limit_hit"

    outcome = resume(
        result.run, "stop", root=tmp_path, cfg=cfg, tracker=tracker
    )
    assert outcome.outcome == "stuck"
    assert outcome.node == "decision_planner"


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
