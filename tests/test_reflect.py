import json
import threading
from pathlib import Path

from langchain_core.messages import AIMessage

from fakes import FakeChatModel, cfg_with
from k0ntrol.memory.layout import MEM, RULES
from k0ntrol.memory.reflect import (
    Reflector,
    apply_mem_decision,
    offer_mem_proposals,
    run_reflect,
)
from k0ntrol.memory.session import (
    add_fact,
    add_mem_proposal,
    list_facts,
    list_pending_mem_proposals,
)


def test_run_reflect_stores_facts_from_the_helper_reply(tmp_path):
    transcript = tmp_path / ".k0-mem" / "sessions" / "run.txt"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text("/ask how does checkout work\n", encoding="utf-8")
    seen = {}

    def fake_invoke(prompt):
        seen["prompt"] = prompt
        return {"facts": ["checkout uses sqlite"], "mem": []}

    run_reflect(root=tmp_path, invoke=fake_invoke, transcript=transcript)
    assert "how does checkout work" in seen["prompt"]
    assert [item["body"] for item in list_facts(tmp_path)] == ["checkout uses sqlite"]


def test_run_reflect_reads_the_transcript_from_the_offset(tmp_path):
    transcript = tmp_path / "session.txt"
    transcript.write_text("old turns\nnew turns\n", encoding="utf-8")
    seen = {}

    def fake_invoke(prompt):
        seen["prompt"] = prompt
        return {"facts": [], "mem": []}

    run_reflect(
        root=tmp_path,
        invoke=fake_invoke,
        transcript=transcript,
        offset=len("old turns\n"),
    )
    assert "new turns" in seen["prompt"]
    assert "old turns" not in seen["prompt"]


def test_run_reflect_ignores_missing_or_wrong_shaped_output(tmp_path):
    def fake_invoke(prompt):
        return {"facts": "not a list", "mem": [{"nested": True}, "  ", "keep me"]}

    run_reflect(root=tmp_path, invoke=fake_invoke)
    assert list_facts(tmp_path) == []
    assert list_pending_mem_proposals(tmp_path) == [
        {"id": 1, "body": "keep me"}
    ]


def test_run_reflect_queues_mem_proposals_without_writing_files(tmp_path):
    mem = tmp_path / ".k0-mem" / "General_mem" / "Mem.md"
    rules = tmp_path / ".k0-mem" / "General_mem" / "Rules.md"
    topology = tmp_path / ".k0-mem" / "Project_topology" / tmp_path.name / "strategy.md"
    mem.parent.mkdir(parents=True, exist_ok=True)
    topology.parent.mkdir(parents=True, exist_ok=True)
    mem.write_text(MEM, encoding="utf-8")
    rules.write_text(RULES, encoding="utf-8")
    topology.write_text("human owned\n", encoding="utf-8")

    def fake_invoke(prompt):
        return {
            "facts": ["checkout uses sqlite"],
            "mem": ["Prefer sqlite for checkout state"],
        }

    run_reflect(root=tmp_path, invoke=fake_invoke)
    assert list_pending_mem_proposals(tmp_path) == [
        {"id": 1, "body": "Prefer sqlite for checkout state"}
    ]
    assert mem.read_text(encoding="utf-8") == MEM
    assert rules.read_text(encoding="utf-8") == RULES
    assert topology.read_text(encoding="utf-8") == "human owned\n"
    assert not (tmp_path / ".k0-mem" / "Project_topology" / tmp_path.name / "code_rules.md").exists()


def test_run_reflect_does_not_requeue_decided_mem_proposals(tmp_path):
    add_mem_proposal(tmp_path, "Prefer sqlite for checkout state")
    apply_mem_decision(tmp_path, 1, False)
    seen = {}

    def fake_invoke(prompt):
        seen["prompt"] = prompt
        return {"facts": [], "mem": ["Prefer sqlite for checkout state"]}

    run_reflect(root=tmp_path, invoke=fake_invoke)
    assert list_pending_mem_proposals(tmp_path) == []
    assert "rejected: Prefer sqlite for checkout state" in seen["prompt"]


def test_offer_mem_proposals_answers_between_commands(tmp_path):
    mem = tmp_path / ".k0-mem" / "General_mem" / "Mem.md"
    mem.parent.mkdir(parents=True, exist_ok=True)
    mem.write_text(MEM, encoding="utf-8")
    add_mem_proposal(tmp_path, "Prefer sqlite for checkout state")
    shown = []

    offer_mem_proposals(tmp_path, iter(["n", "y"]).__next__, shown.append)
    assert shown == ["Mem.md += Prefer sqlite for checkout state? y/n"]
    assert mem.read_text(encoding="utf-8") == MEM
    assert list_pending_mem_proposals(tmp_path) == []


def test_offer_mem_proposals_reprompts_on_other_answers(tmp_path):
    add_mem_proposal(tmp_path, "Prefer sqlite")
    shown = []
    answers = iter(["maybe", "n"])
    offer_mem_proposals(tmp_path, lambda: next(answers), shown.append)
    assert shown == [
        "Mem.md += Prefer sqlite? y/n",
        "y or n",
    ]


def test_apply_mem_decision_appends_approved_body(tmp_path):
    mem = tmp_path / ".k0-mem" / "General_mem" / "Mem.md"
    add_mem_proposal(tmp_path, "Prefer sqlite for checkout state")
    apply_mem_decision(tmp_path, 1, True)
    assert mem.read_text(encoding="utf-8").endswith(
        "- Prefer sqlite for checkout state\n"
    )


def test_apply_mem_decision_appends_after_unterminated_text(tmp_path):
    mem = tmp_path / ".k0-mem" / "General_mem" / "Mem.md"
    mem.parent.mkdir(parents=True, exist_ok=True)
    mem.write_text("# Mem\nno newline", encoding="utf-8")
    add_mem_proposal(tmp_path, "fact two")
    apply_mem_decision(tmp_path, 1, True)
    assert mem.read_text(encoding="utf-8") == "# Mem\nno newline\n- fact two\n"


# --- Reflector: the background scheduler --------------------------------------


def _transcript(root: Path, text: str = "") -> Path:
    path = root / ".k0-mem" / "sessions" / "s.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _events(root: Path) -> list[dict]:
    path = root / ".k0-mem" / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _recording_reflect(calls, body=lambda **_kw: None):
    def reflect(**kwargs):
        calls.append(kwargs)
        body(**kwargs)

    return reflect


def test_delayed_worker_reads_only_dispatched_range(tmp_path):
    transcript = _transcript(tmp_path, "first run\n")
    entered = threading.Event()
    release = threading.Event()
    prompts = []

    def make_invoke(handler):
        entered.set()
        assert release.wait(5)
        return lambda prompt: (prompts.append(prompt) or {"facts": [], "mem": []})

    reflector = Reflector(
        root=tmp_path, transcript=transcript, cfg={}, make_invoke=make_invoke,
    )
    reflector.after_run("run-1")
    try:
        assert entered.wait(5)
        with transcript.open("a") as file:
            file.write("later ask\n")
        reflector.ask_turn()
    finally:
        release.set()
        reflector.finish()
    assert len(prompts) == 2
    assert "first run" in prompts[0]
    assert "later ask" not in prompts[0]
    assert "later ask" in prompts[1]
    assert "first run" not in prompts[1]


def test_after_run_fires_reflect_on_a_background_thread(tmp_path):
    transcript = _transcript(tmp_path, "turn one\n")
    calls = []
    reflector = Reflector(
        root=tmp_path,
        transcript=transcript,
        cfg={},
        reflect=_recording_reflect(calls),
    )
    reflector.after_run("run-1")
    reflector.finish()
    assert len(calls) == 1
    assert calls[0]["transcript"] == transcript
    assert calls[0]["offset"] == 0


def test_workers_reflect_in_dispatch_order_when_invoker_setup_is_reordered(tmp_path):
    transcript = _transcript(tmp_path, "first run\n")
    first_entered = threading.Event()
    release_first = threading.Event()
    second_ready = threading.Event()
    second_invoked = threading.Event()
    prompts = []

    def make_invoke(handler):
        if not first_entered.is_set():
            first_entered.set()
            assert release_first.wait(5)
            label = "first"
        else:
            label = "second"
            second_ready.set()

        def invoke(prompt):
            prompts.append((label, prompt))
            handler.on_llm_end(None, run_id=label)
            if label == "second":
                second_invoked.set()
            return {"facts": [label], "mem": []}

        return invoke

    reflector = Reflector(
        root=tmp_path, transcript=transcript, cfg={}, make_invoke=make_invoke,
    )
    reflector.after_run("run-A")
    try:
        assert first_entered.wait(5)
        with transcript.open("a", encoding="utf-8") as file:
            file.write("second run\n")
        reflector.after_run("run-B")
        assert second_ready.wait(5)
        second_invoked.wait(0.3)
    finally:
        release_first.set()
        reflector.finish()
    assert [label for label, _ in prompts] == ["first", "second"]
    assert "first run" in prompts[0][1] and "second run" not in prompts[0][1]
    assert "second run" in prompts[1][1] and "first run" not in prompts[1][1]
    assert [event["run"] for event in _events(tmp_path)] == ["run-A", "run-B"]
    assert [item["body"] for item in list_facts(tmp_path)] == ["first", "second"]


def test_after_run_returns_before_reflect_finishes(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def slow(**_kw):
        entered.set()
        assert release.wait(5)

    reflector = Reflector(
        root=tmp_path,
        transcript=_transcript(tmp_path, "x\n"),
        cfg={},
        reflect=slow,
    )
    reflector.after_run("run-1")  # returns at once
    assert entered.wait(5)


def test_finish_waits_for_a_running_reflect(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    done = []

    def slow(**_kw):
        entered.set()
        assert release.wait(5)

    reflector = Reflector(
        root=tmp_path,
        transcript=_transcript(tmp_path, "x\n"),
        cfg={},
        reflect=slow,
    )
    reflector.after_run("run-1")
    assert entered.wait(5)
    finisher = threading.Thread(target=lambda: done.append(reflector.finish()))
    finisher.start()
    assert not done
    release.set()
    finisher.join(5)
    assert done == [None]


def test_finish_reflects_ask_turns_once(tmp_path):
    calls = []
    reflector = Reflector(
        root=tmp_path,
        transcript=_transcript(tmp_path, "/ask what\nanswer\n"),
        cfg={},
        reflect=_recording_reflect(calls),
    )
    reflector.ask_turn()
    reflector.finish()
    assert len(calls) == 1


def test_finish_without_ask_turns_fires_nothing(tmp_path):
    calls = []
    reflector = Reflector(
        root=tmp_path,
        transcript=_transcript(tmp_path, "banner only\n"),
        cfg={},
        reflect=_recording_reflect(calls),
    )
    reflector.finish()
    assert calls == []


def test_a_run_reflect_covers_earlier_ask_turns(tmp_path):
    calls = []
    reflector = Reflector(
        root=tmp_path,
        transcript=_transcript(tmp_path, "/ask one\n/fast two\nrun done\n"),
        cfg={},
        reflect=_recording_reflect(calls),
    )
    reflector.ask_turn()
    reflector.after_run("run-1")
    reflector.finish()
    # the run's reflect already read the /ask lines; nothing new at /quit
    assert len(calls) == 1


def test_each_reflect_reads_only_what_is_new(tmp_path):
    transcript = _transcript(tmp_path, "first slice\n")
    calls = []
    reflector = Reflector(
        root=tmp_path,
        transcript=transcript,
        cfg={},
        reflect=_recording_reflect(calls),
    )
    reflector.after_run("run-1")
    reflector.finish()
    with transcript.open("a", encoding="utf-8") as file:
        file.write("second slice\n")
    reflector.ask_turn()
    reflector.finish()
    assert len(calls) == 2
    assert calls[1]["offset"] == len("first slice\n")


def test_reflector_model_events_carry_the_run_they_reflect_on(tmp_path):
    transcript = _transcript(tmp_path, "turn one\n")
    model = FakeChatModel(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "reflect",
                        "args": {"facts": ["uses sqlite"], "mem": []},
                        "id": "c1",
                    }
                ],
            )
        ]
    )
    cfg = cfg_with(helpers={"reflect": {"provider": "anthropic", "model": "m"}})
    reflector = Reflector(
        root=tmp_path,
        transcript=transcript,
        cfg=cfg,
        init=lambda *a, **k: model,
    )
    reflector.after_run("run-42")
    reflector.finish()
    (event,) = _events(tmp_path)
    assert event["kind"] == "model"
    assert event["helper"] == "reflect"
    assert event["run"] == "run-42"
    assert [item["body"] for item in list_facts(tmp_path)] == ["uses sqlite"]


def test_finish_reflect_uses_the_chat_scope(tmp_path):
    transcript = _transcript(tmp_path, "/ask q\na\n")
    model = FakeChatModel(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "reflect", "args": {"facts": [], "mem": []}, "id": "c1"}
                ],
            )
        ]
    )
    cfg = cfg_with(helpers={"reflect": {"provider": "anthropic", "model": "m"}})
    reflector = Reflector(
        root=tmp_path,
        transcript=transcript,
        cfg=cfg,
        init=lambda *a, **k: model,
    )
    reflector.ask_turn()
    reflector.finish()
    (event,) = _events(tmp_path)
    assert event["kind"] == "model"
    assert "chat" in event
    assert "run" not in event


def test_a_failed_reflect_is_retried_once_and_reports_each_failure(tmp_path):
    calls = []
    shown = []

    def boom(**_kw):
        calls.append(None)
        raise RuntimeError("kaput")

    reflector = Reflector(
        root=tmp_path,
        transcript=_transcript(tmp_path, "x\n"),
        cfg={},
        reflect=boom,
        write=shown.append,
    )
    reflector.after_run("run-1")
    reflector.finish()  # no exception; one retry at /quit, no more
    assert len(calls) == 2
    assert shown == ["reflect failed: kaput"] * 2


def test_a_failed_range_is_retried_by_the_next_reflect(tmp_path):
    transcript = _transcript(tmp_path, "first run\n")
    prompts = []
    shown = []

    def make_invoke(handler):
        def invoke(prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                raise RuntimeError("kaput")
            return {"facts": [], "mem": []}

        return invoke

    reflector = Reflector(
        root=tmp_path,
        transcript=transcript,
        cfg={},
        make_invoke=make_invoke,
        write=shown.append,
    )
    reflector.after_run("run-1")
    reflector.finish()
    assert len(prompts) == 2
    assert "first run" in prompts[1]
    assert shown == ["reflect failed: kaput"]

    with transcript.open("a", encoding="utf-8") as file:
        file.write("second run\n")
    reflector.after_run("run-2")
    reflector.finish()
    assert len(prompts) == 3
    assert "second run" in prompts[2]
    assert "first run" not in prompts[2]


def test_the_next_run_reflect_covers_the_failed_range(tmp_path):
    transcript = _transcript(tmp_path, "first run\n")
    failed = threading.Event()
    prompts = []

    def make_invoke(handler):
        def invoke(prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                failed.set()
                raise RuntimeError("kaput")
            return {"facts": [], "mem": []}

        return invoke

    reflector = Reflector(
        root=tmp_path, transcript=transcript, cfg={}, make_invoke=make_invoke
    )
    reflector.after_run("run-1")
    assert failed.wait(5)
    with transcript.open("a", encoding="utf-8") as file:
        file.write("second run\n")
    reflector.after_run("run-2")
    reflector.finish()
    assert len(prompts) == 2
    assert "first run" in prompts[1]
    assert "second run" in prompts[1]


def test_a_retried_range_stores_each_fact_and_proposal_once(tmp_path):
    calls = []

    def flaky(**kwargs):
        calls.append(kwargs)
        add_fact(kwargs["root"], "uses sqlite")
        add_mem_proposal(kwargs["root"], "Prefer sqlite")
        if len(calls) == 1:
            raise RuntimeError("kaput")

    reflector = Reflector(
        root=tmp_path, transcript=_transcript(tmp_path, "turn\n"), cfg={}, reflect=flaky
    )
    reflector.after_run("run-1")
    reflector.finish()
    assert len(calls) == 2
    assert [item["body"] for item in list_facts(tmp_path)] == ["uses sqlite"]
    assert list_pending_mem_proposals(tmp_path) == [{"id": 1, "body": "Prefer sqlite"}]


def test_finish_retries_a_failed_range_once_and_waits(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def flaky(**_kw):
        calls.append(None)
        if len(calls) == 1:
            raise RuntimeError("kaput")
        entered.set()
        assert release.wait(5)

    reflector = Reflector(
        root=tmp_path, transcript=_transcript(tmp_path, "x\n"), cfg={}, reflect=flaky
    )
    reflector.after_run("run-1")
    finisher = threading.Thread(target=reflector.finish)
    finisher.start()
    assert entered.wait(5)  # the retry was fired
    assert finisher.is_alive()  # /quit still waits for it
    release.set()
    finisher.join(5)
    assert not finisher.is_alive()
    assert len(calls) == 2


def test_reflects_run_one_at_a_time_in_transcript_order(tmp_path):
    transcript = _transcript(tmp_path, "first\n")
    entered = threading.Event()
    release = threading.Event()
    second = threading.Event()
    offsets = []

    def slow(**kwargs):
        offsets.append(kwargs["offset"])
        if len(offsets) == 1:
            entered.set()
            assert release.wait(5)
        else:
            second.set()

    reflector = Reflector(
        root=tmp_path, transcript=transcript, cfg={}, reflect=slow
    )
    reflector.after_run("run-1")
    assert entered.wait(5)
    with transcript.open("a", encoding="utf-8") as file:
        file.write("second\n")
    reflector.after_run("run-2")
    assert not second.wait(0.3)  # the second reflect waits for the first
    release.set()
    reflector.finish()
    assert offsets == [0, len("first\n")]


def test_a_run_with_no_new_text_reflects_nothing(tmp_path):
    calls = []
    reflector = Reflector(
        root=tmp_path,
        transcript=_transcript(tmp_path, "x\n"),
        cfg={},
        reflect=_recording_reflect(calls),
    )
    reflector.after_run("run-1")
    reflector.finish()
    reflector.after_run("run-2")  # nothing new since the first reflect
    reflector.finish()
    assert len(calls) == 1
