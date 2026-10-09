import json
import stat
import sys
from pathlib import Path

from k0ntrol.backends.invoke import current_event_context, invoke, set_event_context
from k0ntrol.harness.init import MEM, RULES
from k0ntrol.harness.reflect import apply_mem_decision, run_reflect
from k0ntrol.harness.session_memory import (
    add_mem_proposal,
    list_episodes,
    list_facts,
    list_pending_mem_proposals,
)


def reflect_config():
    return {
        "agents": {
            "reflect": {"model": "gpt-5.6-luna-max", "temperature": 0.2},
        }
    }


def test_run_reflect_stores_facts_and_episodes_from_fake_invoke(tmp_path):
    transcript = tmp_path / ".k0-mem" / "sessions" / "run.txt"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text("/ask how does checkout work\n", encoding="utf-8")
    seen = {}

    def fake_invoke(mode, prompt):
        seen["mode"] = mode.name
        seen["prompt"] = prompt
        seen["context"] = dict(current_event_context())
        return "fact: checkout uses sqlite\nepisode: asked how checkout works\n"

    set_event_context(run="session-run", step="ask")
    run_reflect(
        root=tmp_path,
        config=reflect_config(),
        invoke=fake_invoke,
        transcript=transcript,
    )
    assert seen["mode"] == "reflect"
    assert seen["context"]["step"] == "reflect"
    assert seen["context"]["run"] == "session-run"
    assert "how does checkout work" in seen["prompt"]
    assert [item["body"] for item in list_facts(tmp_path)] == ["checkout uses sqlite"]
    assert [item["body"] for item in list_episodes(tmp_path)] == [
        "asked how checkout works"
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

    def fake_invoke(mode, prompt):
        return (
            "fact: checkout uses sqlite\n"
            "mem: Prefer sqlite for checkout state\n"
        )

    run_reflect(root=tmp_path, config=reflect_config(), invoke=fake_invoke)
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

    def fake_invoke(mode, prompt):
        seen["prompt"] = prompt
        return "mem: Prefer sqlite for checkout state\n"

    run_reflect(root=tmp_path, config=reflect_config(), invoke=fake_invoke)
    assert list_pending_mem_proposals(tmp_path) == []
    assert "rejected: Prefer sqlite for checkout state" in seen["prompt"]


def write_agent(path: Path, body: str) -> None:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_run_reflect_emits_one_jsonl_event(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = tmp_path / "cursor-agent"
    write_agent(
        agent,
        "import json\n"
        "print(json.dumps({'type':'result','subtype':'success','is_error':False,"
        "'result': 'fact: checkout uses sqlite\\n'}))\n",
    )
    monkeypatch.setenv("K0_CURSOR_AGENT", str(agent))
    set_event_context(run="session-run", step="")
    run_reflect(root=tmp_path, config=reflect_config(), invoke=invoke)
    events = [
        json.loads(line)
        for line in (tmp_path / ".k0-mem" / "events.jsonl").read_text().splitlines()
    ]
    reflect_events = [event for event in events if event["mode"] == "reflect"]
    assert len(reflect_events) == 1
    assert reflect_events[0]["ok"] is True
    assert reflect_events[0]["step"] == "reflect"
    assert reflect_events[0]["run"] == "session-run"
    assert [item["body"] for item in list_facts(tmp_path)] == ["checkout uses sqlite"]

