import json
import stat
import sys
from pathlib import Path

from k0ntrol.backends.invoke import current_event_context, invoke, set_event_context
from k0ntrol.harness.reflect import run_reflect
from k0ntrol.harness.session_memory import list_episodes, list_facts


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

