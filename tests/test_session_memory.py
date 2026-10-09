from k0ntrol.harness.retrieve import agent_memory, query_index, rebuild_index
from k0ntrol.harness.session_memory import (
    add_episode,
    add_fact,
    fact_entry_id,
    list_episodes,
    list_facts,
)
from k0ntrol.modespec import load_mode


def test_add_fact_is_retrievable_from_session_sqlite(tmp_path):
    fact_id = add_fact(tmp_path, "checkout uses sqlite")
    assert (tmp_path / ".k0-mem" / "session.sqlite").is_file()
    assert list_facts(tmp_path) == [{"id": fact_id, "body": "checkout uses sqlite"}]
    assert fact_id == 1


def test_episodes_stay_separate_from_facts(tmp_path):
    add_fact(tmp_path, "a fact")
    episode_id = add_episode(tmp_path, "user asked about checkout")
    assert list_episodes(tmp_path) == [
        {"id": episode_id, "body": "user asked about checkout"}
    ]
    assert [item["body"] for item in list_facts(tmp_path)] == ["a fact"]


def test_facts_are_in_fts_and_episodes_are_not(tmp_path):
    fact_id = add_fact(tmp_path, "checkout uses sqlite")
    add_episode(tmp_path, "asked about checkout")
    rebuild_index(tmp_path)
    matches = query_index(tmp_path, "checkout sqlite", 5)
    assert [item["id"] for item in matches] == [fact_entry_id(fact_id)]
    assert matches[0]["body"] == "checkout uses sqlite"
    assert all("asked about checkout" not in item["body"] for item in matches)


def test_agent_memory_retrieves_facts_without_project_context(tmp_path):
    fact_id = add_fact(tmp_path, "checkout uses sqlite")
    node_id = fact_entry_id(fact_id)
    seen = {}

    def fake_invoke(mode, prompt):
        seen["mode"] = mode.name
        seen["prompt"] = prompt
        return f"retrieve: yes\n- {node_id}\n"

    config = {"retrieval_top_k": 5, "agents": {"retrieval_gate": {"model": "m"}}}
    extra = agent_memory(
        "how does checkout work",
        root=tmp_path,
        config=config,
        invoke=fake_invoke,
    )
    assert extra == "checkout uses sqlite"
    assert seen["mode"] == "retrieval_gate"
    assert f"- {node_id}" in seen["prompt"]
    assert node_id.startswith("session:fact:")
    assert "needs code memory" not in seen["prompt"]
    gate = load_mode("retrieval_gate", config).instructions.lower()
    assert "session-fact" in gate or "session fact" in gate
    assert not (tmp_path / ".k0-mem" / "Project_context").exists()

