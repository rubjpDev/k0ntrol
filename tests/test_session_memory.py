from k0ntrol.memory.index import query_index, rebuild_index
from k0ntrol.memory.session import (
    add_fact,
    add_mem_proposal,
    decide_mem_proposal,
    fact_entry_id,
    has_facts,
    list_decided_mem_proposals,
    list_facts,
    list_pending_mem_proposals,
    session_db_path,
)


def test_add_fact_is_retrievable_from_session_sqlite(tmp_path):
    fact_id = add_fact(tmp_path, "checkout uses sqlite")
    assert session_db_path(tmp_path).is_file()
    assert list_facts(tmp_path) == [{"id": fact_id, "body": "checkout uses sqlite"}]
    assert fact_id == 1


def test_fact_bodies_are_unique(tmp_path):
    first = add_fact(tmp_path, "checkout uses sqlite")
    duplicate = add_fact(tmp_path, "checkout uses sqlite")
    assert first == 1
    assert duplicate is None
    assert [item["body"] for item in list_facts(tmp_path)] == ["checkout uses sqlite"]


def test_has_facts(tmp_path):
    assert has_facts(tmp_path) is False
    add_fact(tmp_path, "a fact")
    assert has_facts(tmp_path) is True


def test_mem_proposal_lifecycle(tmp_path):
    assert list_pending_mem_proposals(tmp_path) == []
    proposal_id = add_mem_proposal(tmp_path, "Prefer sqlite for checkout state")
    assert add_mem_proposal(tmp_path, "Prefer sqlite for checkout state") is None
    assert list_pending_mem_proposals(tmp_path) == [
        {"id": proposal_id, "body": "Prefer sqlite for checkout state"}
    ]
    body = decide_mem_proposal(tmp_path, proposal_id, approved=True)
    assert body == "Prefer sqlite for checkout state"
    assert list_pending_mem_proposals(tmp_path) == []
    assert list_decided_mem_proposals(tmp_path) == [
        {"status": "approved", "body": "Prefer sqlite for checkout state"}
    ]


def test_facts_are_in_fts(tmp_path):
    fact_id = add_fact(tmp_path, "checkout uses sqlite")
    other = add_fact(tmp_path, "deploys happen on fridays")
    rebuild_index(tmp_path)
    matches = query_index(tmp_path, "checkout", 5)
    assert [item["id"] for item in matches] == [fact_entry_id(fact_id)]
    assert matches[0]["path"] == "session.sqlite"
    assert matches[0]["body"] == "checkout uses sqlite"
    ids = {item["id"] for item in query_index(tmp_path, "fridays", 5)}
    assert ids == {fact_entry_id(other)}


def test_facts_survive_in_the_index_after_rebuild(tmp_path):
    add_fact(tmp_path, "checkout uses sqlite")
    rebuild_index(tmp_path)
    rebuild_index(tmp_path)
    matches = query_index(tmp_path, "sqlite", 5)
    assert [item["id"] for item in matches] == ["session:fact:1"]


def test_old_session_db_is_trimmed(tmp_path):
    """A 0.0.x session.sqlite loses its episodes table and gains the
    facts.body uniqueness the trimmed session memory requires."""
    import sqlite3

    db = session_db_path(tmp_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY, body TEXT)")
    conn.execute("INSERT INTO episodes (body) VALUES ('asked about x')")
    conn.execute("CREATE TABLE facts (id INTEGER PRIMARY KEY, body TEXT NOT NULL)")
    conn.execute("INSERT INTO facts (body) VALUES ('dup')")
    conn.execute("INSERT INTO facts (body) VALUES ('dup')")
    conn.commit()
    conn.close()

    assert [item["body"] for item in list_facts(tmp_path)] == ["dup"]
    assert add_fact(tmp_path, "dup") is None
    assert add_fact(tmp_path, "new") is not None

    conn = sqlite3.connect(db)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    conn.close()
    assert "episodes" not in tables
    assert [item["body"] for item in list_facts(tmp_path)] == ["dup", "new"]
