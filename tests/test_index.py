from k0ntrol.memory.index import (
    custom_rules_text,
    custom_topology_text,
    has_code_memory,
    query_index,
    rebuild_index,
    rules_text,
)
from k0ntrol.memory.layout import ARCHITECTURE, RULES
from k0ntrol.memory.session import add_fact


def _write_note(root, relative, body):
    path = root / ".k0-mem" / "Project_context" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_rebuild_index_covers_context_notes_with_node_ids(tmp_path):
    repo = tmp_path.name
    _write_note(tmp_path, f"{repo}/files/src/app.py.md", "# app\ncheckout code\n")
    _write_note(tmp_path, f"{repo}/functions/pkg.util.helper.md", "# helper\n")
    _write_note(tmp_path, f"{repo}/classes/pkg.model.Item.md", "# Item\n")

    index_path = rebuild_index(tmp_path)
    assert index_path == tmp_path / ".k0-mem" / "index.sqlite"

    matches = query_index(tmp_path, "checkout", 5)
    assert [(m["id"], m["path"]) for m in matches] == [
        (f"{repo}:file:src/app.py", f"{repo}/files/src/app.py.md")
    ]
    all_entries = query_index(tmp_path, "helper Item", 10)
    assert {m["id"] for m in all_entries} == {
        f"{repo}:fn:pkg.util.helper",
        f"{repo}:class:pkg.model.Item",
    }


def test_query_index_limits_and_falls_back_without_tokens(tmp_path):
    repo = tmp_path.name
    for name in ("a", "b", "c"):
        _write_note(tmp_path, f"{repo}/files/{name}.py.md", f"# {name}\n")
    rebuild_index(tmp_path)
    assert len(query_index(tmp_path, "nothing-matches-this", 5)) == 0
    fallback = query_index(tmp_path, "???", 2)
    assert len(fallback) == 2


def test_index_combines_notes_and_facts(tmp_path):
    repo = tmp_path.name
    _write_note(tmp_path, f"{repo}/files/app.py.md", "# app\ncheckout code\n")
    fact_id = add_fact(tmp_path, "checkout decided in review")
    rebuild_index(tmp_path)
    ids = {item["id"] for item in query_index(tmp_path, "checkout", 5)}
    assert ids == {f"{repo}:file:app.py", f"session:fact:{fact_id}"}


def test_has_code_memory(tmp_path):
    assert has_code_memory(tmp_path) is False
    (tmp_path / ".k0-mem" / "Project_context").mkdir(parents=True)
    assert has_code_memory(tmp_path) is False
    _write_note(tmp_path, "repo/files/a.py.md", "# a\n")
    assert has_code_memory(tmp_path) is True


def test_rules_text(tmp_path):
    assert rules_text(tmp_path) == ""
    rules = tmp_path / ".k0-mem" / "General_mem" / "Rules.md"
    rules.parent.mkdir(parents=True)
    rules.write_text("be nice\n", encoding="utf-8")
    assert rules_text(tmp_path) == "be nice\n"


def test_custom_rules_text_only_when_edited(tmp_path):
    assert custom_rules_text(tmp_path) == ""
    rules = tmp_path / ".k0-mem" / "General_mem" / "Rules.md"
    rules.parent.mkdir(parents=True)
    rules.write_text(RULES, encoding="utf-8")
    assert custom_rules_text(tmp_path) == ""
    rules.write_text("project rules\n", encoding="utf-8")
    assert custom_rules_text(tmp_path) == "project rules\n"


def test_custom_topology_text_only_when_edited(tmp_path):
    assert custom_topology_text(tmp_path) == ""
    topo = tmp_path / ".k0-mem" / "Project_topology" / tmp_path.name
    topo.mkdir(parents=True)
    (topo / "architecture.md").write_text(ARCHITECTURE, encoding="utf-8")
    (topo / "strategy.md").write_text("our strategy\n", encoding="utf-8")
    (topo / "code_rules.md").write_text("our code rules\n", encoding="utf-8")
    assert custom_topology_text(tmp_path) == "our strategy\n\n\nour code rules\n"
