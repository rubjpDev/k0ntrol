"""The ``search_memory`` tool: one FTS lookup over the rebuilt index.

The index is rebuilt once per run or ``/ask`` by the harness; the tool
only queries it — rebuilding lazily solely when no index exists at all.
"""

import json
import sqlite3

from k0ntrol.memory.index import rebuild_index
from k0ntrol.memory.session import add_fact
from k0ntrol.tools.search_memory import make_search_memory


def _note(root, relative, body):
    path = root / ".k0-mem" / "Project_context" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _search(tool, query="checkout", limit=5):
    return json.loads(tool.invoke({"query": query, "limit": limit}))


def test_search_memory_returns_hits_with_id_kind_path_body(tmp_path):
    repo = tmp_path.name
    _note(tmp_path, f"{repo}/files/app.py.md", "# app\ncheckout code\n")
    add_fact(tmp_path, "checkout uses sqlite")
    mem = tmp_path / ".k0-mem" / "General_mem" / "Mem.md"
    mem.parent.mkdir(parents=True, exist_ok=True)
    mem.write_text("- checkout picked sqlite\n", encoding="utf-8")
    rebuild_index(tmp_path)

    tool = make_search_memory(tmp_path)
    assert tool.name == "search_memory"
    hits = _search(tool)

    by_id = {hit["id"]: hit for hit in hits}
    assert by_id[f"{repo}:file:app.py"]["kind"] == "file"
    assert by_id["session:fact:1"]["kind"] == "fact"
    assert by_id["mem:1"]["kind"] == "mem"
    assert by_id["mem:1"]["body"] == "checkout picked sqlite"


def test_search_memory_symbol_hits_list_call_edges(tmp_path):
    repo = tmp_path.name
    _note(tmp_path, f"{repo}/functions/pkg.app.run.md", "# run\ncheckout\n")
    graph = tmp_path / ".k0-mem" / "Project_context" / repo / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "nodes": [],
                "edges": [
                    {
                        "from": f"{repo}:fn:pkg.app.run",
                        "to": f"{repo}:fn:pkg.util.helper",
                        "kind": "calls",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rebuild_index(tmp_path)

    hits = _search(make_search_memory(tmp_path))

    assert hits[0]["calls"] == [f"{repo}:fn:pkg.util.helper"]


def test_search_memory_honors_limit(tmp_path):
    repo = tmp_path.name
    for name in ("a", "b", "c"):
        _note(tmp_path, f"{repo}/files/{name}.py.md", "# sharedword\n")
    rebuild_index(tmp_path)

    hits = _search(make_search_memory(tmp_path), query="sharedword", limit=2)

    assert len(hits) == 2


def test_search_memory_builds_a_missing_index_once(tmp_path):
    """Without a rebuilt index the tool builds it once, not per call."""
    repo = tmp_path.name
    _note(tmp_path, f"{repo}/files/app.py.md", "# app\ncheckout code\n")
    tool = make_search_memory(tmp_path)
    assert not (tmp_path / ".k0-mem" / "index.sqlite").exists()

    first = _search(tool)
    second = _search(tool)

    assert first == second
    assert first[0]["id"] == f"{repo}:file:app.py"


def test_search_memory_does_not_rebuild_an_existing_index(tmp_path):
    """A note added after the rebuild stays invisible: no per-call rebuild."""
    repo = tmp_path.name
    _note(tmp_path, f"{repo}/files/old.py.md", "# old\ncheckout\n")
    rebuild_index(tmp_path)
    _note(tmp_path, f"{repo}/files/new.py.md", "# new\ncheckout\n")

    hits = _search(make_search_memory(tmp_path))

    assert [hit["id"] for hit in hits] == [f"{repo}:file:old.py"]


def test_search_memory_with_no_memory_returns_no_hits(tmp_path):
    tool = make_search_memory(tmp_path)

    assert _search(tool) == []
    # and the lazy index it built is a real, empty FTS index
    conn = sqlite3.connect(tmp_path / ".k0-mem" / "index.sqlite")
    try:
        assert conn.execute("SELECT COUNT(*) FROM memory").fetchone() == (0,)
    finally:
        conn.close()
