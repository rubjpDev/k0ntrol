"""Derived FTS index over project memory, plus prompt-facing memory text.

One index, `.k0-mem/index.sqlite` (gitignored), rebuilt once per run or
`/ask` — never per search. It covers code-graph notes under
`Project_context/`, `Mem.md` bullets and session facts. Every hit carries
`id`, `kind`, `path` and `body`; symbol hits (`fn`, `class`) also list
their `calls` / `called_by` edges from `graph.json`.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from k0ntrol.memory.layout import ARCHITECTURE, CODE_RULES, RULES, STRATEGY
from k0ntrol.memory.session import fact_entry_id, list_facts

INDEX_NAME = "index.sqlite"
KIND_PREFIX = {"files": "file", "functions": "fn", "classes": "class"}
MEM_PATH = "General_mem/Mem.md"
SYMBOL_KINDS = ("fn", "class")


def has_code_memory(root: Path) -> bool:
    context = root / ".k0-mem" / "Project_context"
    return context.is_dir() and any(context.rglob("*.md"))


def index_path(root: Path) -> Path:
    return root / ".k0-mem" / INDEX_NAME


def rules_text(root: Path) -> str:
    path = root / ".k0-mem" / "General_mem" / "Rules.md"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def custom_rules_text(root: Path) -> str:
    text = rules_text(root)
    if not text or text == RULES:
        return ""
    return text


def custom_topology_text(root: Path) -> str:
    topo = root / ".k0-mem" / "Project_topology" / root.name
    sections = [
        _custom_file_text(topo / name, template)
        for name, template in (
            ("architecture.md", ARCHITECTURE),
            ("strategy.md", STRATEGY),
            ("code_rules.md", CODE_RULES),
        )
    ]
    return "\n\n".join(section for section in sections if section)


def _custom_file_text(path: Path, template: str) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    if not text or text == template:
        return ""
    return text


def rebuild_index(root: Path) -> Path:
    mem_dir = root / ".k0-mem"
    mem_dir.mkdir(parents=True, exist_ok=True)
    path = index_path(root)
    conn = sqlite3.connect(path)
    try:
        _replace_index(conn, mem_dir, list_facts(root))
    finally:
        conn.close()
    return path


def query_index(root: Path, question: str, limit: int) -> list[dict[str, object]]:
    conn = sqlite3.connect(index_path(root))
    try:
        entries = _select_matches(conn, question, limit)
    finally:
        conn.close()
    return _with_call_edges(root, entries)


def _replace_index(
    conn: sqlite3.Connection,
    mem_dir: Path,
    facts: list[dict[str, object]],
) -> None:
    conn.execute("DROP TABLE IF EXISTS memory")
    conn.execute("CREATE VIRTUAL TABLE memory USING fts5(id, kind, path, body)")
    rows = [
        _entry_row(path, mem_dir / "Project_context")
        for path in _markdown_files(mem_dir / "Project_context")
    ]
    rows.extend(_mem_rows(mem_dir))
    rows.extend(_fact_rows(facts))
    conn.executemany(
        "INSERT INTO memory (id, kind, path, body) VALUES (?, ?, ?, ?)", rows
    )
    conn.commit()


def _fact_rows(facts: list[dict[str, object]]) -> list[tuple[str, str, str, str]]:
    return [
        (fact_entry_id(int(item["id"])), "fact", "session.sqlite", str(item["body"]))
        for item in facts
    ]


def _mem_rows(mem_dir: Path) -> list[tuple[str, str, str, str]]:
    path = mem_dir / MEM_PATH
    if not path.is_file():
        return []
    return [
        (f"mem:{number}", "mem", MEM_PATH, bullet)
        for number, bullet in enumerate(_bullets(path), start=1)
    ]


def _bullets(path: Path) -> list[str]:
    return [
        line[2:].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("- ") and line[2:].strip()
    ]


def _markdown_files(context_root: Path) -> list[Path]:
    if not context_root.is_dir():
        return []
    return sorted(path for path in context_root.rglob("*.md") if path.is_file())


def _entry_row(path: Path, context_root: Path) -> tuple[str, str, str, str]:
    rel = path.relative_to(context_root).as_posix()
    node_id, kind = _node_id_and_kind(path, context_root)
    return (node_id, kind, rel, path.read_text(encoding="utf-8"))


def _node_id_and_kind(path: Path, context_root: Path) -> tuple[str, str]:
    parts = path.relative_to(context_root).parts
    if len(parts) < 3:
        return path.relative_to(context_root).as_posix(), "note"
    repo, kind, *rest = parts
    stem = Path(*rest).as_posix()
    if stem.endswith(".md"):
        stem = stem[:-3]
    prefix = KIND_PREFIX.get(kind, kind)
    return f"{repo}:{prefix}:{stem}", prefix


def _select_matches(
    conn: sqlite3.Connection,
    question: str,
    limit: int,
) -> list[dict[str, object]]:
    fts = _fts_query(question)
    if not fts:
        rows = conn.execute(
            "SELECT id, kind, path, body FROM memory LIMIT ?",
            (limit,),
        ).fetchall()
        return _rows_as_entries(rows)
    rows = conn.execute(
        "SELECT id, kind, path, body FROM memory "
        "WHERE memory MATCH ? ORDER BY rank LIMIT ?",
        (fts, limit),
    ).fetchall()
    return _rows_as_entries(rows)


def _rows_as_entries(rows: list[tuple[str, str, str, str]]) -> list[dict[str, object]]:
    return [
        {"id": row[0], "kind": row[1], "path": row[2], "body": row[3]}
        for row in rows
    ]


def _fts_query(question: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9_]+", question)
    return " OR ".join(tokens)


def _with_call_edges(
    root: Path, entries: list[dict[str, object]]
) -> list[dict[str, object]]:
    if not any(entry["kind"] in SYMBOL_KINDS for entry in entries):
        return entries
    edges = _call_edges(root)
    for entry in entries:
        if entry["kind"] not in SYMBOL_KINDS:
            continue
        node = edges.get(str(entry["id"]), {})
        entry["calls"] = node.get("calls", [])
        entry["called_by"] = node.get("called_by", [])
    return entries


def _call_edges(root: Path) -> dict[str, dict[str, list[str]]]:
    by_node: dict[str, dict[str, list[str]]] = {}
    for graph in (root / ".k0-mem" / "Project_context").glob("*/graph.json"):
        for edge in _graph_edges(graph):
            by_node.setdefault(edge["from"], {}).setdefault("calls", []).append(
                edge["to"]
            )
            by_node.setdefault(edge["to"], {}).setdefault("called_by", []).append(
                edge["from"]
            )
    return by_node


def _graph_edges(graph_path: Path) -> list[dict]:
    try:
        data = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    edges = data.get("edges")
    if not isinstance(edges, list):
        return []
    return [
        edge
        for edge in edges
        if isinstance(edge, dict)
        and edge.get("kind") == "calls"
        and isinstance(edge.get("from"), str)
        and isinstance(edge.get("to"), str)
    ]
