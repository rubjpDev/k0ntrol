"""Derived FTS index over project memory, plus prompt-facing memory text."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from k0ntrol.memory.layout import ARCHITECTURE, CODE_RULES, RULES, STRATEGY
from k0ntrol.memory.session import fact_entry_id, list_facts

INDEX_NAME = "index.sqlite"
KIND_PREFIX = {"files": "file", "functions": "fn", "classes": "class"}


def has_code_memory(root: Path) -> bool:
    context = root / ".k0-mem" / "Project_context"
    return context.is_dir() and any(context.rglob("*.md"))


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
    index_path = mem_dir / INDEX_NAME
    conn = sqlite3.connect(index_path)
    try:
        _replace_index(conn, mem_dir / "Project_context", list_facts(root))
    finally:
        conn.close()
    return index_path


def query_index(root: Path, question: str, limit: int) -> list[dict[str, str]]:
    conn = sqlite3.connect(root / ".k0-mem" / INDEX_NAME)
    try:
        return _select_matches(conn, question, limit)
    finally:
        conn.close()


def _replace_index(
    conn: sqlite3.Connection,
    context_root: Path,
    facts: list[dict[str, object]],
) -> None:
    conn.execute("DROP TABLE IF EXISTS memory")
    conn.execute("CREATE VIRTUAL TABLE memory USING fts5(id, path, body)")
    rows = [_entry_row(path, context_root) for path in _markdown_files(context_root)]
    rows.extend(_fact_rows(facts))
    conn.executemany("INSERT INTO memory (id, path, body) VALUES (?, ?, ?)", rows)
    conn.commit()


def _fact_rows(facts: list[dict[str, object]]) -> list[tuple[str, str, str]]:
    return [
        (fact_entry_id(int(item["id"])), "session.sqlite", str(item["body"]))
        for item in facts
    ]


def _markdown_files(context_root: Path) -> list[Path]:
    if not context_root.is_dir():
        return []
    return sorted(path for path in context_root.rglob("*.md") if path.is_file())


def _entry_row(path: Path, context_root: Path) -> tuple[str, str, str]:
    rel = path.relative_to(context_root).as_posix()
    return (_node_id(path, context_root), rel, path.read_text(encoding="utf-8"))


def _node_id(path: Path, context_root: Path) -> str:
    parts = path.relative_to(context_root).parts
    if len(parts) < 3:
        return path.relative_to(context_root).as_posix()
    repo, kind, *rest = parts
    stem = Path(*rest).as_posix()
    if stem.endswith(".md"):
        stem = stem[:-3]
    prefix = KIND_PREFIX.get(kind, kind)
    return f"{repo}:{prefix}:{stem}"


def _select_matches(
    conn: sqlite3.Connection,
    question: str,
    limit: int,
) -> list[dict[str, str]]:
    fts = _fts_query(question)
    if not fts:
        rows = conn.execute(
            "SELECT id, path, body FROM memory LIMIT ?",
            (limit,),
        ).fetchall()
        return _rows_as_entries(rows)
    rows = conn.execute(
        "SELECT id, path, body FROM memory WHERE memory MATCH ? ORDER BY rank LIMIT ?",
        (fts, limit),
    ).fetchall()
    return _rows_as_entries(rows)


def _rows_as_entries(rows: list[tuple[str, str, str]]) -> list[dict[str, str]]:
    return [{"id": row[0], "path": row[1], "body": row[2]} for row in rows]


def _fts_query(question: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9_]+", question)
    return " OR ".join(tokens)
