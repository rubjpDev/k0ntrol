"""Derived FTS index and retrieval gate for code memory."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

from k0ntrol.harness.init import RULES
from k0ntrol.modespec import ModeSpec, load_mode

Invoke = Callable[[ModeSpec, str], str]

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


def agent_memory(
    query: str,
    *,
    root: Path,
    config: dict,
    invoke: Invoke,
) -> str:
    """Rules (if customized) plus gated retrieval, for injection into a prompt."""
    sections: list[str] = []
    rules = custom_rules_text(root)
    if rules:
        sections.append(rules)
    if has_code_memory(root):
        retrieved = memory_context(query, root=root, config=config, invoke=invoke)
        if retrieved:
            sections.append(retrieved)
    return "\n\n".join(sections)


def memory_context(
    question: str,
    *,
    root: Path,
    config: dict,
    invoke: Invoke,
) -> str:
    """Rebuild FTS, run the retrieval gate, and return markdown to inject."""
    rebuild_index(root)
    top_k = config.get("retrieval_top_k", 5)
    candidates = query_index(root, question, top_k)
    picked = _run_gate(question, candidates, top_k, config, invoke)
    return _join_entries(picked)


def rebuild_index(root: Path) -> Path:
    mem_dir = root / ".k0-mem"
    mem_dir.mkdir(parents=True, exist_ok=True)
    index_path = mem_dir / INDEX_NAME
    conn = sqlite3.connect(index_path)
    try:
        _replace_index(conn, mem_dir / "Project_context")
    finally:
        conn.close()
    return index_path


def query_index(root: Path, question: str, limit: int) -> list[dict[str, str]]:
    conn = sqlite3.connect(root / ".k0-mem" / INDEX_NAME)
    try:
        return _select_matches(conn, question, limit)
    finally:
        conn.close()


def _replace_index(conn: sqlite3.Connection, context_root: Path) -> None:
    conn.execute("DROP TABLE IF EXISTS memory")
    conn.execute("CREATE VIRTUAL TABLE memory USING fts5(id, path, body)")
    rows = [_entry_row(path, context_root) for path in _markdown_files(context_root)]
    conn.executemany("INSERT INTO memory (id, path, body) VALUES (?, ?, ?)", rows)
    conn.commit()


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


def _run_gate(
    question: str,
    candidates: list[dict[str, str]],
    top_k: int,
    config: dict,
    invoke: Invoke,
) -> list[dict[str, str]]:
    mode = load_mode("retrieval_gate", config)
    raw = invoke(mode, _gate_prompt(question, candidates, top_k))
    return _parse_gate(raw, candidates, top_k)


def _gate_prompt(question: str, candidates: list[dict[str, str]], top_k: int) -> str:
    lines = [
        f"Pick at most {top_k} candidate ids if the question needs code memory.",
        "",
        "Question:",
        question,
        "",
        "Candidates:",
    ]
    lines.extend(_candidate_lines(candidates))
    lines.append("")
    lines.append("Reply with retrieve: yes and ids, or retrieve: no.")
    return "\n".join(lines)


def _candidate_lines(candidates: list[dict[str, str]]) -> list[str]:
    if not candidates:
        return ["- (none)"]
    return [f"- {item['id']} {item['path']}" for item in candidates]


def _parse_gate(
    raw: str,
    candidates: list[dict[str, str]],
    top_k: int,
) -> list[dict[str, str]]:
    if not _wants_retrieve(raw):
        return []
    by_id = {item["id"]: item for item in candidates}
    return _picked_entries(_picked_ids(raw), by_id, top_k)


def _wants_retrieve(raw: str) -> bool:
    for line in raw.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("retrieve:"):
            return stripped.split(":", 1)[1].strip() == "yes"
    return False


def _picked_ids(raw: str) -> list[str]:
    ids = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            ids.append(stripped[2:].strip())
    return ids


def _picked_entries(
    ids: list[str],
    by_id: dict[str, dict[str, str]],
    top_k: int,
) -> list[dict[str, str]]:
    picked: list[dict[str, str]] = []
    for token in ids:
        _append_pick(picked, by_id.get(token), top_k)
        if len(picked) == top_k:
            break
    return picked


def _append_pick(
    picked: list[dict[str, str]],
    item: dict[str, str] | None,
    top_k: int,
) -> None:
    if item is None or item in picked or len(picked) == top_k:
        return
    picked.append(item)


def _join_entries(entries: list[dict[str, str]]) -> str:
    return "\n\n".join(item["body"] for item in entries)
