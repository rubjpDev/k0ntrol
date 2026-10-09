"""Facts and episodes for session memory in SQLite."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_NAME = "session.sqlite"


def add_fact(root: Path, body: str) -> int:
    return _insert(root, "facts", body)


def add_episode(root: Path, body: str) -> int:
    return _insert(root, "episodes", body)


def list_facts(root: Path) -> list[dict[str, object]]:
    return _list_rows(root, "facts")


def list_episodes(root: Path) -> list[dict[str, object]]:
    return _list_rows(root, "episodes")


def has_facts(root: Path) -> bool:
    return bool(list_facts(root))


def fact_entry_id(fact_id: int) -> str:
    return f"session:fact:{fact_id}"


def session_db_path(root: Path) -> Path:
    return root / ".k0-mem" / DB_NAME


def _insert(root: Path, table: str, body: str) -> int:
    conn = _connect(root)
    try:
        cursor = conn.execute(
            f"INSERT INTO {table} (body) VALUES (?)",
            (body,),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def _list_rows(root: Path, table: str) -> list[dict[str, object]]:
    if not session_db_path(root).is_file():
        return []
    conn = _connect(root)
    try:
        rows = conn.execute(f"SELECT id, body FROM {table} ORDER BY id").fetchall()
    finally:
        conn.close()
    return [{"id": row[0], "body": row[1]} for row in rows]


def _connect(root: Path) -> sqlite3.Connection:
    path = session_db_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, body TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY, body TEXT NOT NULL)"
    )
    conn.commit()
