"""Harness step that reflects a session transcript into session memory."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from k0ntrol.backends.invoke import current_event_context, set_event_context
from k0ntrol.harness.session_memory import (
    add_episode,
    add_fact,
    add_mem_proposal,
    decide_mem_proposal,
    list_decided_mem_proposals,
    list_pending_mem_proposals,
)
from k0ntrol.modespec import ModeSpec, load_mode

Invoke = Callable[[ModeSpec, str], str]
MEM_RELATIVE = Path(".k0-mem") / "General_mem" / "Mem.md"


def run_reflect(
    *,
    root: Path,
    config: dict,
    invoke: Invoke,
    transcript: Path | None = None,
    run: str = "",
) -> None:
    """Invoke reflect; store facts, episodes, and pending Mem.md proposals. D22."""
    run = run or current_event_context()["run"]
    set_event_context(run=run, step="reflect")
    mode = load_mode("reflect", config)
    raw = invoke(mode, _reflect_prompt(root, transcript))
    facts, episodes, mems = _parse_reflect(raw)
    for body in facts:
        add_fact(root, body)
    for body in episodes:
        add_episode(root, body)
    for body in mems:
        add_mem_proposal(root, body)


def offer_mem_proposals(
    root: Path,
    read_line: Callable[[], str],
    write: Callable[[str], None],
) -> None:
    """Show pending Mem.md proposals at the session prompt, not mid-turn."""
    for item in list_pending_mem_proposals(root):
        write(f"Mem.md += {item['body']}? y/n")
        apply_mem_decision(root, int(item["id"]), _read_yn(read_line, write) == "y")


def apply_mem_decision(root: Path, proposal_id: int, approved: bool) -> None:
    body = decide_mem_proposal(root, proposal_id, approved)
    if approved:
        _append_mem(root, body)


def _read_yn(read_line: Callable[[], str], write: Callable[[str], None]) -> str:
    while True:
        answer = read_line().strip().lower()
        if answer in {"y", "n"}:
            return answer
        write("y or n")


def _append_mem(root: Path, body: str) -> None:
    path = root / MEM_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    prefix = "" if not existing or existing.endswith("\n") else "\n"
    path.write_text(f"{existing}{prefix}- {body}\n", encoding="utf-8")


def _reflect_prompt(root: Path, transcript: Path | None) -> str:
    sections = ["Session transcript:\n" + (_transcript_text(transcript) or "(empty)")]
    decided = _decided_mem_prompt(root)
    if decided:
        sections.append(decided)
    return "\n\n".join(sections)


def _decided_mem_prompt(root: Path) -> str:
    lines = [f"{row['status']}: {row['body']}" for row in list_decided_mem_proposals(root)]
    if not lines:
        return ""
    return "Already decided Mem.md proposals (do not propose again):\n" + "\n".join(
        lines
    )


def _transcript_text(transcript: Path | None) -> str:
    if transcript is None or not transcript.is_file():
        return ""
    return transcript.read_text(encoding="utf-8")


def _parse_reflect(raw: str) -> tuple[list[str], list[str], list[str]]:
    facts: list[str] = []
    episodes: list[str] = []
    mems: list[str] = []
    for line in raw.splitlines():
        _append_parsed(line, facts, episodes, mems)
    return facts, episodes, mems


def _append_parsed(
    line: str, facts: list[str], episodes: list[str], mems: list[str]
) -> None:
    stripped = line.strip()
    lower = stripped.lower()
    for prefix, bucket in (("fact:", facts), ("episode:", episodes), ("mem:", mems)):
        if lower.startswith(prefix):
            _append_value(bucket, stripped)
            return


def _append_value(bucket: list[str], stripped: str) -> None:
    text = stripped.split(":", 1)[1].strip()
    if text.startswith("- "):
        text = text[2:].strip()
    if text:
        bucket.append(text)
