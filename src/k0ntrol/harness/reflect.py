"""Harness step that reflects a session transcript into session memory."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from k0ntrol.backends.invoke import current_event_context, set_event_context
from k0ntrol.harness.session_memory import add_episode, add_fact
from k0ntrol.modespec import ModeSpec, load_mode

Invoke = Callable[[ModeSpec, str], str]


def run_reflect(
    *,
    root: Path,
    config: dict,
    invoke: Invoke,
    transcript: Path | None = None,
    run: str = "",
) -> None:
    """Invoke the reflect mode and store facts and episodes. D22: harness → mode."""
    run = run or current_event_context()["run"]
    set_event_context(run=run, step="reflect")
    mode = load_mode("reflect", config)
    raw = invoke(mode, _reflect_prompt(transcript))
    facts, episodes = _parse_reflect(raw)
    for body in facts:
        add_fact(root, body)
    for body in episodes:
        add_episode(root, body)


def _reflect_prompt(transcript: Path | None) -> str:
    text = _transcript_text(transcript)
    return "Session transcript:\n" + (text or "(empty)")


def _transcript_text(transcript: Path | None) -> str:
    if transcript is None or not transcript.is_file():
        return ""
    return transcript.read_text(encoding="utf-8")


def _parse_reflect(raw: str) -> tuple[list[str], list[str]]:
    facts: list[str] = []
    episodes: list[str] = []
    for line in raw.splitlines():
        _append_parsed(line, facts, episodes)
    return facts, episodes


def _append_parsed(line: str, facts: list[str], episodes: list[str]) -> None:
    stripped = line.strip()
    lower = stripped.lower()
    if lower.startswith("fact:"):
        _append_value(facts, stripped)
        return
    if lower.startswith("episode:"):
        _append_value(episodes, stripped)


def _append_value(bucket: list[str], stripped: str) -> None:
    text = stripped.split(":", 1)[1].strip()
    if text:
        bucket.append(text)
