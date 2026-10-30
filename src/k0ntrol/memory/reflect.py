"""The reflect helper: distills a session transcript into session memory.

``run_reflect`` turns one slice of the session transcript — everything
written since the last completed reflect, marked by a byte ``offset`` —
into session facts and pending ``Mem.md`` proposals through the ``reflect``
helper's structured output. ``Reflector`` owns the offsets and the
background threads: one fire per run outcome and one at ``/quit`` for
pending ``/ask`` turns, never a graph node, so it is never replayed on
resume and never fails a run.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from pathlib import Path

from langchain.chat_models import init_chat_model

from k0ntrol.events import (
    EventWriter,
    K0EventHandler,
    ProgressRenderer,
    events_path,
    new_chat_id,
)
from k0ntrol.helpers import structured_helper_invoker
from k0ntrol.memory.session import (
    add_fact,
    add_mem_proposal,
    decide_mem_proposal,
    list_decided_mem_proposals,
    list_pending_mem_proposals,
)

Invoke = Callable[[str], Mapping]
MEM_RELATIVE = Path(".k0-mem") / "General_mem" / "Mem.md"

# The reflect helper's structured output shape (with_structured_output).
REFLECT_SCHEMA = {
    "title": "reflect",
    "description": "Facts and Mem.md proposals distilled from a session transcript.",
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "session facts worth remembering",
        },
        "mem": {
            "type": "array",
            "items": {"type": "string"},
            "description": "project-wide Mem.md bullet proposals",
        },
    },
    "required": ["facts", "mem"],
}


def run_reflect(
    *,
    root: Path,
    invoke: Invoke,
    transcript: Path | None = None,
    offset: int = 0,
    end: int | None = None,
) -> None:
    """Invoke the reflect helper; store facts and pending Mem.md proposals."""
    raw = invoke(_reflect_prompt(root, transcript, offset, end))
    for body in _output_bodies(raw, "facts"):
        add_fact(root, body)
    for body in _output_bodies(raw, "mem"):
        add_mem_proposal(root, body)


def _output_bodies(raw: object, key: str) -> list[str]:
    """The helper's string list under ``key``; anything else is ignored."""
    values = raw.get(key) if isinstance(raw, Mapping) else None
    if not isinstance(values, list):
        return []
    return [value.strip() for value in values if isinstance(value, str) and value.strip()]


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


def _reflect_prompt(root: Path, transcript: Path | None, offset: int, end: int | None) -> str:
    sections = [
        "You are the reflect helper. Distill this part of the session "
        "transcript into durable session facts and, only where a fact is "
        "worth keeping project-wide, Mem.md bullet proposals.\n"
        "\nSession transcript:\n"
        + (_transcript_text(transcript, offset, end) or "(empty)")
    ]
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


def _transcript_text(transcript: Path | None, offset: int, end: int | None) -> str:
    if transcript is None or not transcript.is_file():
        return ""
    # Bytes, then a lossy decode: the offset can land mid-codepoint when a
    # reflect fired while the session was still writing the transcript.
    with transcript.open("rb") as file:
        file.seek(offset)
        size = -1 if end is None else max(0, end - offset)
        return file.read(size).decode("utf-8", errors="replace")


class Reflector:
    """Reflects one session transcript on background threads.

    ``after_run`` fires once per run outcome (``done``/``stuck``/``rejected``),
    ``finish`` fires once for ``/ask`` turns pending since the last reflect
    and then waits — ``/quit`` never returns while a reflect is running.
    Two byte offsets track the transcript: ``_dispatched`` ends the range
    handed to the last fire, ``_completed`` ends the prefix reflected
    successfully. A fire covers ``_completed`` up to the transcript size at
    dispatch, and only a success moves ``_completed`` — so a failed range
    is covered again by the next reflect, after a run or at ``/quit``.
    Reflect is not a graph node: it cannot delay an outcome, and a failure
    shows one line instead of failing the run.
    """

    def __init__(
        self,
        *,
        root: Path,
        transcript: Path,
        cfg: dict,
        init: Callable = init_chat_model,
        write: Callable[[str], None] | None = None,
        reflect: Callable = run_reflect,
        make_invoke: Callable[[K0EventHandler], Invoke] | None = None,
    ) -> None:
        self._root = root
        self._transcript = transcript
        self._write = write
        self._reflect = reflect
        self._threads: list[threading.Thread] = []
        self._dispatched = 0
        self._completed = 0
        self._ask_turns = 0
        self._cfg = cfg
        self._init = init
        self._make_invoke = make_invoke or self._reflect_invoker
        self._tail = threading.Event()
        self._tail.set()

    def ask_turn(self) -> None:
        """One ``/ask`` exchange happened since the last reflect check."""
        self._ask_turns += 1

    def after_run(self, run_id: str) -> None:
        """Fire a background reflect for an ended run; never blocks."""
        self._fire(run=run_id)

    def finish(self) -> None:
        """``/quit``: reflect pending ``/ask`` turns once, retry a failed
        range once, then wait."""
        self._wait()
        if self._completed < self._dispatched or (
            self._ask_turns and self._has_new_text()
        ):
            self._fire(chat=new_chat_id())
            self._wait()

    def _fire(self, *, run: str | None = None, chat: str | None = None) -> None:
        self._dispatched = _transcript_size(self._transcript)
        self._ask_turns = 0
        previous = self._tail
        self._tail = threading.Event()
        thread = threading.Thread(
            target=self._run,
            args=(self._dispatched, run, chat, previous, self._tail),
            daemon=True, name="k0-reflect",
        )
        self._threads.append(thread)
        thread.start()

    def _run(
        self, end: int, run: str | None, chat: str | None,
        previous: threading.Event, completed: threading.Event,
    ) -> None:
        try:
            handler = K0EventHandler(
                EventWriter(
                    events_path(self._root),
                    run=run,
                    chat=chat,
                    progress=self._progress(),
                )
            )
            invoke = self._make_invoke(handler)
            previous.wait()
            offset = self._completed
            if offset >= end:
                return
            self._reflect(
                root=self._root,
                invoke=invoke,
                transcript=self._transcript,
                offset=offset,
                end=end,
            )
            self._completed = end
        except Exception as error:
            if self._write is not None:
                self._write(f"reflect failed: {_short(error)}")
        finally:
            previous.wait()
            completed.set()

    def _progress(self) -> ProgressRenderer | None:
        if self._write is None:
            return None
        return ProgressRenderer(self._write)

    def _reflect_invoker(self, handler: K0EventHandler) -> Invoke:
        return structured_helper_invoker(
            self._cfg,
            "reflect",
            REFLECT_SCHEMA,
            init=self._init,
            config={"callbacks": [handler]},
        )

    def _has_new_text(self) -> bool:
        return self._dispatched < _transcript_size(self._transcript)

    def _wait(self) -> None:
        for thread in self._threads:
            thread.join()
        self._threads = []


def _transcript_size(transcript: Path) -> int:
    try:
        return transcript.stat().st_size
    except OSError:
        return 0


def _short(error: BaseException) -> str:
    return (str(error) or type(error).__name__)[:200]
