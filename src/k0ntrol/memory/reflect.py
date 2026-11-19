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
from collections import deque
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
from k0ntrol.otel import build_run_trace, trace_callbacks

Invoke = Callable[[str], Mapping]
MEM_RELATIVE = Path(".k0-mem") / "General_mem" / "Mem.md"
_ENVIRONMENT_KINDS = {
    "gate": "a new deterministic check for a mechanical mistake",
    "rule": "a coding rule for a judgement call",
    "hint": "a navigation hint where an agent got lost",
}

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
        "environment": {
            "type": "object",
            "description": "Read-only advice for the human after a rough run.",
            "properties": {
                kind: {"type": "array", "items": {"type": "string"}, "description": description}
                for kind, description in _ENVIRONMENT_KINDS.items()
            },
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
    rough_run: str | None = None,
) -> list[str]:
    """Store facts and Mem.md proposals; return display-only environment advice."""
    raw = invoke(_reflect_prompt(root, transcript, offset, end, rough_run))
    for body in _output_bodies(raw, "facts"):
        add_fact(root, body)
    for body in _output_bodies(raw, "mem"):
        add_mem_proposal(root, body)
    return _environment_lines(raw) if rough_run else []


def _environment_lines(raw: object) -> list[str]:
    environment = raw.get("environment") if isinstance(raw, Mapping) else None
    return [
        f"[{kind}] {body}"
        for kind in _ENVIRONMENT_KINDS
        for body in map(_one_line, _output_bodies(environment, kind))
        if body
    ]


def _one_line(body: str) -> str:
    return " ".join("".join(char if char.isprintable() else " " for char in body).split())


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
        apply_mem_decision(root, int(str(item["id"])), _read_yn(read_line, write) == "y")


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


def _reflect_prompt(
    root: Path, transcript: Path | None, offset: int, end: int | None, rough_run: str | None,
) -> str:
    instructions = (
        "You are the reflect helper. Distill this part of the session "
        "transcript into durable session facts and, only where a fact is "
        "worth keeping project-wide, Mem.md bullet proposals."
    )
    if rough_run:
        instructions += (
            f"\nThe run {rough_run}. In environment, suggest a gate for a mechanical "
            "mistake, a coding rule for a judgement call, or a navigation hint where "
            "an agent got lost. Only propose changes supported by this run; leave "
            "categories empty when there is no actionable lesson. The human sees "
            "the suggestions but they are never applied; do not copy them into Mem.md proposals."
        )
    else:
        instructions += "\nLeave environment empty; this is not a rough run."
    sections = [
        instructions,
        "Session transcript:\n" + (_transcript_text(transcript, offset, end) or "(empty)"),
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
    Rough runs keep their own end offsets until reflected successfully, so
    a retry can recover their advice without asking for advice on a clean run.
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
        propose: Callable[[list[str]], None] | None = None,
        reflect: Callable = run_reflect,
        make_invoke: Callable[[list], Invoke] | None = None,
        tracing: Callable = build_run_trace,
    ) -> None:
        self._root = root
        self._transcript = transcript
        self._write = write
        self._propose = propose
        self._reflect = reflect
        self._threads: list[threading.Thread] = []
        self._dispatched = 0
        self._rough_ranges: deque[tuple[int, str]] = deque()
        self._completed = 0
        self._ask_turns = 0
        self._cfg = cfg
        self._init = init
        self._make_invoke = make_invoke or self._reflect_invoker
        self._tracing = tracing
        self._tail = threading.Event()
        self._tail.set()

    def ask_turn(self) -> None:
        """One ``/ask`` exchange happened since the last reflect check."""
        self._ask_turns += 1

    def after_run(self, run_id: str, *, stuck: bool = False, redone: bool = False) -> None:
        """Fire a background reflect for an ended run; never blocks."""
        rough_run = "took a redo" if redone else None
        self._fire(run=run_id, rough_run="ended stuck" if stuck else rough_run)

    def finish(self) -> None:
        """``/quit``: reflect pending ``/ask`` turns once, retry a failed
        range once, then wait."""
        self._wait()
        if self._completed < self._dispatched or (
            self._ask_turns and self._has_new_text()
        ):
            self._fire(chat=new_chat_id())
            self._wait()

    def _fire(
        self, *, run: str | None = None, chat: str | None = None,
        rough_run: str | None = None,
    ) -> None:
        self._dispatched = _transcript_size(self._transcript)
        if rough_run is not None:
            self._rough_ranges.append((self._dispatched, rough_run))
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
        # Same trace id as the run or chat reflected on: its calls land in
        # that trace. Never ``finish``: that would overwrite the outcome tags.
        trace = self._tracing(run or chat)
        try:
            handler = K0EventHandler(
                EventWriter(
                    events_path(self._root),
                    run=run,
                    chat=chat,
                    progress=self._progress(),
                )
            )
            invoke = self._make_invoke(_callbacks(handler, trace))
            previous.wait()
            self._reflect_through(end, invoke)
        except Exception as error:
            if self._write is not None:
                self._write(f"reflect failed: {_short(error)}")
        finally:
            trace.flush()
            previous.wait()
            completed.set()

    def _reflect_through(self, end: int, invoke: Invoke) -> None:
        # Dispatch only appends; the ordered workers remove successful ranges.
        while self._rough_ranges and self._rough_ranges[0][0] <= end:
            rough_end, reason = self._rough_ranges[0]
            self._reflect_range(rough_end, invoke, reason)
            self._rough_ranges.popleft()
        self._reflect_range(end, invoke, None)

    def _reflect_range(self, end: int, invoke: Invoke, rough_run: str | None) -> None:
        if self._completed >= end:
            return
        proposals = self._reflect(
            root=self._root,
            invoke=invoke,
            transcript=self._transcript,
            offset=self._completed,
            end=end,
            rough_run=rough_run,
        )
        self._completed = end
        if proposals and self._propose is not None:
            self._propose(proposals)

    def _progress(self) -> ProgressRenderer | None:
        if self._write is None:
            return None
        return ProgressRenderer(self._write)

    def _reflect_invoker(self, callbacks: list) -> Invoke:
        return structured_helper_invoker(
            self._cfg,
            "reflect",
            REFLECT_SCHEMA,
            init=self._init,
            config={"callbacks": callbacks},
        )

    def _has_new_text(self) -> bool:
        return self._dispatched < _transcript_size(self._transcript)

    def _wait(self) -> None:
        for thread in self._threads:
            thread.join()
        self._threads = []


def _callbacks(handler: K0EventHandler, trace) -> list:
    """The events handler plus the optional Langfuse and OTel ones."""
    return [handler, *trace_callbacks(trace)]


def _transcript_size(transcript: Path) -> int:
    try:
        return transcript.stat().st_size
    except OSError:
        return 0


def _short(error: BaseException) -> str:
    return (str(error) or type(error).__name__)[:200]
