"""The run module: the one seam between the REPL and the graphs.

``start(kind, input) -> Result`` and ``resume(run_id, decision) -> Result``,
where ``Result`` is ``Outcome(done|stuck|rejected)`` or
``Pending(gate, what to show)``. Gates never read input inside the graph:
the REPL shows the ``Pending``, reads the answer and calls ``resume``.
Progress reaches the REPL through the event writer, never return values.

Checkpoints are a ``SqliteSaver`` at ``.k0-mem/checkpoints.sqlite`` with
``durability="sync"``, strict msgpack and ``thread_id`` = run id, so a run
survives the process dying: a later ``resume`` compiles a fresh graph on
the same DB and picks up at the gate.

One run per checkout (ADR 0007): any unfinished checkpoint thread refuses
a second ``/fast`` with a git-worktree hint. A second ``/full`` gets the
unfinished run back as ``Unfinished`` (#46): the REPL asks whether to
resume it, and ``abandon`` drops its checkpoint thread on a ``n``.
"""

import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain.chat_models import init_chat_model
from langgraph.types import Command

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

from k0ntrol import __version__  # noqa: E402
from k0ntrol.events import (  # noqa: E402
    EventWriter,
    EventWriteError,
    ProgressRenderer,
    events_path,
    new_run_id,
)
from k0ntrol.langfuse import build_trace  # noqa: E402
from k0ntrol.memory.scan import ScanError, run_scan  # noqa: E402
from k0ntrol.providers import free_checks, has_errors, report_lines  # noqa: E402
from k0ntrol.tracker import TRUSTED_ASSOCIATIONS, TrackerError  # noqa: E402
from k0ntrol.tracker.github import GhTracker, subprocess_runner  # noqa: E402
from k0ntrol.workflow.change import (  # noqa: E402
    current_branch,
    dirty_paths,
    existing_branch,
)
from k0ntrol.workflow.context import RunContext  # noqa: E402
from k0ntrol.workflow.fast import build_fast_graph  # noqa: E402
from k0ntrol.workflow.full import build_full_graph  # noqa: E402
from k0ntrol.workflow.stuck import stuck_update  # noqa: E402

_BUILDERS = {"fast": build_fast_graph, "full": build_full_graph}

# The roster each run kind uses; the free doctor checks gate start on it.
# Every agent gets the bulk_read tool, so the helper is part of each roster
# (#61, same gate /ask already applies).
_ROSTER = {
    "fast": ("agents.tester", "agents.coder", "helpers.bulk_read"),
    "full": (
        "agents.spec",
        "agents.tester",
        "agents.coder",
        "agents.reviewer",
        "helpers.bulk_read",
    ),
}

# Outer recursion limit: each loop iteration costs two super-steps at most.
_OUTER_STEPS_PER_ATTEMPT = 4
_OUTER_STEPS_SLACK = 8


class RunRefused(Exception):
    """The run never started (or a resume answer was rejected)."""

    def __init__(self, lines: str | list[str]) -> None:
        self.lines = tuple([lines] if isinstance(lines, str) else lines)
        super().__init__("\n".join(self.lines))


@dataclass(frozen=True)
class Outcome:
    """A finished run: ``done``, ``stuck`` or ``rejected``."""

    run: str
    outcome: str
    reason: str | None = None
    node: str | None = None


@dataclass(frozen=True)
class Pending:
    """A run waiting at a gate: the question and answers to show."""

    run: str
    gate: str
    question: str
    answers: tuple[str, ...]
    detail: str = ""


@dataclass(frozen=True)
class Unfinished:
    """``/full`` found an unfinished run: resume it or abandon it."""

    run: str
    node: str


Result = Outcome | Pending | Unfinished


def checkpoints_path(root: Path) -> Path:
    """The checkpoint DB for the project rooted at ``root``."""
    return root / ".k0-mem" / "checkpoints.sqlite"


def start(
    kind: str,
    input: str,
    *,
    root: Path,
    cfg: dict,
    write: Callable[[str], None] | None = None,
    init: Callable = init_chat_model,
    check: Callable = free_checks,
    scan: Callable = run_scan,
    tracker=None,
    tracing: Callable = build_trace,
) -> Result:
    """Start a run of ``kind`` for ``input``.

    ``/full <text>`` drafts a brief; ``/full #<n>`` refines issue ``n``.
    Refuses — ``RunRefused`` — on bad usage, a missing tracker or work
    item, an untrusted issue body, a dirty tree (``.k0-mem/`` ignored),
    failed doctor checks, a failed rescan or (for ``/fast``) another open
    run. ``/full`` answers an open run with ``Unfinished`` instead.
    """
    if kind not in _BUILDERS:
        raise RunRefused(f"unknown run: {kind}")
    input_text = _checked_input(kind, input)
    tracker = _tracker_for(kind, cfg, root, tracker)
    conn, saver = _open_checkpoints(root)
    try:
        ctx = RunContext(
            root=root, cfg=cfg, kind=kind, init=init, tracker=tracker, scan=scan
        )
        graph = _BUILDERS[kind](ctx, saver)
        busy = _busy_run(saver, graph, kind)
        if busy is not None:
            return busy
        _doctor_gate(check, cfg, kind, root)
        _rescan(scan, root)
        item = _full_preflight(kind, input_text, tracker, root)
        run_id = new_run_id()
        _bind_events(ctx, root, run_id, write)
        ctx.trace = tracing(run_id)
        session_id = item.qualified_ref if item else None
        config = _runnable(run_id, ctx, session_id)
        try:
            ctx.writer.run_start(
                kind,
                work_item=item.qualified_ref if item else None,
                branch=current_branch(root),
                version=__version__,
            )
            graph.invoke(
                _initial_state(kind, input_text, cfg, item),
                config=config,
                durability="sync",
            )
            return _finish(ctx, graph, run_id, config)
        except EventWriteError:
            return _log_failed(ctx, graph, config, run_id, session_id)
    finally:
        conn.close()


def resume(
    run_id: str,
    decision: str | None,
    *,
    root: Path,
    cfg: dict,
    write: Callable[[str], None] | None = None,
    init: Callable = init_chat_model,
    scan: Callable = run_scan,
    tracker=None,
    tracing: Callable = build_trace,
) -> Result:
    """Continue run ``run_id`` with the human's ``decision``.

    ``decision=None`` answers nothing: a run at a gate shows it again, a
    run paused between nodes continues.

    Rebuilds the graph and checkpoint connection from scratch — this is
    also how a fresh process resumes a run after a crash or a ``/quit``
    at a gate.
    """
    conn, saver = _open_checkpoints(root)
    try:
        kind = _run_kind(saver, run_id)
        if kind is None or kind not in _BUILDERS:
            raise RunRefused(f"no run {run_id} in this checkout")
        tracker = _tracker_for(kind, cfg, root, tracker)
        ctx = RunContext(
            root=root, cfg=cfg, kind=kind, init=init, tracker=tracker, scan=scan
        )
        graph = _BUILDERS[kind](ctx, saver)
        state = graph.get_state(_thread(run_id))
        _refuse_finished(state, run_id)
        _bind_events(ctx, root, run_id, write)
        ctx.trace = tracing(run_id)
        session_id = _session_id(ctx, state.values)
        config = _runnable(run_id, ctx, session_id)
        try:
            ctx.writer.run_start(
                kind, resumed=True, branch=current_branch(root), version=__version__
            )
            _advance(ctx, graph, state, decision, config)
            return _finish(ctx, graph, run_id, config)
        except EventWriteError:
            return _log_failed(ctx, graph, config, run_id, session_id)
    finally:
        conn.close()


def abandon(run_id: str, *, root: Path) -> None:
    """Drop run ``run_id``'s checkpoint thread: the run can no longer resume."""
    conn, saver = _open_checkpoints(root)
    try:
        saver.delete_thread(run_id)
    finally:
        conn.close()


def open_runs(saver: SqliteSaver, graph) -> list[tuple[str, str]]:
    """``[(run_id, pending_node)]`` for every unfinished checkpoint thread."""
    threads = {
        tuple_.config["configurable"]["thread_id"]
        for tuple_ in saver.list(None)
    }
    open_ = []
    for thread_id in sorted(threads):
        state = graph.get_state(_thread(thread_id))
        if state.next:
            open_.append((thread_id, state.next[0]))
    return open_


def _open_checkpoints(root: Path) -> tuple[sqlite3.Connection, SqliteSaver]:
    path = checkpoints_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return conn, SqliteSaver(conn)


def _busy_run(saver: SqliteSaver, graph, kind: str) -> Unfinished | None:
    busy = open_runs(saver, graph)
    if not busy:
        return None
    run_id, node = busy[0]
    if kind == "full":
        return Unfinished(run=run_id, node=node)
    raise RunRefused(
        f"a run is already open in this checkout: {run_id} at {node}\n"
        "resume it, or run k0 from another git worktree: "
        "git worktree add <path>"
    )


def _doctor_gate(check: Callable, cfg: dict, kind: str, root: Path) -> None:
    checks = check(cfg, names=_ROSTER[kind], root=root)
    if has_errors(checks):
        raise RunRefused(report_lines(checks))


def _rescan(scan: Callable, root: Path) -> None:
    try:
        scan(root)
    except ScanError as error:
        raise RunRefused(f"scan failed: {error}") from error


def _bind_events(
    ctx: RunContext, root: Path, run_id: str, write: Callable | None
) -> None:
    path = events_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = EventWriter(
        path,
        run=run_id,
        progress=ProgressRenderer(write) if write is not None else None,
    )
    ctx.bind(writer)


def _checked_input(kind: str, input) -> str:
    """Non-empty input in the kind's accepted forms, or a usage refusal."""
    text = str(input).strip()
    if not text or (kind == "full" and not _full_usage_ok(text)):
        raise RunRefused(_usage(kind))
    return text


def _usage(kind: str) -> str:
    if kind == "full":
        return "usage: /full <task> | /full #<issue>"
    return f"usage: /{kind} <task>"


def _full_usage_ok(input_text: str) -> bool:
    """``/full``'s two forms: free text, or ``#`` plus an issue number."""
    return not input_text.startswith("#") or input_text[1:].strip().isdigit()


def _tracker_for(kind: str, cfg: dict, root: Path, tracker):
    """``/full`` cannot run trackerless: build the configured adapter."""
    if kind != "full" or tracker is not None:
        return tracker
    if cfg.get("tracker") != "github":
        raise RunRefused("/full needs a tracker; run k0 init --tracker github")
    return GhTracker(subprocess_runner(root))


def _log_failed(
    ctx: RunContext, graph, config: dict, run_id: str, session_id
) -> Outcome:
    """A broken event log ends the run stuck (events contract, #16).

    ``/full`` still runs the stuck effects from where the run stood (#46);
    there is no ``run_end`` — the log cannot take it.
    """
    state = graph.get_state(config)
    reason, node = None, None
    if ctx.kind == "full" and state.next:
        state = _stuck_by_log(graph, config, state)
        reason, node = "event_log", state.values.get("node")
    _trace_finish(ctx, "stuck", session_id, state.values)
    return Outcome(run=run_id, outcome="stuck", reason=reason, node=node)


def _stuck_by_log(graph, config: dict, state):
    """Run the stuck effects; a pass already marked ``event_log`` just continues.

    Resuming a run whose log is still broken lands here again, and must
    pick the effects up where the failed tracker write left them.
    """
    if state.values.get("reason") != "event_log":
        graph.update_state(
            config, stuck_update(state.next[0], state.values), as_node="limit_hit"
        )
    graph.invoke(None, config=config, durability="sync")
    return graph.get_state(config)


def _full_preflight(kind: str, input_text: str, tracker, root: Path):
    """``/full``'s last refusals: a dirty tree, then the work item read."""
    if kind != "full":
        return None
    _refuse_dirty(root)
    return _work_item(input_text, tracker, root)


def _refuse_dirty(root: Path) -> None:
    dirty = dirty_paths(root)
    if dirty:
        raise RunRefused(
            ["refusing to run /full on a dirty working tree:", *dirty]
        )


def _work_item(input_text: str, tracker, root: Path):
    """``/full #<n>``'s issue: must exist and come from a trusted author."""
    if not input_text.startswith("#"):
        return None
    ref = input_text[1:].strip()
    try:
        item = tracker.read_work_item(ref)
        _refuse_ended_work(item, tracker, root)
    except TrackerError as error:
        raise RunRefused(str(error)) from error
    if item.author_association not in TRUSTED_ASSOCIATIONS:
        raise RunRefused(
            f"work item #{ref} was written by {item.author} "
            f"({item.author_association}), not a trusted author — restate "
            "the work yourself: /full <task>"
        )
    return item


def _refuse_ended_work(item, tracker, root: Path) -> None:
    """A closed issue, or a merged or closed PR, is not worked on again."""
    if item.state != "open":
        raise TrackerError(f"work item #{item.ref} is {item.state}")
    branch = existing_branch(root, item.ref)
    change = tracker.read_change(branch) if branch else None
    if change is not None and change.state != "open":
        raise TrackerError(
            f"pull request #{change.ref} for {branch} is {change.state}"
        )


def _initial_state(kind: str, input_text: str, cfg: dict, item=None) -> dict:
    state = {
        "kind": kind,
        "task": str(input_text).strip(),
        "start_ms": _now_ms(),
        "max_attempts": cfg.get("max_attempts", 2),
        "test_files": [],
        "code_files": [],
        "tester_attempts": 0,
        "coder_attempts": 0,
        "tests_red": False,
        "validator_ok": False,
        "limit": None,
    }
    if kind == "full":
        state.update(_full_state(state["task"], item))
    return state


def _full_state(task: str, item) -> dict:
    """The ``/full`` fields layered over the shared run state."""
    state = {
        "task": task,
        "item_ref": None,
        "item_title": "",
        "refining": False,
        "branch": "",
        "brief": None,
        "spec_attempts": 0,
        "spec_approved": False,
        "spec_rejections": [],
        "spec_feedback": None,
        "test_approved": False,
        "test_rejections": [],
        "test_feedback": None,
    }
    if item is not None:
        state["task"] = item.body
        state["item_ref"] = item.ref
        state["item_title"] = item.title
        state["refining"] = True
    return state


def _runnable(run_id: str, ctx: RunContext, session_id=None) -> dict:
    config = _thread(run_id)
    config["callbacks"] = [ctx.handler]
    # Always explicit (langgraph's default is a safety net, not a budget):
    # each loop-back costs at most two super-steps per agent node.
    attempts = ctx.cfg.get("max_attempts", 2)
    slack = 16 if ctx.kind == "full" else _OUTER_STEPS_SLACK
    config["recursion_limit"] = _OUTER_STEPS_PER_ATTEMPT * attempts + slack
    if ctx.trace is not None and ctx.trace.handler is not None:
        config["callbacks"].append(ctx.trace.handler)
        config["metadata"] = _langfuse_metadata(ctx.kind, session_id)
    return config


def _langfuse_metadata(kind: str, session_id) -> dict:
    """Root-run metadata the Langfuse handler reads: session + kind tag."""
    metadata = {"langfuse_tags": [kind]}
    if session_id:
        metadata["langfuse_session_id"] = session_id
    return metadata


def _session_id(ctx: RunContext, values: dict) -> str | None:
    """The run's work item as a session id, resolved through the tracker."""
    ref = values.get("item_ref")
    if not ref or ctx.tracker is None:
        return None
    try:
        return ctx.tracker.read_work_item(ref).qualified_ref
    except TrackerError:
        return None


def _trace_metadata(ctx: RunContext, values: dict) -> dict:
    """The trace's metadata (#49): branch, change URL and k0 version."""
    metadata = {"k0_version": __version__}
    branch = values.get("branch") or current_branch(ctx.root)
    if branch:
        metadata["branch"] = branch
    if values.get("change_url"):
        metadata["change_url"] = values["change_url"]
    return metadata


def _trace_finish(ctx: RunContext, outcome: str, session_id, values: dict) -> None:
    """Close the run's trace — enabled-only, never raising (#49)."""
    if ctx.trace is None:
        return
    ctx.trace.finish(
        kind=ctx.kind,
        outcome=outcome,
        session_id=session_id,
        metadata=_trace_metadata(ctx, values),
    )


def _thread(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _run_kind(saver: SqliteSaver, run_id: str) -> str | None:
    """The run's kind, read from the checkpoint before a graph is built."""
    tuple_ = saver.get_tuple(_thread(run_id))
    if tuple_ is None:
        return None
    return tuple_.checkpoint["channel_values"].get("kind")


def _refuse_finished(state, run_id: str) -> None:
    if state.next or state.interrupts:
        return
    outcome = state.values.get("outcome", "finished")
    raise RunRefused(f"run {run_id} already ended {outcome}")


def _advance(
    ctx: RunContext, graph, state, decision: str | None, config: dict
) -> None:
    if state.interrupts and decision is None:
        return  # nothing answered: _finish shows the gate again
    if state.interrupts:
        payload = state.interrupts[0].value
        _check_decision(payload, decision)
        ctx.writer.gate(payload["gate"], _gate_result(decision))
        graph.invoke(Command(resume=decision), config=config, durability="sync")
        return
    # Paused between nodes (crash or kill): continue from the checkpoint.
    graph.invoke(None, config=config, durability="sync")


def _check_decision(payload: dict, decision: str) -> None:
    """The decision's verb must be one of the payload's answers.

    Matching is on the first word so ``retry [feedback]`` answers the
    ``limit_hit`` pause with its feedback intact (#42).
    """
    answers = [str(a) for a in payload.get("answers", [])]
    if answers and _decision_verb(decision) not in [a.lower() for a in answers]:
        raise RunRefused(
            f"answer {decision!r} is not one of: {', '.join(answers)}"
        )


def _decision_verb(decision: str) -> str:
    parts = str(decision).strip().split(maxsplit=1)
    return parts[0].lower() if parts else ""


# The first word decides; the rest is feedback for the looping-back node.
_DECISIONS = {
    "y": "approve",
    "yes": "approve",
    "ok": "approve",
    "n": "reject",
    "no": "reject",
    "redo": "reject",
}


def _gate_result(decision: str) -> str:
    return _DECISIONS.get(_decision_verb(decision), _decision_verb(decision))


def _finish(ctx: RunContext, graph, run_id: str, config: dict) -> Result:
    state = graph.get_state(config)
    if state.interrupts:
        if ctx.trace is not None:
            ctx.trace.flush()
        return _pending(run_id, state.interrupts[0].value)
    if state.next:
        raise RunRefused(
            f"run {run_id} is paused outside a gate; resume it to continue"
        )
    values = state.values
    outcome = values.get("outcome", "stuck")
    ctx.writer.run_end(
        outcome,
        _now_ms() - values.get("start_ms", _now_ms()),
        reason=values.get("reason"),
        node=values.get("node"),
    )
    _trace_finish(ctx, outcome, _session_id(ctx, values), values)
    return Outcome(
        run=run_id,
        outcome=outcome,
        reason=values.get("reason"),
        node=values.get("node"),
    )


def _pending(run_id: str, payload: dict) -> Pending:
    return Pending(
        run=run_id,
        gate=str(payload.get("gate", "")),
        question=str(payload.get("question", "")),
        answers=tuple(str(a) for a in payload.get("answers", ())),
        detail=str(payload.get("detail", "")),
    )


def _now_ms() -> int:
    return round(time.time() * 1000)
