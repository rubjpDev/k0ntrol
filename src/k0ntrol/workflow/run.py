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
a second ``/full`` or ``/fast`` with a git-worktree hint.
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
from k0ntrol.memory.scan import ScanError, run_scan  # noqa: E402
from k0ntrol.providers import free_checks, has_errors, report_lines  # noqa: E402
from k0ntrol.workflow.change import current_branch  # noqa: E402
from k0ntrol.workflow.context import RunContext  # noqa: E402
from k0ntrol.workflow.fast import build_fast_graph  # noqa: E402

_BUILDERS = {"fast": build_fast_graph}

# The roster each run kind uses; the free doctor checks gate start on it.
_ROSTER = {"fast": ("agents.tester", "agents.coder")}

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


Result = Outcome | Pending


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
) -> Result:
    """Start a run of ``kind`` (``"fast"`` today) for ``input``.

    Refuses — ``RunRefused`` — on bad usage, failed doctor checks, a
    failed rescan or another run open in the checkout.
    """
    if kind not in _BUILDERS:
        raise RunRefused(f"unknown run: {kind}")
    if not str(input).strip():
        raise RunRefused(f"usage: /{kind} <task>")
    conn, saver = _open_checkpoints(root)
    try:
        ctx = RunContext(root=root, cfg=cfg, kind=kind, init=init, tracker=tracker)
        graph = _BUILDERS[kind](ctx, saver)
        _refuse_if_busy(saver, graph)
        _doctor_gate(check, cfg, kind, root)
        _rescan(scan, root)
        run_id = new_run_id()
        _bind_events(ctx, root, run_id, write)
        config = _runnable(run_id, ctx)
        try:
            ctx.writer.run_start(
                kind, branch=current_branch(root), version=__version__
            )
            graph.invoke(
                _initial_state(kind, input, cfg), config=config, durability="sync"
            )
            return _finish(ctx, graph, run_id, config)
        except EventWriteError:
            # A broken event log ends the run stuck (events contract, #16).
            return Outcome(run=run_id, outcome="stuck")
    finally:
        conn.close()


def resume(
    run_id: str,
    decision: str,
    *,
    root: Path,
    cfg: dict,
    write: Callable[[str], None] | None = None,
    init: Callable = init_chat_model,
    tracker=None,
) -> Result:
    """Continue run ``run_id`` with the human's ``decision``.

    Rebuilds the graph and checkpoint connection from scratch — this is
    also how a fresh process resumes a run after a crash or a ``/quit``
    at a gate.
    """
    conn, saver = _open_checkpoints(root)
    try:
        kind = _run_kind(saver, run_id)
        if kind is None or kind not in _BUILDERS:
            raise RunRefused(f"no run {run_id} in this checkout")
        ctx = RunContext(root=root, cfg=cfg, kind=kind, init=init, tracker=tracker)
        graph = _BUILDERS[kind](ctx, saver)
        state = graph.get_state(_thread(run_id))
        _refuse_finished(state, run_id)
        _bind_events(ctx, root, run_id, write)
        config = _runnable(run_id, ctx)
        try:
            ctx.writer.run_start(
                kind, resumed=True, branch=current_branch(root), version=__version__
            )
            _advance(ctx, graph, state, decision, config)
            return _finish(ctx, graph, run_id, config)
        except EventWriteError:
            return Outcome(run=run_id, outcome="stuck")
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


def _refuse_if_busy(saver: SqliteSaver, graph) -> None:
    busy = open_runs(saver, graph)
    if not busy:
        return
    run_id, node = busy[0]
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


def _initial_state(kind: str, input: str, cfg: dict) -> dict:
    return {
        "kind": kind,
        "task": str(input).strip(),
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


def _runnable(run_id: str, ctx: RunContext) -> dict:
    config = _thread(run_id)
    config["callbacks"] = [ctx.handler]
    # Always explicit (langgraph's default is a safety net, not a budget):
    # each loop-back costs at most two super-steps per agent node.
    attempts = ctx.cfg.get("max_attempts", 2)
    config["recursion_limit"] = _OUTER_STEPS_PER_ATTEMPT * attempts + _OUTER_STEPS_SLACK
    return config


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


def _advance(ctx: RunContext, graph, state, decision: str, config: dict) -> None:
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


def _gate_result(decision: str) -> str:
    verb = _decision_verb(decision)
    return {"y": "approve", "n": "reject"}.get(verb, verb)


def _finish(ctx: RunContext, graph, run_id: str, config: dict) -> Result:
    state = graph.get_state(config)
    if state.interrupts:
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
