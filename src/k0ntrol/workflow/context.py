"""The per-run context graph nodes close over.

Everything a node needs that cannot live in state (the repo root, the
project config, the tracker, the chat-model seam, the event writer, the
built agents) hangs off ``RunContext``. Nodes are closures over the
context, so a graph compiled for one run never shares agent objects with
another. Agents build lazily on first use: resuming at a gate constructs
no model at all.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model

from k0ntrol.agents import K0Agent, agent_tools, allowlist_for, build_agent
from k0ntrol.events import META_ATTEMPT, EventWriter, K0EventHandler
from k0ntrol.langfuse import RunTrace
from k0ntrol.memory.scan import run_scan
from k0ntrol.otel import otel_callbacks
from k0ntrol.tools.run_tests import DEFAULT_TEST_TIMEOUT
from k0ntrol.tools.slice_guard import SliceStart
from k0ntrol.tools.slice_journal import SliceJournal
from k0ntrol.workflow.change import default_base_branch, worktree_snapshot_dir


@dataclass
class RunContext:
    """What one run's nodes share; bound to a run id by ``bind``."""

    root: Path
    cfg: dict
    kind: str
    init: Callable = init_chat_model
    tracker: Any = None
    scan: Callable = run_scan
    _writer: EventWriter | None = None
    handler: K0EventHandler | None = None
    trace: RunTrace | None = None
    session: str | None = None  # the REPL sitting's id, when there is one
    _agents: dict[str, K0Agent] = field(default_factory=dict)

    def bind(self, writer: EventWriter) -> None:
        """Attach the run's event writer and its callback handler."""
        self._writer = writer
        self.handler = K0EventHandler(writer)

    @property
    def writer(self) -> EventWriter:
        """The bound event writer; ``bind`` runs before any node does."""
        assert self._writer is not None
        return self._writer

    @property
    def timeout(self) -> float:
        return self.cfg.get("test_timeout", DEFAULT_TEST_TIMEOUT)

    @property
    def run_id(self) -> str:
        """The run id the bound event writer carries."""
        run = self.writer.run
        assert run is not None
        return run

    @property
    def base_branch(self) -> str:
        """The branch k0/<n>-<slug> cuts from: configured or origin's HEAD."""
        return self.cfg.get("base_branch") or default_base_branch(self.root)

    def agent(
        self,
        name: str,
        *,
        slot: str | None = None,
        read_only: bool = False,
        slice_start: SliceStart = "green",
    ) -> K0Agent:
        """The named agent, built once per run on first use.

        ``slot`` builds a second instance of the same agent — its own
        tools and slice guard — for a phase that must not share mutable
        tool state with the agent's main instance (``/implement``'s seam
        proposal vs its build). ``read_only`` strips the mutating tools
        for proposal phases. ``slice_start`` is the new instance's guard
        start state — a review redo builds IMPLEMENTER in refactor (#119).
        """
        key = name if slot is None else f"{name}:{slot}"
        if key not in self._agents:
            names = allowlist_for(
                name, tracker=self.tracker is not None, read_only=read_only
            )
            tools = agent_tools(
                name,
                root=self.root,
                cfg=self.cfg,
                tracker=self.tracker,
                init=self.init,
                read_only=read_only,
                slice_start=slice_start,
                journal=self._slice_journal(name),
            )
            self._agents[key] = build_agent(
                name,
                self.cfg,
                tools,
                root=self.root,
                init=self.init,
                allowlist=names,
            )
        return self._agents[key]

    def _slice_journal(self, name: str) -> SliceJournal | None:
        """IMPLEMENTER's slice journal: durable scratch under the run's
        ``.k0-mem/snapshots/<run>/slice`` dir for the recovery gate (#174).
        A new instance means a new slice id, so its first write reinitialises
        whatever an earlier slice or restore left there."""
        writer = self._writer
        if name != "implementer" or writer is None or writer.run is None:
            return None
        return SliceJournal(worktree_snapshot_dir(self.root, writer.run) / "slice")

    def agent_config(self, attempt: int) -> dict:
        """The config for one agent run: handlers + attempt tag.

        The optional Langfuse and OTel handlers sit beside the events
        handler when tracing is on (#49).
        """
        callbacks = [self.handler]
        if self.trace is not None and self.trace.handler is not None:
            callbacks.append(self.trace.handler)
        callbacks.extend(otel_callbacks(self.trace))
        return {
            "callbacks": callbacks,
            "metadata": {META_ATTEMPT: attempt},
        }
