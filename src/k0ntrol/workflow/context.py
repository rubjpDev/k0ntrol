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
from k0ntrol.tools.run_tests import DEFAULT_TEST_TIMEOUT


@dataclass
class RunContext:
    """What one run's nodes share; bound to a run id by ``bind``."""

    root: Path
    cfg: dict
    kind: str
    init: Callable = init_chat_model
    tracker: Any = None
    writer: EventWriter | None = None
    handler: K0EventHandler | None = None
    _agents: dict[str, K0Agent] = field(default_factory=dict)

    def bind(self, writer: EventWriter) -> None:
        """Attach the run's event writer and its callback handler."""
        self.writer = writer
        self.handler = K0EventHandler(writer)

    @property
    def timeout(self) -> float:
        return self.cfg.get("test_timeout", DEFAULT_TEST_TIMEOUT)

    def agent(self, name: str) -> K0Agent:
        """The named agent, built once per run on first use."""
        if name not in self._agents:
            names = allowlist_for(name, tracker=self.tracker is not None)
            tools = agent_tools(
                name,
                root=self.root,
                cfg=self.cfg,
                tracker=self.tracker,
                init=self.init,
            )
            self._agents[name] = build_agent(
                name,
                self.cfg,
                tools,
                root=self.root,
                init=self.init,
                allowlist=names,
            )
        return self._agents[name]

    def agent_config(self, attempt: int) -> dict:
        """The config for one agent run: events handler + attempt tag."""
        return {
            "callbacks": [self.handler],
            "metadata": {META_ATTEMPT: attempt},
        }
