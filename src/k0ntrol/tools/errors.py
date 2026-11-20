"""The structured tool-error contract shared with the event log (#37).

Every failure a tool wants an agent to see is a ``ToolError`` (a
``ToolException``; ``ToolErrorMiddleware`` in ``agents.py`` turns it into an
error ``ToolMessage`` the agent sees, since the tool node alone would
re-raise it and end the run).
A refusal carries ``denied`` — ``"path_guard"`` for a path-guard refusal,
``"slice_guard"`` for a write the slice guard's state refuses,
``"network_guard"`` for a URL the network guard refuses (unsafe scheme or
address), ``"allowlist"`` for a call to a tool outside the agent's
allowlist — which
the event handler copies into the ``tool`` event's ``denied`` field.
Ordinary errors (file not found, ambiguous edit, bad regex) leave it ``None``.
"""

from typing import Iterable, Literal

from langchain_core.tools import ToolException

Denied = Literal[
    "allowlist", "network_guard", "path_guard", "slice_guard", "slice_recovery"
]


class ToolError(ToolException):
    """A tool failure the agent sees; ``denied`` tags refusals for events."""

    def __init__(self, message: str, *, denied: Denied | None = None) -> None:
        super().__init__(message)
        self.denied = denied


class SliceRecoveryNeeded(Exception):
    """The slice guard's bound on denied test writes ran out (#174).

    Not a ``ToolError``: it must propagate out of the agent run (past
    ``ToolErrorMiddleware``, which converts only ``ToolError``) so the
    workflow can pause for the human's recovery decision. ``denied``
    tags the ``tool`` event like a refusal; ``slice`` and ``paths``
    carry the journal identity the recovery gate restores.
    """

    denied: Denied = "slice_recovery"

    def __init__(
        self, message: str, *, slice: str | None = None, paths: Iterable[str] = ()
    ) -> None:
        super().__init__(message)
        self.slice = slice
        self.paths = list(paths)
