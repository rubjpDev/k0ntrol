"""The structured tool-error contract shared with the event log (#37).

Every failure a tool wants an agent to see is a ``ToolError`` (a
``ToolException``, so the tool node surfaces the message to the agent).
A refusal carries ``denied`` — ``"path_guard"`` for a path-guard refusal,
``"allowlist"`` for a call to a tool outside the agent's allowlist — which
the event handler copies into the ``tool`` event's ``denied`` field.
Ordinary errors (file not found, ambiguous edit, bad regex) leave it ``None``.
"""

from typing import Literal

from langchain_core.tools import ToolException

Denied = Literal["allowlist", "path_guard"]


class ToolError(ToolException):
    """A tool failure the agent sees; ``denied`` tags refusals for events."""

    def __init__(self, message: str, *, denied: Denied | None = None) -> None:
        super().__init__(message)
        self.denied = denied
