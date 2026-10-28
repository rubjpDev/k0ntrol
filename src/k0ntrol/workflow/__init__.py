"""The workflow package: run module, graphs, gates, routers, side effects.

The run module is the seam between the REPL and the graphs
(``k0ntrol.workflow`` exposes ``start``/``resume`` and the ``Result``
types). #41 ships the ``/fast`` graph; ``/full`` lands in #44/#45.
"""

from k0ntrol.workflow.run import (
    Outcome,
    Pending,
    Result,
    RunRefused,
    resume,
    start,
)

__all__ = ["Outcome", "Pending", "Result", "RunRefused", "resume", "start"]
