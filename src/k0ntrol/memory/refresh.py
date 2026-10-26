"""The once-per-run memory refresh seam.

``/full``, ``/fast`` and ``/ask`` call this before their agents run:
the hash-based rescan (describing changed files through the ``bulk_read``
helper's one-shot ``invoke`` when given) plus one FTS index rebuild.
``search_memory`` then only queries the index — never rebuilds per call.
"""

from pathlib import Path

from k0ntrol.memory.index import rebuild_index
from k0ntrol.memory.scan import Invoke, run_scan


def refresh_memory(
    root: Path, *, config: dict | None = None, invoke: Invoke | None = None
) -> str:
    """Rescan the code graph and rebuild the FTS index, once."""
    status = run_scan(root, config=config, invoke=invoke)
    rebuild_index(root)
    return status
