"""The ``search_memory`` tool: one FTS lookup over project + session memory.

Every agent gets it. It queries `.k0-mem/index.sqlite` — code-graph notes,
`Mem.md` bullets and session facts — which the harness rebuilds once per
run or `/ask`. The tool only fills in a missing index; it never rebuilds
per call.
"""

import json
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from k0ntrol.memory.index import index_path, query_index, rebuild_index


def make_search_memory(root: Path) -> BaseTool:
    """Build the ``search_memory`` tool bound to ``root``."""

    @tool
    def search_memory(query: str, limit: int = 5) -> str:
        """Search project memory for `query`. Returns a JSON list of hits with
        id, kind, path and body; symbol hits also list calls and called_by."""
        _ensure_index(root)
        return json.dumps(query_index(root, query, limit), indent=2)

    return search_memory


def _ensure_index(root: Path) -> None:
    if not index_path(root).is_file():
        rebuild_index(root)
