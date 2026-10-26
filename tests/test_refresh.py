"""``refresh_memory``: the once-per-run/``/ask`` memory rebuild seam.

It rescans the code graph (describing changed files through the
``bulk_read`` helper when given) and rebuilds the FTS index once — the
seam ``/full``, ``/fast`` and ``/ask`` call before their agents run.
"""

import subprocess
from pathlib import Path

from k0ntrol.memory.index import query_index
from k0ntrol.memory.refresh import refresh_memory


def _git_tracked(root: Path, files: dict[str, str]) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        subprocess.run(
            ["git", "add", "--", relative],
            cwd=root,
            check=True,
            capture_output=True,
        )


def test_refresh_memory_rescans_and_rebuilds_the_index(tmp_path):
    _git_tracked(tmp_path, {"src/app.py": "def checkout():\n    return 1\n"})

    assert refresh_memory(tmp_path) == "scanned"

    assert (tmp_path / ".k0-mem" / "index.sqlite").exists()
    graph = tmp_path / ".k0-mem" / "Project_context" / tmp_path.name / "graph.json"
    assert graph.exists()


def test_refresh_memory_indexes_the_notes_the_helper_wrote(tmp_path):
    _git_tracked(tmp_path, {"src/app.py": "def checkout():\n    return 1\n"})
    replies = iter(
        [
            "- FILE src/app.py\n"
            "  - desc: checkout entry\n"
            "  - hace: wires the cart\n"
            "  - gotchas: none\n"
        ]
    )

    refresh_memory(tmp_path, invoke=lambda prompt: next(replies))

    hits = query_index(tmp_path, "wires the cart", 5)
    assert [(h["id"], h["kind"]) for h in hits] == [
        (f"{tmp_path.name}:file:src/app.py", "file")
    ]


def test_refresh_memory_indexes_mem_bullets_and_facts(tmp_path):
    _git_tracked(tmp_path, {"a.py": "x = 1\n"})
    mem = tmp_path / ".k0-mem" / "General_mem" / "Mem.md"
    mem.parent.mkdir(parents=True, exist_ok=True)
    mem.write_text("- checkout picked sqlite\n", encoding="utf-8")

    refresh_memory(tmp_path)

    hits = query_index(tmp_path, "sqlite", 5)
    assert [(h["id"], h["kind"]) for h in hits] == [("mem:1", "mem")]
