import json
import subprocess
from pathlib import Path

import pytest

from k0ntrol.cli import main
from k0ntrol.memory.scan import ScanError, run_scan


def git_tracked(root: Path, files: dict[str, str | bytes]) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        subprocess.run(["git", "add", "--", relative], cwd=root, check=True, capture_output=True)


def _graph(root: Path) -> dict:
    graph_path = root / ".k0-mem" / "Project_context" / root.name / "graph.json"
    return json.loads(graph_path.read_text(encoding="utf-8"))


def _memory(root: Path) -> Path:
    return root / ".k0-mem" / "Project_context" / root.name


def test_scan_writes_the_code_graph_without_file_notes(tmp_path):
    git_tracked(tmp_path, {"src/app.py": "def checkout():\n    return 1\n"})

    assert run_scan(tmp_path) == "scanned"

    repo = tmp_path.name
    memory = _memory(tmp_path)
    graph = _graph(tmp_path)
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges = {(edge["from"], edge["kind"], edge["to"]) for edge in graph["edges"]}

    file_node = nodes[f"{repo}:file:src/app.py"]
    assert file_node["kind"] == "File"
    assert file_node["path"] == "src/app.py"
    assert file_node["desc"] == ""
    assert file_node["hash"]
    assert nodes[f"{repo}:fn:src.app.checkout"]["kind"] == "Function"
    assert nodes[f"{repo}:fn:src.app.checkout"]["desc"] == ""
    assert (f"{repo}:file:src/app.py", "defines", f"{repo}:fn:src.app.checkout") in edges

    note_files = list(memory.rglob("*.md"))
    assert note_files == []


def test_unchanged_scan_leaves_the_graph_untouched(tmp_path):
    git_tracked(tmp_path, {"src/app.py": "def checkout():\n    return 1\n"})
    assert run_scan(tmp_path) == "scanned"
    graph_path = _memory(tmp_path) / "graph.json"
    before = graph_path.read_bytes()
    mtime = graph_path.stat().st_mtime_ns

    assert run_scan(tmp_path) == "scanned"
    assert graph_path.read_bytes() == before
    assert graph_path.stat().st_mtime_ns == mtime


def test_scan_keeps_descriptions_of_unchanged_files(tmp_path):
    git_tracked(
        tmp_path,
        {
            "src/app.py": "def checkout():\n    return 1\n",
            "src/other.py": "def other():\n    return 2\n",
        },
    )
    run_scan(tmp_path)
    repo = tmp_path.name
    graph_path = _memory(tmp_path) / "graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    for node in graph["nodes"]:
        if node["id"] in {f"{repo}:file:src/app.py", f"{repo}:fn:src.app.checkout"}:
            node["desc"] = "checkout helper"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    (tmp_path / "src" / "other.py").write_text("def other():\n    return 3\n")
    subprocess.run(
        ["git", "add", "--", "src/other.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    assert run_scan(tmp_path) == "scanned"
    nodes = {node["id"]: node for node in _graph(tmp_path)["nodes"]}
    assert nodes[f"{repo}:file:src/app.py"]["desc"] == "checkout helper"
    assert nodes[f"{repo}:fn:src.app.checkout"]["desc"] == "checkout helper"
    assert nodes[f"{repo}:file:src/other.py"]["desc"] == ""


def test_scan_skips_binary_and_untracked_files(tmp_path):
    git_tracked(tmp_path, {"ok.py": "print(1)\n", "blob.bin": b"a\x00b"})
    (tmp_path / "loose.py").write_text("print(2)\n", encoding="utf-8")

    run_scan(tmp_path)
    graph = _graph(tmp_path)
    assert [node["path"] for node in graph["nodes"]] == ["ok.py"]


def test_scan_without_git_does_not_write_empty_graph(tmp_path):
    (tmp_path / "a.py").write_text("print(1)\n", encoding="utf-8")
    with pytest.raises(ScanError):
        run_scan(tmp_path)
    assert not _memory(tmp_path).exists()


def test_scan_cli_writes_the_graph(tmp_path, monkeypatch):
    git_tracked(tmp_path, {"a.py": "print(1)\n"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert main(["scan"]) == 0
    graph = _graph(tmp_path)
    assert graph["nodes"][0]["id"] == f"{tmp_path.name}:file:a.py"


def test_scan_python_package_extracts_nodes_and_edges(tmp_path):
    git_tracked(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/util.py": "def helper():\n    return 1\n",
            "pkg/app.py": (
                "from pkg.util import helper\n"
                "\n"
                "def run():\n"
                "    helper()\n"
                "    missing()\n"
            ),
            "pkg/model.py": (
                "class Item:\n"
                "    def value(self):\n"
                "        return 2\n"
            ),
            "notes.txt": "plain text\n",
        },
    )

    assert run_scan(tmp_path) == "scanned"

    repo = tmp_path.name
    graph = _graph(tmp_path)
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges = {(edge["from"], edge["kind"], edge["to"]) for edge in graph["edges"]}

    assert nodes[f"{repo}:fn:pkg.util.helper"]["kind"] == "Function"
    assert nodes[f"{repo}:fn:pkg.app.run"]["kind"] == "Function"
    assert nodes[f"{repo}:class:pkg.model.Item"]["kind"] == "Class"
    assert nodes[f"{repo}:fn:pkg.model.Item.value"]["kind"] == "Function"
    assert nodes[f"{repo}:file:notes.txt"]["kind"] == "File"
    assert not any("missing" in node_id for node_id in nodes)

    assert (f"{repo}:file:pkg/util.py", "defines", f"{repo}:fn:pkg.util.helper") in edges
    assert (f"{repo}:file:pkg/app.py", "defines", f"{repo}:fn:pkg.app.run") in edges
    assert (f"{repo}:file:pkg/model.py", "defines", f"{repo}:class:pkg.model.Item") in edges
    assert (f"{repo}:file:pkg/model.py", "defines", f"{repo}:fn:pkg.model.Item.value") in edges
    assert (f"{repo}:file:pkg/app.py", "imports", f"{repo}:file:pkg/util.py") in edges
    assert (f"{repo}:fn:pkg.app.run", "calls", f"{repo}:fn:pkg.util.helper") in edges
    assert all("missing" not in edge[2] for edge in edges)


def test_scan_syntax_error_keeps_file_node(tmp_path):
    git_tracked(tmp_path, {"broken.py": "def broken(\n"})

    assert run_scan(tmp_path) == "scanned"

    repo = tmp_path.name
    graph = _graph(tmp_path)
    assert [node["kind"] for node in graph["nodes"]] == ["File"]
    assert graph["nodes"][0]["id"] == f"{repo}:file:broken.py"
    assert graph["edges"] == []


def test_deleted_file_drops_its_nodes_and_edges(tmp_path):
    git_tracked(
        tmp_path,
        {
            "pkg.py": "def gone():\n    return 1\n\nclass Gone:\n    pass\n",
            "keep.py": "def keep():\n    return 1\n",
        },
    )
    assert run_scan(tmp_path) == "scanned"
    subprocess.run(
        ["git", "rm", "-f", "--", "pkg.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    assert run_scan(tmp_path) == "scanned"

    graph = _graph(tmp_path)
    ids = {node["id"] for node in graph["nodes"]}
    repo = tmp_path.name
    assert f"{repo}:file:pkg.py" not in ids
    assert f"{repo}:fn:pkg.gone" not in ids
    assert f"{repo}:class:pkg.Gone" not in ids
    assert f"{repo}:file:keep.py" in ids
    assert f"{repo}:fn:keep.keep" in ids
    assert not any(
        edge["from"] == f"{repo}:file:pkg.py"
        or edge["to"] in {f"{repo}:fn:pkg.gone", f"{repo}:class:pkg.Gone"}
        for edge in graph["edges"]
    )


def test_renamed_symbols_drop_stale_nodes(tmp_path):
    git_tracked(tmp_path, {"pkg.py": "def gone():\n    return 1\n"})
    assert run_scan(tmp_path) == "scanned"
    (tmp_path / "pkg.py").write_text("def kept():\n    return 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", "pkg.py"], cwd=tmp_path, check=True, capture_output=True
    )

    assert run_scan(tmp_path) == "scanned"

    ids = {node["id"] for node in _graph(tmp_path)["nodes"]}
    repo = tmp_path.name
    assert f"{repo}:fn:pkg.gone" not in ids
    assert f"{repo}:fn:pkg.kept" in ids
    assert f"{repo}:file:pkg.py" in ids
