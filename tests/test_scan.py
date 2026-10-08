import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from k0ntrol.cli import main
from k0ntrol.harness.scan import ScanError, rescan_changed, run_scan


def scan_config(**overrides):
    cfg = {
        "threshold_lines": 350,
        "scan_confirm_over": 50,
        "agents": {"bulk_reader": {"model": "m"}},
    }
    cfg.update(overrides)
    return cfg


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


def write_agent(path: Path, body: str) -> None:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_scan_writes_file_memory_and_graph(tmp_path):
    git_tracked(tmp_path, {"src/app.py": "def checkout():\n    return 1\n"})
    calls = []

    def fake_bulk(question, paths, **kwargs):
        calls.append(paths)
        return (
            "- FILE src/app.py\n"
            "  - desc: checkout helper\n"
            "  - hace: returns 1 from checkout\n"
            "  - gotchas: none\n"
        )

    status = run_scan(
        root=tmp_path,
        config=scan_config(),
        bulk_read=fake_bulk,
        invoke=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("invoke")),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("confirm")),
        write=lambda _line: None,
    )

    assert status == "scanned"
    assert len(calls) == 1
    memory = tmp_path / ".k0-mem" / "Project_context" / tmp_path.name
    md = (memory / "files" / "src" / "app.py.md").read_text(encoding="utf-8")
    assert "## hace" in md
    assert "returns 1 from checkout" in md
    assert "## gotchas" in md
    graph = json.loads((memory / "graph.json").read_text(encoding="utf-8"))
    node = graph["nodes"][0]
    assert node["id"] == f"{tmp_path.name}:file:src/app.py"
    assert node["kind"] == "File"
    assert node["path"] == "src/app.py"
    assert node["desc"] == "checkout helper"
    assert node["hash"]


def test_unchanged_scan_makes_zero_model_calls(tmp_path):
    git_tracked(tmp_path, {"src/app.py": "def checkout():\n    return 1\n"})
    calls = []

    def fake_bulk(question, paths, **kwargs):
        calls.append(paths)
        return (
            "- FILE src/app.py\n"
            "  - desc: checkout helper\n"
            "  - hace: returns 1 from checkout\n"
            "  - gotchas: none\n"
        )

    kwargs = dict(
        root=tmp_path,
        config=scan_config(),
        bulk_read=fake_bulk,
        invoke=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("invoke")),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("confirm")),
        write=lambda _line: None,
    )
    assert run_scan(**kwargs) == "scanned"
    assert len(calls) == 1
    assert run_scan(**kwargs) == "scanned"
    assert len(calls) == 1
    memory = tmp_path / ".k0-mem" / "Project_context" / tmp_path.name
    node = json.loads((memory / "graph.json").read_text(encoding="utf-8"))["nodes"][0]
    assert node["desc"] == "checkout helper"


def test_unchanged_scan_skips_extract_and_graph_rewrite(tmp_path, monkeypatch):
    git_tracked(tmp_path, {"src/app.py": "def checkout():\n    return 1\n"})
    calls = []

    def fake_bulk(question, paths, **kwargs):
        calls.append(paths)
        return (
            "- FILE src/app.py\n"
            "  - desc: checkout helper\n"
            "  - hace: returns 1 from checkout\n"
            "  - gotchas: none\n"
        )

    kwargs = dict(
        root=tmp_path,
        config=scan_config(),
        bulk_read=fake_bulk,
        invoke=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("invoke")),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("confirm")),
        write=lambda _line: None,
    )
    assert run_scan(**kwargs) == "scanned"
    assert len(calls) == 1
    graph_path = tmp_path / ".k0-mem" / "Project_context" / tmp_path.name / "graph.json"
    before = graph_path.read_bytes()
    mtime = graph_path.stat().st_mtime_ns

    monkeypatch.setattr(
        "k0ntrol.harness.scan.extract_python",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("extract_python")),
    )
    assert rescan_changed(**kwargs) == "scanned"
    assert len(calls) == 1
    assert graph_path.read_bytes() == before
    assert graph_path.stat().st_mtime_ns == mtime


def test_scan_batches_files_up_to_threshold_lines(tmp_path):
    git_tracked(
        tmp_path,
        {
            "a.py": "a1\na2\n",
            "b.py": "b1\nb2\n",
            "c.py": "c1\n",
        },
    )
    batches = []

    def fake_bulk(question, paths, **kwargs):
        batches.append([path.name for path in paths])
        return "\n".join(
            f"- FILE {path.name}\n  - desc: {path.name}\n  - hace: {path.name}\n  - gotchas: none"
            for path in paths
        )

    run_scan(
        root=tmp_path,
        config=scan_config(threshold_lines=4),
        bulk_read=fake_bulk,
        invoke=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("invoke")),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("confirm")),
        write=lambda _line: None,
    )
    assert batches == [["a.py", "b.py"], ["c.py"]]


def test_scan_skips_binary_and_untracked_files(tmp_path):
    git_tracked(tmp_path, {"ok.py": "print(1)\n", "blob.bin": b"a\x00b"})
    (tmp_path / "loose.py").write_text("print(2)\n", encoding="utf-8")
    seen = []

    def fake_bulk(question, paths, **kwargs):
        seen.extend(path.name for path in paths)
        return "- FILE ok.py\n  - desc: ok\n  - hace: ok\n  - gotchas: none\n"

    run_scan(
        root=tmp_path,
        config=scan_config(),
        bulk_read=fake_bulk,
        invoke=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("invoke")),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("confirm")),
        write=lambda _line: None,
    )
    assert seen == ["ok.py"]
    graph = json.loads(
        (tmp_path / ".k0-mem" / "Project_context" / tmp_path.name / "graph.json").read_text(
            encoding="utf-8"
        )
    )
    assert [node["path"] for node in graph["nodes"]] == ["ok.py"]


def test_scan_asks_when_calls_exceed_confirm_limit(tmp_path):
    git_tracked(tmp_path, {"a.py": "a\n", "b.py": "b\n"})
    prompts = []

    def fake_bulk(question, paths, **kwargs):
        raise AssertionError("bulk")

    status = run_scan(
        root=tmp_path,
        config=scan_config(threshold_lines=1, scan_confirm_over=1),
        bulk_read=fake_bulk,
        invoke=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("invoke")),
        read_line=lambda: "n",
        write=prompts.append,
    )
    assert status == "cancelled"
    assert prompts == ["scan needs 2 model calls (limit 1). continue? y/n"]
    assert not (tmp_path / ".k0-mem" / "Project_context" / tmp_path.name / "graph.json").exists()


def test_scan_without_git_does_not_write_empty_graph(tmp_path):
    (tmp_path / "a.py").write_text("print(1)\n", encoding="utf-8")
    with pytest.raises(ScanError):
        run_scan(
            root=tmp_path,
            config=scan_config(),
            bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bulk")),
            invoke=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("invoke")),
            read_line=lambda: (_ for _ in ()).throw(AssertionError("confirm")),
            write=lambda _line: None,
        )
    assert not (tmp_path / ".k0-mem" / "Project_context" / tmp_path.name / "graph.json").exists()


def test_scan_cli_writes_memory(tmp_path, monkeypatch):
    git_tracked(tmp_path, {"a.py": "print(1)\n"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    def fake_invoke(mode, prompt):
        assert mode.name == "bulk_reader"
        return (
            "- FILE a.py\n"
            "  - desc: prints 1\n"
            "  - hace: prints 1\n"
            "  - gotchas: none\n"
        )

    monkeypatch.setattr("k0ntrol.backends.invoke.invoke", fake_invoke)
    assert main(["scan"]) == 0
    md = (
        tmp_path / ".k0-mem" / "Project_context" / tmp_path.name / "files" / "a.py.md"
    ).read_text(encoding="utf-8")
    assert "## hace" in md
    assert "prints 1" in md


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _file_notes(root: Path, paths: list[Path]) -> str:
    return "\n".join(
        f"- FILE {_rel(path, root)}\n"
        f"  - desc: {_rel(path, root)}\n"
        f"  - hace: {_rel(path, root)}\n"
        f"  - gotchas: none"
        for path in paths
    )


def _run_scan(root: Path) -> str:
    return run_scan(
        root=root,
        config=scan_config(),
        bulk_read=lambda question, paths, **kwargs: _file_notes(root, paths),
        invoke=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("invoke")),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("confirm")),
        write=lambda _line: None,
    )


def _memory(root: Path) -> Path:
    return root / ".k0-mem" / "Project_context" / root.name


def _write_leftover_symbol_markdown(memory: Path) -> None:
    for folder, name in (("functions", "pkg.gone"), ("classes", "pkg.Gone")):
        path = memory / folder / f"{name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# {name}\n\n## hace\nleftover\n\n## gotchas\nnone\n",
            encoding="utf-8",
        )


def _seed_pkg_gone(root: Path) -> Path:
    git_tracked(
        root,
        {
            "pkg.py": "def gone():\n    return 1\n\nclass Gone:\n    pass\n",
            "keep.py": "def keep():\n    return 1\n",
        },
    )
    assert _run_scan(root) == "scanned"
    memory = _memory(root)
    _write_leftover_symbol_markdown(memory)
    return memory


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

    status = run_scan(
        root=tmp_path,
        config=scan_config(),
        bulk_read=lambda question, paths, **kwargs: _file_notes(tmp_path, paths),
        invoke=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("invoke")),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("confirm")),
        write=lambda _line: None,
    )

    assert status == "scanned"
    repo = tmp_path.name
    memory = tmp_path / ".k0-mem" / "Project_context" / repo
    graph = json.loads((memory / "graph.json").read_text(encoding="utf-8"))
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

    helper_md = (memory / "functions" / "pkg.util.helper.md").read_text(encoding="utf-8")
    assert helper_md.startswith("# pkg.util.helper\n")
    assert "## hace" in helper_md
    item_md = (memory / "classes" / "pkg.model.Item.md").read_text(encoding="utf-8")
    assert item_md.startswith("# pkg.model.Item\n")
    assert (memory / "files" / "notes.txt.md").is_file()


def test_scan_syntax_error_keeps_file_node(tmp_path):
    git_tracked(tmp_path, {"broken.py": "def broken(\n"})

    status = run_scan(
        root=tmp_path,
        config=scan_config(),
        bulk_read=lambda question, paths, **kwargs: _file_notes(tmp_path, paths),
        invoke=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("invoke")),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("confirm")),
        write=lambda _line: None,
    )

    assert status == "scanned"
    repo = tmp_path.name
    memory = tmp_path / ".k0-mem" / "Project_context" / repo
    graph = json.loads((memory / "graph.json").read_text(encoding="utf-8"))
    assert [node["kind"] for node in graph["nodes"]] == ["File"]
    assert graph["nodes"][0]["id"] == f"{repo}:file:broken.py"
    assert graph["edges"] == []
    assert not (memory / "functions").exists()
    assert not (memory / "classes").exists()
    assert (memory / "files" / "broken.py.md").is_file()


def test_deleted_python_file_drops_stale_symbol_markdown_and_nodes(tmp_path):
    memory = _seed_pkg_gone(tmp_path)
    subprocess.run(
        ["git", "rm", "-f", "--", "pkg.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    assert _run_scan(tmp_path) == "scanned"

    graph = json.loads((memory / "graph.json").read_text(encoding="utf-8"))
    ids = {node["id"] for node in graph["nodes"]}
    repo = tmp_path.name
    assert f"{repo}:file:pkg.py" not in ids
    assert f"{repo}:fn:pkg.gone" not in ids
    assert f"{repo}:class:pkg.Gone" not in ids
    assert f"{repo}:file:keep.py" in ids
    assert f"{repo}:fn:keep.keep" in ids
    assert not (memory / "functions" / "pkg.gone.md").exists()
    assert not (memory / "classes" / "pkg.Gone.md").exists()
    assert (memory / "functions" / "keep.keep.md").is_file()
    assert not any(
        edge["from"] == f"{repo}:file:pkg.py" or edge["to"] in {
            f"{repo}:fn:pkg.gone",
            f"{repo}:class:pkg.Gone",
        }
        for edge in graph["edges"]
    )


def test_renamed_symbols_drop_stale_markdown_and_nodes(tmp_path):
    memory = _seed_pkg_gone(tmp_path)
    (tmp_path / "pkg.py").write_text("def kept():\n    return 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", "pkg.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    assert _run_scan(tmp_path) == "scanned"

    graph = json.loads((memory / "graph.json").read_text(encoding="utf-8"))
    ids = {node["id"] for node in graph["nodes"]}
    repo = tmp_path.name
    assert f"{repo}:fn:pkg.gone" not in ids
    assert f"{repo}:class:pkg.Gone" not in ids
    assert f"{repo}:fn:pkg.kept" in ids
    assert f"{repo}:file:pkg.py" in ids
    assert not (memory / "functions" / "pkg.gone.md").exists()
    assert not (memory / "classes" / "pkg.Gone.md").exists()
    assert (memory / "functions" / "pkg.kept.md").is_file()


def test_syntax_error_file_drops_stale_symbol_markdown_and_nodes(tmp_path):
    memory = _seed_pkg_gone(tmp_path)
    (tmp_path / "pkg.py").write_text("def gone(\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "--", "pkg.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    assert _run_scan(tmp_path) == "scanned"

    graph = json.loads((memory / "graph.json").read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    repo = tmp_path.name
    assert nodes[f"{repo}:file:pkg.py"]["kind"] == "File"
    assert f"{repo}:fn:pkg.gone" not in nodes
    assert f"{repo}:class:pkg.Gone" not in nodes
    assert not any(
        edge["from"] in {f"{repo}:fn:pkg.gone", f"{repo}:class:pkg.Gone"}
        or edge["to"] in {f"{repo}:fn:pkg.gone", f"{repo}:class:pkg.Gone"}
        for edge in graph["edges"]
    )
    assert not (memory / "functions" / "pkg.gone.md").exists()
    assert not (memory / "classes" / "pkg.Gone.md").exists()
    assert (memory / "files" / "pkg.py.md").is_file()


def test_scan_describes_functions_from_bulk_read(tmp_path):
    git_tracked(tmp_path, {"src/app.py": "def checkout():\n    return 1\n"})
    calls = []

    def fake_bulk(question, paths, **kwargs):
        calls.append(paths)
        return (
            "- FILE src/app.py\n"
            "  - desc: checkout helper\n"
            "  - hace: returns 1 from checkout\n"
            "  - gotchas: none\n"
            "- FUNCTION src.app.checkout\n"
            "  - desc: returns one\n"
            "  - hace: returns 1\n"
            "  - gotchas: none\n"
        )

    kwargs = dict(
        root=tmp_path,
        config=scan_config(),
        bulk_read=fake_bulk,
        invoke=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("invoke")),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("confirm")),
        write=lambda _line: None,
    )
    assert run_scan(**kwargs) == "scanned"
    assert len(calls) == 1
    repo = tmp_path.name
    memory = tmp_path / ".k0-mem" / "Project_context" / repo
    graph = json.loads((memory / "graph.json").read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert nodes[f"{repo}:file:src/app.py"]["desc"] == "checkout helper"
    assert nodes[f"{repo}:fn:src.app.checkout"]["desc"] == "returns one"
    md = (memory / "functions" / "src.app.checkout.md").read_text(encoding="utf-8")
    assert "returns 1" in md

    assert run_scan(**kwargs) == "scanned"
    assert len(calls) == 1
    graph = json.loads((memory / "graph.json").read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert nodes[f"{repo}:fn:src.app.checkout"]["desc"] == "returns one"


def test_scan_cli_emits_one_jsonl_event(tmp_path, monkeypatch):

    git_tracked(tmp_path, {"a.py": "print(1)\n"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    agent = tmp_path / "cursor-agent"
    write_agent(
        agent,
        "import json\n"
        "print(json.dumps({'type':'result','subtype':'success','is_error':False,"
        "'result':'- FILE a.py\\n  - desc: prints 1\\n  - hace: prints 1\\n  - gotchas: none\\n'}))\n",
    )
    monkeypatch.setenv("K0_CURSOR_AGENT", str(agent))
    assert main(["scan"]) == 0
    lines = (tmp_path / ".k0-mem" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["mode"] == "bulk_reader"
    assert event["ok"] is True
    assert event["step"] == "scan"

