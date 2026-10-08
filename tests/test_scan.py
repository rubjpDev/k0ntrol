import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from k0ntrol.cli import main
from k0ntrol.harness.scan import ScanError, run_scan


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

