import json
import subprocess
from pathlib import Path

import pytest

from k0ntrol.backends.invoke import BackendError
from k0ntrol.frontends.session import run_session
from k0ntrol.harness.scan import run_scan


def config():
    return {
        "backend": "cursor",
        "threshold_lines": 350,
        "scan_confirm_over": 50,
        "agents": {
            "bulk_reader": {"model": "gpt-secret-model"},
            "retrieval_gate": {"model": "gpt-secret-model"},
        },
    }


def git_tracked(root: Path, files: dict[str, str]) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        subprocess.run(["git", "add", "--", relative], cwd=root, check=True, capture_output=True)


def seed_scan(root: Path, notes: dict[str, str]) -> None:
    def fake_bulk(question, paths, **kwargs):
        return "".join(
            f"- FILE {path.relative_to(root).as_posix()}\n"
            f"  - desc: {notes[path.relative_to(root).as_posix()]}\n"
            f"  - hace: {notes[path.relative_to(root).as_posix()]}\n"
            f"  - gotchas: none\n"
            for path in paths
        )

    run_scan(
        root=root,
        config=config(),
        bulk_read=fake_bulk,
        invoke=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("invoke")),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("confirm")),
        write=lambda _line: None,
    )


def memory_dir(root: Path) -> Path:
    return root / ".k0-mem" / "Project_context" / root.name


def forbid(mode, prompt, **kwargs):
    raise AssertionError("model called")


def test_help_stays_on_screen_and_quit_exits(tmp_path):
    lines = iter(["/help", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        invoke=forbid,
        bulk_read=forbid,
    )
    assert code == 0
    text = "\n".join(shown)
    assert "› /help" in text
    assert "/ask" in text
    assert "/fast" in text
    assert "/full" in text
    assert "\x1b[?1049h" not in text


def test_blank_and_unknown_stay_in_session(tmp_path):
    lines = iter(["", "/nope", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        invoke=forbid,
        bulk_read=forbid,
    )
    assert code == 0
    text = "\n".join(shown)
    assert "unknown command: /nope" in text


def test_full_in_the_session_writes_spec_on_yes(tmp_path):
    def fake_invoke(mode, prompt):
        assert mode.name == "spec"
        return "## Goal\nhello\n## Files\n"

    cfg = {
        "backend": "cursor",
        "threshold_lines": 350,
        "test_cmd": "pytest -q",
        "max_retries": 2,
        "agents": {
            name: {"model": "m"}
            for name in ("spec", "tester", "coder", "validator", "bulk_reader")
        },
    }
    lines = iter(["/full add checkout", "y", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=cfg,
        invoke=fake_invoke,
        bulk_read=forbid,
    )
    assert code == 0
    assert (tmp_path / "spec.md").read_text(encoding="utf-8").startswith("## Goal\nhello")
    assert "## Goal" in "\n".join(shown)


def test_full_in_the_session_runs_the_coder_and_validator_path(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("original\n", encoding="utf-8")
    names = []

    def fake_invoke(mode, prompt):
        names.append(mode.name)
        if mode.name == "spec":
            return "## Goal\ng\n## Files\n- tests/test_app.py\n- app.py\n"
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            return "FILE app.py\nchanged\n"
        assert mode.name == "reviewer"
        return "FILE review.md\nLooks good.\n"

    cfg = {
        "backend": "cursor",
        "threshold_lines": 350,
        "test_cmd": "pytest -q",
        "max_retries": 2,
        "agents": {
            name: {"model": "m"}
            for name in (
                "spec",
                "tester",
                "coder",
                "validator",
                "reviewer",
                "bulk_reader",
            )
        },
    }
    lines = iter(["/full change the app", "y", "y", "ok", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=cfg,
        invoke=fake_invoke,
        bulk_read=forbid,
        run_suite=lambda root: (0, "ok\n"),
    )

    assert code == 0
    assert names == ["spec", "tester", "coder", "reviewer"]
    assert target.read_text(encoding="utf-8") == "changed\n"
    assert "app.py\nchanged" in "\n".join(shown)
    assert "done" in shown


def test_full_in_the_session_prints_stuck_after_validation_limit(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("original\n", encoding="utf-8")
    names = []

    def fake_invoke(mode, prompt):
        names.append(mode.name)
        if mode.name == "spec":
            return "## Goal\ng\n## Files\n- tests/test_app.py\n- app.py\n"
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            return "FILE app.py\nbroken\n"
        assert mode.name == "validator"
        return "- test_app failed\n"

    cfg = {
        "backend": "cursor",
        "threshold_lines": 350,
        "test_cmd": "pytest -q",
        "max_retries": 0,
        "agents": {
            name: {"model": "m"}
            for name in (
                "spec",
                "tester",
                "coder",
                "validator",
                "reviewer",
                "bulk_reader",
            )
        },
    }
    lines = iter(["/full change the app", "y", "y", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=cfg,
        invoke=fake_invoke,
        bulk_read=forbid,
        run_suite=lambda root: (1, "FAILED tests/test_app.py::test_app\n"),
    )

    assert code == 0
    assert "stuck" in shown
    assert names == ["spec", "tester", "coder", "validator"]


def test_ctrl_c_and_eof_exit_zero(tmp_path):
    def interrupt():
        raise KeyboardInterrupt

    assert (
        run_session(
            interrupt,
            lambda _line: None,
            root=tmp_path,
            config=config(),
            invoke=forbid,
            bulk_read=lambda *args, **kwargs: forbid(None, None),
        )
        == 0
    )

    def end():
        raise EOFError

    assert (
        run_session(
            end,
            lambda _line: None,
            root=tmp_path,
            config=config(),
            invoke=forbid,
            bulk_read=lambda *args, **kwargs: forbid(None, None),
        )
        == 0
    )


def test_ask_without_memory_suggests_k0_scan_once(tmp_path):
    (tmp_path / "app.py").write_text("x\n", encoding="utf-8")

    def fake_invoke(mode, prompt):
        return "line 1 is x\n"

    lines = iter(
        [
            "/ask what is line 1? @app.py",
            "/ask what is line 1? @app.py",
            "/quit",
        ]
    )
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        invoke=fake_invoke,
        bulk_read=forbid,
    )
    assert code == 0
    text = "\n".join(shown)
    assert text.count("memory not built; run k0 scan") == 1
    assert text.count("line 1 is x") == 2


def test_ask_with_unchanged_memory_makes_no_scan_calls(tmp_path):
    git_tracked(tmp_path, {"app.py": "x\n"})
    seed_scan(tmp_path, {"app.py": "the app"})

    def fake_invoke(mode, prompt):
        return "still x\n"

    lines = iter(["/ask what is here? @app.py", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        invoke=fake_invoke,
        bulk_read=forbid,
    )
    assert code == 0
    assert "still x" in "\n".join(shown)
    graph = json.loads((memory_dir(tmp_path) / "graph.json").read_text(encoding="utf-8"))
    assert graph["nodes"][0]["desc"] == "the app"


def test_ask_rescan_drops_deleted_file_nodes_and_markdown(tmp_path):
    git_tracked(tmp_path, {"keep.py": "k\n", "gone.py": "g\n"})
    seed_scan(tmp_path, {"keep.py": "keeper", "gone.py": "goner"})
    files = memory_dir(tmp_path) / "files"
    assert (files / "gone.py.md").is_file()
    subprocess.run(["git", "rm", "-f", "gone.py"], cwd=tmp_path, check=True, capture_output=True)

    def fake_invoke(mode, prompt):
        return "kept\n"

    lines = iter(["/ask what is keep? @keep.py", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        invoke=fake_invoke,
        bulk_read=forbid,
    )
    assert code == 0
    assert "kept" in "\n".join(shown)
    graph = json.loads((memory_dir(tmp_path) / "graph.json").read_text(encoding="utf-8"))
    assert [node["path"] for node in graph["nodes"]] == ["keep.py"]
    assert (files / "keep.py.md").is_file()
    assert not (files / "gone.py.md").exists()


def test_ask_rescan_updates_changed_and_added_files(tmp_path):
    git_tracked(tmp_path, {"app.py": "old\n"})
    seed_scan(tmp_path, {"app.py": "old app"})
    (tmp_path / "app.py").write_text("new\n", encoding="utf-8")
    (tmp_path / "extra.py").write_text("e\n", encoding="utf-8")
    subprocess.run(["git", "add", "--", "extra.py"], cwd=tmp_path, check=True, capture_output=True)
    seen = []

    def fake_bulk(question, paths, **kwargs):
        seen.extend(path.name for path in paths)
        return "".join(
            f"- FILE {path.relative_to(tmp_path).as_posix()}\n"
            f"  - desc: {path.stem} now\n"
            f"  - hace: {path.stem} now\n"
            f"  - gotchas: none\n"
            for path in paths
        )

    def fake_invoke(mode, prompt):
        return "answered\n"

    lines = iter(["/ask what is here? @app.py", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        invoke=fake_invoke,
        bulk_read=fake_bulk,
    )
    assert code == 0
    assert "answered" in "\n".join(shown)
    assert sorted(seen) == ["app.py", "extra.py"]
    graph = json.loads((memory_dir(tmp_path) / "graph.json").read_text(encoding="utf-8"))
    by_path = {node["path"]: node for node in graph["nodes"]}
    assert by_path["app.py"]["desc"] == "app now"
    assert by_path["extra.py"]["desc"] == "extra now"
    files = memory_dir(tmp_path) / "files"
    assert "app now" in (files / "app.py.md").read_text(encoding="utf-8")
    assert "extra now" in (files / "extra.py.md").read_text(encoding="utf-8")


def test_ask_rescan_asks_when_calls_exceed_confirm_limit(tmp_path):
    git_tracked(tmp_path, {"a.py": "a\n", "b.py": "b\n"})
    seed_scan(tmp_path, {"a.py": "old a", "b.py": "old b"})
    (tmp_path / "a.py").write_text("A\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("B\n", encoding="utf-8")
    cfg = config()
    cfg["threshold_lines"] = 1
    cfg["scan_confirm_over"] = 1

    def fake_invoke(mode, prompt):
        return "still answering\n"

    lines = iter(["/ask what is a? @a.py", "n", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=cfg,
        invoke=fake_invoke,
        bulk_read=forbid,
    )
    assert code == 0
    text = "\n".join(shown)
    assert "scan needs 2 model calls (limit 1). continue? y/n" in text
    assert "still answering" in text
    graph = json.loads((memory_dir(tmp_path) / "graph.json").read_text(encoding="utf-8"))
    by_path = {node["path"]: node for node in graph["nodes"]}
    assert by_path["a.py"]["desc"] == "old a"


def test_fast_rescan_drops_deleted_file_markdown(tmp_path):
    git_tracked(tmp_path, {"keep.py": "k\n", "gone.py": "g\n"})
    seed_scan(tmp_path, {"keep.py": "keeper", "gone.py": "goner"})
    subprocess.run(["git", "rm", "-f", "gone.py"], cwd=tmp_path, check=True, capture_output=True)

    def fake_invoke(mode, prompt):
        if mode.name == "tester":
            return "FILE tests/test_keep.py\ndef test_keep():\n    assert True\n"
        assert mode.name == "coder"
        return "FILE keep.py\nchanged\n"

    cfg = config()
    cfg.update(
        {
            "test_cmd": "pytest -q",
            "max_retries": 2,
            "agents": {
                name: {"model": "m"}
                for name in ("spec", "tester", "coder", "validator", "bulk_reader")
            },
        }
    )
    lines = iter(["/fast change @keep.py", "y", "/quit"])
    code = run_session(
        lambda: next(lines),
        lambda _line: None,
        root=tmp_path,
        config=cfg,
        invoke=fake_invoke,
        bulk_read=forbid,
        run_suite=lambda root: (0, "ok\n"),
    )
    assert code == 0
    files = memory_dir(tmp_path) / "files"
    assert (files / "keep.py.md").is_file()
    assert not (files / "gone.py.md").exists()


def test_full_rescan_drops_deleted_file_markdown(tmp_path):
    git_tracked(tmp_path, {"keep.py": "k\n", "gone.py": "g\n"})
    seed_scan(tmp_path, {"keep.py": "keeper", "gone.py": "goner"})
    subprocess.run(["git", "rm", "-f", "gone.py"], cwd=tmp_path, check=True, capture_output=True)

    def fake_invoke(mode, prompt):
        assert mode.name == "spec"
        return "## Goal\nhello\n## Files\n"

    cfg = config()
    cfg.update(
        {
            "test_cmd": "pytest -q",
            "max_retries": 2,
            "agents": {
                name: {"model": "m"}
                for name in ("spec", "tester", "coder", "validator", "bulk_reader")
            },
        }
    )
    lines = iter(["/full add checkout", "y", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=cfg,
        invoke=fake_invoke,
        bulk_read=forbid,
    )
    assert code == 0
    files = memory_dir(tmp_path) / "files"
    assert not (files / "gone.py.md").exists()
    assert (tmp_path / "spec.md").read_text(encoding="utf-8").startswith("## Goal\nhello")


def test_ask_prints_the_answer_and_a_missing_file_stays_in_session(tmp_path, monkeypatch):
    target = tmp_path / "app.py"
    target.write_text("x\n")

    def fake_invoke(mode, prompt):
        return "- app (file) line 1: x\n"

    lines = iter(["/ask what is line 1? @app.py", "/ask @missing.py", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        invoke=fake_invoke,
        bulk_read=forbid,
    )
    assert code == 0
    text = "\n".join(shown)
    assert "- app (file) line 1: x" in text
    assert "missing.py" in text


def test_transcript_rereads_what_was_shown(tmp_path):
    lines = iter(["/help", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        invoke=forbid,
        bulk_read=forbid,
    )
    assert code == 0
    files = list((tmp_path / ".k0-mem" / "sessions").glob("*.txt"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    for chunk in shown:
        assert chunk in text


def test_session_prints_the_banner_once(tmp_path):
    from k0ntrol import __version__

    lines = iter(["/quit"])
    shown = []
    run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        invoke=forbid,
        bulk_read=forbid,
        color=False,
    )
    text = "\n".join(shown)
    assert text.count("▄█████▙") == 1
    assert f"k0ntrol v{__version__}" in text
    assert str(tmp_path) in text
    assert "cursor" in text
    assert "gpt-secret-model" not in text
    assert "\x1b" not in text


def test_terminal_echo_is_not_reprinted_but_is_recorded(tmp_path):
    lines = iter(["/help", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        invoke=forbid,
        bulk_read=forbid,
        echo_input=False,
    )
    assert code == 0
    output = "\n".join(shown)
    assert "› /help" not in output
    transcript = next((tmp_path / ".k0-mem" / "sessions").glob("*.txt"))
    assert "› /help" in transcript.read_text(encoding="utf-8")


def test_backend_error_stays_in_session(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("x\n")

    def fail(mode, prompt):
        raise BackendError("Workspace Trust Required")

    lines = iter(["/ask what is here? @app.py", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=config(),
        invoke=fail,
        bulk_read=forbid,
    )
    assert code == 0
    text = "\n".join(shown)
    assert "backend error: Workspace Trust Required" in text
    assert "Traceback" not in text


def test_fast_in_the_session_skips_spec(tmp_path):
    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")
    names = []

    def fake_invoke(mode, prompt):
        names.append(mode.name)
        if mode.name == "coder":
            return "FILE a.py\na\n"
        return "FILE tests/test_fast.py\ndef test_fast():\n    assert False\n"

    cfg = {
        "backend": "cursor",
        "threshold_lines": 350,
        "test_cmd": "pytest -q",
        "max_retries": 2,
        "agents": {
            name: {"model": "m"}
            for name in ("spec", "tester", "coder", "validator", "bulk_reader")
        },
    }
    lines = iter(["/fast change @a.py", "/quit"])
    code = run_session(
        lambda: next(lines),
        lambda _line: None,
        root=tmp_path,
        config=cfg,
        invoke=fake_invoke,
        bulk_read=forbid,
    )
    assert code == 0
    assert "spec" not in names
    assert "tester" in names
    assert "change" in (tmp_path / "spec.md").read_text(encoding="utf-8")


def test_fast_in_the_session_prints_done_without_reviewer(tmp_path):
    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")
    names = []

    def fake_invoke(mode, prompt):
        names.append(mode.name)
        if mode.name == "tester":
            return "FILE tests/test_fast.py\ndef test_fast():\n    assert True\n"
        assert mode.name == "coder"
        return "FILE a.py\nchanged\n"

    cfg = {
        "backend": "cursor",
        "threshold_lines": 350,
        "test_cmd": "pytest -q",
        "max_retries": 2,
        "agents": {
            name: {"model": "m"}
            for name in (
                "spec",
                "tester",
                "coder",
                "validator",
                "reviewer",
                "bulk_reader",
            )
        },
    }
    lines = iter(["/fast change @a.py", "y", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=cfg,
        invoke=fake_invoke,
        bulk_read=forbid,
        run_suite=lambda root: (0, "ok\n"),
    )

    assert code == 0
    assert names == ["tester", "coder"]
    assert "reviewer" not in names
    assert "done" in shown


def test_ask_sets_one_run_and_the_ask_step(tmp_path):
    from k0ntrol.backends.invoke import current_event_context

    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")
    seen = []

    def fake_invoke(mode, prompt):
        seen.append(dict(current_event_context()))
        return "ok\n"

    lines = iter(["/ask what @a.py", "/quit"])
    code = run_session(
        lambda: next(lines),
        lambda _line: None,
        root=tmp_path,
        config=config(),
        invoke=fake_invoke,
        bulk_read=forbid,
    )
    assert code == 0
    assert seen[0]["step"] == "ask"
    assert seen[0]["run"]
