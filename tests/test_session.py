import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from k0ntrol.backends.invoke import BackendError
from k0ntrol.frontends.session import run_session
from k0ntrol.harness.init import MEM, RULES
from k0ntrol.harness.scan import run_scan
from k0ntrol.harness.session_memory import (
    add_mem_proposal,
    list_facts,
    list_pending_mem_proposals,
)

# Positive waits return as soon as the event fires, so a generous cap only matters on slow CI runners.
THREAD_TIMEOUT = 5.0


def config():
    return {
        "backend": "cursor",
        "threshold_lines": 350,
        "scan_confirm_over": 50,
        "reflect_every": 10,
        "agents": {
            "bulk_reader": {"model": "gpt-secret-model"},
            "retrieval_gate": {"model": "gpt-secret-model"},
            "reflect": {"model": "gpt-secret-model"},
        },
    }


def reflect_cfg(**overrides):
    cfg = config()
    cfg.update(overrides)
    return cfg


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


def allow_reflect(fn):
    def invoke(mode, prompt, **kwargs):
        if getattr(mode, "name", None) == "reflect":
            return ""
        return fn(mode, prompt, **kwargs)

    return invoke


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
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        assert mode.name == "spec"
        return (
            "## Goal\nhello\n## Files\n"
            "- tests/test_app.py (create)\n"
            "- app.py (create)\n"
        )

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
                "bulk_reader",
                "reflect",
            )
        },
    }
    lines = iter(["/full add checkout", "y", "y", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=cfg,
        invoke=allow_reflect(fake_invoke),
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
            return (
                "## Goal\ng\n## Files\n"
                "- tests/test_app.py (create)\n"
                "- app.py (change)\n"
            )
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
                "reflect",
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
        invoke=allow_reflect(fake_invoke),
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
            return (
                "## Goal\ng\n## Files\n"
                "- tests/test_app.py (create)\n"
                "- app.py (change)\n"
            )
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
                "reflect",
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
        invoke=allow_reflect(fake_invoke),
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
        if mode.name == "retrieval_gate":
            return "retrieve: no\n"
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
                for name in (
                    "spec",
                    "tester",
                    "coder",
                    "validator",
                    "bulk_reader",
                    "retrieval_gate",
                    "reflect",
                )
            },
        }
    )
    lines = iter(["/fast change @keep.py", "y", "/quit"])
    code = run_session(
        lambda: next(lines),
        lambda _line: None,
        root=tmp_path,
        config=cfg,
        invoke=allow_reflect(fake_invoke),
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
        if mode.name == "retrieval_gate":
            return "retrieve: no\n"
        if mode.name == "spec":
            return (
                "## Goal\nhello\n## Files\n"
                "- tests/test_app.py (create)\n"
                "- keep.py (change)\n"
            )
        return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"

    cfg = config()
    cfg.update(
        {
            "test_cmd": "pytest -q",
            "max_retries": 2,
            "agents": {
                name: {"model": "m"}
                for name in (
                    "spec",
                    "tester",
                    "coder",
                    "validator",
                    "bulk_reader",
                    "retrieval_gate",
                    "reflect",
                )
            },
        }
    )
    lines = iter(["/full add checkout", "y", "y", "/quit"])
    shown = []
    code = run_session(
        lambda: next(lines),
        shown.append,
        root=tmp_path,
        config=cfg,
        invoke=allow_reflect(fake_invoke),
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
        invoke=allow_reflect(fail),
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
            for name in (
                "spec",
                "tester",
                "coder",
                "validator",
                "bulk_reader",
                "reflect",
            )
        },
    }
    lines = iter(["/fast change @a.py", "/quit"])
    code = run_session(
        lambda: next(lines),
        lambda _line: None,
        root=tmp_path,
        config=cfg,
        invoke=allow_reflect(fake_invoke),
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
                "reflect",
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
        invoke=allow_reflect(fake_invoke),
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
        invoke=allow_reflect(fake_invoke),
        bulk_read=forbid,
    )
    assert code == 0
    assert seen[0]["step"] == "ask"
    assert seen[0]["run"]


def test_quit_reflects_pending_ask_turns(tmp_path):
    (tmp_path / "app.py").write_text("x\n", encoding="utf-8")
    names = []

    def fake_invoke(mode, prompt):
        names.append(mode.name)
        if mode.name == "reflect":
            return "fact: line 1 is x\n"
        return "line 1 is x\n"

    lines = iter(["/ask what is line 1? @app.py", "/quit"])
    code = run_session(
        lambda: next(lines),
        lambda _line: None,
        root=tmp_path,
        config=reflect_cfg(),
        invoke=fake_invoke,
        bulk_read=forbid,
    )
    assert code == 0
    assert names[-1] == "reflect"
    assert [item["body"] for item in list_facts(tmp_path)] == ["line 1 is x"]


def test_help_and_unknown_are_not_turns(tmp_path):
    names = []

    def fake_invoke(mode, prompt):
        names.append(mode.name)
        return "fact: should not run\n"

    lines = iter(["/help", "/nope", "/quit"])
    code = run_session(
        lambda: next(lines),
        lambda _line: None,
        root=tmp_path,
        config=reflect_cfg(),
        invoke=fake_invoke,
        bulk_read=forbid,
    )
    assert code == 0
    assert names == []
    assert list_facts(tmp_path) == []


def test_reflect_every_turns_include_ask(tmp_path):
    (tmp_path / "app.py").write_text("x\n", encoding="utf-8")
    names = []

    def fake_invoke(mode, prompt):
        names.append(mode.name)
        if mode.name == "reflect":
            return "fact: asked twice\n"
        return "ok\n"

    lines = iter(
        [
            "/ask what is line 1? @app.py",
            "/ask what is line 1? @app.py",
            "/quit",
        ]
    )
    code = run_session(
        lambda: next(lines),
        lambda _line: None,
        root=tmp_path,
        config=reflect_cfg(reflect_every=2),
        invoke=fake_invoke,
        bulk_read=forbid,
    )
    assert code == 0
    assert names.count("reflect") == 1
    assert [item["body"] for item in list_facts(tmp_path)] == ["asked twice"]


def test_done_fast_reflects_once_and_quit_does_not_double_fire(tmp_path):
    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")
    names = []

    def fake_invoke(mode, prompt):
        names.append(mode.name)
        if mode.name == "reflect":
            return "fact: fast finished\n"
        if mode.name == "tester":
            return "FILE tests/test_fast.py\ndef test_fast():\n    assert True\n"
        if mode.name == "coder":
            return "FILE a.py\nchanged\n"
        raise AssertionError(mode.name)

    cfg = reflect_cfg(test_cmd="pytest -q", max_retries=2, reflect_every=1)
    cfg["agents"] = {
        name: {"model": "m"}
        for name in (
            "spec",
            "tester",
            "coder",
            "validator",
            "reviewer",
            "bulk_reader",
            "retrieval_gate",
            "reflect",
        )
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
    assert "done" in shown
    assert names.count("reflect") == 1
    assert [item["body"] for item in list_facts(tmp_path)] == ["fast finished"]


def test_tested_fast_is_not_done_reflect(tmp_path):
    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")
    reflected = threading.Event()

    def fake_invoke(mode, prompt):
        if mode.name == "reflect":
            reflected.set()
            return "fact: too early\n"
        if mode.name == "tester":
            return "FILE tests/test_fast.py\ndef test_fast():\n    assert True\n"
        if mode.name == "coder":
            return "FILE a.py\nchanged\n"
        raise AssertionError(mode.name)

    cfg = reflect_cfg(test_cmd="pytest -q", max_retries=2, reflect_every=10)
    cfg["agents"] = {
        name: {"model": "m"}
        for name in (
            "spec",
            "tester",
            "coder",
            "validator",
            "bulk_reader",
            "reflect",
        )
    }
    lines = iter(["/fast change @a.py", "/help", "/quit"])

    def read_line():
        line = next(lines)
        if line == "/help":
            assert not reflected.wait(0.05)
        return line

    code = run_session(
        read_line,
        lambda _line: None,
        root=tmp_path,
        config=cfg,
        invoke=fake_invoke,
        bulk_read=forbid,
    )
    assert code == 0
    assert reflected.is_set()
    assert [item["body"] for item in list_facts(tmp_path)] == ["too early"]


def test_prompt_does_not_wait_for_reflect(tmp_path):
    (tmp_path / "app.py").write_text("x\n", encoding="utf-8")
    continued = threading.Event()

    def fake_invoke(mode, prompt):
        if mode.name == "reflect":
            assert continued.wait(THREAD_TIMEOUT)
            return "fact: async\n"
        return "ok\n"

    lines = iter(["/ask what is line 1? @app.py", "/help", "/quit"])

    def read_line():
        line = next(lines)
        if line == "/help":
            continued.set()
        return line

    code = run_session(
        read_line,
        lambda _line: None,
        root=tmp_path,
        config=reflect_cfg(reflect_every=1),
        invoke=fake_invoke,
        bulk_read=forbid,
    )
    assert code == 0
    assert [item["body"] for item in list_facts(tmp_path)] == ["async"]


def test_quit_waits_for_in_flight_reflect(tmp_path):
    (tmp_path / "app.py").write_text("x\n", encoding="utf-8")
    started = threading.Event()
    release = threading.Event()

    def fake_invoke(mode, prompt):
        if mode.name == "reflect":
            started.set()
            assert release.wait(THREAD_TIMEOUT)
            return "fact: flushed\n"
        return "ok\n"

    lines = iter(["/ask what is line 1? @app.py", "/quit"])
    finished = []

    def run():
        code = run_session(
            lambda: next(lines),
            lambda _line: None,
            root=tmp_path,
            config=reflect_cfg(reflect_every=1),
            invoke=fake_invoke,
            bulk_read=forbid,
        )
        finished.append(code)

    worker = threading.Thread(target=run)
    worker.start()
    assert started.wait(THREAD_TIMEOUT)
    assert worker.is_alive()
    assert list_facts(tmp_path) == []
    release.set()
    worker.join(THREAD_TIMEOUT)
    assert not worker.is_alive()
    assert finished == [0]
    assert [item["body"] for item in list_facts(tmp_path)] == ["flushed"]


def _pending_mem_arrives(root: Path, timeout: float = THREAD_TIMEOUT) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if list_pending_mem_proposals(root):
            return True
        time.sleep(0.01)
    return bool(list_pending_mem_proposals(root))


def _seed_mem_files(root: Path) -> tuple[Path, Path, Path]:
    mem = root / ".k0-mem" / "General_mem" / "Mem.md"
    rules = root / ".k0-mem" / "General_mem" / "Rules.md"
    topology = root / ".k0-mem" / "Project_topology" / root.name / "strategy.md"
    mem.parent.mkdir(parents=True, exist_ok=True)
    topology.parent.mkdir(parents=True, exist_ok=True)
    mem.write_text(MEM, encoding="utf-8")
    rules.write_text(RULES, encoding="utf-8")
    topology.write_text("human owned\n", encoding="utf-8")
    return mem, rules, topology


def test_session_y_at_prompt_appends_mem_proposal(tmp_path):
    mem, rules, topology = _seed_mem_files(tmp_path)
    add_mem_proposal(tmp_path, "Prefer sqlite for checkout state")
    lines = iter(["y", "/quit"])
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
    assert "Mem.md += Prefer sqlite for checkout state? y/n" in shown
    assert mem.read_text(encoding="utf-8") == MEM + "- Prefer sqlite for checkout state\n"
    assert rules.read_text(encoding="utf-8") == RULES
    assert topology.read_text(encoding="utf-8") == "human owned\n"
    assert list_pending_mem_proposals(tmp_path) == []


def test_session_n_at_prompt_drops_mem_proposal(tmp_path):
    mem, rules, topology = _seed_mem_files(tmp_path)
    add_mem_proposal(tmp_path, "Prefer sqlite for checkout state")
    lines = iter(["n", "/quit"])
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
    assert "Mem.md += Prefer sqlite for checkout state? y/n" in shown
    assert mem.read_text(encoding="utf-8") == MEM
    assert rules.read_text(encoding="utf-8") == RULES
    assert topology.read_text(encoding="utf-8") == "human owned\n"
    assert list_pending_mem_proposals(tmp_path) == []


def test_mem_proposal_is_shown_at_next_prompt_not_mid_turn(tmp_path):
    (tmp_path / "app.py").write_text("x\n", encoding="utf-8")
    mem, rules, topology = _seed_mem_files(tmp_path)
    spec_started = threading.Event()

    def fake_invoke(mode, prompt):
        if mode.name == "reflect":
            assert spec_started.wait(THREAD_TIMEOUT)
            return "mem: Prefer sqlite for checkout state\n"
        if mode.name == "spec":
            spec_started.set()
            assert _pending_mem_arrives(tmp_path)
            return (
                "## Goal\nhello\n## Files\n"
                "- tests/test_app.py (create)\n"
                "- app.py (change)\n"
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        return "ok\n"

    cfg = reflect_cfg(reflect_every=1)
    cfg["agents"]["spec"] = {"model": "m"}
    cfg["agents"]["tester"] = {"model": "m"}
    cfg["max_retries"] = 2
    cfg["test_cmd"] = "pytest -q"
    lines = iter(["/ask what is line 1? @app.py", "/full add checkout", "y", "y", "n", "/quit"])
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
    assert (tmp_path / "spec.md").read_text(encoding="utf-8").startswith("## Goal\nhello")
    assert "Mem.md += Prefer sqlite for checkout state? y/n" in shown
    assert text.index("## Goal\nhello") < text.index(
        "Mem.md += Prefer sqlite for checkout state? y/n"
    )
    assert mem.read_text(encoding="utf-8") == MEM
    assert rules.read_text(encoding="utf-8") == RULES
    assert topology.read_text(encoding="utf-8") == "human owned\n"

