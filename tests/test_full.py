import json
import stat
import sys

import pytest

from k0ntrol.backends.invoke import current_event_context, invoke
from k0ntrol.harness.init import ARCHITECTURE, CODE_RULES, RULES, STRATEGY
from k0ntrol.harness.loop import run_full


def loop_config(**overrides):
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
    cfg.update(overrides)
    return cfg


def memory_loop_config(**overrides):
    cfg = loop_config()
    cfg["retrieval_top_k"] = 5
    cfg["agents"] = dict(cfg["agents"])
    cfg["agents"]["retrieval_gate"] = {"model": "m"}
    cfg.update(overrides)
    return cfg


def write_file_memory(root, relative, body):
    path = (
        root
        / ".k0-mem"
        / "Project_context"
        / root.name
        / "files"
        / f"{relative}.md"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return f"{root.name}:file:{relative}"


def files_spec(*entries):
    lines = "\n".join(f"- {path} ({action})" for path, action in entries)
    return f"## Goal\ng\n## Files\n{lines}\n"


def seed_product(tmp_path, relative="src/app.py", body="original\n"):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_full_rewrites_spec_until_yes(tmp_path):
    seed_product(tmp_path)
    calls = []

    def fake_invoke(mode, prompt):
        calls.append((mode.name, prompt))
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        spec_count = sum(1 for name, _prompt in calls if name == "spec")
        return (
            f"## Goal\nv{spec_count}\n## Files\n"
            "- tests/test_app.py (create)\n"
            "- src/app.py (change)\n"
        )

    lines = iter(["n", "y", "y"])
    shown = []
    status = run_full(
        "add checkout",
        root=tmp_path,
        config=loop_config(),
        read_line=lambda: next(lines),
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bulk")),
    )
    assert status == "done"
    assert [name for name, _ in calls] == ["spec", "spec", "tester"]
    assert "rejected" in calls[1][1]
    spec = (tmp_path / "spec.md").read_text(encoding="utf-8")
    assert spec.startswith("## Goal\nv2\n")
    assert "src/app.py (change)" in spec
    assert "## Goal" in "\n".join(shown)
    assert "y/n" in "\n".join(shown)


def test_full_trims_spec_narration_before_goal(tmp_path):
    seed_product(tmp_path)
    shown = []

    def fake_invoke(mode, prompt):
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        return (
            "I'll read X.Now Y.## Goal\n"
            "g\n"
            "## Files\n"
            "- tests/test_app.py (create)\n"
            "- src/app.py (change)\n"
        )

    status = run_full(
        "add checkout",
        root=tmp_path,
        config=loop_config(),
        read_line=lambda: "y",
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )
    assert status == "done"
    spec = (tmp_path / "spec.md").read_text(encoding="utf-8")
    assert spec.startswith("## Goal")
    assert shown[0].startswith("## Goal")
    assert "I'll read" not in spec
    assert "I'll read" not in shown[0]


def test_full_keeps_spec_without_goal_for_files_check(tmp_path):
    seed_product(tmp_path)

    def fake_invoke(mode, prompt):
        return "I'll narrate only.\n## Files\n- src/app.py (change)\n"

    shown = []
    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(max_retries=0),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("gate")),
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )

    assert status == "stuck"
    spec = (tmp_path / "spec.md").read_text(encoding="utf-8")
    assert spec.startswith("I'll narrate only.")
    assert shown[0].startswith("I'll narrate only.")
    assert "## Files needs a test path" in "\n".join(shown)
    assert "## Files needs a test path" in spec
    assert "stuck" in shown


def test_full_trims_spec_narration_on_files_retry(tmp_path):
    seed_product(tmp_path)
    shown = []

    def fake_invoke(mode, prompt):
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if "## Files needs a test path" in prompt:
            return (
                "I'll read X.Now Y.## Goal\n"
                "g\n"
                "## Files\n"
                "- tests/test_app.py (create)\n"
                "- src/app.py (change)\n"
            )
        return (
            "I'll read X.Now Y.## Goal\n"
            "g\n"
            "## Files\n"
            "- src/app.py (change)\n"
        )

    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(max_retries=1),
        read_line=lambda: "y",
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )

    assert status == "done"
    spec_writes = [text for text in shown if "## Goal" in text]
    assert spec_writes
    assert all(text.startswith("## Goal") for text in spec_writes)
    assert all("I'll read" not in text for text in spec_writes)
    spec = (tmp_path / "spec.md").read_text(encoding="utf-8")
    assert spec.startswith("## Goal")
    assert "I'll read" not in spec


def test_full_delegates_a_large_file_before_spec(tmp_path):
    seed_product(tmp_path)
    big = tmp_path / "big.py"
    big.write_text("x\n" * 351, encoding="utf-8")

    def fake_bulk(question, paths, **kwargs):
        assert list(paths) == [big]
        return "SUMMARY\n"

    def fake_invoke(mode, prompt):
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        assert mode.name == "spec"
        assert "SUMMARY" in prompt
        assert "x\n" * 351 not in prompt
        return files_spec(
            ("tests/test_app.py", "create"),
            ("src/app.py", "change"),
        )

    status = run_full(
        "what @big.py",
        root=tmp_path,
        config=loop_config(),
        read_line=lambda: "y",
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=fake_bulk,
    )
    assert status == "done"


def test_full_missing_path_does_not_invoke(tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("called")

    shown = []
    status = run_full(
        "what @missing.py",
        root=tmp_path,
        config=loop_config(),
        read_line=lambda: "y",
        write=shown.append,
        invoke=boom,
        bulk_read=boom,
    )
    assert status == "stuck"
    assert any("missing.py" in line for line in shown)


def test_full_yes_writes_tests_and_not_product_code(tmp_path):
    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "create"),
            )
        assert mode.name == "tester"
        return "FILE tests/test_app.py\ndef test_app():\n    assert False\n"

    shown = []
    status = run_full(
        "add a test",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y"]).__next__,
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bulk")),
    )
    assert "y/n" in "\n".join(shown)
    assert (tmp_path / "tests" / "test_app.py").read_text(encoding="utf-8").startswith(
        "def test_app"
    )
    assert not (tmp_path / "src" / "app.py").exists()
    assert status == "done"


def test_two_noes_retry_the_tester_once_then_stuck(tmp_path):
    tester_prompts = []

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "create"),
            )
        tester_prompts.append(prompt)
        return "FILE tests/test_app.py\ndef test_app():\n    assert False\n"

    lines = iter(["y", "n", "n"])
    shown = []
    status = run_full(
        "add a test",
        root=tmp_path,
        config=loop_config(),
        read_line=lambda: next(lines),
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bulk")),
    )
    assert status == "stuck"
    assert len(tester_prompts) == 2
    assert "rejected by human" in tester_prompts[1]
    spec = (tmp_path / "spec.md").read_text(encoding="utf-8")
    assert spec.count("rejected by human") == 2
    assert "tests/test_app.py" in "\n".join(shown)
    assert "def test_app" in "\n".join(shown)


def test_full_continues_through_coder_validator_to_diff_gate(tmp_path):
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("original\n", encoding="utf-8")
    seen = []

    def fake_invoke(mode, prompt):
        seen.append((mode.name, current_event_context()["step"]))
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            assert "original\n" in prompt
            return "FILE src/app.py\nchanged\n"
        if mode.name == "reviewer":
            assert "## Goal" in prompt
            assert "tests/test_app.py" in prompt
            assert "Implementation diff:" in prompt
            assert "+changed" in prompt
            return "FILE review.md\nLooks good.\n"
        raise AssertionError(mode.name)

    shown = []
    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y", "ok"]).__next__,
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (0, "ok\n"),
    )

    assert status == "done"
    assert [name for name, _step in seen] == [
        "spec",
        "tester",
        "coder",
        "reviewer",
    ]
    assert [step for _name, step in seen] == [
        "spec",
        "tester",
        "coder",
        "reviewer",
    ]
    assert target.read_text(encoding="utf-8") == "changed\n"
    assert "src/app.py\nchanged" in "\n".join(shown)
    assert "--- a/src/app.py" in "\n".join(shown)
    assert (tmp_path / "review.md").read_text(encoding="utf-8") == "Looks good.\n"
    assert "y/n" in "\n".join(shown)
    assert "ok / spec / code" in "\n".join(shown)


def test_full_reviewer_bulk_reads_large_test_context(tmp_path):
    seed_product(tmp_path)
    bulk_paths = []
    large_test = "def test_app():\n    assert True\n" + "# filler\n" * 351

    def fake_bulk(question, paths, **kwargs):
        bulk_paths.extend(paths)
        return "TEST SUMMARY\n"

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            return f"FILE tests/test_app.py\n{large_test}"
        if mode.name == "coder":
            return "FILE src/app.py\nchanged\n"
        assert mode.name == "reviewer"
        assert "TEST SUMMARY" in prompt
        assert "# filler" not in prompt
        return "FILE review.md\nLooks good.\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y", "ok"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=fake_bulk,
        run_suite=lambda root: (0, "ok\n"),
    )

    assert status == "done"
    assert bulk_paths == [tmp_path / "tests" / "test_app.py"]


def test_full_reviewer_bulk_reads_an_oversized_diff(tmp_path):
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("original\n" * 351, encoding="utf-8")
    bulk_paths = []

    def fake_bulk(question, paths, **kwargs):
        bulk_paths.extend(paths)
        assert "implementation diff" in question.lower()
        return "DIFF SUMMARY\n"

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            return "FILE src/app.py\nchanged\n"
        assert mode.name == "reviewer"
        assert "DIFF SUMMARY" in prompt
        assert "original\n" not in prompt
        return "FILE review.md\nLooks good.\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y", "ok"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=fake_bulk,
        run_suite=lambda root: (0, "ok\n"),
    )

    assert status == "done"
    assert bulk_paths == [target]


def test_full_reviewer_reads_all_test_paths(tmp_path):
    test_paths = iter(["tests/test_app.py", "tests/test_other.py"])

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("tests/test_other.py", "create"),
                ("src/app.py", "create"),
            )
        if mode.name == "tester":
            path = next(test_paths)
            return f"FILE {path}\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            return "FILE src/app.py\nchanged\n"
        assert mode.name == "reviewer"
        assert "tests/test_app.py" in prompt
        assert "tests/test_other.py" in prompt
        return "FILE review.md\nLooks good.\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y", "ok"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (0, "ok\n"),
    )

    assert status == "done"
    assert (tmp_path / "tests/test_app.py").is_file()
    assert (tmp_path / "tests/test_other.py").is_file()


def test_full_stops_when_tester_writes_the_wrong_test_path(tmp_path):
    names = []

    def fake_invoke(mode, prompt):
        names.append(mode.name)
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("tests/test_other.py", "create"),
                ("src/app.py", "create"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        raise AssertionError(f"unexpected mode: {mode.name}")

    shown = []
    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y"]).__next__,
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )

    assert status == "stuck"
    assert names == ["spec", "tester", "tester", "tester", "tester"]
    assert not (tmp_path / "tests/test_other.py").exists()
    error = (
        "FILE path is not the expected test path: tests/test_app.py "
        "(expected tests/test_other.py)"
    )
    assert error in shown
    assert shown.count("y/n") == 1


def test_full_retries_tester_when_file_is_the_wrong_test_path(tmp_path):
    seed_product(tmp_path)
    test_body = "def test_app():\n    assert False\n"
    seed_product(tmp_path, "tests/test_app.py", test_body)
    tester_prompts = []

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "change"),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            tester_prompts.append(prompt)
            if len(tester_prompts) == 1:
                return "FILE tests/test_other.py\ndef test_other():\n    assert True\n"
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        raise AssertionError(f"unexpected mode: {mode.name}")

    shown = []
    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y"]).__next__,
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )

    assert status == "done"
    assert f"Target:\n{test_body}" in tester_prompts[0]
    assert "Target:\n<new file>" not in tester_prompts[0]
    error = (
        "FILE path is not the expected test path: tests/test_other.py "
        "(expected tests/test_app.py)"
    )
    assert error in shown
    assert error in (tmp_path / "spec.md").read_text(encoding="utf-8")
    assert len(tester_prompts) == 2
    assert shown.count("y/n") == 2
    assert not (tmp_path / "tests/test_other.py").exists()


def test_full_stops_when_tester_omits_the_file_prefix(tmp_path):
    seed_product(tmp_path)

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            return "def test_app():\n    assert True\n"
        raise AssertionError(f"unexpected mode: {mode.name}")

    shown = []
    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y"]).__next__,
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )

    assert status == "stuck"
    text = "\n".join(shown)
    assert "response must start with FILE <relative-path>" in text
    assert "FILE path is not a test path" not in text
    assert shown.count("y/n") == 1


def test_full_stops_when_tester_file_is_not_a_test_path(tmp_path):
    seed_product(tmp_path)

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            return "FILE src/app.py\nchanged\n"
        raise AssertionError(f"unexpected mode: {mode.name}")

    shown = []
    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y"]).__next__,
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )

    assert status == "stuck"
    text = "\n".join(shown)
    assert "FILE path is not a test path: src/app.py" in text
    assert "expected" not in text
    assert shown.count("y/n") == 1


def test_full_review_code_runs_a_fresh_coder_and_reviewer(tmp_path):
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("original\n", encoding="utf-8")
    names = []
    coder_count = 0
    reviewer_count = 0

    def fake_invoke(mode, prompt):
        nonlocal coder_count, reviewer_count
        names.append(mode.name)
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            coder_count += 1
            if coder_count == 2:
                assert "Review:\nFix the code." in prompt
            return f"FILE src/app.py\nchanged {coder_count}\n"
        reviewer_count += 1
        if reviewer_count == 1:
            return "FILE review.md\nFix the code.\n"
        return "FILE review.md\nLooks good.\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y", "code", "ok"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (0, "ok\n"),
    )

    assert status == "done"
    assert names == ["spec", "tester", "coder", "reviewer", "coder", "reviewer"]
    assert target.read_text(encoding="utf-8") == "changed 2\n"
    assert "Fix the code." in (tmp_path / "spec.md").read_text(encoding="utf-8")


def test_full_review_code_redoes_only_review_picked_targets(tmp_path):
    seed_product(tmp_path, "src/app.py", "app original\n")
    seed_product(tmp_path, "src/util.py", "util original\n")
    coder_files = []
    redo_prompts = []

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
                ("src/util.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            if "Target:\napp " in prompt:
                coder_files.append("src/app.py")
                return "FILE src/app.py\napp v1\n"
            if "Target:\nutil " in prompt:
                coder_files.append("src/util.py")
                if "Review:" in prompt:
                    redo_prompts.append(prompt)
                    return "FILE src/util.py\nutil v2\n"
                return "FILE src/util.py\nutil v1\n"
            raise AssertionError(prompt)
        return (
            "FILE review.md\n"
            "Fix util.\n"
            "redo:\n"
            "- src/util.py\n"
        )

    events = tmp_path / ".k0-mem" / "events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(
        json.dumps({"step": "reviewer", "ok": True}) + "\n",
        encoding="utf-8",
    )
    shown = []
    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y", "code", "ok"]).__next__,
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (0, "ok\n"),
    )

    assert status == "done"
    assert coder_files == ["src/app.py", "src/util.py", "src/util.py"]
    assert "↻ coder: src/util.py" in shown
    assert json.loads(events.read_text(encoding="utf-8").splitlines()[-1])[
        "redo"
    ] == ["src/util.py"]
    assert redo_prompts
    assert "app v1" in redo_prompts[0]
    assert (tmp_path / "src" / "app.py").read_text(encoding="utf-8") == "app v1\n"
    assert (tmp_path / "src" / "util.py").read_text(encoding="utf-8") == "util v2\n"


def test_full_review_code_redoes_all_coder_targets(tmp_path):
    seed_product(tmp_path, "src/app.py", "app original\n")
    seed_product(tmp_path, "src/util.py", "util original\n")
    names = []
    coder_files = []

    def fake_invoke(mode, prompt):
        names.append(mode.name)
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
                ("src/util.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            if "Target:\napp " in prompt:
                coder_files.append("src/app.py")
                return "FILE src/app.py\napp changed\n"
            if "Target:\nutil " in prompt:
                coder_files.append("src/util.py")
                return "FILE src/util.py\nutil changed\n"
            raise AssertionError(prompt)
        return "FILE review.md\nLooks good.\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y", "code", "ok"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (0, "ok\n"),
    )

    assert status == "done"
    assert names == [
        "spec",
        "tester",
        "coder",
        "coder",
        "reviewer",
        "coder",
        "coder",
        "reviewer",
    ]
    assert coder_files == [
        "src/app.py",
        "src/util.py",
        "src/app.py",
        "src/util.py",
    ]


def test_full_review_code_has_no_automatic_loop_limit(tmp_path):
    seed_product(tmp_path)
    counts = {"coder": 0, "reviewer": 0}

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            counts["coder"] += 1
            return "FILE src/app.py\nchanged\n"
        counts["reviewer"] += 1
        return f"FILE review.md\nreview {counts['reviewer']}\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(max_retries=0),
        read_line=iter(["y", "y", "code", "code", "ok"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (0, "ok\n"),
    )

    assert status == "done"
    assert counts == {"coder": 3, "reviewer": 3}


def test_full_review_spec_returns_to_spec_before_repeating_the_loop(tmp_path):
    seed_product(tmp_path)
    names = []
    spec_prompts = []
    reviewer_count = 0

    def fake_invoke(mode, prompt):
        nonlocal reviewer_count
        names.append(mode.name)
        if mode.name == "spec":
            spec_prompts.append(prompt)
            test_action = (
                "change"
                if (tmp_path / "tests" / "test_app.py").is_file()
                else "create"
            )
            return files_spec(
                ("tests/test_app.py", test_action),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            return "FILE src/app.py\nchanged\n"
        reviewer_count += 1
        if reviewer_count == 1:
            return "FILE review.md\nThe spec is wrong.\n"
        return "FILE review.md\nLooks good.\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y", "spec", "y", "y", "ok"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (0, "ok\n"),
    )

    assert status == "done"
    assert names == [
        "spec",
        "tester",
        "coder",
        "reviewer",
        "spec",
        "tester",
        "coder",
        "reviewer",
    ]
    assert "The spec is wrong." in spec_prompts[1]


def test_full_red_suite_retries_coder_then_stuck(tmp_path):
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("original\n", encoding="utf-8")
    counts = {"coder": 0, "validator": 0}
    steps = []
    coder_prompts = []

    def fake_invoke(mode, prompt):
        steps.append((mode.name, current_event_context()["step"]))
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            counts["coder"] += 1
            coder_prompts.append(prompt)
            return "FILE src/app.py\nbroken\n"
        counts["validator"] += 1
        assert "FAILED tests/test_app.py::test_app" in prompt
        return "- test_app failed\n"

    shown = []
    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y"]).__next__,
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (1, "FAILED tests/test_app.py::test_app\n"),
    )

    assert status == "stuck"
    assert counts == {"coder": 3, "validator": 3}
    assert "Target:\noriginal\n" in coder_prompts[0]
    assert all("Target:\nbroken\n" in prompt for prompt in coder_prompts[1:])
    assert [name for name, _step in steps] == [
        "spec",
        "tester",
        "coder",
        "validator",
        "coder",
        "validator",
        "coder",
        "validator",
    ]
    assert [step for _name, step in steps] == [
        "spec",
        "tester",
        "coder",
        "validator",
        "coder",
        "validator",
        "coder",
        "validator",
    ]
    spec = (tmp_path / "spec.md").read_text(encoding="utf-8")
    assert spec.count("- test_app failed") == 3
    assert "stuck" in spec
    assert "stuck" in "\n".join(shown)


def test_full_coder_writes_every_non_test_file_and_shows_combined_diff(tmp_path):
    seed_product(tmp_path, "src/app.py", "app original\n")
    seed_product(tmp_path, "src/util.py", "util original\n")
    coder_prompts = []
    steps = []

    def fake_invoke(mode, prompt):
        steps.append((mode.name, current_event_context()["step"]))
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
                ("src/util.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            coder_prompts.append(prompt)
            if "Target:\napp original\n" in prompt:
                return "FILE src/app.py\napp changed\n"
            if "Target:\nutil original\n" in prompt:
                return "FILE src/util.py\nutil changed\n"
            raise AssertionError(prompt)
        return "FILE review.md\nLooks good.\n"

    shown = []
    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y", "ok"]).__next__,
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (0, "ok\n"),
    )

    assert status == "done"
    assert len(coder_prompts) == 2
    assert "Target:\napp original\n" in coder_prompts[0]
    assert "app changed" in coder_prompts[1]
    assert [step for name, step in steps if name == "coder"] == ["coder", "coder"]
    assert (tmp_path / "src" / "app.py").read_text(encoding="utf-8") == "app changed\n"
    assert (tmp_path / "src" / "util.py").read_text(encoding="utf-8") == "util changed\n"
    text = "\n".join(shown)
    assert text.index("--- a/src/app.py") < text.index("--- a/src/util.py")
    assert "+app changed" in text
    assert "+util changed" in text


def test_full_red_suite_redoes_all_coder_targets(tmp_path):
    seed_product(tmp_path, "src/app.py", "app original\n")
    seed_product(tmp_path, "src/util.py", "util original\n")
    counts = {"coder": 0, "validator": 0}
    coder_files = []

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
                ("src/util.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            counts["coder"] += 1
            if "Target:\napp " in prompt:
                coder_files.append("src/app.py")
                return "FILE src/app.py\napp broken\n"
            if "Target:\nutil " in prompt:
                coder_files.append("src/util.py")
                return "FILE src/util.py\nutil broken\n"
            raise AssertionError(prompt)
        counts["validator"] += 1
        return "- test_app failed\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (1, "FAILED tests/test_app.py::test_app\n"),
    )

    assert status == "stuck"
    assert counts == {"coder": 6, "validator": 3}
    assert coder_files == [
        "src/app.py",
        "src/util.py",
        "src/app.py",
        "src/util.py",
        "src/app.py",
        "src/util.py",
    ]


def test_full_red_suite_redoes_only_validator_picked_targets(tmp_path):
    seed_product(tmp_path, "src/app.py", "app original\n")
    seed_product(tmp_path, "src/util.py", "util original\n")
    coder_files = []
    redo_prompts = []

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
                ("src/util.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            if "Target:\napp " in prompt:
                coder_files.append("src/app.py")
                return "FILE src/app.py\napp v1\n"
            if "Target:\nutil " in prompt:
                coder_files.append("src/util.py")
                if "test_app failed" in prompt:
                    redo_prompts.append(prompt)
                    return "FILE src/util.py\nutil v2\n"
                return "FILE src/util.py\nutil v1\n"
            raise AssertionError(prompt)
        if mode.name == "validator":
            return "- test_app failed\nredo:\n- src/util.py\n"
        return "FILE review.md\nLooks good.\n"

    shown = []

    def suite(root):
        util = (root / "src" / "util.py").read_text(encoding="utf-8")
        if "util v2" in util:
            return 0, "ok\n"
        return 1, "FAILED tests/test_app.py::test_app\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y", "ok"]).__next__,
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=suite,
    )

    assert status == "done"
    assert coder_files == ["src/app.py", "src/util.py", "src/util.py"]
    assert "↻ coder: src/util.py" in shown
    assert redo_prompts
    assert "app v1" in redo_prompts[0]
    assert (tmp_path / "src" / "app.py").read_text(encoding="utf-8") == "app v1\n"
    assert (tmp_path / "src" / "util.py").read_text(encoding="utf-8") == "util v2\n"


def test_full_records_picked_paths_on_validator_jsonl_event(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seed_product(tmp_path, "src/app.py", "app original\n")
    seed_product(tmp_path, "src/util.py", "util original\n")

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
                ("src/util.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            if "Target:\napp " in prompt:
                return "FILE src/app.py\napp v1\n"
            if "Target:\nutil " in prompt:
                body = "util v2\n" if "test_app failed" in prompt else "util v1\n"
                return f"FILE src/util.py\n{body}"
            raise AssertionError(prompt)
        if mode.name == "validator":
            events = tmp_path / ".k0-mem" / "events.jsonl"
            events.parent.mkdir(parents=True, exist_ok=True)
            events.write_text(
                json.dumps({"step": "validator", "mode": "validator", "ok": True})
                + "\n",
                encoding="utf-8",
            )
            return "- test_app failed\nredo:\n- src/util.py\n"
        return "FILE review.md\nLooks good.\n"

    def suite(root):
        if "util v2" in (root / "src" / "util.py").read_text(encoding="utf-8"):
            return 0, "ok\n"
        return 1, "FAILED tests/test_app.py::test_app\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y", "ok"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=suite,
    )

    assert status == "done"
    events = [
        json.loads(line)
        for line in (tmp_path / ".k0-mem" / "events.jsonl").read_text().splitlines()
    ]
    validator = [event for event in events if event.get("step") == "validator"]
    assert validator[-1]["redo"] == ["src/util.py"]


def test_full_last_retry_redoes_all_coder_targets(tmp_path):
    seed_product(tmp_path, "src/app.py", "app original\n")
    seed_product(tmp_path, "src/util.py", "util original\n")
    coder_files = []

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
                ("src/util.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            if "Target:\napp " in prompt:
                coder_files.append("src/app.py")
                return "FILE src/app.py\napp broken\n"
            if "Target:\nutil " in prompt:
                coder_files.append("src/util.py")
                return "FILE src/util.py\nutil broken\n"
            raise AssertionError(prompt)
        return "- test_app failed\nredo:\n- src/util.py\n"

    shown = []
    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y"]).__next__,
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (1, "FAILED tests/test_app.py::test_app\n"),
    )

    assert status == "stuck"
    assert coder_files == [
        "src/app.py",
        "src/util.py",
        "src/util.py",
        "src/app.py",
        "src/util.py",
    ]
    assert shown.count("↻ coder: src/util.py") == 1
    assert shown.count("↻ coder: src/app.py src/util.py") == 1


def python_cmd(source: str) -> str:
    return f"{sys.executable} -c {source!r}"


def test_full_green_gates_skip_validator(tmp_path):
    seed_product(tmp_path)
    names = []

    def fake_invoke(mode, prompt):
        names.append(mode.name)
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            return "FILE src/app.py\nchanged\n"
        if mode.name == "reviewer":
            return "FILE review.md\nLooks good.\n"
        raise AssertionError(mode.name)

    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(
            gates=[
                {
                    "name": "complexity",
                    "cmd": python_cmd("open('complexity.ok','w').write('ok')"),
                },
                {
                    "name": "coverage",
                    "cmd": python_cmd("open('coverage.ok','w').write('ok')"),
                },
            ]
        ),
        read_line=iter(["y", "y", "ok"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (0, "ok\n"),
    )

    assert status == "done"
    assert names == ["spec", "tester", "coder", "reviewer"]
    assert (tmp_path / "complexity.ok").read_text(encoding="utf-8") == "ok"
    assert (tmp_path / "coverage.ok").read_text(encoding="utf-8") == "ok"


def test_full_red_gate_retries_coder_then_passes(tmp_path):
    seed_product(tmp_path)
    coder_count = 0
    validator_prompts = []

    def fake_invoke(mode, prompt):
        nonlocal coder_count
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            coder_count += 1
            if coder_count == 1:
                return "FILE src/app.py\nbroken\n"
            return "FILE src/app.py\nfixed\n"
        if mode.name == "validator":
            validator_prompts.append(prompt)
            assert "complexity" in prompt
            assert "too complex" in prompt
            return "- complexity failed\nredo:\n- src/app.py\n"
        return "FILE review.md\nLooks good.\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(
            gates=[
                {
                    "name": "complexity",
                    "cmd": python_cmd(
                        "from pathlib import Path; "
                        "text = Path('src/app.py').read_text(); "
                        "print('too complex'); "
                        "raise SystemExit(0 if 'fixed' in text else 1)"
                    ),
                }
            ]
        ),
        read_line=iter(["y", "y", "ok"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (0, "ok\n"),
    )

    assert status == "done"
    assert coder_count == 2
    assert validator_prompts
    assert (tmp_path / "src" / "app.py").read_text(encoding="utf-8") == "fixed\n"


def test_full_red_gate_retries_coder_then_stuck(tmp_path):
    seed_product(tmp_path)
    counts = {"coder": 0, "validator": 0}

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            counts["coder"] += 1
            return "FILE src/app.py\nbroken\n"
        counts["validator"] += 1
        assert "complexity" in prompt
        assert "too complex" in prompt
        return "- complexity failed\n"

    shown = []
    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(
            gates=[
                {
                    "name": "complexity",
                    "cmd": python_cmd("print('too complex'); raise SystemExit(1)"),
                }
            ]
        ),
        read_line=iter(["y", "y"]).__next__,
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (0, "ok\n"),
    )

    assert status == "stuck"
    assert counts == {"coder": 3, "validator": 3}
    spec = (tmp_path / "spec.md").read_text(encoding="utf-8")
    assert spec.count("- complexity failed") == 3
    assert "stuck" in spec
    assert "stuck" in "\n".join(shown)


def test_full_red_suite_skips_gates(tmp_path):
    seed_product(tmp_path)

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            return "FILE src/app.py\nbroken\n"
        return "- test_app failed\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(
            max_retries=0,
            gates=[
                {
                    "name": "complexity",
                    "cmd": python_cmd("open('gate_ran','w').write('x')"),
                }
            ],
        ),
        read_line=iter(["y", "y"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (1, "FAILED tests/test_app.py::test_app\n"),
    )

    assert status == "stuck"
    assert not (tmp_path / "gate_ran").exists()


def test_full_red_suite_redoes_all_when_validator_pick_is_invalid(tmp_path):
    seed_product(tmp_path, "src/app.py", "app original\n")
    seed_product(tmp_path, "src/util.py", "util original\n")
    coder_files = []

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
                ("src/util.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            if "Target:\napp " in prompt:
                coder_files.append("src/app.py")
                return "FILE src/app.py\napp v1\n"
            if "Target:\nutil " in prompt:
                coder_files.append("src/util.py")
                return "FILE src/util.py\nutil v1\n"
            raise AssertionError(prompt)
        if mode.name == "validator":
            return "- test_app failed\nredo:\n- tests/test_app.py\n- missing.py\n"
        return "FILE review.md\nLooks good.\n"

    shown = []

    def suite(root):
        if coder_files.count("src/app.py") >= 2:
            return 0, "ok\n"
        return 1, "FAILED tests/test_app.py::test_app\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y", "ok"]).__next__,
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=suite,
    )

    assert status == "done"
    assert coder_files == ["src/app.py", "src/util.py", "src/app.py", "src/util.py"]
    assert "↻ coder: src/app.py src/util.py" in shown


def test_full_red_suite_redoes_all_when_validator_pick_is_empty(tmp_path):
    seed_product(tmp_path, "src/app.py", "app original\n")
    seed_product(tmp_path, "src/util.py", "util original\n")
    coder_files = []

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
                ("src/util.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            if "Target:\napp " in prompt:
                coder_files.append("src/app.py")
                return "FILE src/app.py\napp v1\n"
            if "Target:\nutil " in prompt:
                coder_files.append("src/util.py")
                return "FILE src/util.py\nutil v1\n"
            raise AssertionError(prompt)
        if mode.name == "validator":
            return "- test_app failed\nredo:\n"
        return "FILE review.md\nLooks good.\n"

    shown = []

    def suite(root):
        if coder_files.count("src/app.py") >= 2:
            return 0, "ok\n"
        return 1, "FAILED tests/test_app.py::test_app\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y", "ok"]).__next__,
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=suite,
    )

    assert status == "done"
    assert coder_files == ["src/app.py", "src/util.py", "src/app.py", "src/util.py"]
    assert "↻ coder: src/app.py src/util.py" in shown


def test_full_coder_can_create_a_new_product_file(tmp_path):
    seen = []

    def fake_invoke(mode, prompt):
        seen.append((mode.name, prompt))
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/new.py", "create"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "reviewer":
            return "FILE review.md\nLooks good.\n"
        assert mode.name == "coder"
        assert "Target:\n<new file>" in prompt
        return "FILE src/new.py\ncreated\n"

    status = run_full(
        "create the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y", "ok"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (0, "ok\n"),
    )

    assert status == "done"
    assert [name for name, _prompt in seen] == ["spec", "tester", "coder", "reviewer"]
    assert (tmp_path / "src" / "new.py").read_text(encoding="utf-8") == "created\n"


def test_full_spec_injects_memory_when_gate_says_yes(tmp_path):
    node_id = write_file_memory(
        tmp_path,
        "src/app.py",
        "# src/app.py\n\n## hace\nreturns 1 from checkout\n\n## gotchas\nnone\n",
    )
    seen = {}

    def fake_invoke(mode, prompt):
        if mode.name == "retrieval_gate":
            seen["gate_prompt"] = prompt
            return f"retrieve: yes\n- {node_id}\n"
        assert mode.name == "spec"
        seen["prompt"] = prompt
        return "## Goal\ng\n## Files\n"

    status = run_full(
        "what does checkout return?",
        root=tmp_path,
        config=memory_loop_config(max_retries=0),
        read_line=lambda: "y",
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )
    assert status == "stuck"
    assert node_id in seen["gate_prompt"]
    assert "returns 1 from checkout" in seen["prompt"]
    assert "what does checkout return?" in seen["prompt"]


def test_full_loop_agents_inject_memory_and_validator_does_not(tmp_path):
    node_id = write_file_memory(
        tmp_path,
        "src/app.py",
        "# src/app.py\n\n## hace\nreturns 1 from checkout\n\n## gotchas\nnone\n",
    )
    rules = tmp_path / ".k0-mem" / "General_mem" / "Rules.md"
    rules.parent.mkdir(parents=True, exist_ok=True)
    rules.write_text("# Rules\n\nNever log secrets.\n", encoding="utf-8")
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("original\n", encoding="utf-8")
    prompts = {}

    def fake_invoke(mode, prompt):
        if mode.name == "retrieval_gate":
            return f"retrieve: yes\n- {node_id}\n"
        prompts[mode.name] = prompt
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            return "FILE src/app.py\nbroken\n"
        assert mode.name == "validator"
        return "- test_app failed\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=memory_loop_config(max_retries=0),
        read_line=iter(["y", "y"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (1, "FAILED tests/test_app.py::test_app\n"),
    )
    assert status == "stuck"
    memory = "returns 1 from checkout"
    assert memory in prompts["spec"]
    assert memory in prompts["tester"]
    assert memory in prompts["coder"]
    assert "# Rules\n\nNever log secrets.\n" in prompts["coder"]
    assert memory not in prompts["validator"]
    assert "Never log secrets" not in prompts["validator"]
    assert "reviewer" not in prompts


def test_full_reviewer_injects_memory(tmp_path):
    node_id = write_file_memory(
        tmp_path,
        "src/app.py",
        "# src/app.py\n\n## hace\nreturns 1 from checkout\n\n## gotchas\nnone\n",
    )
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("original\n", encoding="utf-8")
    seen = {}

    def fake_invoke(mode, prompt):
        if mode.name == "retrieval_gate":
            return f"retrieve: yes\n- {node_id}\n"
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            return "FILE src/app.py\nchanged\n"
        assert mode.name == "reviewer"
        seen["prompt"] = prompt
        return "FILE review.md\nLooks good.\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=memory_loop_config(),
        read_line=iter(["y", "y", "ok"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (0, "ok\n"),
    )
    assert status == "done"
    assert "returns 1 from checkout" in seen["prompt"]


def test_full_injects_custom_rules_outside_the_gate(tmp_path):
    write_file_memory(
        tmp_path,
        "src/app.py",
        "# src/app.py\n\n## hace\nreturns 1 from checkout\n\n## gotchas\nnone\n",
    )
    rules = tmp_path / ".k0-mem" / "General_mem" / "Rules.md"
    rules.parent.mkdir(parents=True, exist_ok=True)
    rules.write_text("# Rules\n\nNever log secrets.\n", encoding="utf-8")
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("original\n", encoding="utf-8")
    prompts = {}

    def fake_invoke(mode, prompt):
        if mode.name == "retrieval_gate":
            return "retrieve: no\n"
        prompts[mode.name] = prompt
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            return "FILE src/app.py\nchanged\n"
        assert mode.name == "reviewer"
        return "FILE review.md\nLooks good.\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=memory_loop_config(),
        read_line=iter(["y", "y", "ok"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (0, "ok\n"),
    )
    assert status == "done"
    for name in ("spec", "tester", "coder", "reviewer"):
        assert "# Rules\n\nNever log secrets.\n" in prompts[name]
        assert "returns 1 from checkout" not in prompts[name]


def test_full_injects_custom_topology_outside_the_gate(tmp_path):
    write_file_memory(
        tmp_path,
        "src/app.py",
        "# src/app.py\n\n## hace\nreturns 1 from checkout\n\n## gotchas\nnone\n",
    )
    topo = tmp_path / ".k0-mem" / "Project_topology" / tmp_path.name
    topo.mkdir(parents=True, exist_ok=True)
    (topo / "architecture.md").write_text(
        "# Architecture\n\nHexagonal core.\n",
        encoding="utf-8",
    )
    (topo / "strategy.md").write_text(
        "# Strategy\n\nSmall PRs.\n",
        encoding="utf-8",
    )
    (topo / "code_rules.md").write_text(CODE_RULES, encoding="utf-8")
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target.write_text("original\n", encoding="utf-8")
    prompts = {}

    def fake_invoke(mode, prompt):
        if mode.name == "retrieval_gate":
            return "retrieve: no\n"
        prompts[mode.name] = prompt
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            return "FILE src/app.py\nchanged\n"
        assert mode.name == "reviewer"
        return "FILE review.md\nLooks good.\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=memory_loop_config(),
        read_line=iter(["y", "y", "ok"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (0, "ok\n"),
    )
    assert status == "done"
    for name in ("spec", "tester", "coder", "reviewer"):
        assert "# Architecture\n\nHexagonal core.\n" in prompts[name]
        assert "# Strategy\n\nSmall PRs.\n" in prompts[name]
        assert "Local coding constraints" not in prompts[name]
        assert "returns 1 from checkout" not in prompts[name]
    assert "validator" not in prompts or "Hexagonal core" not in prompts["validator"]


def test_full_skips_template_topology(tmp_path):
    write_file_memory(
        tmp_path,
        "src/app.py",
        "# src/app.py\n\n## hace\nreturns 1 from checkout\n\n## gotchas\nnone\n",
    )
    topo = tmp_path / ".k0-mem" / "Project_topology" / tmp_path.name
    topo.mkdir(parents=True, exist_ok=True)
    (topo / "architecture.md").write_text(ARCHITECTURE, encoding="utf-8")
    (topo / "strategy.md").write_text(STRATEGY, encoding="utf-8")
    (topo / "code_rules.md").write_text(CODE_RULES, encoding="utf-8")
    seen = {}

    def fake_invoke(mode, prompt):
        if mode.name == "retrieval_gate":
            return "retrieve: no\n"
        seen["prompt"] = prompt
        return "## Goal\ng\n## Files\n"

    status = run_full(
        "what does checkout return?",
        root=tmp_path,
        config=memory_loop_config(max_retries=0),
        read_line=lambda: "y",
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )
    assert status == "stuck"
    assert seen["prompt"] == "what does checkout return?"
    assert "System shape" not in seen["prompt"]
    assert "How we change this codebase" not in seen["prompt"]
    assert "Local coding constraints" not in seen["prompt"]


def test_full_skips_template_rules(tmp_path):
    write_file_memory(
        tmp_path,
        "src/app.py",
        "# src/app.py\n\n## hace\nreturns 1 from checkout\n\n## gotchas\nnone\n",
    )
    rules = tmp_path / ".k0-mem" / "General_mem" / "Rules.md"
    rules.parent.mkdir(parents=True, exist_ok=True)
    rules.write_text(RULES, encoding="utf-8")
    seen = {}

    def fake_invoke(mode, prompt):
        if mode.name == "retrieval_gate":
            return "retrieve: no\n"
        seen["prompt"] = prompt
        return "## Goal\ng\n## Files\n"

    status = run_full(
        "what does checkout return?",
        root=tmp_path,
        config=memory_loop_config(max_retries=0),
        read_line=lambda: "y",
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )
    assert status == "stuck"
    assert seen["prompt"] == "what does checkout return?"
    assert "Global rules for every agent" not in seen["prompt"]


def test_full_coder_keeps_whole_target_when_memory_is_injected(tmp_path):
    write_file_memory(
        tmp_path,
        "src/app.py",
        "# src/app.py\n\n## hace\nreturns 1 from checkout\n\n## gotchas\nnone\n",
    )
    target = tmp_path / "src" / "app.py"
    target.parent.mkdir()
    target_body = "x\n" * 400
    target.write_text(target_body, encoding="utf-8")
    seen = {}

    def fake_invoke(mode, prompt):
        if mode.name == "retrieval_gate":
            return f"retrieve: yes\n- {tmp_path.name}:file:src/app.py\n"
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
            )
        if mode.name == "tester":
            return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"
        if mode.name == "coder":
            seen["prompt"] = prompt
            return "FILE src/app.py\nchanged\n"
        return "FILE review.md\nLooks good.\n"

    status = run_full(
        "change the app",
        root=tmp_path,
        config=memory_loop_config(),
        read_line=iter(["y", "y", "ok"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: "SUMMARY\n",
        run_suite=lambda root: (0, "ok\n"),
    )
    assert status == "done"
    assert f"Target:\n{target_body}" in seen["prompt"]
    assert "returns 1 from checkout" in seen["prompt"]
    assert seen["prompt"].index(f"Target:\n{target_body}") < seen["prompt"].index(
        "returns 1 from checkout"
    )


def test_full_spec_gate_shows_parsed_files_list(tmp_path):
    seed_product(tmp_path)

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return files_spec(
                ("tests/test_app.py", "create"),
                ("src/app.py", "change"),
            )
        return "FILE tests/test_app.py\ndef test_app():\n    assert True\n"

    shown = []
    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y"]).__next__,
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )

    assert status == "done"
    text = "\n".join(shown)
    assert "tests/test_app.py (create)" in text
    assert "src/app.py (change)" in text
    assert "y/n" in text


def test_full_accepts_backticked_files_from_release_check(tmp_path):
    seed_product(tmp_path, "calc.py", "def add(a, b):\n    return a + b\n")
    seed_product(tmp_path, "test_calc.py", "def test_add():\n    assert True\n")

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return (
                "## Goal\ng\n## Files\n"
                "- `calc.py` (change)\n"
                "- `test_calc.py` (change)\n"
            )
        if mode.name == "tester":
            return "FILE test_calc.py\ndef test_add():\n    assert True\n"
        if mode.name == "coder":
            return "FILE calc.py\nchanged\n"
        return "FILE review.md\nLooks good.\n"

    shown = []
    status = run_full(
        "change add",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y", "ok"]).__next__,
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
        run_suite=lambda root: (0, "ok\n"),
    )

    assert status == "done"
    assert "calc.py (change)" in shown
    assert "test_calc.py (change)" in shown


def test_full_rejects_annotated_files_from_release_check(tmp_path):
    names = []

    def fake_invoke(mode, prompt):
        names.append(mode.name)
        return (
            "## Goal\ng\n## Files\n"
            "- `tests/test_x.py`: change. add a test\n"
            "- `src/app.py` (change)\n"
        )

    shown = []
    status = run_full(
        "add a test",
        root=tmp_path,
        config=loop_config(max_retries=0),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("gate")),
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )

    assert status == "stuck"
    assert names == ["spec"]
    text = "\n".join(shown)
    assert "invalid ## Files line" in text
    assert "stuck" in text
    assert "done" not in shown
    spec = (tmp_path / "spec.md").read_text(encoding="utf-8")
    assert "invalid ## Files line" in spec


def test_full_files_need_a_test_path(tmp_path):
    seed_product(tmp_path)

    def fake_invoke(mode, prompt):
        return files_spec(("src/app.py", "change"))

    shown = []
    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(max_retries=0),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("gate")),
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )

    assert status == "stuck"
    assert "## Files needs a test path" in "\n".join(shown)


def test_full_files_need_a_non_test_path(tmp_path):
    def fake_invoke(mode, prompt):
        return files_spec(("tests/test_app.py", "create"))

    shown = []
    status = run_full(
        "add a test",
        root=tmp_path,
        config=loop_config(max_retries=0),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("gate")),
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )

    assert status == "stuck"
    assert "## Files needs a non-test path" in "\n".join(shown)


def test_full_files_change_path_must_exist(tmp_path):
    def fake_invoke(mode, prompt):
        return files_spec(
            ("tests/test_app.py", "create"),
            ("src/app.py", "change"),
        )

    shown = []
    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(max_retries=0),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("gate")),
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )

    assert status == "stuck"
    assert "change path does not exist: src/app.py" in "\n".join(shown)


def test_full_files_create_path_must_not_exist(tmp_path):
    seed_product(tmp_path)

    def fake_invoke(mode, prompt):
        return files_spec(
            ("tests/test_app.py", "create"),
            ("src/app.py", "create"),
        )

    shown = []
    status = run_full(
        "create the app",
        root=tmp_path,
        config=loop_config(max_retries=0),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("gate")),
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )

    assert status == "stuck"
    assert "create path already exists: src/app.py" in "\n".join(shown)


def test_full_files_path_must_stay_inside_the_repo(tmp_path):
    def fake_invoke(mode, prompt):
        return files_spec(
            ("tests/test_app.py", "create"),
            ("../secret.py", "create"),
        )

    shown = []
    status = run_full(
        "write outside",
        root=tmp_path,
        config=loop_config(max_retries=0),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("gate")),
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )

    assert status == "stuck"
    assert "path is outside the repo: ../secret.py" in "\n".join(shown)


def test_full_files_path_must_stay_outside_k0_mem(tmp_path):
    mem = tmp_path / ".k0-mem" / "secret.py"
    mem.parent.mkdir(parents=True, exist_ok=True)
    mem.write_text("nope\n", encoding="utf-8")

    def fake_invoke(mode, prompt):
        return files_spec(
            ("tests/test_app.py", "create"),
            (".k0-mem/secret.py", "change"),
        )

    shown = []
    status = run_full(
        "edit memory",
        root=tmp_path,
        config=loop_config(max_retries=0),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("gate")),
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )

    assert status == "stuck"
    assert "path is inside .k0-mem/: .k0-mem/secret.py" in "\n".join(shown)


def test_full_invalid_files_retry_then_stuck(tmp_path):
    seed_product(tmp_path)
    prompts = []

    def fake_invoke(mode, prompt):
        prompts.append(prompt)
        return files_spec(("src/app.py", "change"))

    shown = []
    status = run_full(
        "change the app",
        root=tmp_path,
        config=loop_config(max_retries=1),
        read_line=lambda: (_ for _ in ()).throw(AssertionError("gate")),
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )

    assert status == "stuck"
    assert len(prompts) == 2
    assert "## Files needs a test path" in prompts[1]
    assert (tmp_path / "spec.md").read_text(encoding="utf-8").count(
        "## Files needs a test path"
    ) == 1
    assert shown.count("## Files needs a test path") == 2
    assert "stuck" in shown


def write_agent(path, body):
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_full_each_gate_decision_emits_one_jsonl_event(tmp_path, monkeypatch):
    write_file_memory(
        tmp_path,
        "src/app.py",
        "# src/app.py\n\n## hace\nreturns 1 from checkout\n\n## gotchas\nnone\n",
    )
    monkeypatch.chdir(tmp_path)
    agent = tmp_path / "cursor-agent"
    write_agent(
        agent,
        "import json, sys\n"
        "prompt = sys.argv[-1]\n"
        "if 'Pick at most' in prompt:\n"
        "    result = 'retrieve: no\\n'\n"
        "else:\n"
        "    result = '## Goal\\ng\\n## Files\\n'\n"
        "print(json.dumps({'type':'result','subtype':'success','is_error':False,"
        "'result': result}))\n",
    )
    monkeypatch.setenv("K0_CURSOR_AGENT", str(agent))
    status = run_full(
        "what does checkout return?",
        root=tmp_path,
        config=memory_loop_config(max_retries=0),
        read_line=lambda: "y",
        write=lambda _line: None,
        invoke=invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bulk")
        ),
    )
    assert status == "stuck"
    events = [
        json.loads(line)
        for line in (tmp_path / ".k0-mem" / "events.jsonl").read_text().splitlines()
    ]
    gate_events = [event for event in events if event["mode"] == "retrieval_gate"]
    assert len(gate_events) == 1
    assert gate_events[0]["ok"] is True
