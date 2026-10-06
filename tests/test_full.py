import pytest

from k0ntrol.harness.loop import run_full


def loop_config(**overrides):
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
    cfg.update(overrides)
    return cfg


def test_full_rewrites_spec_until_yes(tmp_path):
    calls = []

    def fake_invoke(mode, prompt):
        calls.append((mode.name, prompt))
        return f"## Goal\nv{len(calls)}\n## Files\n"

    lines = iter(["n", "y"])
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
    assert status == "tested"
    assert [name for name, _ in calls] == ["spec", "spec"]
    assert "rejected" in calls[1][1]
    assert (tmp_path / "spec.md").read_text(encoding="utf-8") == "## Goal\nv2\n## Files\n"
    assert "## Goal" in "\n".join(shown)


def test_full_delegates_a_large_file_before_spec(tmp_path):
    big = tmp_path / "big.py"
    big.write_text("x\n" * 351, encoding="utf-8")

    def fake_bulk(question, paths, **kwargs):
        assert list(paths) == [big]
        return "SUMMARY\n"

    def fake_invoke(mode, prompt):
        assert mode.name == "spec"
        assert "SUMMARY" in prompt
        assert "x\n" * 351 not in prompt
        return "## Goal\nv\n## Files\n"

    status = run_full(
        "what @big.py",
        root=tmp_path,
        config=loop_config(),
        read_line=lambda: "y",
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=fake_bulk,
    )
    assert status == "tested"


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
            return "## Goal\ng\n## Files\n- tests/test_app.py\n- src/app.py\n"
        assert mode.name == "tester"
        return "FILE tests/test_app.py\ndef test_app():\n    assert False\n"

    status = run_full(
        "add a test",
        root=tmp_path,
        config=loop_config(),
        read_line=iter(["y", "y"]).__next__,
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bulk")),
    )
    assert (tmp_path / "tests" / "test_app.py").read_text(encoding="utf-8").startswith(
        "def test_app"
    )
    assert not (tmp_path / "src" / "app.py").exists()
    assert status == "done"


def test_two_noes_retry_the_tester_once_then_stuck(tmp_path):
    tester_prompts = []

    def fake_invoke(mode, prompt):
        if mode.name == "spec":
            return "## Goal\ng\n## Files\n- tests/test_app.py\n"
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
