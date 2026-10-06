from k0ntrol.harness.loop import run_fast


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


def test_fast_writes_literal_spec_and_a_test(tmp_path):
    (tmp_path / "a.py").write_text("original\n", encoding="utf-8")
    names = []

    def fake_invoke(mode, prompt):
        names.append(mode.name)
        if mode.name == "tester":
            return "FILE tests/test_fast.py\ndef test_fast():\n    assert False\n"
        if mode.name == "coder":
            return "FILE a.py\nchanged\n"
        raise AssertionError(mode.name)

    status = run_fast(
        "change the app @a.py",
        root=tmp_path,
        config=loop_config(),
        read_line=lambda: "y",
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bulk")),
    )
    assert status == "tested"
    assert names == ["tester", "coder"]
    assert (tmp_path / "spec.md").read_text(encoding="utf-8") == (
        "change the app\n\n## Files\n- a.py\n"
    )
    assert (tmp_path / "tests" / "test_fast.py").is_file()
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "changed\n"


def test_coder_gets_the_whole_target_and_bulk_reads_only_the_reference(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    target = source / "app.py"
    target.write_text("x\n" * 351, encoding="utf-8")
    notes = tmp_path / "docs"
    notes.mkdir()
    (notes / "notes.md").write_text("n\n" * 400, encoding="utf-8")
    bulk_paths = []

    def fake_bulk(question, paths, **kwargs):
        bulk_paths.extend(paths)
        return "SUMMARY\n"

    def fake_invoke(mode, prompt):
        if mode.name == "tester":
            assert "SUMMARY" in prompt
            assert "x\n" * 351 not in prompt
            return "FILE tests/test_app.py\ndef test_app():\n    assert False\n"
        assert mode.name == "coder"
        assert "x\n" * 351 in prompt
        assert "SUMMARY" in prompt
        return "FILE src/app.py\nprint('ok')\n"

    run_fast(
        "change app @src/app.py @docs/notes.md",
        root=tmp_path,
        config=loop_config(),
        read_line=lambda: "y",
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=fake_bulk,
    )
    assert [path.name for path in bulk_paths] == ["notes.md"]
    assert (target).read_text(encoding="utf-8") == "print('ok')\n"


import sys


def test_red_suite_retries_coder_then_stuck(tmp_path):
    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")
    counts = {"coder": 0, "validator": 0}

    def fake_invoke(mode, prompt):
        if mode.name == "tester":
            return "FILE tests/test_fast.py\ndef test_fast():\n    assert False\n"
        if mode.name == "coder":
            counts["coder"] += 1
            return "FILE a.py\nBREAK\n"
        counts["validator"] += 1
        assert "FAILED tests/test_old.py::test_old" in prompt
        return "- test_old failed\n"

    def suite(root):
        text = (root / "a.py").read_text(encoding="utf-8")
        if "BREAK" in text:
            return 1, "FAILED tests/test_old.py::test_old"
        return 0, "ok\n"

    status = run_fast(
        "change @a.py",
        root=tmp_path,
        config=loop_config(),
        read_line=lambda: "y",
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bulk")),
        run_suite=suite,
    )
    assert status == "stuck"
    assert counts["coder"] == 3
    assert counts["validator"] == 3
    assert "stuck" in (tmp_path / "spec.md").read_text(encoding="utf-8")


def test_green_suite_reaches_the_gate(tmp_path):
    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")

    def fake_invoke(mode, prompt):
        if mode.name == "tester":
            return "FILE tests/test_fast.py\ndef test_fast():\n    assert False\n"
        if mode.name == "coder":
            return "FILE a.py\nprint('ok')\n"
        raise AssertionError(mode.name)

    shown = []
    status = run_fast(
        "change @a.py",
        root=tmp_path,
        config=loop_config(),
        read_line=lambda: "y",
        write=shown.append,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bulk")),
        run_suite=lambda root: (0, "ok\n"),
    )
    assert status == "done"
    assert "+print('ok')" in "\n".join(shown)


def test_no_on_the_green_gate_is_stuck(tmp_path):
    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")

    def fake_invoke(mode, prompt):
        if mode.name == "tester":
            return "FILE tests/test_fast.py\ndef test_fast():\n    assert False\n"
        return "FILE a.py\nprint('ok')\n"

    status = run_fast(
        "change @a.py",
        root=tmp_path,
        config=loop_config(),
        read_line=lambda: "n",
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bulk")),
        run_suite=lambda root: (0, "ok\n"),
    )
    assert status == "stuck"
    assert "rejected at final gate" in (tmp_path / "spec.md").read_text(encoding="utf-8")


def test_configured_suite_returns_the_process_code(tmp_path):
    from k0ntrol.harness.loop import run_configured_suite

    code, _output = run_configured_suite(
        tmp_path,
        f'{sys.executable} -c "raise SystemExit(3)"',
    )
    assert code == 3


def test_fast_and_validator_steps(tmp_path):
    from k0ntrol.backends.invoke import current_event_context
    from k0ntrol.harness.loop import run_fast

    (tmp_path / "a.py").write_text("a\n", encoding="utf-8")
    seen = []

    def fake_invoke(mode, prompt):
        seen.append((mode.name, current_event_context()["step"], current_event_context()["run"]))
        if mode.name == "tester":
            return "FILE tests/test_fast.py\ndef test_fast():\n    assert False\n"
        if mode.name == "coder":
            return "FILE a.py\nBREAK\n"
        return "- test_old failed\n"

    def suite(root):
        text = (root / "a.py").read_text(encoding="utf-8")
        if "BREAK" in text:
            return 1, "FAILED tests/test_old.py::test_old"
        return 0, "ok\n"

    cfg = {
        "backend": "cursor",
        "threshold_lines": 350,
        "test_cmd": "pytest -q",
        "max_retries": 0,
        "agents": {
            name: {"model": "m"}
            for name in ("spec", "tester", "coder", "validator", "bulk_reader")
        },
    }
    status = run_fast(
        "change @a.py",
        root=tmp_path,
        config=cfg,
        read_line=lambda: "y",
        write=lambda _line: None,
        invoke=fake_invoke,
        bulk_read=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bulk")),
        run_suite=suite,
    )
    assert status == "stuck"
    assert [item[0] for item in seen] == ["tester", "coder", "validator"]
    assert [item[1] for item in seen] == ["tester", "coder", "validator"]
    assert len({item[2] for item in seen}) == 1
    assert seen[0][2]
