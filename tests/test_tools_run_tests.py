"""One scrubbed, timed-out, truncated runner behind run_tests/test_cmd/gates."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from k0ntrol.tools.run_tests import make_run_tests
from k0ntrol.tools.runner import OUTPUT_LIMIT, run_command, scrubbed_env


@pytest.fixture
def root(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("X = 1\n")
    return tmp_path


def test_run_command_returns_exit_code_and_output(root):
    result = run_command([sys.executable, "-c", "print('hello')"], root=root, timeout=30)
    assert result.returncode == 0
    assert "hello" in result.output
    assert result.ok
    assert not result.timed_out


def test_run_command_accepts_a_string_and_splits_it(root):
    result = run_command(f'{sys.executable} -c "print(1 + 1)"', root=root, timeout=30)
    assert result.returncode == 0
    assert "2" in result.output


def test_run_command_reports_a_nonzero_exit(root):
    result = run_command(
        [sys.executable, "-c", "import sys; sys.exit(3)"], root=root, timeout=30
    )
    assert result.returncode == 3
    assert not result.ok


def test_run_command_scrubs_secret_shaped_and_vendor_env_vars(root):
    environ = {
        "ANTHROPIC_API_KEY": "sk-ant",
        "MY_TOKEN": "tok",
        "DB_SECRET": "s",
        "GOOGLE_APPLICATION_CREDENTIALS": "/cred.json",
        "AWS_ACCESS_KEY_ID": "AKIA",
        "KEEP_ME": "visible",
        "PATH": "/usr/bin:/bin",
    }
    result = run_command(
        [sys.executable, "-c", "import os; print(sorted(os.environ))"],
        root=root,
        timeout=30,
        environ=environ,
    )
    assert "KEEP_ME" in result.output
    for leaked in ("ANTHROPIC_API_KEY", "MY_TOKEN", "DB_SECRET",
                   "GOOGLE_APPLICATION_CREDENTIALS", "AWS_ACCESS_KEY_ID"):
        assert leaked not in result.output


def test_scrubbed_env_removes_suffixes_and_vendor_names():
    scrubbed = scrubbed_env(
        {"A_KEY": "1", "B_TOKEN": "2", "C_SECRET": "3", "PLAIN": "4"}
    )
    assert scrubbed == {"PLAIN": "4"}


def test_run_command_times_out_and_a_timeout_is_red(root):
    result = run_command(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        root=root,
        timeout=0.5,
    )
    assert result.timed_out
    assert not result.ok


def test_run_command_truncates_long_output_keeping_the_tail(root):
    result = run_command(
        [sys.executable, "-c", "print('x' * 50000); print('THE-TAIL')"],
        root=root,
        timeout=30,
    )
    assert "THE-TAIL" in result.output
    assert len(result.output) <= OUTPUT_LIMIT + 200


def test_run_command_keeps_partial_output_on_timeout(root):
    result = run_command(
        [sys.executable, "-c",
         "import sys, time; print('partial'); sys.stdout.flush(); time.sleep(30)"],
        root=root,
        timeout=0.5,
    )
    assert result.timed_out
    assert "partial" in result.output


def test_run_command_survives_a_missing_binary(root):
    result = run_command(["definitely-not-a-binary-xyz"], root=root, timeout=30)
    assert not result.ok


def test_run_command_never_tests_a_stale_pyc(root):
    """An edit in the same second, same size, must not hit the old bytecode."""
    module = root / "calc.py"
    module.write_text("ANSWER = 42\n")
    probe = [sys.executable, "-c", "import calc; print(calc.ANSWER)"]
    assert "42" in run_command(probe, root=root, timeout=30).output
    stamp = module.stat().st_mtime_ns
    module.write_text("ANSWER = 41\n")
    os.utime(module, ns=(stamp, stamp))

    assert "41" in run_command(probe, root=root, timeout=30).output


def _plain_python_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}


def test_run_command_ignores_a_pyc_cached_before_the_call(root):
    """A .pyc written by plain Python must not mask a same-size, same-mtime edit."""
    module = root / "calc.py"
    module.write_text("ANSWER = 42\n")
    probe = [sys.executable, "-c", "import calc; print(calc.ANSWER)"]
    subprocess.run(probe, cwd=root, env=_plain_python_env(), check=True, capture_output=True)
    assert list((root / "__pycache__").glob("calc.*.pyc"))
    stamp = module.stat().st_mtime_ns
    module.write_text("ANSWER = 41\n")
    os.utime(module, ns=(stamp, stamp))

    assert run_command(probe, root=root, timeout=30).output.strip() == "41"


def test_run_command_pytest_ignores_a_cached_assertion_rewrite(root):
    """pytest's assertion rewriter caches its own .pyc: it must be bypassed too."""
    test_file = root / "tests" / "test_answer.py"
    test_file.write_text("def test_answer():\n    assert 42 == 42\n")
    pytest_cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test_file)]
    subprocess.run(pytest_cmd, cwd=root, env=_plain_python_env(), check=True, capture_output=True)
    assert list((root / "tests" / "__pycache__").glob("test_answer*.pyc"))
    stamp = test_file.stat().st_mtime_ns
    test_file.write_text("def test_answer():\n    assert 41 == 42\n")
    os.utime(test_file, ns=(stamp, stamp))

    result = run_command(pytest_cmd, root=root, timeout=60)

    assert not result.ok
    assert "41 == 42" in result.output


def _same_stamp_edit(module: Path, text: str) -> None:
    """Rewrite ``module`` with same-size ``text`` and its old mtime (#70)."""
    stamp = module.stat().st_mtime_ns
    module.write_text(text)
    os.utime(module, ns=(stamp, stamp))


def test_run_command_reuses_bytecode_compiled_outside_the_root(root, tmp_path_factory):
    """Dependencies compile once per process: the second call reuses the cache.

    The stale value proves reuse without timing: only a cached .pyc can still
    say 42 after a same-size, same-mtime edit outside the root. The cache is
    k0's own, so a ``PYTHONDONTWRITEBYTECODE`` in the user's env must not
    switch it off.
    """
    outside = tmp_path_factory.mktemp("site")
    module = outside / "dep.py"
    module.write_text("ANSWER = 42\n")
    environ = {**os.environ, "PYTHONPATH": str(outside), "PYTHONDONTWRITEBYTECODE": "1"}
    probe = [sys.executable, "-c", "import dep; print(dep.ANSWER)"]
    assert run_command(probe, root=root, timeout=30, environ=environ).output.strip() == "42"
    _same_stamp_edit(module, "ANSWER = 41\n")

    result = run_command(probe, root=root, timeout=30, environ=environ)

    assert result.output.strip() == "42"


@pytest.mark.parametrize("via", ["cwd", "link"])
def test_run_command_sees_a_same_stamp_edit_under_a_symlinked_root(root, tmp_path_factory, via):
    """The cache mirrors whichever path the child imports through.

    Through the cwd the child sees the real path; through ``sys.path`` it
    sees the link. Both mirrors must go before each command.
    """
    link = tmp_path_factory.mktemp("links") / "checkout"
    link.symlink_to(root, target_is_directory=True)
    module = root / "calc.py"
    module.write_text("ANSWER = 42\n")
    head = f"import sys; sys.path.insert(0, {str(link)!r}); " if via == "link" else ""
    probe = [sys.executable, "-c", head + "import calc; print(calc.ANSWER)"]
    assert run_command(probe, root=link, timeout=30).output.strip() == "42"
    _same_stamp_edit(module, "ANSWER = 41\n")

    assert run_command(probe, root=link, timeout=30).output.strip() == "41"


def test_run_command_leaves_no_new_repo_bytecode(root):
    (root / "calc.py").write_text("ANSWER = 42\n")
    before = sorted(root.rglob("*.pyc"))
    probe = [sys.executable, "-c", "import calc; print(calc.ANSWER)"]

    assert run_command(probe, root=root, timeout=30).ok

    assert sorted(root.rglob("*.pyc")) == before


_PRINT_CACHE_DIR = """
import sys
from pathlib import Path
from k0ntrol.tools.runner import run_command
probe = [sys.executable, "-c", "import os; print(os.environ['PYTHONPYCACHEPREFIX'])"]
cache = run_command(probe, root=Path(sys.argv[1]), timeout=30).output.strip()
print(cache, Path(cache).is_dir())
"""


def test_run_command_cache_dir_is_gone_after_the_process_exits(root):
    child = subprocess.run(
        [sys.executable, "-c", _PRINT_CACHE_DIR, str(root)],
        capture_output=True, text=True, check=True,
    )
    cache, existed = child.stdout.split()

    assert existed == "True"
    assert not Path(cache).exists()


def test_run_command_still_runs_after_a_timeout(root):
    timed_out = run_command(
        [sys.executable, "-c", "import time; time.sleep(30)"], root=root, timeout=0.5
    )
    assert timed_out.timed_out
    assert not timed_out.ok

    after = run_command([sys.executable, "-m", "pytest", "-q"], root=root, timeout=60)

    assert after.ok, after.output


def test_run_tests_runs_the_configured_test_cmd(root):
    tool = make_run_tests(root, {"test_cmd": f'{sys.executable} -c "print(\'ran\')"'})
    output = tool.invoke({"paths": []})
    assert "ran" in output
    assert "exit 0" in output


def test_run_tests_narrows_to_existing_test_paths(root):
    tool = make_run_tests(
        root,
        {"test_cmd": f'{sys.executable} -c "import sys; print(\'ARGS:\' + \'|\'.join(sys.argv[1:]))"'},
    )
    output = tool.invoke(
        {"paths": ["tests/test_a.py", "src/app.py", "tests/missing.py", "../outside.py"]}
    )
    assert "ARGS:tests/test_a.py" in output
    assert "src/app.py" not in output
    assert "missing.py" not in output


def test_run_tests_runs_the_full_command_when_no_paths_qualify(root):
    tool = make_run_tests(
        root,
        {"test_cmd": f'{sys.executable} -c "import sys; print(\'ARGS:\' + \'|\'.join(sys.argv[1:]))"'},
    )
    output = tool.invoke({"paths": ["src/app.py", "tests/missing.py"]})
    assert "ARGS:" in output
    assert "app.py" not in output


def test_run_tests_reports_a_timeout_as_red(root):
    tool = make_run_tests(
        root,
        {"test_cmd": f'{sys.executable} -c "import time; time.sleep(30)"',
         "test_timeout": 0.5},
    )
    output = tool.invoke({"paths": []})
    assert "timed out" in output
