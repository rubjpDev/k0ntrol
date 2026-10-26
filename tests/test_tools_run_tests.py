"""One scrubbed, timed-out, truncated runner behind run_tests/test_cmd/gates."""

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
