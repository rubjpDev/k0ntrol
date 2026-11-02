"""The stdlib stub runner the workflow fixtures use as ``test_cmd`` (ADR 0009).

It runs as a real subprocess, like any ``test_cmd``: these tests run it
in a temp repo and read only its exit code and output, as k0 does.
"""

import subprocess
import sys
from pathlib import Path

STUB_RUNNER = Path(__file__).parent / "stub_runner.py"

PASSING = "def test_one():\n    assert 1 == 1\n\n\ndef test_two():\n    assert True\n"
FAILING = "def test_bad():\n    assert 1 == 2, 'one is not two'\n"


def stub(root: Path, *files: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(STUB_RUNNER), *files],
        cwd=root, capture_output=True, text=True, check=False,
    )


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_passing_tests_print_the_count_and_exit_zero(tmp_path):
    write(tmp_path, "tests/test_ok.py", PASSING)

    done = stub(tmp_path, "tests/test_ok.py")

    assert done.returncode == 0, done.stdout
    assert "2 passed" in done.stdout


def test_a_failing_test_prints_its_name_and_error_and_exits_non_zero(tmp_path):
    write(tmp_path, "tests/test_bad.py", FAILING)

    done = stub(tmp_path, "tests/test_bad.py")

    assert done.returncode != 0
    assert "FAILED tests/test_bad.py::test_bad" in done.stdout
    assert "AssertionError: one is not two" in done.stdout


def test_an_import_error_prints_the_file_and_exits_non_zero(tmp_path):
    write(tmp_path, "tests/test_calc.py", "from calc import answer\n\n\ndef test_a():\n    pass\n")

    done = stub(tmp_path, "tests/test_calc.py")

    assert done.returncode != 0
    assert "FAILED tests/test_calc.py" in done.stdout
    assert "No module named 'calc'" in done.stdout


def test_modules_at_the_repo_root_are_importable(tmp_path):
    write(tmp_path, "calc.py", "def answer():\n    return 42\n")
    write(tmp_path, "tests/test_calc.py", "from calc import answer\n\n\ndef test_a():\n    assert answer() == 42\n")

    done = stub(tmp_path, "tests/test_calc.py")

    assert done.returncode == 0, done.stdout
    assert "1 passed" in done.stdout


def test_methods_of_test_classes_run(tmp_path):
    write(
        tmp_path,
        "tests/test_cls.py",
        "class TestCalc:\n    def test_ok(self):\n        pass\n\n"
        "    def test_bad(self):\n        assert False\n\n"
        "    def helper(self):\n        assert False\n",
    )

    done = stub(tmp_path, "tests/test_cls.py")

    assert done.returncode != 0
    assert "FAILED tests/test_cls.py::TestCalc::test_bad" in done.stdout
    assert "helper" not in done.stdout
    assert "1 failed, 1 passed" in done.stdout


def test_only_the_appended_files_run(tmp_path):
    write(tmp_path, "tests/test_ok.py", PASSING)
    write(tmp_path, "tests/test_bad.py", FAILING)

    done = stub(tmp_path, "tests/test_ok.py")

    assert done.returncode == 0, done.stdout
    assert "test_bad" not in done.stdout


def test_without_files_it_finds_the_test_files_itself(tmp_path):
    write(tmp_path, "tests/test_ok.py", PASSING)
    write(tmp_path, "tests/test_bad.py", FAILING)
    write(tmp_path, "tests/helpers.py", "raise RuntimeError('not a test file')\n")

    done = stub(tmp_path)

    assert done.returncode != 0
    assert "FAILED tests/test_bad.py::test_bad" in done.stdout
    assert "1 failed, 2 passed" in done.stdout


def test_no_tests_at_all_is_not_green(tmp_path):
    write(tmp_path, "tests/test_empty.py", "def helper():\n    pass\n")

    done = stub(tmp_path, "tests/test_empty.py")

    assert done.returncode != 0
    assert "no tests ran" in done.stdout
