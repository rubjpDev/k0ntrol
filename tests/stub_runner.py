"""A stdlib test runner: the workflow fixtures' ``test_cmd`` (ADR 0009).

It stands in for ``python -m pytest -q`` where pytest is not the seam under
test: k0 reads only a command's exit code and output. It still runs as a
real subprocess over the real test files, from the repo root.

    python stub_runner.py [test files...]

With no files it runs ``tests/test_*.py``. It runs every ``test_*``
function and the ``test_*`` methods of ``Test*`` classes, prints
``FAILED <name>`` plus the error for each failure, then a summary
(``N passed`` or ``F failed, N passed``). It exits 1 on any failure or
import error, and 5 (as pytest does) when no tests ran.
"""

import importlib.util
import inspect
import sys
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

Test = tuple[str, Callable[[], object]]


def main(argv: list[str]) -> int:
    root = Path.cwd()
    sys.path[0] = str(root)  # import from the repo, not from this file's dir
    files = argv or sorted(str(p.relative_to(root)) for p in root.glob("tests/test_*.py"))
    passed, failed = 0, 0
    for path in files:
        file_passed, file_failed = run_file(path)
        passed, failed = passed + file_passed, failed + file_failed
    return summarize(passed, failed)


def run_file(path: str) -> tuple[int, int]:
    try:
        module = load(Path(path))
    except Exception:  # noqa: BLE001 - an import error is a red test file
        report(path)
        return 0, 1
    results = [run_one(name, call) for name, call in collect(path, module)]
    return results.count(True), results.count(False)


def load(path: Path) -> ModuleType:
    sys.path.insert(1, str(path.parent.resolve()))  # sibling imports, as pytest
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def collect(path: str, module: ModuleType) -> Iterator[Test]:
    for name, obj in list(vars(module).items()):
        if name.startswith("test_") and inspect.isfunction(obj):
            yield f"{path}::{name}", obj
        elif name.startswith("Test") and inspect.isclass(obj):
            yield from collect_class(f"{path}::{name}", obj)


def collect_class(prefix: str, cls: type) -> Iterator[Test]:
    for name, obj in list(vars(cls).items()):
        if name.startswith("test_") and inspect.isfunction(obj):
            yield f"{prefix}::{name}", getattr(cls(), name)


def run_one(name: str, call: Callable[[], object]) -> bool:
    try:
        call()
    except Exception:  # noqa: BLE001 - any error fails the test
        report(name)
        return False
    return True


def report(name: str) -> None:
    print(f"FAILED {name}")
    print(traceback.format_exc())


def summarize(passed: int, failed: int) -> int:
    if not passed and not failed:
        print("no tests ran")
        return 5
    print(f"{failed} failed, {passed} passed" if failed else f"{passed} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
