"""The ``run_tests`` tool: TESTER and CODER's only command (ADR 0001).

It runs the configured ``test_cmd`` as an argv list, optionally narrowed to
existing test paths, through the same scrubbed/timed-out/truncated runner
the red check and VALIDATOR use. There is no shell tool.
"""

from pathlib import Path

from langchain_core.tools import BaseTool, tool

from k0ntrol.tools.errors import ToolError
from k0ntrol.tools.path_guard import guard_path, is_test_path
from k0ntrol.tools.runner import CommandResult, run_command

DEFAULT_TEST_TIMEOUT = 600


def make_run_tests(root: Path, config: dict) -> BaseTool:
    """Build the ``run_tests`` tool bound to ``root`` and the project config."""
    test_cmd = config["test_cmd"]
    timeout = config.get("test_timeout", DEFAULT_TEST_TIMEOUT)

    @tool
    def run_tests(paths: list[str] | None = None) -> str:
        """Run the project's test command. `paths` may narrow it to test files."""
        narrowed = _existing_test_paths(paths or [], root)
        result = run_command(test_cmd, root=root, timeout=timeout, extra_args=narrowed)
        return _format(result)

    return run_tests


def _existing_test_paths(paths: list[str], root: Path) -> list[str]:
    """The given paths that exist, sit inside the root and are test paths."""
    return [relative for path in paths if (relative := _test_path(path, root))]


def _test_path(path: str, root: Path) -> str | None:
    try:
        resolved = guard_path(path, root)
    except ToolError:
        return None
    relative = resolved.relative_to(root.resolve()).as_posix()
    if not resolved.is_file() or not is_test_path(relative):
        return None
    return relative


def _format(result: CommandResult) -> str:
    status = "timed out" if result.timed_out else f"exit {result.returncode}"
    return f"$ {' '.join(result.argv)}\n{status}\n{result.output}".rstrip()
