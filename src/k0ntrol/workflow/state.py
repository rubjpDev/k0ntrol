"""The workflow graph's state.

Strict msgpack is on (``LANGGRAPH_STRICT_MSGPACK``), so state stays plain
types: strings, numbers, bools, lists, dicts and ``None``. Agent message
lists never enter the outer state — each agent run starts fresh from a
brief built out of these fields (ADR 0002).
"""

from typing import Any, TypedDict


class RunState(TypedDict, total=False):
    """One run's state; every field is a plain msgpack-safe value."""

    kind: str  # "fast" today; "full" in #44
    task: str  # the /fast task text
    start_ms: int  # epoch ms when the run started (run_end.ms)
    max_attempts: int  # agent runs allowed per loop (default 2)

    test_files: list[str]  # test paths TESTER wrote
    code_files: list[str]  # non-test paths CODER wrote
    tester_attempts: int  # TESTER agent runs so far (red-check loop)
    coder_attempts: int  # CODER agent runs so far (VALIDATOR loop)

    tests_red: bool  # last red check: test_cmd failed on the test files
    validator_ok: bool  # last VALIDATOR run: test_cmd + gates all green
    red_check_output: str  # last red-check output (TESTER retry context)
    validator_output: str  # last VALIDATOR raw output (CODER retry input)

    limit: dict[str, Any] | None  # set when the last node can hit a limit
    outcome: str  # "done" | "stuck" | "rejected", set at the run's end
    reason: str | None  # "model_call_limit" | "max_attempts" on stuck
    node: str | None  # the node a stuck run came from
