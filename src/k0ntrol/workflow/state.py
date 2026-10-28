"""The workflow graph's state.

Strict msgpack is on (``LANGGRAPH_STRICT_MSGPACK``), so state stays plain
types: strings, numbers, bools, lists, dicts and ``None``. Agent message
lists never enter the outer state — each agent run starts fresh from a
brief built out of these fields (ADR 0002).
"""

from typing import Any, TypedDict


class RunState(TypedDict, total=False):
    """One run's state; every field is a plain msgpack-safe value."""

    kind: str  # "fast" | "full"
    task: str  # the /fast or /full task text (issue body for /full #<n>)
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

    # /full (#44): the work item the run carries, its branch and gates.
    item_ref: str | None  # the issue number (None until the spec gate passes)
    item_title: str
    branch: str  # k0/<n>-<slug>
    brief: dict | None  # SPEC's structured AgentBrief as a plain dict
    spec_attempts: int  # SPEC agent runs so far
    spec_approved: bool  # the human ok'd the brief
    spec_rejections: list[str]  # earlier spec-gate feedback, for the comment
    spec_feedback: str | None  # the feedback of the last spec-gate reject
    test_approved: bool  # the human ok'd the tests
    test_rejections: list[str]  # earlier test-gate feedback
    test_feedback: str | None  # the feedback of the last test-gate reject

    limit: dict[str, Any] | None  # set when the last node can hit a limit
    retry_node: str | None  # the node a limit_hit retry sends back to
    retry_context: str | None  # limit note + output tail + human feedback
    outcome: str  # "done" | "stuck" | "rejected", set at the run's end
    reason: str | None  # "model_call_limit" | "max_attempts" on stuck
    node: str | None  # the node a stuck run came from
