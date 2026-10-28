"""The ``/fast`` workflow graph.

TESTER → red check → CODER → VALIDATOR → diff gate, on the working
tree: no tracker, branch, PR, SPEC or REVIEWER. TESTER and CODER come
from the one agent builder with ``/fast`` allowlists (no tracker reads);
the red check and VALIDATOR are deterministic nodes over the shared
runner, and the diff gate is a single ``interrupt()`` node.
"""

from langgraph.graph import END, START, StateGraph

from k0ntrol.workflow.context import RunContext
from k0ntrol.workflow.nodes import (
    limit_hit,
    make_agent_node,
    make_diff_gate,
    make_red_check,
    make_validator,
)
from k0ntrol.workflow.routers import (
    route_after_coder,
    route_after_tester,
    route_red_check,
    route_validator,
)
from k0ntrol.workflow.state import RunState


def build_fast_graph(ctx: RunContext, checkpointer):
    """Compile the ``/fast`` graph for one run context."""
    graph = StateGraph(RunState)
    graph.add_node("tester", make_agent_node(ctx, "tester", "test_files", tester_brief))
    graph.add_node("red_check", make_red_check(ctx))
    graph.add_node("coder", make_agent_node(ctx, "coder", "code_files", coder_brief))
    graph.add_node("validator", make_validator(ctx))
    graph.add_node("diff_gate", make_diff_gate(ctx))
    graph.add_node("limit_hit", limit_hit)
    graph.add_edge(START, "tester")
    graph.add_conditional_edges("tester", route_after_tester)
    graph.add_conditional_edges("red_check", route_red_check)
    graph.add_conditional_edges("coder", route_after_coder)
    graph.add_conditional_edges("validator", route_validator)
    graph.add_edge("diff_gate", END)
    graph.add_edge("limit_hit", END)
    return graph.compile(checkpointer=checkpointer)


def tester_brief(state: dict) -> str:
    """TESTER's fresh context: the task, plus the red-check failure on retry."""
    brief = (
        f"Task: {state['task']}\n\n"
        "Write tests that prove this task done — and that FAIL right now, "
        "before the change exists. Write test files only."
    )
    if state.get("tester_attempts"):
        brief += (
            "\n\nThe red check ran the suite on your tests and it PASSED — "
            "tests that pass without the change prove nothing. "
            f"Output:\n{state.get('red_check_output', '')}\n"
            "Revise the tests so they fail."
        )
    return brief


def coder_brief(state: dict) -> str:
    """CODER's fresh context: task, test files, VALIDATOR output on retry."""
    files = ", ".join(state.get("test_files", [])) or "(none)"
    brief = (
        f"Task: {state['task']}\n\n"
        f"The tests defining the contract live in: {files}. "
        "Implement the task so they pass. Write non-test files only."
    )
    if state.get("coder_attempts"):
        brief += (
            "\n\nVALIDATOR is still red. Its raw output:\n"
            f"{state.get('validator_output', '')}\nFix it."
        )
    return brief
