"""The graph's routers: pure functions of state, no side effects.

``limit_hit`` is the routing target for every limit (agent model-call
limit, loop out of ``max_attempts``); the pause's ``retry`` routes back
through ``retry_node`` (#42).
"""

from langgraph.graph import END

DEFAULT_MAX_ATTEMPTS = 2


def route_after_tester(state: dict) -> str:
    """A TESTER run that hit its model-call limit never reaches the check."""
    if state.get("limit"):
        return "limit_hit"
    return "red_check"


def route_red_check(state: dict) -> str:
    """Red → CODER; still green → TESTER while attempts remain."""
    if state.get("tests_red"):
        return "coder"
    if _exhausted(state, "tester_attempts"):
        return "limit_hit"
    return "tester"


def route_after_coder(state: dict) -> str:
    if state.get("limit"):
        return "limit_hit"
    return "validator"


def route_validator(state: dict) -> str:
    """Green → the diff gate; red → CODER while attempts remain."""
    if state.get("validator_ok"):
        return "diff_gate"
    if _exhausted(state, "coder_attempts"):
        return "limit_hit"
    return "coder"


# --- /full (#44): spec gate, then tester → red check → test gate ---------------


def route_after_spec(state: dict) -> str:
    """A SPEC run that hit its model-call limit never reaches the gate."""
    if state.get("limit"):
        return "limit_hit"
    return "spec_gate"


def route_spec_gate(state: dict) -> str:
    """Redo → SPEC; approved → publish (create the issue for /full <text>)."""
    if not state.get("spec_approved"):
        return "spec"
    if state.get("item_ref"):
        return "post_brief"
    return "create_issue"


def route_full_red_check(state: dict) -> str:
    """Red → the human test gate; still green → TESTER while attempts remain."""
    if state.get("tests_red"):
        return "test_gate"
    if _exhausted(state, "tester_attempts"):
        return "limit_hit"
    return "tester"


def route_test_gate(state: dict) -> str:
    """Approve → commit the tests; redo → TESTER with the feedback."""
    return "commit_tests" if state.get("test_approved") else "tester"


def route_full_validator(state: dict) -> str:
    """Green → publish; red → CODER while attempts remain (same as /fast)."""
    if state.get("validator_ok"):
        return "publish"
    if _exhausted(state, "coder_attempts"):
        return "limit_hit"
    return "coder"


def route_after_reviewer(state: dict) -> str:
    """A REVIEWER run that hit its model-call limit never posts."""
    if state.get("limit"):
        return "limit_hit"
    return "post_review"


def route_review_gate(state: dict) -> str:
    """The gate's decision names the next node: ok → record + ready,
    redo → CODER with the review, spec → SPEC with the review."""
    return state.get("review_route") or END


def route_limit_hit(state: dict) -> str:
    """``stop`` ends the run; ``retry`` goes back to the limited node."""
    return state.get("retry_node") or END



def _exhausted(state: dict, key: str) -> bool:
    return state.get(key, 0) >= state.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
