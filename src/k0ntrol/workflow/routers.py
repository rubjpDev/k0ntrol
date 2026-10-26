"""The graph's routers: pure functions of state, no side effects.

``limit_hit`` is already a routing target for every limit (agent
model-call limit, loop out of ``max_attempts``); #42 makes the node a
pause without touching these routers.
"""

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


def _exhausted(state: dict, key: str) -> bool:
    return state.get(key, 0) >= state.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
