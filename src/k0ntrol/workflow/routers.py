"""The graph's routers: pure functions of state, no side effects.

``limit_hit`` is the routing target for every limit (agent model-call
limit, loop out of ``max_attempts``); the pause's ``retry`` routes back
through ``retry_node`` (#42).
"""

from langgraph.graph import END

DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_MAX_QUESTIONS = 25


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


def route_ci_gate(state: dict) -> str:
    """Cleared → push (with a gate comment when the human approved);
    rejected → record it, then CODER with the feedback."""
    return _route_ci(state, prefix="", skipped="coder")


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


def route_full_limit_hit(state: dict) -> str:
    """``/full``: ``retry`` reruns the node; ``stop`` runs the stuck effects."""
    if state.get("retry_node"):
        return state["retry_node"]
    return "stuck_commit" if state.get("branch") else "stuck_label"


def route_stuck_ci_gate(state: dict) -> str:
    """The stuck path's CI gate: no CODER, so a reject (or a gate nobody
    decided) just leaves HEAD unpushed and goes on to the label."""
    return _route_ci(state, prefix="stuck_", skipped="stuck_label")


def _route_ci(state: dict, *, prefix: str, skipped: str) -> str:
    """The CI gate's outcomes for one path; ``prefix`` names its nodes.

    ``skipped`` is where HEAD goes uncleared with nobody having decided
    (a run stuck by its event log never asks).
    """
    if state.get("ci_cleared_sha"):
        return f"{prefix}ci_comment" if state.get("ci_approved") else f"{prefix}push"
    if state.get("ci_feedback") is None:
        return skipped
    return f"{prefix}ci_reject_comment"


# --- /implement (#116, #117): the working-tree and tracker paths --------------


def route_implement_start(state: dict) -> str:
    """``/implement #<n>`` takes the tracker path; ``<task>`` the seam gate."""
    return "branch_setup" if state.get("item_ref") else "seam"


def route_after_seam(state: dict) -> str:
    """A proposal run that hit its model-call limit never reaches the gate."""
    if state.get("limit"):
        return "limit_hit"
    return "seam_gate"


def route_seam_gate(state: dict) -> str:
    """Approve → IMPLEMENTER builds; redo → propose again with feedback."""
    return "implementer" if state.get("seam_approved") else "seam"


def route_after_implementer(state: dict) -> str:
    if state.get("limit"):
        return "limit_hit"
    return "validator"


def route_implement_validator(state: dict) -> str:
    """Green → publish on the tracker path, the diff gate on the tree."""
    if state.get("validator_ok"):
        return "publish" if state.get("item_ref") else "diff_gate"
    if _exhausted(state, "implementer_attempts"):
        return "limit_hit"
    return "implementer"


def route_implement_ci_gate(state: dict) -> str:
    """Cleared → push (recording an approval); rejected → IMPLEMENTER."""
    return _route_ci(state, prefix="", skipped="implementer")


def route_after_conventions_review(state: dict) -> str:
    """A limited REVIEWER run never posts; else the Conventions review posts."""
    if state.get("limit"):
        return "limit_hit"
    return "post_conventions"


def route_after_contract_review(state: dict) -> str:
    """A limited REVIEWER run never posts; else the Contract review posts."""
    if state.get("limit"):
        return "limit_hit"
    return "post_contract"


def _exhausted(state: dict, key: str) -> bool:
    return state.get(key, 0) >= state.get("max_attempts", DEFAULT_MAX_ATTEMPTS)


# --- /plan (#120, #121): the Q&A's rounds, then the spec gate ---------------------


def route_after_planner(state: dict) -> str:
    """A limited round → the pause; a round with no questions ends the
    Q&A and drafts the spec (#121); a round that would pass
    ``max_questions`` stops at the question-limit gate (#127); else the
    round gate asks it."""
    if state.get("limit"):
        return "limit_hit"
    questions = state.get("questions") or []
    if not questions:
        return "spec"
    if _questions_asked(state) + len(questions) > _max_questions(state):
        return "question_limit"
    return "round_gate"


def route_question_limit(state: dict) -> str:
    """The limit gate's decision names the next node: end → the spec
    draft, handoff → publish the Map, continue → the pending round."""
    return state.get("question_limit_route") or "spec"


def _questions_asked(state: dict) -> int:
    """Questions shown at round gates so far — the pending round is
    unasked and never counts."""
    return sum(len(round_.get("questions", [])) for round_ in state.get("rounds", []))


def _max_questions(state: dict) -> int:
    return int(state.get("max_questions", DEFAULT_MAX_QUESTIONS))


def route_after_plan_spec(state: dict) -> str:
    """A limited draft → the pause; a drafted spec → its gate; anything
    else → PLANNER drafts again while attempts remain."""
    if state.get("limit"):
        return "limit_hit"
    return "spec_gate" if state.get("spec") else "spec"


def route_plan_spec_gate(state: dict) -> str:
    """``redo`` → draft again with the feedback; ``ok`` → the ticket
    breakdown (#122) — END until it lands, the run already ``done``."""
    return END if state.get("spec_approved") else "spec"


# --- /research (#124): chart the map, then publish ---------------------------------


def route_research_start(state: dict) -> str:
    """``/research #<n>`` works the map's next decision; ``<question>`` charts."""
    return "pick_decision" if state.get("item_ref") else "planner"


def route_after_chart(state: dict) -> str:
    """A limited or attempts-exhausted chart → the pause; a charted map →
    publish; anything else → PLANNER charts again."""
    if state.get("limit"):
        return "limit_hit"
    return "publish_map" if state.get("map") else "planner"


# --- /research #<n> (#125): work the map's next decision ticket --------------------

#: A decision ticket's type names the node that answers it; anything else
#: (``human``, a child with no ``## Type``) goes straight to the human.
_ANSWERERS = {"research": "researcher", "Q&A": "decision_planner"}


def route_after_pick(state: dict) -> str:
    """Nothing workable → the run ends done; else the type's answerer."""
    decision = state.get("decision")
    if not decision:
        return END
    return _ANSWERERS.get(decision["type"], "decision_gate")


def route_after_researcher(state: dict) -> str:
    """A limited RESEARCHER run never comments nor reaches the gate."""
    if state.get("limit"):
        return "limit_hit"
    return "comment_findings"


def route_after_decision_planner(state: dict) -> str:
    """A limited round → the pause; questions → the round gate; a
    proposal or an empty round → the decision gate."""
    if state.get("limit"):
        return "limit_hit"
    return "round_gate" if state.get("questions") else "decision_gate"


# --- /triage (#128): the Q&A's rounds, the outcome gate, then apply ------------


def route_after_triage_planner(state: dict) -> str:
    """Same shape as ``/plan``'s rounds; an empty round drafts the outcome."""
    if state.get("limit"):
        return "limit_hit"
    return "round_gate" if state.get("questions") else "outcome"


def route_after_outcome(state: dict) -> str:
    """A limited draft → the pause; a proposal → its gate; anything else
    → PLANNER drafts again while attempts remain."""
    if state.get("limit"):
        return "limit_hit"
    return "outcome_gate" if state.get("triage_outcome") else "outcome"


def route_outcome_gate(state: dict) -> str:
    """``ok`` applies the labels and comment; ``redo`` drafts again."""
    return "apply" if state.get("triage_approved") else "outcome"
