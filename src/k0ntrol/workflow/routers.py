"""The graph's routers: pure functions of state, no side effects.

``limit_hit`` is the routing target for every limit (agent model-call
limit, loop out of ``max_attempts``); the pause's ``retry`` routes back
through ``retry_node`` (#42).
"""

from langgraph.graph import END

DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_MAX_QUESTIONS = 25


def route_review_gate(state: dict) -> str:
    """The gate's decision names the next node: ok → record + ready,
    redo → IMPLEMENTER with the reviews, plan → comment + unready.
    A stored ``ok`` the verdicts cannot back routes back to the gate
    itself, which re-shows without ``ok`` (#175)."""
    return state.get("review_route") or END


def route_mark_ready(state: dict) -> str:
    """Ready → END (the run is done); a failed readiness check re-shows
    the review gate — a checkpoint resumed straight into ``mark_ready``
    gets its verdicts re-checked in code (#175)."""
    return END if state.get("outcome") == "done" else "review_gate"


def route_limit_hit(state: dict) -> str:
    """``stop`` ends the run; ``retry`` goes back to the limited node."""
    return state.get("retry_node") or END


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


def route_after_branch_setup(state: dict) -> str:
    """A change already published under review re-enters at its reviews —
    build, validation and push are not rerun (#178)."""
    return "conventions_review" if state.get("change_ref") else "implementer"


def route_after_seam(state: dict) -> str:
    """A proposal run that hit its model-call limit never reaches the gate."""
    if state.get("limit"):
        return "limit_hit"
    return "seam_gate"


def route_seam_gate(state: dict) -> str:
    """Approve → IMPLEMENTER builds; redo → propose again with feedback."""
    return "implementer" if state.get("seam_approved") else "seam"


def route_after_implementer(state: dict) -> str:
    """A slice-guard escalation waits on the human's recovery decision;
    a limit goes to the pause; anything else runs VALIDATOR (#174)."""
    if state.get("recovery"):
        return "recovery_gate"
    if state.get("limit"):
        return "limit_hit"
    return "validator"


def route_recovery_gate(state: dict) -> str:
    """The gate's decision names the next node: recover → IMPLEMENTER
    retries, a refused journal asks again, stop → END (#174)."""
    return state.get("recovery_route") or END


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


def route_plan_start(state: dict) -> str:
    """START's first hop: a decided Map skips the Q&A straight to the
    spec draft (#126); every other idea opens a round."""
    return "spec" if state.get("decided_map") else "planner"


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
    breakdown (#122)."""
    return "tickets" if state.get("spec_approved") else "spec"


def route_after_tickets(state: dict) -> str:
    """A limited breakdown → the pause; a cut → its gate; anything else
    → PLANNER cuts again while attempts remain (#122)."""
    if state.get("limit"):
        return "limit_hit"
    return "ticket_gate" if state.get("tickets") else "tickets"


def route_ticket_gate(state: dict) -> str:
    """``redo`` → cut again with the feedback; ``ok`` → publish the
    spec and its tickets (#123)."""
    return "publish_spec" if state.get("tickets_approved") else "tickets"


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
    """A limited round → the pause; a round that would pass
    ``max_questions`` is never asked — the gate decides with what's
    known; questions → the round gate; a proposal or an empty round →
    the decision gate."""
    if state.get("limit"):
        return "limit_hit"
    questions = state.get("questions") or []
    if not questions:
        return "decision_gate"
    if _questions_asked(state) + len(questions) > _max_questions(state):
        return "decision_gate"
    return "round_gate"


# --- /triage (#128): the Q&A's rounds, the outcome gate, then apply ------------


def route_after_triage_planner(state: dict) -> str:
    """Same shape as ``/plan``'s rounds; an empty round — or one that
    would pass ``max_questions`` — drafts the outcome."""
    if state.get("limit"):
        return "limit_hit"
    questions = state.get("questions") or []
    if not questions:
        return "outcome"
    if _questions_asked(state) + len(questions) > _max_questions(state):
        return "outcome"
    return "round_gate"


def route_after_outcome(state: dict) -> str:
    """A limited draft → the pause; a proposal → its gate; anything else
    → PLANNER drafts again while attempts remain."""
    if state.get("limit"):
        return "limit_hit"
    return "outcome_gate" if state.get("triage_outcome") else "outcome"


def route_outcome_gate(state: dict) -> str:
    """``ok`` applies the labels and comment; ``redo`` drafts again."""
    return "apply" if state.get("triage_approved") else "outcome"
