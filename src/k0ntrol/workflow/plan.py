"""The ``/plan`` graph: the Q&A definition's rounds (#120), the
question-limit gate (#127), then the spec gate (#121).

Each round is a fresh PLANNER run given the idea and the questions and
answers so far. It returns at most ``max_round_questions`` numbered
questions — each with its recommended answer — shown at the round gate;
the human answers in one line and the next round sees every earlier
question and answer. A round with no questions ends the Q&A: a fresh
PLANNER run then drafts the Spec — its own brief carrying every retained
round — and the spec gate takes ``ok`` or ``redo <feedback>``.

A round that would pass ``max_questions`` is never asked: the run pauses
at the question-limit gate with the open questions shown — ``end``
drafts the spec now (they land under Not yet decided), ``handoff``
charts them to a Map for ``/research``, ``continue`` grants ten more
questions. The ticket breakdown, its gate and publishing are later
tickets (#122, #123).
"""

from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from k0ntrol.agents import Spec
from k0ntrol.workflow.context import RunContext
from k0ntrol.workflow.nodes import (
    _attempt_limit,
    _invoke_counted,
    _last_output,
    make_limit_hit,
)
from k0ntrol.workflow.qa import (
    DEFAULT_MAX_ROUND_QUESTIONS,
    _answered_lines,
    _cap,
    _questions,
    _round_detail,
    make_round_gate,
)
from k0ntrol.workflow.research import (
    _BLANK,
    _bullets,
    _prose,
    make_publish_decisions,
    make_publish_map,
)
from k0ntrol.workflow.routers import (
    _exhausted,
    _max_questions,
    _questions_asked,
    route_after_plan_spec,
    route_after_planner,
    route_limit_hit,
    route_plan_spec_gate,
    route_question_limit,
)
from k0ntrol.workflow.state import RunState

# The questions a ``continue`` at the question-limit gate grants (#127).
_CONTINUE_QUESTIONS = 10


def build_plan_graph(
    ctx: RunContext, checkpointer: BaseCheckpointSaver | None
) -> CompiledStateGraph:
    """Compile the ``/plan`` graph for one run context."""
    graph = StateGraph(RunState)
    graph.add_node("planner", make_planner_node(ctx))
    graph.add_node("round_gate", make_round_gate(ctx))
    graph.add_node("question_limit", make_question_limit(ctx))
    graph.add_node("spec", make_spec_node(ctx))
    graph.add_node("spec_gate", make_spec_gate(ctx))
    graph.add_node("publish_map", make_publish_map(ctx))
    graph.add_node("publish_decisions", make_publish_decisions(ctx))
    graph.add_node("limit_hit", make_limit_hit(ctx))
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_after_planner)
    graph.add_edge("round_gate", "planner")
    graph.add_conditional_edges("question_limit", route_question_limit)
    graph.add_conditional_edges("spec", route_after_plan_spec)
    graph.add_conditional_edges("spec_gate", route_plan_spec_gate)
    graph.add_edge("publish_map", "publish_decisions")
    graph.add_edge("publish_decisions", END)
    graph.add_conditional_edges("limit_hit", route_limit_hit)
    return graph.compile(checkpointer=checkpointer)


def make_planner_node(ctx: RunContext) -> Callable[[dict], dict]:
    """One round of the Q&A: a fresh PLANNER run for the next questions.

    A limited run never reaches the gate and never uses up an attempt;
    a round with no questions ends the Q&A — the spec draft follows.
    """

    def node(state: dict) -> dict:
        result, updates = _invoke_counted(ctx, "planner", state, planner_brief)
        if result.limit_hit:
            return updates
        updates["questions"] = _questions(result, _cap(state))
        return updates

    return node


# --- the question-limit gate (#127) -----------------------------------------------


def make_question_limit(ctx: RunContext) -> Callable[[dict], dict]:
    """The question-limit gate: the round that would pass
    ``max_questions`` is never asked and stays in ``questions`` as the
    open questions. ``end`` → the spec draft, which lists them under
    Not yet decided; ``handoff`` → the Map they chart for ``/research``;
    ``continue`` → the pending round, with ten more questions granted.
    """

    def node(state: dict) -> dict:
        decision = interrupt(_limit_payload(ctx, state))
        verb = str(decision).strip().partition(" ")[0].lower()
        while verb == "handoff" and ctx.tracker is None:
            # The answered gate offered the hand off but this resume has
            # no tracker to publish the Map — ask again without it.
            decision = interrupt(_limit_payload(ctx, state))
            verb = str(decision).strip().partition(" ")[0].lower()
        if verb == "handoff":
            return {
                "question_limit_route": "publish_map",
                "map": _handoff_map(state),
            }
        if verb == "continue":
            return {
                "question_limit_route": "round_gate",
                "max_questions": (
                    _questions_asked(state) + _CONTINUE_QUESTIONS
                ),
            }
        return {"question_limit_route": "spec"}

    return node


def _limit_payload(ctx: RunContext, state: dict) -> dict:
    """The gate's interrupt payload: the open questions, and the choices
    the run can take — the hand off needs a tracker to publish the Map."""
    options = [("end", "end the plan as is"), ("continue", "continue for 10 more")]
    if ctx.tracker is not None:
        options.insert(1, ("handoff", "hand off to research"))
    return {
        "gate": "question_limit",
        "question": (
            f"the Q&A reached the question limit "
            f"({_questions_asked(state)}/{_max_questions(state)}) — "
            f"{_either([label for _, label in options])}?"
        ),
        "answers": [verb for verb, _ in options],
        "detail": _round_detail(state.get("questions", [])),
    }


def _either(options: list[str]) -> str:
    if len(options) == 2:
        return " or ".join(options)
    return f"{options[0]}, {options[1]} or {options[2]}"


def _handoff_map(state: dict) -> dict:
    """The Map the hand off publishes, built in code: one Q&A decision
    ticket per open question — the questions PLANNER still meant to ask
    the human (spec #102, story 30)."""
    idea = " ".join(_idea(state).split())
    source = state.get("item_title") or idea
    return {
        "title": f"Open questions: {source[:72]}",
        "destination": (
            "decide each open question so the plan's spec can be drafted"
        ),
        "notes": f"Charted at a /plan Q&A's question limit.\n\nIdea: {idea}",
        "decisions": [
            {"question": q["question"], "type": "Q&A", "blocked_by": []}
            for q in state.get("questions", [])
        ],
        "not_yet_specified": [],
        "out_of_scope": [],
    }


# --- the spec gate (#121) -------------------------------------------------------


def make_spec_node(ctx: RunContext) -> Callable[[dict], dict]:
    """The spec draft: a fresh PLANNER run given the idea and every
    answered round — ``spec_brief``, never the round's brief, so an early
    Spec output during the Q&A is re-drafted here (#121).

    A run that answers with another structured output drafts again while
    attempts remain, then the limit pause — the same resumable path a
    model-call limit takes (``limit_node="spec"`` names this node).
    """

    def node(state: dict) -> dict:
        result, updates = _invoke_counted(
            ctx,
            "planner",
            state,
            spec_brief,
            attempts_key="spec_attempts",
            limit_node="spec",
        )
        if result.limit_hit:
            return updates
        draft = result.structured_response
        if isinstance(draft, Spec):
            updates["spec"] = _with_open_questions(
                draft.model_dump(), state.get("questions", [])
            )
            return updates
        merged = {**state, **updates}
        if _exhausted(merged, "spec_attempts"):
            updates["limit"] = _attempt_limit(
                merged, "spec", "spec", _last_output(result)
            )
        return updates

    return node


def make_spec_gate(ctx: RunContext) -> Callable[[dict], dict]:
    """The spec gate: the draft in its format, ``ok`` or ``redo
    <feedback>``. ``ok`` moves on to the ticket breakdown (#122) — the
    run ends ``done`` until it lands, nothing published. ``redo``
    rejects the draft: the next PLANNER run gets the feedback."""

    def node(state: dict) -> dict:
        decision = interrupt(
            {
                "gate": "spec",
                "question": "approve this spec?",
                "answers": ["ok", "redo"],
                "detail": render_spec(state.get("spec")),
            }
        )
        word, _, feedback = str(decision).strip().partition(" ")
        if word.lower() == "ok":
            return {
                "spec_approved": True,
                "spec_feedback": None,
                "outcome": "done",
            }
        return {
            "spec": None,
            "spec_approved": False,
            "spec_attempts": 0,
            "spec_feedback": feedback.strip(),
            "spec_rejections": [
                *state.get("spec_rejections", []), feedback.strip()
            ],
        }

    return node


def _with_open_questions(spec: dict, questions: list[dict]) -> dict:
    """Fold the questions the limit left unasked into Not yet decided —
    in code, so ending as is never silently drops one (#127)."""
    if not questions:
        return spec
    undecided = list(spec.get("not_yet_decided", []))
    undecided += [
        q["question"] for q in questions if q["question"] not in undecided
    ]
    return {**spec, "not_yet_decided": undecided}


def render_spec(spec: dict | None) -> str:
    """The draft as the gate shows it — every section of the spec format
    (#102); the same rendering becomes the published body in #123."""
    if not spec:
        return "(PLANNER produced no spec)"
    return "\n\n".join(
        [
            f"**Title:** {spec.get('title', '')}",
            _prose("Problem Statement", spec.get("problem_statement", "")),
            _prose("Solution", spec.get("solution", "")),
            _bullets("User Stories", spec.get("user_stories", [])),
            _bullets(
                "Implementation Decisions", spec.get("implementation_decisions", [])
            ),
            _testing(spec),
            _bullets("Glossary and decisions", spec.get("glossary_and_decisions", [])),
            _bullets("Not yet decided", spec.get("not_yet_decided", [])),
            _bullets("Out of Scope", spec.get("out_of_scope", [])),
        ]
    )


def _testing(spec: dict) -> str:
    """Testing Decisions with the seams listed under them."""
    decisions = "\n".join(f"- {item}" for item in spec.get("testing_decisions", []))
    seams = "\n".join(f"- {item}" for item in spec.get("seams", []))
    return f"## Testing Decisions\n\n{decisions or _BLANK}\n\nSeams:\n{seams or _BLANK}"


def planner_brief(state: dict) -> str:
    """PLANNER's fresh context for one round: the idea, then every earlier
    round's questions and the line the human answered."""
    lines = [
        f"Idea: {_idea(state)}",
        "",
        f"Ask the next round of questions — at most {_cap(state)}, each "
        "numbered with your recommended answer — or an empty QuestionRound "
        "when nothing is left to decide.",
    ]
    for index, round_ in enumerate(state.get("rounds", []), 1):
        lines.append(f"\nRound {index}:")
        lines.extend(_answered_lines(round_))
    return "\n".join(lines)


def spec_brief(state: dict) -> str:
    """PLANNER's fresh context for the spec draft: the idea and every
    answered round, and on a gate ``redo`` the human's feedback."""
    lines = [
        f"Idea: {_idea(state)}",
        "",
        "The Q&A is over — draft the spec: answer with Spec — the title, "
        "problem statement, solution, user stories, implementation "
        "decisions, testing decisions with the seams, glossary and "
        "decisions, not yet decided, and out of scope.",
    ]
    for index, round_ in enumerate(state.get("rounds", []), 1):
        lines.append(f"\nRound {index}:")
        lines.extend(_answered_lines(round_))
    open_ = state.get("questions") or []
    if open_:
        lines.append(
            "\nThe question limit left these questions unasked — they "
            "are not yet decided:"
        )
        lines.extend(f"  - {question['question']}" for question in open_)
    if state.get("spec_feedback"):
        lines.append(
            "\nYour last draft was rejected with this feedback: "
            f"{state['spec_feedback']}"
        )
    return "\n".join(lines)


def _idea(state: dict) -> str:
    """The idea the Q&A defined: the task text, prefixed by its work
    item when ``/plan #<n>`` carries one."""
    idea = state["task"]
    if state.get("item_ref"):
        idea = f"#{state['item_ref']} {state.get('item_title', '')}\n\n{idea}"
    return idea
