"""The ``/plan`` graph: the Q&A definition's rounds (#120), then the
spec gate (#121).

Each round is a fresh PLANNER run given the idea and the questions and
answers so far. It returns at most ``max_round_questions`` numbered
questions — each with its recommended answer — shown at the round gate;
the human answers in one line and the next round sees every earlier
question and answer. A round with no questions ends the Q&A: a fresh
PLANNER run then drafts the Spec — its own brief carrying every retained
round — and the spec gate takes ``ok`` or ``redo <feedback>``. The ticket
breakdown, its gate and publishing are later tickets (#122, #123).
"""

from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
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
from k0ntrol.workflow.research import _BLANK, _bullets, _prose
from k0ntrol.workflow.routers import (
    _exhausted,
    route_after_plan_spec,
    route_after_planner,
    route_limit_hit,
    route_plan_spec_gate,
)
from k0ntrol.workflow.state import RunState

DEFAULT_MAX_ROUND_QUESTIONS = 7


def build_plan_graph(
    ctx: RunContext, checkpointer: BaseCheckpointSaver | None
) -> CompiledStateGraph:
    """Compile the ``/plan`` graph for one run context."""
    graph = StateGraph(RunState)
    graph.add_node("planner", make_planner_node(ctx))
    graph.add_node("round_gate", make_round_gate(ctx))
    graph.add_node("spec", make_spec_node(ctx))
    graph.add_node("spec_gate", make_spec_gate(ctx))
    graph.add_node("limit_hit", make_limit_hit(ctx))
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_after_planner)
    graph.add_edge("round_gate", "planner")
    graph.add_conditional_edges("spec", route_after_plan_spec)
    graph.add_conditional_edges("spec_gate", route_plan_spec_gate)
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


def make_round_gate(ctx: RunContext) -> Callable[[dict], dict]:
    """The round gate: the round's numbered questions, each with its
    recommended answer; the human's one line answers the whole round and
    joins the Q&A every later PLANNER run sees. Free text, so the gate
    takes any line — ``_check_decision`` only polices listed answers.
    """

    def node(state: dict) -> dict:
        decision = interrupt(
            {
                "gate": "round",
                "question": "answer the round's questions in one line",
                "answers": [],
                "detail": _round_detail(state.get("questions", [])),
            }
        )
        answered = {
            "questions": state.get("questions", []),
            "answer": str(decision).strip(),
        }
        return {
            "rounds": [*state.get("rounds", []), answered],
            "questions": [],
        }

    return node


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
            updates["spec"] = draft.model_dump()
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


def _questions(result, cap: int) -> list[dict]:
    """The round's questions as plain dicts, capped in code, never in the
    prompt alone — and numbered in code too, so the numbers the human
    answers by are always ``1..n`` whatever the model emitted.

    PLANNER's other outputs (Spec, TicketList) carry no questions:
    answering early ends the Q&A like an empty round.
    """
    questions = getattr(result.structured_response, "questions", None) or []
    return [
        {**question.model_dump(), "number": index}
        for index, question in enumerate(questions[:cap], 1)
    ]


def _cap(state: dict) -> int:
    return int(state.get("max_round_questions", DEFAULT_MAX_ROUND_QUESTIONS))


def _round_detail(questions: list[dict]) -> str:
    """What the gate shows: ``n. question`` over ``recommended: answer
    (why)``, one pair of lines per question."""
    lines = []
    for question in questions:
        lines.append(f"{question['number']}. {question['question']}")
        recommended = f"   recommended: {question['recommendation']}"
        if question.get("rationale"):
            recommended += f" ({question['rationale']})"
        lines.append(recommended)
    return "\n".join(lines)


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


def _answered_lines(round_: dict) -> list[str]:
    """One answered round as PLANNER reads it back: its questions, then the
    line the human answered them with."""
    lines = [
        f"  {question['number']}. {question['question']} "
        f"(recommended: {question['recommendation']})"
        for question in round_["questions"]
    ]
    return [*lines, f"  answer: {round_['answer']}"]
