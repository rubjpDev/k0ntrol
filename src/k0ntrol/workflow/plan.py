"""The ``/plan`` graph: the Q&A definition's rounds (#120).

Each round is a fresh PLANNER run given the idea and the questions and
answers so far. It returns at most ``max_round_questions`` numbered
questions — each with its recommended answer — shown at the round gate;
the human answers in one line and the next round sees every earlier
question and answer. A round with no questions ends the Q&A. The spec
draft, its gate, the ticket gate and publishing are later tickets
(#121, #122, #123).
"""

from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from k0ntrol.workflow.context import RunContext
from k0ntrol.workflow.nodes import _invoke_counted, make_limit_hit
from k0ntrol.workflow.routers import route_after_planner, route_limit_hit
from k0ntrol.workflow.state import RunState

DEFAULT_MAX_ROUND_QUESTIONS = 7


def build_plan_graph(
    ctx: RunContext, checkpointer: BaseCheckpointSaver | None
) -> CompiledStateGraph:
    """Compile the ``/plan`` graph for one run context."""
    graph = StateGraph(RunState)
    graph.add_node("planner", make_planner_node(ctx))
    graph.add_node("round_gate", make_round_gate(ctx))
    graph.add_node("limit_hit", make_limit_hit(ctx))
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_after_planner)
    graph.add_edge("round_gate", "planner")
    graph.add_conditional_edges("limit_hit", route_limit_hit)
    return graph.compile(checkpointer=checkpointer)


def make_planner_node(ctx: RunContext) -> Callable[[dict], dict]:
    """One round of the Q&A: a fresh PLANNER run for the next questions.

    A limited run never reaches the gate and never uses up an attempt;
    a round with no questions ends the definition ``done``.
    """

    def node(state: dict) -> dict:
        result, updates = _invoke_counted(ctx, "planner", state, planner_brief)
        if result.limit_hit:
            return updates
        questions = _questions(result, _cap(state))
        updates["questions"] = questions
        if not questions:
            updates["outcome"] = "done"
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


def _questions(result, cap: int) -> list[dict]:
    """The round's questions as plain dicts, capped in code, never in the
    prompt alone — and numbered in code too, so the numbers the human
    answers by are always ``1..n`` whatever the model emitted.

    PLANNER's other outputs (Spec, TicketList) carry no questions:
    answering early ends the definition like an empty round.
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
    idea = state["task"]
    if state.get("item_ref"):
        idea = f"#{state['item_ref']} {state.get('item_title', '')}\n\n{idea}"
    lines = [
        f"Idea: {idea}",
        "",
        f"Ask the next round of questions — at most {_cap(state)}, each "
        "numbered with your recommended answer — or an empty QuestionRound "
        "when nothing is left to decide.",
    ]
    for index, round_ in enumerate(state.get("rounds", []), 1):
        lines.append(f"\nRound {index}:")
        lines.extend(_answered_lines(round_))
    return "\n".join(lines)


def _answered_lines(round_: dict) -> list[str]:
    """One answered round as PLANNER reads it back: its questions, then the
    line the human answered them with."""
    lines = [
        f"  {question['number']}. {question['question']} "
        f"(recommended: {question['recommendation']})"
        for question in round_["questions"]
    ]
    return [*lines, f"  answer: {round_['answer']}"]
