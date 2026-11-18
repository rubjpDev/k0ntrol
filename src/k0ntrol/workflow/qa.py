"""The Q&A round machinery ``/plan``'s Q&A definition and ``/research
#<n>``'s Q&A decision tickets share (#120, #125).

A round is one fresh agent run's numbered questions — each with its
recommended answer — shown at the round gate; the human answers in one
line and the next run's fresh context carries every earlier question
and the line that answered it. The numbers the human answers by are
always ``1..n``, assigned in code whatever the model emitted.
"""

from collections.abc import Callable

from langgraph.types import interrupt

from k0ntrol.workflow.context import RunContext

DEFAULT_MAX_ROUND_QUESTIONS = 7


def make_round_gate(ctx: RunContext) -> Callable[[dict], dict]:
    """The round gate: the round's numbered questions, each with its
    recommended answer; the human's one line answers the whole round and
    joins the Q&A every later agent run sees. Free text, so the gate
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


def _answered_lines(round_: dict) -> list[str]:
    """One answered round as a fresh agent run reads it back: its
    questions, then the line the human answered them with."""
    lines = [
        f"  {question['number']}. {question['question']} "
        f"(recommended: {question['recommendation']})"
        for question in round_["questions"]
    ]
    return [*lines, f"  answer: {round_['answer']}"]
