"""The ``/plan`` graph: the Q&A definition's rounds (#120), the
question-limit gate (#127), the spec gate (#121), then the ticket
gate (#122).

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
questions.

An approved spec is cut into tickets by one more fresh PLANNER run,
sized by ``ticket_size``. The harness prepends the first ticket in
code — it records the spec's glossary and ADR changes and blocks every
other ticket (PLANNER has no write tools) — and the ticket gate lists
each ticket's title, blockers, deliverable and criteria count, flagging
any over its size's count; it takes ``ok`` or ``redo <feedback>``.
Publishing the spec and its tickets is #123.
"""

from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from k0ntrol.agents import Spec, TicketList
from k0ntrol.workflow.context import RunContext
from k0ntrol.workflow.nodes import (
    _attempt_limit,
    _invoke_counted,
    _last_output,
    make_limit_hit,
)
from k0ntrol.workflow.qa import (
    DEFAULT_MAX_ROUND_QUESTIONS,
    _answered_rounds,
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
    route_after_tickets,
    route_limit_hit,
    route_plan_spec_gate,
    route_plan_start,
    route_question_limit,
    route_ticket_gate,
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
    graph.add_node("tickets", make_tickets_node(ctx))
    graph.add_node("ticket_gate", make_ticket_gate(ctx))
    graph.add_node("publish_map", make_publish_map(ctx))
    graph.add_node("publish_decisions", make_publish_decisions(ctx))
    graph.add_node("publish_spec", make_publish_spec(ctx))
    graph.add_node("publish_tickets", make_publish_tickets(ctx))
    graph.add_node("limit_hit", make_limit_hit(ctx))
    graph.add_conditional_edges(START, route_plan_start)
    graph.add_conditional_edges("planner", route_after_planner)
    graph.add_edge("round_gate", "planner")
    graph.add_conditional_edges("question_limit", route_question_limit)
    graph.add_conditional_edges("spec", route_after_plan_spec)
    graph.add_conditional_edges("spec_gate", route_plan_spec_gate)
    graph.add_conditional_edges("tickets", route_after_tickets)
    graph.add_conditional_edges("ticket_gate", route_ticket_gate)
    graph.add_edge("publish_map", "publish_decisions")
    graph.add_edge("publish_decisions", END)
    graph.add_edge("publish_spec", "publish_tickets")
    graph.add_edge("publish_tickets", END)
    graph.add_conditional_edges("limit_hit", route_limit_hit)
    return graph.compile(checkpointer=checkpointer)


def make_planner_node(
    ctx: RunContext, brief: Callable[[dict], str] | None = None
) -> Callable[[dict], dict]:
    """One round of the Q&A: a fresh PLANNER run for the next questions.

    ``brief`` is the run kind's round brief — ``planner_brief`` for
    ``/plan``; ``/triage`` passes its own (#128). A limited run never
    reaches the gate and never uses up an attempt; a round with no
    questions ends the Q&A — the spec draft follows.
    """

    def node(state: dict) -> dict:
        result, updates = _invoke_counted(
            ctx, "planner", state, brief or planner_brief
        )
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
        verb = _limit_decision(ctx, state)
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


def _limit_decision(ctx: RunContext, state: dict) -> str:
    """The gate's verb, re-asking when a trackerless resume cannot publish
    the Map a ``handoff`` answer would need."""
    decision = interrupt(_limit_payload(ctx, state))
    verb = str(decision).strip().partition(" ")[0].lower()
    while verb == "handoff" and ctx.tracker is None:
        decision = interrupt(_limit_payload(ctx, state))
        verb = str(decision).strip().partition(" ")[0].lower()
    return verb


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
            f"{_either_line([label for _, label in options])}?"
        ),
        "answers": [verb for verb, _ in options],
        "detail": _round_detail(state.get("questions", [])),
    }


def _either_line(options: list[str]) -> str:
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
            updates["spec"] = _drafted_spec(draft, state)
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
    <feedback>``. ``ok`` moves on to the ticket breakdown (#122).
    ``redo`` rejects the draft: the next PLANNER run gets the
    feedback."""

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


def _drafted_spec(draft: Spec, state: dict) -> dict:
    """The draft as a state dict: open questions folded in, and on a
    decided Map the map's ref, so the spec it publishes links the map
    it came from (#126)."""
    spec = _with_open_questions(
        draft.model_dump(), state.get("questions", [])
    )
    if state.get("decided_map"):
        spec["map_ref"] = state["item_ref"]
    return spec


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
    sections = [f"**Title:** {spec.get('title', '')}"]
    if spec.get("map_ref"):
        sections.append(f"**Map:** #{spec['map_ref']}")
    return "\n\n".join(
        [
            *sections,
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


# --- the ticket gate (#122) -------------------------------------------------------


#: A ``ticket_size``'s acceptance-criteria cap — small also means one
#: seam; the gate flags a ticket over its size's count (#122).
_TICKET_SIZE_CRITERIA = {"small": 3, "medium": 5, "large": 8}
_TICKET_SIZE_RULE = {
    "small": "one seam and at most 3 acceptance criteria",
    "medium": "at most 5 acceptance criteria",
    "large": "at most 8 acceptance criteria",
}


def make_tickets_node(ctx: RunContext) -> Callable[[dict], dict]:
    """The ticket breakdown: a fresh PLANNER run cuts the approved spec,
    sized by ``ticket_size``. The first ticket is prepended in code —
    the glossary-and-ADR ticket, blocking every other.

    A run that answers with another structured output cuts again while
    attempts remain, then the limit pause — the same resumable path the
    spec draft takes (``limit_node="tickets"`` names this node).
    """

    def node(state: dict) -> dict:
        result, updates = _invoke_counted(
            ctx,
            "planner",
            state,
            tickets_brief,
            attempts_key="tickets_attempts",
            limit_node="tickets",
        )
        if result.limit_hit:
            return updates
        cut = result.structured_response
        if isinstance(cut, TicketList):
            updates["tickets"] = _with_glossary_ticket(
                cut, state.get("spec") or {}
            )
            return updates
        return _uncut(result, updates, state)

    return node


def _uncut(result: object, updates: dict, state: dict) -> dict:
    """No TicketList this run: cut again while attempts remain, then
    the attempts pause."""
    merged = {**state, **updates}
    if _exhausted(merged, "tickets_attempts"):
        updates["limit"] = _attempt_limit(
            merged, "tickets", "tickets", _last_output(result)
        )
    return updates


def _with_glossary_ticket(cut: TicketList, spec: dict) -> list[dict]:
    """PLANNER's tickets after the code-made first ticket: its indexes
    shift one and the glossary ticket blocks each of them — blockers
    still only point backwards (#122)."""
    tickets = [ticket.model_dump() for ticket in cut.tickets]
    for ticket in tickets:
        ticket["blocked_by"] = sorted(
            {0, *(blocker + 1 for blocker in ticket["blocked_by"])}
        )
    return [_glossary_ticket(spec), *tickets]


def _glossary_ticket(spec: dict) -> dict:
    """The spec's first ticket, made in code: record the plan's glossary
    and ADR changes (PLANNER has no write tools) (#122)."""
    entries = spec.get("glossary_and_decisions") or []
    lines = "\n".join(f"- {entry}" for entry in entries) or "- none"
    return {
        "title": "Record the spec's glossary and ADR changes",
        "what_to_build": (
            "record the plan's glossary entries in GLOSSARY.md and its "
            "decisions as ADRs under docs/adr/:\n" + lines
        ),
        "seams": ["GLOSSARY.md", "docs/adr/"],
        "acceptance_criteria": [
            "GLOSSARY.md and the ADRs record the spec's glossary and "
            "decisions"
        ],
        "blocked_by": [],
    }


def make_ticket_gate(ctx: RunContext) -> Callable[[dict], dict]:
    """The ticket gate: every ticket's title, blockers, deliverable and
    criteria count — a ticket over its ``ticket_size`` count flagged —
    then ``ok`` or ``redo <feedback>``. ``ok`` publishes the spec and
    its tickets (#123); ``redo`` recuts with the feedback."""

    def node(state: dict) -> dict:
        decision = interrupt(
            {
                "gate": "ticket",
                "question": "approve these tickets?",
                "answers": ["ok", "redo"],
                "detail": render_tickets(
                    state.get("tickets") or [],
                    state.get("ticket_size", "small"),
                ),
            }
        )
        word, _, feedback = str(decision).strip().partition(" ")
        if word.lower() == "ok":
            return {
                "tickets_approved": True,
                "ticket_feedback": None,
            }
        return {
            "tickets": None,
            "tickets_approved": False,
            "tickets_attempts": 0,
            "ticket_feedback": feedback.strip(),
            "ticket_rejections": [
                *state.get("ticket_rejections", []), feedback.strip()
            ],
        }

    return node


# --- publishing (#123) ---------------------------------------------------------------

READY_LABEL = "ready-for-agent"


def make_publish_spec(ctx: RunContext) -> Callable[[dict], dict]:
    """The Spec work item, marker-idempotent (carrier contract, #13):
    ``/plan <idea>`` opens it; ``/plan #<n>`` on a plain item updates
    its plan — the spec posted on it — instead. A decided Map is never
    the spec's parent: its children are decision tickets, so the spec
    gets its own item that links the map (#126, #123). A trackerless
    run has nothing to publish on and ends done."""

    def node(state: dict) -> dict:
        if ctx.tracker is None:
            return {"outcome": "done"}
        if state.get("item_ref") and not state.get("decided_map"):
            ctx.tracker.comment_work_item(
                state["item_ref"],
                render_spec(state["spec"]),
                run=ctx.run_id,
                node="publish_spec",
            )
            return {}
        item = ctx.tracker.ensure_work_item(
            state["spec"]["title"],
            render_spec(state["spec"]),
            run=ctx.run_id,
            node="publish_spec",
        )
        return {"item_ref": item.ref, "item_title": item.title}

    return node


def make_publish_tickets(ctx: RunContext) -> Callable[[dict], dict]:
    """Each ticket as a marked child of the spec item, blockers first,
    then its native blocker links and the ready-for-agent label — one
    node marker per ticket so a resume recovers each on its own (#123)."""

    def node(state: dict) -> dict:
        if ctx.tracker is None:
            return {"outcome": "done"}
        ctx.tracker.ensure_label(READY_LABEL)
        refs: list[str] = []
        for index, ticket in enumerate(state["tickets"]):
            refs.append(
                _publish_ticket(ctx, state["item_ref"], ticket, index, refs)
            )
        return {"ticket_refs": refs, "outcome": "done"}

    return node


def _publish_ticket(
    ctx: RunContext, item_ref: str, ticket: dict, index: int, refs: list[str]
) -> str:
    """One ticket as a marked child of the spec item — blockers first —
    then its native blocker links and the ready-for-agent label."""
    child = ctx.tracker.create_child_work_item(
        item_ref,
        ticket["title"],
        ticket_body(ticket, item_ref, refs),
        run=ctx.run_id,
        node=f"publish_ticket_{index}",
    )
    for blocker in ticket.get("blocked_by", []):
        ctx.tracker.link_blocked_by(child.ref, refs[blocker])
    ctx.tracker.add_label(child.ref, READY_LABEL)
    return child.ref


def ticket_body(ticket: dict, item_ref: str, refs: list[str]) -> str:
    """A ticket's body — the one format tickets and agent briefs share:
    what to build, seams, acceptance criteria, blocked by, and the
    parent (ADR 0012, #123)."""
    return "\n\n".join(
        [
            _prose("What to build", ticket["what_to_build"]),
            _bullets("Seams", ticket.get("seams", [])),
            _bullets(
                "Acceptance criteria", ticket.get("acceptance_criteria", [])
            ),
            _bullets(
                "Blocked by",
                [f"#{refs[i]}" for i in ticket.get("blocked_by", [])],
            ),
            f"Parent: #{item_ref}",
        ]
    )


def render_tickets(tickets: list[dict], ticket_size: str) -> str:
    """The breakdown as the gate shows it: title, deliverable, blockers
    and criteria count per ticket, oversized ones flagged (#122)."""
    cap = _TICKET_SIZE_CRITERIA.get(ticket_size, _TICKET_SIZE_CRITERIA["small"])
    lines = [f"{len(tickets)} tickets (ticket_size: {ticket_size})"]
    for index, ticket in enumerate(tickets):
        count = len(ticket.get("acceptance_criteria", []))
        flag = f" — over {ticket_size} ({cap} max)" if count > cap else ""
        blockers = ", ".join(
            str(blocker + 1) for blocker in ticket.get("blocked_by", [])
        ) or "—"
        lines.append("")
        lines.append(f"## {index + 1} · {ticket.get('title', '')}")
        lines.append(f"Delivers: {ticket.get('what_to_build', '')}")
        lines.append(f"Blocked by: {blockers}")
        lines.append(f"Acceptance criteria: {count}{flag}")
        lines.extend(f"- {c}" for c in ticket.get("acceptance_criteria", []))
    return "\n".join(lines)


def tickets_brief(state: dict) -> str:
    """PLANNER's fresh context for the breakdown: the approved spec, the
    ``ticket_size`` rule, and on a gate ``redo`` the human's feedback.

    The first ticket is not PLANNER's to write — the harness prepends
    the glossary-and-ADR ticket, so a ``blocked_by`` index only ever
    names an earlier ticket in PLANNER's own list.
    """
    size = state.get("ticket_size", "small")
    lines = [
        f"Idea: {_idea(state)}",
        "",
        "The spec below is approved — cut it into tickets: answer with "
        f"TicketList, blockers first. Size each ticket {size}: "
        f"{_TICKET_SIZE_RULE[size]}.",
        "",
        "The first ticket is not yours to write: the harness adds the "
        "glossary-and-ADR ticket itself and makes it block every ticket "
        "you cut — a blocked_by index only names an earlier ticket in "
        "your own list.",
        "",
        render_spec(state.get("spec")),
    ]
    if state.get("ticket_feedback"):
        lines.append(
            "\nYour last breakdown was rejected with this feedback: "
            f"{state['ticket_feedback']}"
        )
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
    lines.extend(_answered_rounds(state))
    return "\n".join(lines)


def _spec_lead(state: dict) -> str:
    """The brief's lead: a decided Map drafts straight from its
    destination and decisions, no Q&A having run (#126)."""
    if state.get("decided_map"):
        return (
            "This map's decision tickets are all decided — draft the "
            "spec from its destination and decisions"
        )
    return "The Q&A is over — draft the spec"


def spec_brief(state: dict) -> str:
    """PLANNER's fresh context for the spec draft: the idea and every
    answered round — or, on a decided Map, the map's destination and
    decided tickets, no Q&A having run (#126) — and on a gate ``redo``
    the human's feedback."""
    lines = [
        f"Idea: {_idea(state)}",
        "",
        f"{_spec_lead(state)}: answer with Spec — the title, "
        "problem statement, solution, user stories, implementation "
        "decisions, testing decisions with the seams, glossary and "
        "decisions, not yet decided, and out of scope.",
    ]
    lines.extend(_answered_rounds(state))
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
