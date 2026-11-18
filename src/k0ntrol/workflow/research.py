"""The ``/research`` graph: PLANNER charts a Map (#124), the map's decision
tickets get worked one run at a time (#125).

``/research <question>`` runs one fresh PLANNER pass that charts the
question's open decisions into a ``ChartedMap``: the map's title,
destination and notes, and one decision ticket per open question typed
research, Q&A or human. The harness then publishes the Map work item —
destination, notes, an empty Decisions so far — with each decision
ticket as its child and the blocker links where one decision waits on
another.

``/research #<n>`` picks the map's first open decision ticket whose
blockers are all closed. A research ticket gets a fresh RESEARCHER run
whose cited findings are commented before the decision gate; a Q&A
ticket a short Q&A with PLANNER; a human ticket goes straight to the
gate. ``ok`` at the gate takes the proposal, any other line is the
human's own decision: the ticket closes with it and the map's Decisions
so far gains its one-line gist and link. The REPL command is #129.
"""

from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from k0ntrol.agents import ChartedMap, DecisionProposal
from k0ntrol.tracker import (
    TRUSTED_ASSOCIATIONS,
    ChildWorkItem,
    WorkItem,
    comment_marker,
    is_k0_work_item,
)
from k0ntrol.workflow.context import RunContext
from k0ntrol.workflow.nodes import (
    _attempt_limit,
    _invoke_counted,
    _last_output,
    make_limit_hit,
)
from k0ntrol.workflow.qa import (
    _answered_lines,
    _cap,
    _questions,
    make_round_gate,
)
from k0ntrol.workflow.routers import (
    _exhausted,
    route_after_chart,
    route_after_decision_planner,
    route_after_pick,
    route_after_researcher,
    route_limit_hit,
    route_research_start,
)
from k0ntrol.workflow.state import RunState

_BLANK = "—"


def build_research_graph(
    ctx: RunContext, checkpointer: BaseCheckpointSaver | None
) -> CompiledStateGraph:
    """Compile the ``/research`` graph for one run context."""
    graph = StateGraph(RunState)
    graph.add_node("planner", make_chart_node(ctx))
    graph.add_node("publish_map", make_publish_map(ctx))
    graph.add_node("publish_decisions", make_publish_decisions(ctx))
    graph.add_node("pick_decision", make_pick_decision(ctx))
    graph.add_node("researcher", make_researcher_node(ctx))
    graph.add_node("comment_findings", make_comment_findings(ctx))
    graph.add_node("decision_planner", make_decision_planner_node(ctx))
    graph.add_node("round_gate", make_round_gate(ctx))
    graph.add_node("decision_gate", make_decision_gate(ctx))
    graph.add_node("record_decision", make_record_decision(ctx))
    graph.add_node("limit_hit", make_limit_hit(ctx))
    graph.add_conditional_edges(START, route_research_start)
    graph.add_conditional_edges("planner", route_after_chart)
    graph.add_edge("publish_map", "publish_decisions")
    graph.add_edge("publish_decisions", END)
    graph.add_conditional_edges("pick_decision", route_after_pick)
    graph.add_conditional_edges("researcher", route_after_researcher)
    graph.add_edge("comment_findings", "decision_gate")
    graph.add_conditional_edges("decision_planner", route_after_decision_planner)
    graph.add_edge("round_gate", "decision_planner")
    graph.add_edge("decision_gate", "record_decision")
    graph.add_edge("record_decision", END)
    graph.add_conditional_edges("limit_hit", route_limit_hit)
    return graph.compile(checkpointer=checkpointer)


def make_chart_node(ctx: RunContext) -> Callable[[dict], dict]:
    """One charting pass: a fresh PLANNER run returning a ChartedMap.

    A run that answers with another structured output charts again while
    attempts remain — the limit pause carries the last output, the same
    resumable path a model-call limit takes.
    """

    def node(state: dict) -> dict:
        result, updates = _invoke_counted(ctx, "planner", state, chart_brief)
        if result.limit_hit:
            return updates
        chart = result.structured_response
        if isinstance(chart, ChartedMap):
            updates["map"] = chart.model_dump()
            return updates
        merged = {**state, **updates}
        if _exhausted(merged, "planner_attempts"):
            updates["limit"] = _attempt_limit(
                merged, "planner", "planner", _last_output(result)
            )
        return updates

    return node


def make_publish_map(ctx: RunContext) -> Callable[[dict], dict]:
    """The Map work item, marker-idempotent (carrier contract, #13)."""

    def node(state: dict) -> dict:
        chart = state["map"]
        item = ctx.tracker.ensure_work_item(
            chart["title"], map_body(chart), run=ctx.run_id, node="publish_map"
        )
        return {"item_ref": item.ref, "item_title": item.title}

    return node


def make_publish_decisions(ctx: RunContext) -> Callable[[dict], dict]:
    """Each decision ticket as a marked child of the Map, then its blocker
    links — one node marker per decision so a resume recovers each child
    on its own.
    """

    def node(state: dict) -> dict:
        refs = []
        for index, decision in enumerate(state["map"]["decisions"]):
            child = ctx.tracker.create_child_work_item(
                state["item_ref"],
                decision["question"],
                decision_body(decision),
                run=ctx.run_id,
                node=f"publish_decision_{index}",
            )
            refs.append(child.ref)
            for blocker in decision.get("blocked_by", []):
                ctx.tracker.link_blocked_by(child.ref, refs[blocker])
        return {"decision_refs": refs, "outcome": "done"}

    return node


def map_body(chart: dict) -> str:
    """The Map body: Destination, Notes, Decisions so far (empty — the
    confirmed decisions land one line at a time, #125), Not yet
    specified, Out of scope."""
    parts = [
        _prose("Destination", chart["destination"]),
        _prose("Notes", chart.get("notes", "")),
        "## Decisions so far",
        _bullets("Not yet specified", chart.get("not_yet_specified", [])),
        _bullets("Out of scope", chart.get("out_of_scope", [])),
    ]
    return "\n\n".join(parts)


def decision_body(decision: dict) -> str:
    """A decision ticket body: its question and its type."""
    return f"## Question\n\n{decision['question']}\n\n## Type\n\n{decision['type']}"


def chart_brief(state: dict) -> str:
    """PLANNER's fresh context for the chart: the research question, and
    on a re-chart the note that its last answer was no map."""
    text = (
        f"Research question: {state['task']}\n\n"
        "Chart the map for this question: answer with ChartedMap — the "
        "map's title, its destination, notes, one decision ticket per "
        "open question (typed research, Q&A or human, with the "
        "blocked_by indexes of the earlier decisions they wait on), "
        "what is not yet specified and what is out of scope."
    )
    if state.get("planner_attempts"):
        text += (
            "\n\nYour last answer did not chart a map — answer with ChartedMap."
        )
    return text


def _prose(title: str, text: str) -> str:
    return f"## {title}\n\n{text or _BLANK}"


def _bullets(title: str, lines: list[str]) -> str:
    body = "\n".join(f"- {line}" for line in lines) or _BLANK
    return f"## {title}\n\n{body}"


# --- /research #<n>: work the map's next decision ticket (#125) ------------------


def make_pick_decision(ctx: RunContext) -> Callable[[dict], dict]:
    """The map's first open decision ticket whose blockers are all closed.

    Children come in tracker order: a closed child is done, an open child
    with an open blocker still waits. The picked ticket's type — the
    ``## Type`` section k0 wrote, ``human`` when there is none or the
    author is not trusted (ADR 0006) — decides who answers it.
    """

    def node(state: dict) -> dict:
        for child in ctx.tracker.list_children(state["item_ref"]):
            if child.state != "open" or _blocked(ctx, child.ref):
                continue
            item = ctx.tracker.read_work_item(child.ref)
            return {"decision": _decision_dict(child, item)}
        return {"decision": None, "outcome": "done"}

    return node


def _blocked(ctx: RunContext, ref: str) -> bool:
    return any(
        blocker.state != "closed" for blocker in ctx.tracker.list_blockers(ref)
    )


def _decision_dict(child: ChildWorkItem, item: WorkItem) -> dict:
    return {
        "ref": child.ref,
        "question": item.title,
        "type": _decision_type(item),
        "url": item.url,
    }


_DECISION_TYPES = ("research", "Q&A", "human")


def _decision_type(item: WorkItem) -> str:
    """The ticket's ``## Type``: who answers it.

    A child with no readable type — or one an untrusted author filed —
    is worked as a human decision: only a gate ever reads its content,
    never an agent (ADR 0006).
    """
    trusted = (
        is_k0_work_item(item)
        or item.author_association in TRUSTED_ASSOCIATIONS
    )
    lines = _section(item.body, "Type").splitlines()
    type_ = lines[0].strip() if lines else ""
    return type_ if trusted and type_ in _DECISION_TYPES else "human"


def _section(body: str, title: str) -> str:
    """The text between ``## <title>`` and the next ``##`` heading."""
    lines = body.splitlines()
    if f"## {title}" not in lines:
        return ""
    collected = []
    for line in lines[lines.index(f"## {title}") + 1 :]:
        if line.startswith("## "):
            break
        collected.append(line)
    return "\n".join(collected).strip()


def make_researcher_node(ctx: RunContext) -> Callable[[dict], dict]:
    """A fresh RESEARCHER run on the picked research ticket: its cited
    findings and proposed decision feed the comment and the gate."""

    def node(state: dict) -> dict:
        result, updates = _invoke_counted(
            ctx, "researcher", state, researcher_brief
        )
        if result.limit_hit:
            return updates
        finding = result.structured_response
        updates["finding"] = finding.model_dump()
        updates["proposal"] = finding.proposed_decision
        return updates

    return node


def researcher_brief(state: dict) -> str:
    """RESEARCHER's fresh context: the decision ticket to answer."""
    decision = state["decision"]
    return (
        f"Decision ticket #{decision['ref']} on map #{state['item_ref']} "
        f"({state.get('item_title', '')}):\n"
        f"{decision['question']}\n\n"
        "Answer it from primary sources and answer with ResearchFinding — "
        "the cited findings, your proposed decision and each source's "
        "title and URL."
    )


def make_comment_findings(ctx: RunContext) -> Callable[[dict], dict]:
    """RESEARCHER's cited findings as a marked comment on the ticket: the
    evidence sits on the work item before the decision gate asks."""

    def node(state: dict) -> dict:
        ctx.tracker.comment_work_item(
            state["decision"]["ref"],
            _findings_comment(state["finding"]),
            run=ctx.run_id,
            node="comment_findings",
        )
        return {}

    return node


def _findings_comment(finding: dict) -> str:
    lines = ["## Findings", ""]
    lines += [f"- {item}" for item in finding["findings"]]
    lines += ["", "## Citations", ""]
    lines += [
        f"- [{citation['title']}]({citation['url']})"
        for citation in finding["citations"]
    ]
    lines += ["", "## Proposed decision", "", finding["proposed_decision"]]
    return "\n".join(lines)


def make_decision_planner_node(ctx: RunContext) -> Callable[[dict], dict]:
    """A Q&A ticket's PLANNER run: the next round of questions, or a
    DecisionProposal once it can propose the decision.

    The shared round machinery counts each run under ``planner_attempts``
    and caps a round's questions in code; an off-shape or empty answer
    ends the Q&A at the decision gate, like /plan's (#120).
    """

    def node(state: dict) -> dict:
        result, updates = _invoke_counted(
            ctx,
            "planner",
            state,
            decision_brief,
            limit_node="decision_planner",
        )
        if result.limit_hit:
            return updates
        proposal = result.structured_response
        if isinstance(proposal, DecisionProposal):
            updates["proposal"] = _proposal_text(proposal)
            updates["questions"] = []
            return updates
        updates["questions"] = _questions(result, _cap(state))
        return updates

    return node


def _proposal_text(proposal: DecisionProposal) -> str:
    text = proposal.proposed_decision
    if proposal.rationale:
        text += f"\n\n{proposal.rationale}"
    return text


def decision_brief(state: dict) -> str:
    """PLANNER's fresh context for a Q&A decision ticket: the question,
    then every earlier round's questions and the line that answered
    them."""
    decision = state["decision"]
    lines = [
        f"Decision ticket #{decision['ref']} on map #{state['item_ref']} "
        f"({state.get('item_title', '')}):",
        f"{decision['question']}",
        "",
        f"Ask the next round of questions — at most {_cap(state)}, each "
        "numbered with your recommended answer — or answer with "
        "DecisionProposal once you can propose the decision.",
    ]
    for index, round_ in enumerate(state.get("rounds", []), 1):
        lines.append(f"\nRound {index}:")
        lines.extend(_answered_lines(round_))
    return "\n".join(lines)


def make_decision_gate(ctx: RunContext) -> Callable[[dict], dict]:
    """The decision gate: the proposal — or, for a human ticket, the bare
    question. ``ok`` takes the proposal; any other line is the human's
    own decision — free text, like the round gate."""

    def node(state: dict) -> dict:
        decision = state["decision"]
        proposal = state.get("proposal", "")
        answer = str(
            interrupt(
                {
                    "gate": "decision",
                    "question": _gate_question(decision, proposal),
                    "answers": [],
                    "detail": _gate_detail(state),
                }
            )
        ).strip()
        if proposal and answer.lower() == "ok":
            return {"decided": proposal}
        return {"decided": answer}

    return node


def _gate_question(decision: dict, proposal: str) -> str:
    base = f"#{decision['ref']}: {decision['question']}"
    if proposal:
        return f"{base} — ok takes the proposal, anything else is yours"
    return base


def _gate_detail(state: dict) -> str:
    finding = state.get("finding")
    if finding:
        return _findings_comment(finding)
    return state.get("proposal") or state["decision"]["question"]


def make_record_decision(ctx: RunContext) -> Callable[[dict], dict]:
    """Close the ticket with the confirmed decision and index it on the
    map: one line — gist and link — under Decisions so far,
    marker-idempotent like every k0 write (carrier contract, #13)."""

    def node(state: dict) -> dict:
        decision = state["decision"]
        ctx.tracker.close_work_item(
            decision["ref"],
            f"## Decision\n\n{state['decided']}",
            run=ctx.run_id,
            node="record_decision",
        )
        _index_on_map(ctx, state)
        return {"outcome": "done"}

    return node


def _index_on_map(ctx: RunContext, state: dict) -> None:
    marker = comment_marker(ctx.run_id, "record_decision")
    item = ctx.tracker.read_work_item(state["item_ref"])
    if marker in item.body:
        return
    line = _map_line(state["decision"], state["decided"], marker)
    ctx.tracker.edit_work_item_body(
        state["item_ref"], _with_decision_line(item.body, line)
    )


def _map_line(decision: dict, decided: str, marker: str) -> str:
    gist = decided.splitlines()[0].strip()
    return f"- [{decision['question']}]({decision['url']}): {gist} {marker}"


def _with_decision_line(body: str, line: str) -> str:
    """Append ``line`` under ``## Decisions so far`` — kept entries first;
    a hand-made map without the section gets it."""
    lines = body.splitlines()
    if "## Decisions so far" not in lines:
        return f"{body.rstrip()}\n\n## Decisions so far\n\n{line}"
    head = lines.index("## Decisions so far")
    end = next(
        (i for i in range(head + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    entries = [l for l in lines[head + 1 : end] if l.strip()]
    return "\n".join([*lines[: head + 1], "", *entries, line, "", *lines[end:]])
