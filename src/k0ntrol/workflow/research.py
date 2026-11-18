"""The ``/research`` graph: PLANNER charts a Map (#124).

``/research <question>`` runs one fresh PLANNER pass that charts the
question's open decisions into a ``ChartedMap``: the map's title,
destination and notes, and one decision ticket per open question typed
research, Q&A or human. The harness then publishes the Map work item —
destination, notes, an empty Decisions so far — with each decision
ticket as its child and the blocker links where one decision waits on
another. Working the map's tickets is #125; the REPL command is #129.
"""

from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from k0ntrol.agents import ChartedMap
from k0ntrol.workflow.context import RunContext
from k0ntrol.workflow.nodes import (
    _attempt_limit,
    _invoke_counted,
    _last_output,
    make_limit_hit,
)
from k0ntrol.workflow.routers import _exhausted, route_after_chart, route_limit_hit
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
    graph.add_node("limit_hit", make_limit_hit(ctx))
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_after_chart)
    graph.add_edge("publish_map", "publish_decisions")
    graph.add_edge("publish_decisions", END)
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
