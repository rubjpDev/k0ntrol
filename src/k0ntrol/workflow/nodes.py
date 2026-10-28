"""The graph's node factories.

Every node is a closure over the run context returning a state update —
plain types only (strict msgpack). Routers stay pure, so each node that
can lead to ``limit_hit`` also writes a ``limit`` field carrying
everything the pause needs: the reason, the node to show, the node a
``retry`` reruns, the limit description and the tail of the last output.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import interrupt

from k0ntrol.tools.path_guard import is_test_path
from k0ntrol.tools.runner import run_command
from k0ntrol.workflow.change import diff_stat, worktree_summary
from k0ntrol.workflow.context import RunContext

_WRITE_TOOLS = ("write_file", "edit_file")

# Chars of a failing output kept at the pause and in the retry context.
_OUTPUT_TAIL = 2_000


def make_agent_node(
    ctx: RunContext, name: str, files_key: str, brief: Callable[[dict], str]
) -> Callable[[dict], dict]:
    """An agent node: fresh-context invoke, written files, limit flag.

    The node counts the agent run (``<name>_attempts``), merges the files
    the agent actually wrote into ``files_key`` and flags
    ``model_call_limit`` when the agent hit its limit — a limited run
    never falls through to a check and never uses up an attempt (#20).
    """

    def node(state: dict) -> dict:
        attempts_key = f"{name}_attempts"
        attempt = state.get(attempts_key, 0) + 1
        result = ctx.agent(name).invoke(
            [HumanMessage(_brief_with_retry(brief, state))],
            config=ctx.agent_config(attempt),
        )
        updates: dict[str, Any] = {
            attempts_key: attempt,
            files_key: _merge(state.get(files_key), _written(result)),
            "limit": None,
            "retry_node": None,
            "retry_context": None,
        }
        if result.limit_hit:
            updates[attempts_key] = state.get(attempts_key, 0)
            updates["limit"] = _call_limit(ctx, name, result)
        return updates

    return node


def _brief_with_retry(brief: Callable[[dict], str], state: dict) -> str:
    """The agent's brief, plus the pause context on a ``limit_hit`` retry."""
    text = brief(state)
    if state.get("retry_context"):
        text += f"\n\n{state['retry_context']}"
    return text


def _call_limit(ctx: RunContext, name: str, result) -> dict:
    """The ``limit`` field for an agent that hit its model-call limit."""
    calls = ctx.agent(name).model_calls
    return {
        "reason": "model_call_limit",
        "node": name,
        "retry_node": name,
        "what": "model calls",
        "used": calls,
        "max": calls,
        "output": _last_output(result),
    }


def _attempt_limit(state: dict, node: str, retry_node: str, output: str) -> dict:
    """The ``limit`` field for a loop that ran out of ``max_attempts``."""
    return {
        "reason": "max_attempts",
        "node": node,
        "retry_node": retry_node,
        "what": "attempts",
        "used": state.get(f"{retry_node}_attempts", 0),
        "max": state.get("max_attempts", 2),
        "output": _tail(output),
    }


def _tail(text: str) -> str:
    return text[-_OUTPUT_TAIL:]


def _last_output(result) -> str:
    """What the limited agent last produced: its answer, else its last
    tool result — the closest thing to a failing output it has."""
    if result.text:
        return _tail(result.text)
    for message in reversed(result.messages):
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            return _tail(content)
    return ""


def make_red_check(ctx: RunContext) -> Callable[[dict], dict]:
    """The red check: ``test_cmd`` on TESTER's files must fail (ADR 0001)."""

    def node(state: dict) -> dict:
        files = [p for p in state.get("test_files", []) if _test_file(ctx, p)]
        result = run_command(
            ctx.cfg["test_cmd"],
            root=ctx.root,
            timeout=ctx.timeout,
            extra_args=files,
        )
        red = bool(files) and not result.ok
        ctx.writer.gate("red_check", "pass" if red else "fail")
        return _red_update(red, result, state)

    return node


def _red_update(red: bool, result, state: dict) -> dict:
    return {
        "tests_red": red,
        "red_check_output": result.output,
        "limit": (
            None if red else _attempt_limit(state, "red_check", "tester", result.output)
        ),
    }


def make_validator(ctx: RunContext) -> Callable[[dict], dict]:
    """VALIDATOR: ``test_cmd`` plus the configured ``gates``, all in code."""

    def node(state: dict) -> dict:
        results = _run_checks(ctx)
        ok = all(result.ok for _, result in results)
        _emit_gate_events(ctx, results, ok)
        return {
            "validator_ok": ok,
            "validator_output": _format(results),
            "limit": (
                None
                if ok
                else _attempt_limit(state, "validator", "coder", _format(results))
            ),
        }

    return node


def _emit_gate_events(ctx: RunContext, results: list[tuple[str, Any]], ok: bool) -> None:
    for name, result in results[1:]:
        ctx.writer.gate(name, "pass" if result.ok else "fail")
    ctx.writer.gate("validator", "pass" if ok else "fail")


def make_diff_gate(ctx: RunContext) -> Callable[[dict], dict]:
    """The diff gate: show the working tree, ``y`` done / ``n`` rejected."""

    def node(state: dict) -> dict:
        decision = interrupt(
            {
                "gate": "diff",
                "question": "tests are green — keep the change?",
                "answers": ["y", "n"],
                "detail": worktree_summary(ctx.root),
            }
        )
        approved = str(decision).strip().lower() in ("y", "yes")
        return {"outcome": "done" if approved else "rejected"}

    return node


def make_limit_hit(ctx: RunContext) -> Callable[[dict], dict]:
    """The ``limit_hit`` pause: an ``interrupt()``, never a gate (#20).

    Shown: the node, the limit (``model calls 60/60`` or
    ``attempts 2/2``), the tail of the last failing output and
    ``git diff --stat``. ``retry [feedback]`` reruns the node that hit
    the limit — fresh context, same limit, counter reset, files kept —
    and ``stop`` ends the run ``stuck``. Retries are unlimited.
    """

    def node(state: dict) -> dict:
        limit = state.get("limit") or {}
        decision = interrupt(
            {
                "gate": "limit_hit",
                "question": (
                    f"{limit.get('node', '?')} hit its limit "
                    f"({_limit_desc(limit)}) — retry or stop?"
                ),
                "answers": ["retry", "stop"],
                "detail": _limit_detail(ctx, limit),
            }
        )
        verb, _, feedback = str(decision).strip().partition(" ")
        if verb.lower() == "retry":
            return _retry_update(limit, feedback.strip())
        return {
            "outcome": "stuck",
            "reason": limit.get("reason"),
            "node": limit.get("node"),
            "retry_node": None,
        }

    return node


def _limit_desc(limit: dict) -> str:
    return f"{limit.get('what', 'attempts')} {limit.get('used', '?')}/{limit.get('max', '?')}"


def _limit_detail(ctx: RunContext, limit: dict) -> str:
    parts = []
    output = str(limit.get("output") or "").rstrip()
    if output:
        parts.append(f"last output:\n{output}")
    parts.append(diff_stat(ctx.root))
    return "\n\n".join(parts)


def _retry_update(limit: dict, feedback: str) -> dict:
    retry_node = str(limit.get("retry_node") or "tester")
    return {
        "limit": None,
        "retry_node": retry_node,
        "retry_context": _retry_context(limit, feedback),
        f"{retry_node}_attempts": 0,
    }


def _retry_context(limit: dict, feedback: str) -> str:
    parts = [
        f"The previous run stopped at its limit: {_limit_desc(limit)} "
        f"at {limit.get('node', '?')}. Files written so far are kept."
    ]
    output = str(limit.get("output") or "").rstrip()
    if output:
        parts.append(f"Last output:\n{output}")
    if feedback:
        parts.append(f"Human feedback: {feedback}")
    return "\n\n".join(parts)


def _run_checks(ctx: RunContext) -> list[tuple[str, Any]]:
    results = [
        (
            "test_cmd",
            run_command(ctx.cfg["test_cmd"], root=ctx.root, timeout=ctx.timeout),
        )
    ]
    for gate in ctx.cfg.get("gates") or []:
        results.append(
            (
                gate["name"],
                run_command(gate["cmd"], root=ctx.root, timeout=ctx.timeout),
            )
        )
    return results


def _format(results: list[tuple[str, Any]]) -> str:
    return "\n".join(
        f"$ {' '.join(result.argv)}\n{result.output}".rstrip()
        for _, result in results
    )


def _test_file(ctx: RunContext, path: str) -> bool:
    return is_test_path(path) and (ctx.root / path).is_file()


def _merge(existing: list[str] | None, new: list[str]) -> list[str]:
    return sorted(set(existing or []) | set(new))


def _written(result) -> list[str]:
    """The paths the agent's write tools actually wrote (never failed calls)."""
    succeeded = _successful_calls(result.messages)
    return [
        path
        for message in result.messages
        for path in _written_paths(message, succeeded)
    ]


def _successful_calls(messages: list) -> set:
    return {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage) and message.status != "error"
    }


def _written_paths(message, succeeded: set) -> list[str]:
    if not isinstance(message, AIMessage):
        return []
    return [
        call["args"]["path"]
        for call in message.tool_calls
        if _is_successful_write(call, succeeded)
    ]


def _is_successful_write(call: dict, succeeded: set) -> bool:
    return (
        call["name"] in _WRITE_TOOLS
        and call["id"] in succeeded
        and bool(call["args"].get("path"))
    )
