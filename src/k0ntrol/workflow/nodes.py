"""The graph's node factories.

Every node is a closure over the run context returning a state update —
plain types only (strict msgpack). Routers stay pure, so each node that
can lead to ``limit_hit`` also writes a ``limit`` field saying what that
limit would mean; ``limit_hit`` reads it. Today ``limit_hit`` only ends
the run ``stuck``; #42 turns it into a pause (`retry`/`stop` interrupt)
without touching the routers.
"""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import interrupt

from k0ntrol.tools.path_guard import is_test_path
from k0ntrol.tools.runner import run_command
from k0ntrol.workflow.change import worktree_summary
from k0ntrol.workflow.context import RunContext

_WRITE_TOOLS = ("write_file", "edit_file")


def make_agent_node(
    ctx: RunContext, name: str, files_key: str, brief: Callable[[dict], str]
) -> Callable[[dict], dict]:
    """An agent node: fresh-context invoke, written files, limit flag.

    The node counts the agent run (``<name>_attempts``), merges the files
    the agent actually wrote into ``files_key`` and flags
    ``model_call_limit`` when the agent hit its limit.
    """

    def node(state: dict) -> dict:
        attempts_key = f"{name}_attempts"
        attempt = state.get(attempts_key, 0) + 1
        result = ctx.agent(name).invoke(
            [HumanMessage(brief(state))], config=ctx.agent_config(attempt)
        )
        updates: dict[str, Any] = {
            attempts_key: attempt,
            files_key: _merge(state.get(files_key), _written(result)),
            "limit": None,
        }
        if result.limit_hit:
            updates["limit"] = {"reason": "model_call_limit", "node": name}
        return updates

    return node


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
        return _red_update(red, result)

    return node


def _red_update(red: bool, result) -> dict:
    return {
        "tests_red": red,
        "red_check_output": result.output,
        "limit": None if red else _limit("max_attempts", "red_check"),
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
            "limit": None if ok else _limit("max_attempts", "validator"),
        }

    return node


def _emit_gate_events(ctx: RunContext, results: list[tuple[str, Any]], ok: bool) -> None:
    for name, result in results[1:]:
        ctx.writer.gate(name, "pass" if result.ok else "fail")
    ctx.writer.gate("validator", "pass" if ok else "fail")


def _limit(reason: str, node: str) -> dict:
    return {"reason": reason, "node": node}


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


def limit_hit(state: dict) -> dict:
    """Where a hit limit ends the run — a plain ``stuck`` for now.

    #42 turns this node into the ``limit_hit`` pause: an ``interrupt()``
    offering ``retry [feedback]`` / ``stop``. Routers already send limits
    here, so no edge changes are needed then.
    """
    limit = state.get("limit") or {}
    return {
        "outcome": "stuck",
        "reason": limit.get("reason"),
        "node": limit.get("node"),
    }


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
