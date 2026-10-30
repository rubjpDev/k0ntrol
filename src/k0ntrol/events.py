"""The always-on, content-free event log.

One append-only file, `.k0-mem/events.jsonl`, one JSON line per event. The
schema is the contract in spec #33 ("Events and Langfuse"): every line has
`ts` (ISO UTC, ms), `kind`, and `run` — or `chat` for `/ask`. Fields that do
not apply are omitted, never `null`. No prompts, completions, file contents
or tool arguments other than `path` are ever written.

`EventWriter` appends lines for harness events (`run_start`, `gate`,
`run_end`). `K0EventHandler` — the one callback attached to every graph and
helper model call — emits the `model` and `tool` events through the same
writer.

Shared contract for tool refusals (the `denied` field): a tool refuses a
call by raising an exception that carries a `denied` attribute with value
`"allowlist"` or `"path_guard"`. The canonical class is
`k0ntrol.tools.errors.ToolError` (raised with `denied="path_guard"` /
`denied="allowlist"`); the handler accepts any exception with a truthy
`denied` attribute, so tool code never depends on this module.
"""

import json
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent

from k0ntrol.providers import REGISTRY

ERR_MAX_CHARS = 500
MASKED = "***"
# Values shorter than this are not masked: a short env value would mangle
# ordinary words inside error text. Real credentials are much longer.
MASK_MIN_LENGTH = 8

# Env-var name suffixes whose *values* are treated as secrets for masking.
# Same rule the command runner uses to scrub its environment.
_SECRET_NAME_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET")

# Metadata keys the run module / agent builder put in config["metadata"] so
# the handler can fill the event fields. `k0_agent` xor `k0_helper` decides
# the `agent`/`helper` field; `langgraph_node` is the fallback for agents.
META_AGENT = "k0_agent"
META_HELPER = "k0_helper"
META_ATTEMPT = "k0_attempt"
META_EFFORT = "k0_effort"


def events_path(root: Path) -> Path:
    """The event log for the project rooted at `root`."""
    return root / ".k0-mem" / "events.jsonl"


def new_run_id() -> str:
    """A fresh run id: 32 lowercase hex, also the Langfuse trace id."""
    return uuid4().hex


def new_chat_id() -> str:
    """A fresh `chat` id for a `/ask` exchange. Same format as a run id."""
    return new_run_id()


def secret_env_values(env: Mapping[str, str] | None = None) -> list[str]:
    """Exact values of known secret env vars, used for `err` masking."""
    source = os.environ if env is None else env
    known_names = {name for spec in REGISTRY.values() for name in spec.credentials}
    values = [
        value
        for name, value in source.items()
        if (name in known_names or name.endswith(_SECRET_NAME_SUFFIXES))
        and len(value) >= MASK_MIN_LENGTH
    ]
    return values


class EventWriteError(Exception):
    """A JSONL write failed. The run treats this as `stuck`."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _mask(text: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, MASKED)
    return text[:ERR_MAX_CHARS]


class EventWriter:
    """Appends one JSON line per event to `path`.

    Bound to exactly one scope: a `run` id, or a `chat` id for `/ask`.
    `progress`, when given, is called with each written event (for the
    REPL's progress lines); it may never break a write.
    """

    def __init__(
        self,
        path: Path,
        *,
        run: str | None = None,
        chat: str | None = None,
        mask: Iterable[str] | None = None,
        progress: Callable[[dict], None] | None = None,
    ) -> None:
        if (run is None) == (chat is None):
            raise ValueError("EventWriter needs exactly one of run= or chat=")
        self._path = path
        self._run = run
        self._chat = chat
        self._mask = list(mask) if mask is not None else secret_env_values()
        self._progress = progress
        self._lock = threading.Lock()

    @property
    def run(self) -> str | None:
        """The run id this writer is bound to (`None` for a chat writer)."""
        return self._run

    def emit(self, kind: str, **fields: Any) -> dict:
        """Append one event. `None` fields are omitted; `err` is masked."""
        event = {"ts": _now_iso(), "kind": kind}
        event["run" if self._run else "chat"] = self._run or self._chat
        err = fields.pop("err", None)
        for key, value in fields.items():
            if value is not None:
                event[key] = value
        if err:
            event["err"] = _mask(str(err), self._mask)
        line = json.dumps(event, separators=(",", ":"))
        with self._lock:
            try:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError as exc:
                raise EventWriteError(f"events.jsonl write failed: {exc}") from exc
        self._notify(event)
        return event

    def _notify(self, event: dict) -> None:
        if self._progress is None:
            return
        try:
            self._progress(event)
        except Exception:
            pass

    def run_start(
        self,
        run_kind: Literal["full", "fast"],
        *,
        work_item: str | None = None,
        branch: str | None = None,
        version: str | None = None,
        resumed: bool | None = None,
    ) -> dict:
        """Emit `run_start`. `work_item` and `resumed` only when they apply."""
        return self.emit(
            "run_start",
            run_kind=run_kind,
            work_item=work_item,
            branch=branch,
            version=version,
            resumed=resumed,
        )

    def gate(
        self,
        gate: str,
        result: str,
        *,
        redo: list[str] | None = None,
    ) -> dict:
        """Emit `gate`: `pass`/`fail` for code gates, `waiting` then a
        decision (`approve`/`reject`/`retry`/`stop`) for human gates."""
        return self.emit("gate", gate=gate, result=result, redo=redo)

    def run_end(
        self,
        outcome: Literal["done", "stuck", "rejected"],
        ms: int,
        *,
        reason: str | None = None,
        node: str | None = None,
    ) -> dict:
        """Emit `run_end`. `reason`/`node` only for `stuck` from a limit."""
        return self.emit(
            "run_end", outcome=outcome, ms=ms, reason=reason, node=node
        )


class K0EventHandler(GraphCallbackHandler):
    """The one callback handler attached to every graph and helper call.

    Emits `model` events on chat-model end/error and `tool` events on tool
    end/error through the bound `EventWriter` (which carries the `run` or
    `chat` scope). Per-call context comes from `config["metadata"]`:
    `k0_agent` xor `k0_helper`, `k0_attempt`, `k0_effort`; `langgraph_node`
    is the fallback for `agent`.

    `raise_error` is on so a failed JSONL write propagates and the run can
    end `stuck`; `run_inline` keeps line order under async dispatch.

    `on_interrupt` emits `gate`/`waiting` when a gate node's `interrupt()`
    value is a mapping carrying a `"gate"` name — the convention gate
    payloads follow. Decisions (`approve`/`reject`/`retry`/`stop`) are
    harness events emitted through `EventWriter.gate`.
    """

    run_inline = True
    raise_error = True

    def __init__(self, writer: EventWriter) -> None:
        self._writer = writer
        self._llm_starts: dict[Any, dict] = {}
        self._tool_starts: dict[Any, dict] = {}

    # -- model events ------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict,
        messages: list,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ) -> None:
        self._llm_starts[run_id] = _call_start(metadata)

    def on_llm_start(
        self,
        serialized: dict,
        prompts: list,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ) -> None:
        # k0 uses chat models, but a plain LLM call still gets its event.
        self._llm_starts[run_id] = _call_start(metadata)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list | None = None,
        **kwargs: Any,
    ) -> None:
        start = self._llm_starts.pop(run_id, {})
        message = _llm_message(response)
        usage = getattr(message, "usage_metadata", None) or {}
        self._writer.emit(
            "model",
            agent=start.get("agent"),
            helper=start.get("helper"),
            provider=start.get("provider"),
            model=start.get("model") or _response_model(response, message),
            effort=start.get("effort"),
            attempt=start.get("attempt"),
            ms=_elapsed_ms(start),
            tokens_in=usage.get("input_tokens"),
            tokens_out=usage.get("output_tokens"),
            cost_usd=_response_cost(message),
            ok=True,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list | None = None,
        **kwargs: Any,
    ) -> None:
        start = self._llm_starts.pop(run_id, {})
        self._writer.emit(
            "model",
            agent=start.get("agent"),
            helper=start.get("helper"),
            provider=start.get("provider"),
            model=start.get("model"),
            effort=start.get("effort"),
            attempt=start.get("attempt"),
            ms=_elapsed_ms(start),
            ok=False,
            err=str(error),
        )

    # -- tool events --------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list | None = None,
        metadata: dict | None = None,
        inputs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        metadata = metadata or {}
        path = inputs.get("path") if isinstance(inputs, dict) else None
        self._tool_starts[run_id] = {
            "t0": time.monotonic(),
            "agent": _agent_name(metadata),
            "tool": (serialized or {}).get("name"),
            "path": str(path) if path else None,
        }

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        start = self._tool_starts.pop(run_id, {})
        self._writer.emit(
            "tool",
            agent=start.get("agent"),
            tool=start.get("tool"),
            path=start.get("path"),
            ms=_elapsed_ms(start),
            ok=True,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        start = self._tool_starts.pop(run_id, {})
        self._writer.emit(
            "tool",
            agent=start.get("agent"),
            tool=start.get("tool"),
            path=start.get("path"),
            ms=_elapsed_ms(start),
            ok=False,
            err=str(error),
            denied=getattr(error, "denied", None),
        )

    # -- gate events ---------------------------------------------------------

    def on_interrupt(self, event: GraphInterruptEvent) -> None:
        for interrupt in event.interrupts:
            value = interrupt.value
            if isinstance(value, Mapping) and value.get("gate"):
                self._writer.gate(str(value["gate"]), "waiting")


def _call_start(metadata: Mapping | None) -> dict:
    metadata = metadata or {}
    return {
        "t0": time.monotonic(),
        "helper": metadata.get(META_HELPER),
        "agent": _agent_name(metadata),
        "provider": metadata.get("ls_provider"),
        "model": metadata.get("ls_model_name"),
        "effort": metadata.get(META_EFFORT),
        "attempt": metadata.get(META_ATTEMPT),
    }


def _agent_name(metadata: Mapping) -> Any:
    if metadata.get(META_HELPER):
        return None
    return metadata.get(META_AGENT) or metadata.get("langgraph_node")


def _elapsed_ms(start: Mapping) -> int:
    t0 = start.get("t0")
    if t0 is None:
        return 0
    return round((time.monotonic() - t0) * 1000)


def _llm_message(response: Any) -> Any | None:
    try:
        generation = response.generations[0][0]
    except (IndexError, TypeError, AttributeError):
        return None
    return getattr(generation, "message", None)


def _response_model(response: Any, message: Any | None) -> Any | None:
    llm_output = getattr(response, "llm_output", None) or {}
    if llm_output.get("model_name"):
        return llm_output["model_name"]
    meta = getattr(message, "response_metadata", None) or {}
    return meta.get("model_name")


def _response_cost(message: Any | None) -> Any | None:
    meta = getattr(message, "response_metadata", None) or {}
    return meta.get("cost")


# --- Progress renderer ----------------------------------------------------

_ERR_SNIP = 80


def render_event(event: dict) -> str | None:
    """One progress line for an event, or `None` when there is nothing to
    say (unknown or absent `kind`, e.g. old 0.0.x lines)."""
    renderer = _RENDERERS.get(event.get("kind", ""))
    if renderer is None:
        return None
    return renderer(event)


class ProgressRenderer:
    """Turns each event into at most one progress line via `write`."""

    def __init__(self, write: Callable[[str], None]) -> None:
        self._write = write

    def __call__(self, event: dict) -> None:
        line = render_event(event)
        if line is not None:
            self._write(line)


def _snip(text: str) -> str:
    return text[:_ERR_SNIP]


def _render_run_start(e: dict) -> str:
    parts = ["run", e.get("run_kind", "?")]
    if e.get("work_item"):
        parts.append(e["work_item"])
    parts.append("started")
    if e.get("branch"):
        parts.append("on " + e["branch"])
    if e.get("resumed"):
        parts.append("(resumed)")
    return " ".join(parts)


def _render_model(e: dict) -> str:
    who = e.get("agent") or e.get("helper") or "?"
    desc = _model_desc(e)
    if e.get("ok"):
        return _render_model_ok(e, who, desc)
    return f"model {who} {desc} failed: {_snip(e.get('err', ''))} ({e.get('ms', 0)}ms)"


def _model_desc(e: dict) -> str:
    desc = e.get("model") or "?"
    if e.get("provider"):
        return f"{e['provider']}/{desc}"
    return desc


def _render_model_ok(e: dict, who: str, desc: str) -> str:
    usage = ""
    if "tokens_in" in e or "tokens_out" in e:
        usage = f" ({e.get('tokens_in', 0)} in, {e.get('tokens_out', 0)} out)"
    return f"model {who} {desc} ok {e.get('ms', 0)}ms{usage}"


def _render_tool(e: dict) -> str:
    head = f"tool {e.get('agent', '?')} {e.get('tool', '?')}"
    if e.get("path"):
        head += f" {e['path']}"
    if e.get("denied"):
        return f"{head} denied {e['denied']} ({e.get('ms', 0)}ms)"
    if e.get("ok"):
        return f"{head} ok {e.get('ms', 0)}ms"
    return f"{head} failed: {_snip(e.get('err', ''))} ({e.get('ms', 0)}ms)"


def _render_gate(e: dict) -> str:
    line = f"gate {e.get('gate', '?')} {e.get('result', '?')}"
    if e.get("redo"):
        line += " redo: " + ", ".join(e["redo"])
    return line


def _render_run_end(e: dict) -> str:
    line = f"run ended {e.get('outcome', '?')} ({e.get('ms', 0)}ms)"
    if e.get("reason"):
        line += f" {e['reason']}"
        if e.get("node"):
            line += f" at {e['node']}"
    return line


_RENDERERS: dict[str, Callable[[dict], str]] = {
    "run_start": _render_run_start,
    "model": _render_model,
    "tool": _render_tool,
    "gate": _render_gate,
    "run_end": _render_run_end,
}
