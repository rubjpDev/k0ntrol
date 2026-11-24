"""Optional plain OpenTelemetry export: a third, opt-in callback handler.

Set the standard ``OTEL_EXPORTER_OTLP_ENDPOINT`` (and optionally
``OTEL_EXPORTER_OTLP_HEADERS``) and every run also exports spans over
OTLP/HTTP to that endpoint (Jaeger, Tempo, Honeycomb, a collector). With
the variable unset, or the ``otel`` (or ``langfuse``) extra missing,
nothing is imported or constructed. The trace id is the run id (the chat
id for ``/ask``), so a span joins its ``events.jsonl`` lines on one key.

The handler is a small callback-to-span bridge written on the OTel API.
It runs on its own ``TracerProvider``, never the global one: Langfuse's
default span filter also exports ``gen_ai.*`` spans, so sharing a
provider would send every model call to Langfuse twice, and registering
a global provider would depend on who initialises first. It sets no
"current span", so it cannot orphan Langfuse's spans. It exports names,
models, token counts and error text only (masked with the same rule as
the event log), never prompts, tool inputs or outputs. Every SDK call is
guarded: an export error never changes a run's outcome.
"""

import functools
import importlib
import os
import random
import threading
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from k0ntrol import __version__
from k0ntrol.events import MASKED, secret_env_values
from k0ntrol.langfuse import build_trace

ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
_KEY_ENV = (
    ENDPOINT_ENV,
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_SERVICE_NAME",
)
_REQUIRED_MODULES = (
    "opentelemetry.sdk.trace",
    "opentelemetry.exporter.otlp.proto.http.trace_exporter",
)
HIDDEN_TAG = "langsmith:hidden"  # LangGraph's internal plumbing chains
FLUSH_TIMEOUT_MS = 5000

_PROVIDERS: dict[tuple, Any] = {}
_PROVIDERS_LOCK = threading.Lock()


def otel_enabled(env: Mapping[str, str] | None = None) -> bool:
    """The endpoint set, the SDK not switched off, the packages importable."""
    source = os.environ if env is None else env
    if not source.get(ENDPOINT_ENV):
        return False
    if source.get("OTEL_SDK_DISABLED", "").lower() == "true":
        return False
    try:
        for module in _REQUIRED_MODULES:
            importlib.import_module(module)
    except ImportError:
        return False
    return True


def _resource_attributes(env: Mapping[str, str]) -> dict[str, str]:
    attributes = {"service.version": __version__}
    if not env.get("OTEL_SERVICE_NAME"):
        attributes["service.name"] = "k0"
    return attributes


def _otlp_exporter():
    """The OTLP/HTTP exporter; it reads the standard env vars itself."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    return OTLPSpanExporter()


def _new_provider(env: Mapping[str, str], exporter: Any):
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create(_resource_attributes(env)))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


def _build_provider(env: Mapping[str, str], exporter: Any = None):
    """A private provider; one per endpoint config for the whole process."""
    if exporter is not None:
        return _new_provider(env, exporter)
    key = tuple(env.get(name) for name in _KEY_ENV)
    with _PROVIDERS_LOCK:
        if key not in _PROVIDERS:
            _PROVIDERS[key] = _new_provider(env, _otlp_exporter())
        return _PROVIDERS[key]


def _guarded(method: Callable) -> Callable:
    """Swallow every error: tracing never changes a run's outcome."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with suppress(Exception):
            return method(self, *args, **kwargs)

    return wrapper


def _root_context(trace_id: str, span_id: int):
    """A context whose remote parent pins every span to ``trace_id``."""
    from opentelemetry import trace

    parent = trace.SpanContext(
        trace_id=int(trace_id, 16),
        span_id=span_id,
        is_remote=True,
        trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
    )
    return trace.set_span_in_context(trace.NonRecordingSpan(parent))


def _serialized_name(serialized: Any) -> str | None:
    return serialized.get("name") if isinstance(serialized, dict) else None


def _usage_of(generation: Any) -> dict:
    message = getattr(generation, "message", None)
    return getattr(message, "usage_metadata", None) or {}


def _token_usage(response: Any) -> dict[str, int]:
    """Input/output token attributes summed over the response, or ``{}``."""
    usages = [_usage_of(gen) for gens in response.generations for gen in gens]
    if not any(usages):
        return {}
    return {
        "gen_ai.usage.input_tokens": sum(u.get("input_tokens", 0) for u in usages),
        "gen_ai.usage.output_tokens": sum(u.get("output_tokens", 0) for u in usages),
    }


class K0OtelHandler(BaseCallbackHandler):
    """LangChain run events as OTel spans on one trace id.

    Parents are explicit (run id → context), never the ambient "current
    span". Chains hidden by LangGraph are skipped and their children
    attach to the nearest visible ancestor.
    """

    run_inline = True
    raise_error = False

    def __init__(self, tracer: Any, trace_id: str, secrets: list[str]) -> None:
        self._tracer = tracer
        self._secrets = secrets
        self._root = _root_context(trace_id, random.getrandbits(63) + 1)
        self._spans: dict[Any, Any] = {}
        self._contexts: dict[Any, Any] = {}

    def record(self, name: str, attributes: dict) -> None:
        """One instant span on the trace (the run-outcome marker)."""
        self._tracer.start_span(
            name, context=self._root, attributes=self._clean(attributes)
        ).end()

    @_guarded
    def on_chain_start(
        self, serialized, inputs, *, run_id, parent_run_id=None, tags=None,
        metadata=None, **kwargs,
    ):
        if HIDDEN_TAG in (tags or ()):
            self._contexts[run_id] = self._parent_context(parent_run_id)
            return
        metadata = metadata or {}
        name = kwargs.get("name") or _serialized_name(serialized) or "chain"
        attributes = {
            "k0.agent": metadata.get("k0_agent"),
            "langgraph.node": metadata.get("langgraph_node"),
        }
        self._start(run_id, parent_run_id, name, attributes)

    @_guarded
    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None,
        metadata=None, **kwargs,
    ):
        self._llm_start(run_id, parent_run_id, metadata, kwargs)

    @_guarded
    def on_llm_start(
        self, serialized, prompts, *, run_id, parent_run_id=None,
        metadata=None, **kwargs,
    ):
        self._llm_start(run_id, parent_run_id, metadata, kwargs)

    @_guarded
    def on_tool_start(
        self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs
    ):
        tool = kwargs.get("name") or _serialized_name(serialized) or "tool"
        attributes = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tool}
        self._start(run_id, parent_run_id, f"execute_tool {tool}", attributes)

    @_guarded
    def on_llm_end(self, response, *, run_id, **kwargs):
        span = self._spans.get(run_id)
        usage = _token_usage(response)
        if span is not None and usage:
            span.set_attributes(usage)
        self._end(run_id)

    @_guarded
    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._end(run_id)

    @_guarded
    def on_tool_end(self, output, *, run_id, **kwargs):
        self._end(run_id)

    @_guarded
    def on_chain_error(self, error, *, run_id, **kwargs):
        self._fail(run_id, error)

    @_guarded
    def on_llm_error(self, error, *, run_id, **kwargs):
        self._fail(run_id, error)

    @_guarded
    def on_tool_error(self, error, *, run_id, **kwargs):
        self._fail(run_id, error)

    def _llm_start(self, run_id, parent_run_id, metadata, kwargs) -> None:
        params = kwargs.get("invocation_params") or {}
        metadata = metadata or {}
        model = (
            params.get("model") or params.get("model_name")
            or metadata.get("ls_model_name") or "unknown"
        )
        attributes = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": model,
            "gen_ai.provider.name": metadata.get("ls_provider"),
        }
        self._start(run_id, parent_run_id, f"chat {model}", attributes)

    def _parent_context(self, parent_run_id):
        return self._contexts.get(parent_run_id, self._root)

    def _start(self, run_id, parent_run_id, name: str, attributes: dict) -> None:
        from opentelemetry import trace

        span = self._tracer.start_span(
            self._mask(name),
            context=self._parent_context(parent_run_id),
            attributes=self._clean(attributes),
        )
        self._spans[run_id] = span
        self._contexts[run_id] = trace.set_span_in_context(span)

    def _end(self, run_id) -> None:
        self._contexts.pop(run_id, None)
        span = self._spans.pop(run_id, None)
        if span is not None:
            span.end()

    def _fail(self, run_id, error: BaseException) -> None:
        from opentelemetry.trace import Status, StatusCode

        span = self._spans.get(run_id)
        if span is not None:
            span.set_attribute("error.type", type(error).__name__)
            span.set_status(Status(StatusCode.ERROR, self._mask(str(error))))
        self._end(run_id)

    def _mask(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, MASKED)
        return text

    def _clean(self, attributes: dict) -> dict:
        return {
            key: self._mask(value) if isinstance(value, str) else value
            for key, value in attributes.items()
            if value is not None
        }


@dataclass
class OtelTrace:
    """The OTel side of one run's trace; ``handler`` is ``None`` when off."""

    trace_id: str
    handler: Any = None
    _provider: Any = None

    def flush(self) -> None:
        """Push queued spans to the endpoint; enabled-only, never raising."""
        if self._provider is None:
            return
        with suppress(Exception):
            self._provider.force_flush(FLUSH_TIMEOUT_MS)

    def finish(
        self,
        *,
        kind: str,
        outcome: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        **_: Any,
    ) -> None:
        """Mark the run's end with one span, then ``flush()``."""
        if self.handler is None:
            return
        attributes = {
            "k0.kind": kind,
            "k0.outcome": outcome,
            "k0.session_id": session_id,
        }
        for key, value in (metadata or {}).items():
            attributes["k0." + key.removeprefix("k0_")] = value
        with suppress(Exception):
            self.handler.record(f"k0 {kind} end", attributes)
        self.flush()


def build_otel_trace(
    trace_id: str,
    *,
    env: Mapping[str, str] | None = None,
    exporter: Any = None,
) -> OtelTrace:
    """The OTel side of a run's trace; off (no handler) unless enabled.

    Never raises: any failure (import, provider, handler) degrades to
    tracing off. ``exporter`` replaces the OTLP one (tests).
    """
    try:
        return _build_otel_trace(trace_id, env, exporter)
    except Exception:
        return OtelTrace(trace_id=trace_id)


def _build_otel_trace(trace_id: str, env, exporter) -> OtelTrace:
    source = os.environ if env is None else env
    if not otel_enabled(source):
        return OtelTrace(trace_id=trace_id)
    provider = _build_provider(source, exporter)
    tracer = provider.get_tracer("k0ntrol", __version__)
    handler = K0OtelHandler(tracer, trace_id, secret_env_values(source))
    return OtelTrace(trace_id=trace_id, handler=handler, _provider=provider)


@dataclass
class OtelRunTrace:
    """Langfuse's trace facade plus the OTel one, behind the same calls.

    ``handler`` stays Langfuse's, so existing callers work unchanged;
    ``otel_handler`` is the extra callback. Anything else asked of it
    (new facade methods) goes to the Langfuse trace.
    """

    trace_id: str
    langfuse: Any
    otel: OtelTrace

    @property
    def handler(self) -> Any:
        return self.langfuse.handler

    @property
    def otel_handler(self) -> Any:
        return self.otel.handler

    def flush(self) -> None:
        for part in (self.langfuse, self.otel):
            with suppress(Exception):
                part.flush()

    def finish(self, **kwargs: Any) -> None:
        for part in (self.langfuse, self.otel):
            with suppress(Exception):
                part.finish(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.langfuse, name)


def build_run_trace(
    trace_id: str,
    *args: Any,
    env: Mapping[str, str] | None = None,
    exporter: Any = None,
    **kwargs: Any,
):
    """The run's trace facade: Langfuse's, plus OTel when enabled.

    With ``OTEL_EXPORTER_OTLP_ENDPOINT`` unset this returns exactly what
    ``build_trace`` returns. Never raises.
    """
    base = build_trace(trace_id, *args, env=env, **kwargs)
    otel = build_otel_trace(trace_id, env=env, exporter=exporter)
    if otel.handler is None:
        return base
    return OtelRunTrace(trace_id=trace_id, langfuse=base, otel=otel)


def otel_callbacks(trace: Any) -> list:
    """The OTel callback a run's ``trace`` carries, as a list to append."""
    handler = getattr(trace, "otel_handler", None)
    return [] if handler is None else [handler]


def trace_callbacks(trace: Any) -> list:
    """Every tracing callback ``trace`` carries: Langfuse's, then OTel's.

    Append it after the always-on events handler.
    """
    langfuse = [] if trace.handler is None else [trace.handler]
    return langfuse + otel_callbacks(trace)
