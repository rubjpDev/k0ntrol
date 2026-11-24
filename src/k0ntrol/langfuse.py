"""Optional Langfuse tracing: the second, opt-in callback handler.

The handler exists only when ``LANGFUSE_PUBLIC_KEY`` and
``LANGFUSE_SECRET_KEY`` are both set and the ``langfuse`` extra imports
— otherwise it is never constructed (#49). The trace id is the run id
(the chat id for ``/ask``); tags carry the run kind and outcome;
metadata carries the branch, change URL and k0 version. Every call
into the SDK is guarded: a Langfuse error never changes a run's
outcome.
"""

import importlib
import os
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from k0ntrol.events import MASKED, secret_env_values

LANGFUSE_ENV_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")

# The only run-metadata keys allowed into a trace's metadata: the
# SDK would otherwise copy harness plumbing (thread_id, ls_integration,
# k0_*) into the trace alongside them.
TRACE_METADATA_KEYS = ("branch", "change_url", "k0_version")


def langfuse_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Both keys set *and* the ``langfuse`` extra importable."""
    source = os.environ if env is None else env
    if not all(source.get(key) for key in LANGFUSE_ENV_KEYS):
        return False
    try:
        importlib.import_module("langfuse.langchain")
    except ImportError:
        return False
    return True


def build_mask(env: Mapping[str, str] | None = None) -> Callable:
    """The SDK ``mask`` hook: every known secret env value → ``***``.

    Same rule the event handler uses (``secret_env_values``): exact
    values of ``*_KEY``/``*_TOKEN``/``*_SECRET`` vars long enough to be
    credentials, replaced anywhere in the payload.
    """
    secrets = secret_env_values(env)

    def mask(*, data: Any, **kwargs: Any) -> Any:
        return _mask_data(data, secrets)

    return mask


def _mask_data(data: Any, secrets: list[str]) -> Any:
    if isinstance(data, dict):
        return _mask_dict(data, secrets)
    if isinstance(data, (list, tuple)):
        return type(data)(_mask_data(item, secrets) for item in data)
    return _mask_leaf(data, secrets)


def _mask_dict(data: dict, secrets: list[str]) -> dict:
    return {key: _mask_data(value, secrets) for key, value in data.items()}


def _mask_leaf(data: Any, secrets: list[str]) -> Any:
    if not isinstance(data, str):
        return data
    for secret in secrets:
        data = data.replace(secret, MASKED)
    return data


@dataclass
class RunTrace:
    """One trace per run (or ``/ask`` chat) id.

    ``handler`` is the LangChain callback the run modules append to
    ``config["callbacks"]``; ``None`` when tracing is off, and then
    ``finish``/``flush`` are no-ops.
    """

    trace_id: str
    handler: Any = None
    _langfuse: Any = None

    def flush(self) -> None:
        """Flush queued spans to Langfuse; enabled-only, never raising."""
        if self._langfuse is None:
            return
        with suppress(Exception):
            self._langfuse.get_client().flush()

    def finish(
        self,
        *,
        kind: str,
        outcome: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Close the trace at run end, then ``flush()``.

        Emits one last observation on the trace carrying the session,
        the kind+outcome tags and the metadata — the only v5 SDK path
        that sets trace attributes after spans ran.
        """
        if self._langfuse is None:
            return
        with suppress(Exception):
            self._end_observation(kind, outcome, session_id, metadata)
        self.flush()

    def _end_observation(self, kind, outcome, session_id, metadata) -> None:
        attrs = {"tags": [kind, outcome] if outcome else [kind]}
        if session_id:
            attrs["session_id"] = session_id
        if metadata:
            attrs["metadata"] = metadata
        with self._langfuse.propagate_attributes(**attrs):
            observation = self._langfuse.get_client().start_observation(
                name=f"k0 {kind} end",
                trace_context={"trace_id": self.trace_id},
            )
            observation.end()


def build_trace(
    trace_id: str, *, env: Mapping[str, str] | None = None
) -> RunTrace:
    """The run's trace facade; a bare ``RunTrace`` when tracing is off.

    Never raises: any SDK failure (import, client init, handler
    construction) degrades to tracing off.
    """
    try:
        return _build_trace(trace_id, env)
    except Exception:
        return RunTrace(trace_id=trace_id)


def _build_trace(trace_id: str, env: Mapping[str, str] | None) -> RunTrace:
    if not langfuse_enabled(env):
        return RunTrace(trace_id=trace_id)
    langfuse = importlib.import_module("langfuse")
    langfuse.Langfuse(mask=build_mask(env))
    handler_cls = _handler_class()
    return RunTrace(
        trace_id=trace_id,
        handler=handler_cls(trace_context={"trace_id": trace_id}),
        _langfuse=langfuse,
    )


def _handler_class():
    """The SDK handler, narrowed to our trace-metadata allowlist."""
    handler_cls = importlib.import_module("langfuse.langchain").CallbackHandler

    class K0CallbackHandler(handler_cls):
        def _parse_langfuse_trace_attributes(self, *, metadata=None, tags=None):
            attributes = super()._parse_langfuse_trace_attributes(
                metadata=metadata, tags=tags
            )
            attributes["metadata"] = {
                key: value
                for key, value in (attributes.get("metadata") or {}).items()
                if key in TRACE_METADATA_KEYS
            }
            return attributes

    return K0CallbackHandler
