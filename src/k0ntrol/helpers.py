"""Helpers: one-shot model calls configured under ``helpers:``.

A helper is not an agent: one prompt in, one answer out, no tools and no
model-call limit. ``helper_invoker`` returns a ``prompt -> text`` callable
for one helper's model, built lazily on first call so tool construction
stays free and a missing ``helpers.<name>`` entry fails the call, not the
build.

Calls emit ``model`` events with ``helper`` set (metadata key
``k0_helper``): inside an agent's tool call the invoker inherits the run's
callbacks and metadata, so the event lands in the same scope — the
inherited ``agent``/``attempt``/``effort`` keys are cleared or replaced
with the helper's own. Outside a graph (``k0 scan``, reflect) the caller
passes ``config={"callbacks": [K0EventHandler(...)]}`` itself.
"""

from collections.abc import Callable

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from k0ntrol.events import META_AGENT, META_ATTEMPT, META_EFFORT, META_HELPER
from k0ntrol.providers import build_chat_model
from k0ntrol.tools.errors import ToolError

Invoke = Callable[[str], str]


def helper_invoker(
    cfg: dict | None,
    name: str,
    *,
    init: Callable = init_chat_model,
    config: dict | None = None,
) -> Invoke:
    """A ``prompt -> answer`` one-shot call to helper ``name``'s model."""
    cache: dict = {}

    def invoke(prompt: str) -> str:
        if "model" not in cache:
            cache["entry"] = _helper_entry(cfg, name)
            cache["model"] = build_chat_model(cfg or {}, cache["entry"], init=init)
        reply = cache["model"].invoke(
            [HumanMessage(prompt)],
            config=_call_config(name, cache["entry"], config),
        )
        return _text(reply)

    return invoke


def _helper_entry(cfg: dict | None, name: str) -> dict:
    entry = (cfg or {}).get("helpers", {}).get(name)
    if not isinstance(entry, dict) or not entry.get("provider") or not entry.get("model"):
        raise ToolError(f"helpers.{name} is not configured")
    return entry


def _call_config(name: str, entry: dict, config: dict | None) -> dict:
    merged = dict(config or {})
    metadata = dict(merged.get("metadata") or {})
    metadata[META_AGENT] = None
    metadata[META_ATTEMPT] = None
    metadata[META_EFFORT] = entry.get("effort")
    metadata[META_HELPER] = name
    merged["metadata"] = metadata
    return merged


def _text(reply) -> str:
    """A reply's text, whether plain or content blocks."""
    content = getattr(reply, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""
