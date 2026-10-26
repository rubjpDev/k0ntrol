"""Shared fakes for provider and doctor tests: no network, no extras needed."""


class FakeChatModel:
    """Scripted stand-in for a ``BaseChatModel``.

    ``profile`` mirrors ``model.profile`` (a ``ModelProfile`` TypedDict or
    ``None``); ``bind_tools`` records and returns self so ``--live`` tests can
    drive ``invoke`` through the same object.
    """

    def __init__(self, model, model_provider, kwargs, profile=None, reply=None):
        self.model = model
        self.model_provider = model_provider
        self.kwargs = kwargs
        self.profile = profile
        self.reply = reply
        self.bound_tools = None
        self.bind_kwargs = None
        self.invocations = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        self.bind_kwargs = kwargs
        return self

    def invoke(self, messages):
        self.invocations.append(list(messages))
        return self.reply


class FakeReply:
    """The message a fake chat model returns; ``tool_calls`` like AIMessage."""

    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls or []
        self.content = content


def recording_init(calls, profiles=None, replies=None):
    """A fake ``init_chat_model`` recording every call.

    ``profiles`` maps ``model_provider`` (or ``"any"``) to a profile dict;
    ``replies`` maps ``model`` to the ``invoke`` return value.
    """

    def init(model, *, model_provider=None, **kwargs):
        calls.append({"model": model, "model_provider": model_provider, "kwargs": kwargs})
        profile = (profiles or {}).get(model_provider, (profiles or {}).get("any"))
        reply = (replies or {}).get(model)
        return FakeChatModel(model, model_provider, kwargs, profile=profile, reply=reply)

    return init


def cfg_with(agent_entries=None, helpers=None, providers=None):
    """A config dict in the 0.1.0 shape, filled with only what a test needs."""
    cfg = {"agents": {}, "helpers": {}}
    for name, entry in (agent_entries or {}).items():
        entry.setdefault("model_calls", 10)
        cfg["agents"][name] = entry
    cfg["helpers"] = helpers or {}
    if providers is not None:
        cfg["providers"] = providers
    return cfg
