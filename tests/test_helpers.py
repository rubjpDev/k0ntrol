"""Helpers: one-shot model calls configured under ``helpers:``.

A helper is not an agent: one prompt in, one answer out, no tools, no
model-call limit. Calls emit ``model`` events with ``helper`` set through
whatever ``K0EventHandler`` is in the call's callbacks — inherited inside
an agent's tool call, passed explicitly by harness code like ``k0 scan``.
"""

import json

import pytest
from langchain_core.messages import AIMessage

from fakes import FakeChatModel, cfg_with, recording_init
from k0ntrol.events import EventWriter, K0EventHandler
from k0ntrol.helpers import helper_invoker
from k0ntrol.tools.errors import ToolError


def _init(model):
    return lambda _model, *, model_provider=None, **kwargs: model


def _cfg():
    return cfg_with(helpers={"bulk_read": {"provider": "anthropic", "model": "m"}})


def test_helper_invoker_answers_with_the_configured_model():
    model = FakeChatModel(reply=AIMessage(content="the answer"))
    invoke = helper_invoker(_cfg(), "bulk_read", init=_init(model))

    assert invoke("what do the files say?") == "the answer"
    prompt_messages = model.invocations[0]
    assert prompt_messages[-1].content == "what do the files say?"


def test_helper_invoker_builds_the_model_lazily_once():
    calls = []
    init = recording_init(calls, replies={"m": AIMessage(content="ok")})
    invoke = helper_invoker(_cfg(), "bulk_read", init=init)
    assert calls == []

    invoke("one")
    invoke("two")

    assert len(calls) == 1


def test_helper_invoker_needs_a_helpers_entry():
    invoke = helper_invoker({}, "bulk_read", init=_init(FakeChatModel()))
    with pytest.raises(ToolError, match="helpers.bulk_read"):
        invoke("q")


def test_helper_call_emits_a_model_event_with_helper(tmp_path):
    events_path = tmp_path / "events.jsonl"
    writer = EventWriter(events_path, run="run-1", mask=[])
    invoke = helper_invoker(
        _cfg(),
        "bulk_read",
        init=_init(FakeChatModel(reply=AIMessage(content="ok"))),
        config={"callbacks": [K0EventHandler(writer)]},
    )

    invoke("prompt")

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert len(events) == 1
    event = events[0]
    assert event["kind"] == "model"
    assert event["helper"] == "bulk_read"
    assert event["ok"] is True
    assert "agent" not in event
    assert event["run"] == "run-1"


def test_helper_call_clears_agent_metadata_it_inherits(tmp_path):
    """Inside an agent's tool call the helper inherits the parent's metadata;
    the event must still say helper, never agent."""
    events_path = tmp_path / "events.jsonl"
    writer = EventWriter(events_path, chat="chat-1", mask=[])
    invoke = helper_invoker(
        _cfg(),
        "bulk_read",
        init=_init(FakeChatModel(reply=AIMessage(content="ok"))),
        config={
            "callbacks": [K0EventHandler(writer)],
            "metadata": {"k0_agent": "chatty", "k0_effort": "high", "k0_attempt": 2},
        },
    )

    invoke("prompt")

    (event,) = [
        json.loads(line) for line in events_path.read_text().splitlines()
    ]
    assert event["helper"] == "bulk_read"
    assert "agent" not in event
    assert "effort" not in event
    assert "attempt" not in event


def test_helper_call_keeps_the_helpers_own_effort(tmp_path):
    cfg = cfg_with(
        helpers={"bulk_read": {"provider": "anthropic", "model": "m", "effort": "low"}}
    )
    events_path = tmp_path / "events.jsonl"
    writer = EventWriter(events_path, run="run-1", mask=[])
    invoke = helper_invoker(
        cfg,
        "bulk_read",
        init=_init(FakeChatModel(reply=AIMessage(content="ok"))),
        config={"callbacks": [K0EventHandler(writer)]},
    )

    invoke("prompt")

    (event,) = [
        json.loads(line) for line in events_path.read_text().splitlines()
    ]
    assert event["effort"] == "low"
