"""Tests for the event log: writer, callback handler, progress renderer.

Contract: spec #33 "Events and Langfuse" and #16's resolution. The schema
table in the spec is authoritative.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel, LangSmithParams
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.callbacks import GraphCallbackHandler, GraphInterruptEvent
from langgraph.types import Interrupt

from k0ntrol.events import (
    EventWriter,
    EventWriteError,
    K0EventHandler,
    ProgressRenderer,
    ToolDeniedError,
    events_path,
    new_run_id,
    render_event,
)

RUN = "a" * 32


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def make_writer(tmp_path: Path, **kwargs) -> EventWriter:
    path = tmp_path / ".k0-mem" / "events.jsonl"
    path.parent.mkdir(parents=True)
    return EventWriter(path, run=RUN, **kwargs)


# --- EventWriter ---------------------------------------------------------


def test_run_start_line_carries_required_fields(tmp_path):
    writer = make_writer(tmp_path)
    writer.run_start(
        "full",
        work_item="rubjpDev/k0ntrol#37",
        branch="k0/37-event-log",
        version="0.1.0.dev0",
    )
    (event,) = read_events(tmp_path / ".k0-mem" / "events.jsonl")
    assert event["kind"] == "run_start"
    assert event["run"] == RUN
    assert "chat" not in event
    assert event["run_kind"] == "full"
    assert event["work_item"] == "rubjpDev/k0ntrol#37"
    assert event["branch"] == "k0/37-event-log"
    assert event["version"] == "0.1.0.dev0"
    ts = datetime.fromisoformat(event["ts"].replace("Z", "+00:00"))
    assert ts.tzinfo is not None
    assert "." in event["ts"]  # millisecond precision


def test_absent_fields_are_omitted_never_null(tmp_path):
    writer = make_writer(tmp_path)
    writer.run_start("fast", branch="main", version="0.1.0.dev0")
    writer.emit("model", agent="coder", ok=True, effort=None)
    events = read_events(tmp_path / ".k0-mem" / "events.jsonl")
    for event in events:
        assert all(value is not None for value in event.values())
    assert "work_item" not in events[0]
    assert "resumed" not in events[0]
    assert "effort" not in events[1]


def test_every_emit_appends_one_line(tmp_path):
    writer = make_writer(tmp_path)
    writer.gate("validator", "pass")
    writer.gate("spec", "waiting")
    writer.run_end("done", ms=1234)
    events = read_events(tmp_path / ".k0-mem" / "events.jsonl")
    assert [e["kind"] for e in events] == ["gate", "gate", "run_end"]
    assert events[0]["gate"] == "validator"
    assert events[0]["result"] == "pass"
    assert events[1]["result"] == "waiting"
    assert events[2]["outcome"] == "done"
    assert events[2]["ms"] == 1234


def test_chat_binding_writes_chat_not_run(tmp_path):
    path = tmp_path / ".k0-mem" / "events.jsonl"
    path.parent.mkdir(parents=True)
    writer = EventWriter(path, chat="b" * 32)
    writer.emit("model", agent="chatty", ok=True)
    (event,) = read_events(path)
    assert event["chat"] == "b" * 32
    assert "run" not in event


def test_writer_requires_exactly_one_of_run_or_chat(tmp_path):
    path = tmp_path / "events.jsonl"
    with pytest.raises(ValueError):
        EventWriter(path)
    with pytest.raises(ValueError):
        EventWriter(path, run=RUN, chat="b" * 32)


def test_err_masked_by_secret_env_values(tmp_path, monkeypatch):
    monkeypatch.setenv("ACME_API_KEY", "sk-acme-9f8e7d6c5b")
    writer = make_writer(tmp_path)
    writer.emit("model", ok=False, err="401 from sk-acme-9f8e7d6c5b key")
    (event,) = read_events(tmp_path / ".k0-mem" / "events.jsonl")
    assert "sk-acme-9f8e7d6c5b" not in event["err"]
    assert "***" in event["err"]


def test_err_cut_to_500_chars(tmp_path):
    writer = make_writer(tmp_path)
    writer.emit("tool", ok=False, err="x" * 900)
    (event,) = read_events(tmp_path / ".k0-mem" / "events.jsonl")
    assert event["err"] == "x" * 500


def test_failed_write_raises_event_write_error(tmp_path):
    path = tmp_path / "missing-dir" / "events.jsonl"
    writer = EventWriter(path, run=RUN)
    with pytest.raises(EventWriteError):
        writer.emit("run_start")


def test_concurrent_writes_produce_whole_lines(tmp_path):
    writer = make_writer(tmp_path)

    def emit_many(n: int) -> None:
        for i in range(50):
            writer.emit("gate", gate=f"g{n}", result=str(i))

    threads = [threading.Thread(target=emit_many, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    events = read_events(tmp_path / ".k0-mem" / "events.jsonl")
    assert len(events) == 200


def test_progress_sink_receives_each_written_event(tmp_path):
    seen: list[dict] = []
    writer = make_writer(tmp_path, progress=seen.append)
    writer.gate("validator", "pass")
    writer.run_end("done", ms=1)
    assert [e["kind"] for e in seen] == ["gate", "run_end"]


def test_progress_failure_does_not_fail_the_write(tmp_path):
    def broken(_event: dict) -> None:
        raise RuntimeError("progress boom")

    writer = make_writer(tmp_path, progress=broken)
    writer.emit("gate", gate="validator", result="pass")
    assert len(read_events(tmp_path / ".k0-mem" / "events.jsonl")) == 1


def test_new_run_id_is_32_hex():
    run = new_run_id()
    assert len(run) == 32
    int(run, 16)


def test_events_path_is_under_k0_mem():
    assert events_path(Path("/repo")) == Path("/repo/.k0-mem/events.jsonl")


# --- Progress renderer ----------------------------------------------------


def test_render_event_run_start():
    line = render_event(
        {
            "kind": "run_start",
            "run_kind": "full",
            "work_item": "o/r#1",
            "branch": "k0/1-x",
            "version": "0.1.0.dev0",
        }
    )
    assert line is not None
    assert "\n" not in line
    assert "full" in line and "o/r#1" in line and "k0/1-x" in line


def test_render_event_model_ok_and_error():
    ok_line = render_event(
        {
            "kind": "model",
            "agent": "coder",
            "provider": "anthropic",
            "model": "claude-sonnet-5-5",
            "ms": 100,
            "ok": True,
        }
    )
    err_line = render_event(
        {
            "kind": "model",
            "helper": "reflect",
            "model": "m",
            "ms": 5,
            "ok": False,
            "err": "boom",
        }
    )
    assert "coder" in ok_line and "claude-sonnet-5-5" in ok_line
    assert "reflect" in err_line and "boom" in err_line


def test_render_event_tool_denied():
    line = render_event(
        {
            "kind": "tool",
            "agent": "coder",
            "tool": "write_file",
            "path": "src/x.py",
            "ms": 3,
            "ok": False,
            "denied": "path_guard",
            "err": "outside root",
        }
    )
    assert "write_file" in line and "src/x.py" in line and "denied" in line


def test_render_event_gate_and_run_end():
    gate = render_event({"kind": "gate", "gate": "validator", "result": "fail"})
    end = render_event(
        {
            "kind": "run_end",
            "outcome": "stuck",
            "ms": 7,
            "reason": "model_call_limit",
            "node": "coder",
        }
    )
    assert "validator" in gate and "fail" in gate
    assert "stuck" in end and "model_call_limit" in end and "coder" in end


def test_render_event_skips_unknown_kinds():
    assert render_event({"kind": "mystery"}) is None
    assert render_event({"ts": "x"}) is None


def test_progress_renderer_writes_at_most_one_line():
    written: list[str] = []
    renderer = ProgressRenderer(written.append)
    renderer({"kind": "gate", "gate": "validator", "result": "pass"})
    renderer({"kind": "unknown-kind"})
    assert len(written) == 1
    assert "validator" in written[0]


# --- K0EventHandler --------------------------------------------------------

SECRET = "sk-test-0123456789abcdef"


class ScriptedChatModel(BaseChatModel):
    """Minimal scripted chat model for driving the real callback manager."""

    fail: bool = False
    cost: float | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _get_ls_params(self, stop=None, **kwargs) -> LangSmithParams:
        params = super()._get_ls_params(stop, **kwargs)
        params["ls_provider"] = "testprovider"
        params["ls_model_name"] = "fake-model"
        return params

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.fail:
            raise RuntimeError(f"provider blew up: {SECRET}")
        message = AIMessage(content="done")
        message.usage_metadata = {
            "input_tokens": 3,
            "output_tokens": 5,
            "total_tokens": 8,
        }
        if self.cost is not None:
            message.response_metadata["cost"] = self.cost
        return ChatResult(generations=[ChatGeneration(message=message)])


@tool
def read_file(path: str) -> str:
    """Read a file."""
    return "file contents"


@tool
def write_file(path: str, content: str) -> str:
    """Write a file. Always refused by the path guard in these tests."""
    raise ToolDeniedError("path_guard", "path outside root")


@tool
def boom(path: str) -> str:
    """A tool that fails without a refusal."""
    raise ValueError("kaboom")


def make_handler(tmp_path: Path, **kwargs) -> tuple[K0EventHandler, Path]:
    path = tmp_path / ".k0-mem" / "events.jsonl"
    path.parent.mkdir(parents=True)
    writer = EventWriter(path, run=RUN, **kwargs)
    return K0EventHandler(writer), path


def test_handler_is_a_graph_callback_handler(tmp_path):
    handler, _ = make_handler(tmp_path)
    assert isinstance(handler, GraphCallbackHandler)
    assert isinstance(handler, BaseCallbackHandler)
    assert handler.run_inline is True
    # A failed JSONL write must reach the run, which ends it `stuck`.
    assert handler.raise_error is True


def test_model_event_fields(tmp_path):
    handler, path = make_handler(tmp_path)
    ScriptedChatModel().invoke(
        "hi",
        config={
            "callbacks": [handler],
            "metadata": {
                "k0_agent": "coder",
                "k0_attempt": 2,
                "k0_effort": "medium",
            },
        },
    )
    (event,) = read_events(path)
    assert event["kind"] == "model"
    assert event["run"] == RUN
    assert event["agent"] == "coder"
    assert "helper" not in event
    assert event["provider"] == "testprovider"
    assert event["model"] == "fake-model"
    assert event["effort"] == "medium"
    assert event["attempt"] == 2
    assert event["tokens_in"] == 3
    assert event["tokens_out"] == 5
    assert event["ok"] is True
    assert isinstance(event["ms"], int)
    assert "err" not in event


def test_model_event_for_a_helper_call(tmp_path):
    handler, path = make_handler(tmp_path)
    ScriptedChatModel().invoke(
        "hi",
        config={
            "callbacks": [handler],
            "metadata": {"k0_helper": "bulk_read"},
        },
    )
    (event,) = read_events(path)
    assert event["helper"] == "bulk_read"
    assert "agent" not in event


def test_model_event_uses_langgraph_node_as_agent_fallback(tmp_path):
    handler, path = make_handler(tmp_path)
    ScriptedChatModel().invoke(
        "hi",
        config={
            "callbacks": [handler],
            "metadata": {"langgraph_node": "tester"},
        },
    )
    (event,) = read_events(path)
    assert event["agent"] == "tester"


def test_model_event_records_provider_cost(tmp_path):
    handler, path = make_handler(tmp_path)
    ScriptedChatModel(cost=0.0021).invoke(
        "hi", config={"callbacks": [handler], "metadata": {"k0_agent": "coder"}}
    )
    (event,) = read_events(path)
    assert event["cost_usd"] == 0.0021


def test_model_error_masks_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("TESTPROVIDER_API_KEY", SECRET)
    handler, path = make_handler(tmp_path)
    with pytest.raises(RuntimeError):
        ScriptedChatModel(fail=True).invoke(
            "hi", config={"callbacks": [handler], "metadata": {"k0_agent": "coder"}}
        )
    (event,) = read_events(path)
    assert event["ok"] is False
    assert SECRET not in event["err"]
    assert "***" in event["err"]


def test_tool_event_ok_with_path_only(tmp_path):
    handler, path = make_handler(tmp_path)
    read_file.invoke(
        {"path": "src/x.py"},
        config={"callbacks": [handler], "metadata": {"k0_agent": "coder"}},
    )
    (event,) = read_events(path)
    assert event["kind"] == "tool"
    assert event["agent"] == "coder"
    assert event["tool"] == "read_file"
    assert event["path"] == "src/x.py"
    assert event["ok"] is True
    assert "err" not in event
    # No tool arguments other than `path` may be logged.
    assert "file contents" not in json.dumps(event)


def test_tool_event_denied_on_path_guard_refusal(tmp_path):
    handler, path = make_handler(tmp_path)
    with pytest.raises(ToolDeniedError):
        write_file.invoke(
            {"path": "../escape.py", "content": "secret source"},
            config={"callbacks": [handler], "metadata": {"k0_agent": "coder"}},
        )
    (event,) = read_events(path)
    assert event["denied"] == "path_guard"
    assert event["ok"] is False
    assert "outside root" in event["err"]
    # The refused call's other arguments are never written.
    assert "secret source" not in json.dumps(event)


def test_tool_error_without_denial_has_no_denied(tmp_path):
    handler, path = make_handler(tmp_path)
    with pytest.raises(ValueError):
        boom.invoke(
            {"path": "src/x.py"},
            config={"callbacks": [handler], "metadata": {"k0_agent": "tester"}},
        )
    (event,) = read_events(path)
    assert event["ok"] is False
    assert "denied" not in event
    assert "kaboom" in event["err"]


def test_gate_waiting_from_interrupt_payload(tmp_path):
    handler, path = make_handler(tmp_path)
    handler.on_interrupt(
        GraphInterruptEvent(
            run_id=None,
            status="done",
            checkpoint_id="ckpt-1",
            checkpoint_ns=(),
            interrupts=(Interrupt(value={"gate": "spec"}),),
        )
    )
    (event,) = read_events(path)
    assert event == {
        "ts": event["ts"],
        "kind": "gate",
        "run": RUN,
        "gate": "spec",
        "result": "waiting",
    }


def test_interrupt_without_a_gate_payload_emits_nothing(tmp_path):
    handler, path = make_handler(tmp_path)
    handler.on_interrupt(
        GraphInterruptEvent(
            run_id=None,
            status="done",
            checkpoint_id="ckpt-1",
            checkpoint_ns=(),
            interrupts=(
                Interrupt(value={"question": "y/n?"}),
                Interrupt(value="bare"),
            ),
        )
    )
    assert not path.exists() or read_events(path) == []


def test_tool_denied_error_is_the_shared_refusal_contract():
    err = ToolDeniedError("allowlist", "tool not allowed")
    assert err.denied == "allowlist"
    assert str(err) == "tool not allowed"


def test_model_name_falls_back_to_llm_output_and_response_metadata(tmp_path):
    from uuid import uuid4

    from langchain_core.outputs import LLMResult

    handler, path = make_handler(tmp_path)
    run_a, run_b = uuid4(), uuid4()
    handler.on_chat_model_start({}, [], run_id=run_a, metadata={"k0_agent": "coder"})
    handler.on_llm_end(
        LLMResult(generations=[[]], llm_output={"model_name": "from-llm-output"}),
        run_id=run_a,
    )
    message = AIMessage(content="x")
    message.response_metadata["model_name"] = "from-response-metadata"
    handler.on_chat_model_start({}, [], run_id=run_b, metadata={"k0_agent": "coder"})
    handler.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=message)]]),
        run_id=run_b,
    )
    events = read_events(path)
    assert events[0]["model"] == "from-llm-output"
    assert events[1]["model"] == "from-response-metadata"


def test_model_error_without_a_start_still_logs_the_call(tmp_path):
    from uuid import uuid4

    handler, path = make_handler(tmp_path)
    handler.on_llm_error(RuntimeError("lost start"), run_id=uuid4())
    (event,) = read_events(path)
    assert event["kind"] == "model"
    assert event["ok"] is False
    assert event["ms"] == 0
    assert "lost start" in event["err"]


def test_plain_llm_start_also_produces_a_model_event(tmp_path):
    from uuid import uuid4

    from langchain_core.outputs import LLMResult

    handler, path = make_handler(tmp_path)
    run_id = uuid4()
    handler.on_llm_start({}, ["prompt"], run_id=run_id, metadata={"k0_agent": "spec"})
    handler.on_llm_end(LLMResult(generations=[[]]), run_id=run_id)
    (event,) = read_events(path)
    assert event["kind"] == "model"
    assert event["agent"] == "spec"
    assert event["ok"] is True


def test_render_variants(tmp_path):
    resumed = render_event(
        {"kind": "run_start", "run_kind": "full", "branch": "b", "resumed": True}
    )
    tool_ok = render_event(
        {"kind": "tool", "agent": "a", "tool": "grep", "ms": 1, "ok": True}
    )
    tool_err = render_event(
        {"kind": "tool", "agent": "a", "tool": "grep", "ms": 1, "ok": False, "err": "e"}
    )
    gate_redo = render_event(
        {"kind": "gate", "gate": "validator", "result": "fail", "redo": ["x.py"]}
    )
    usage = render_event(
        {
            "kind": "model",
            "agent": "coder",
            "model": "m",
            "ms": 1,
            "ok": True,
            "tokens_in": 3,
            "tokens_out": 5,
        }
    )
    assert "resumed" in resumed
    assert "ok" in tool_ok
    assert "failed" in tool_err
    assert "x.py" in gate_redo
    assert "3 in" in usage and "5 out" in usage
