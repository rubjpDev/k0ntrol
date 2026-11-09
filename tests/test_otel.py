"""Optional plain OpenTelemetry export: a second, opt-in callback handler.

The off path is tested without the ``otel`` packages. The on path needs
them (the ``otel`` or ``langfuse`` extra) and skips cleanly when they are
missing; it drives an in-memory span exporter instead of a network one.
"""

import subprocess
import sys
import textwrap
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from fakes import STUB_TEST_CMD, FakeChatModel, cfg_with
from k0ntrol.events import MASKED
from k0ntrol.frontends.ask import answer_ask
from k0ntrol.otel import (
    build_otel_trace,
    build_run_trace,
    otel_enabled,
    otel_callbacks,
)
from k0ntrol.providers import Check
from k0ntrol.workflow import Pending, start

RUN = "0123456789abcdef0123456789abcdef"
ENV = {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318"}


def _installed() -> bool:
    try:
        import opentelemetry.exporter.otlp.proto.http.trace_exporter  # noqa: F401
        import opentelemetry.sdk.trace  # noqa: F401
    except ImportError:
        return False
    return True


needs_otel = pytest.mark.skipif(
    not _installed(), reason="the otel (or langfuse) extra is not installed"
)


@pytest.fixture(autouse=True)
def _fresh_providers(monkeypatch):
    """Each test builds its own providers instead of reusing a cached one."""
    monkeypatch.setattr("k0ntrol.otel._PROVIDERS", {})


def _memory():
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    return InMemorySpanExporter()


def _spans(trace, exporter):
    trace.flush()
    return {span.name: span for span in exporter.get_finished_spans()}


# --- the off path (runs without the packages) -----------------------------


def test_disabled_without_the_endpoint():
    assert otel_enabled({}) is False
    assert otel_enabled({"OTEL_EXPORTER_OTLP_HEADERS": "a=b"}) is False


def test_disabled_by_the_sdk_kill_switch():
    assert otel_enabled({**ENV, "OTEL_SDK_DISABLED": "true"}) is False


def test_disabled_when_the_packages_are_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", None)
    assert otel_enabled(ENV) is False


@needs_otel
def test_enabled_with_the_endpoint_and_the_packages():
    assert otel_enabled(ENV) is True


def test_off_builds_a_trace_without_a_handler():
    trace = build_otel_trace(RUN, env={})
    assert trace.handler is None
    trace.finish(kind="fast", outcome="done")  # no-ops, no raise
    trace.flush()


def test_off_run_trace_is_the_langfuse_one_unchanged(monkeypatch):
    sentinel = SimpleNamespace(handler=None)
    monkeypatch.setattr("k0ntrol.otel.build_trace", lambda *a, **k: sentinel)
    assert build_run_trace(RUN, env={}) is sentinel


def test_off_imports_and_builds_nothing():
    code = textwrap.dedent(
        """
        import sys
        from k0ntrol.otel import build_run_trace
        before = set(sys.modules)  # langsmith may already pull opentelemetry in
        trace = build_run_trace("0123456789abcdef0123456789abcdef", env={})
        trace.finish(kind="fast", outcome="done")
        trace.flush()
        added = set(sys.modules) - before
        assert not added, sorted(added)
        """
    )
    env = {"PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_a_failing_exporter_degrades_to_off(monkeypatch):
    monkeypatch.setattr("k0ntrol.otel.otel_enabled", lambda env=None: True)

    def boom(*args, **kwargs):
        raise RuntimeError("no sdk")

    monkeypatch.setattr("k0ntrol.otel._build_provider", boom)
    trace = build_otel_trace(RUN, env=ENV)
    assert trace.handler is None


def test_otel_callbacks_is_empty_without_a_handler():
    assert otel_callbacks(None) == []
    assert otel_callbacks(SimpleNamespace(handler="lf")) == []
    assert otel_callbacks(SimpleNamespace(otel_handler=None)) == []
    handler = BaseCallbackHandler()
    assert otel_callbacks(SimpleNamespace(otel_handler=handler)) == [handler]


# --- the on path ----------------------------------------------------------


@needs_otel
def test_handler_is_a_callback_handler_that_never_raises():
    trace = build_otel_trace(RUN, env=ENV, exporter=_memory())
    assert isinstance(trace.handler, BaseCallbackHandler)
    assert trace.handler.raise_error is False
    assert trace.handler.run_inline is True


@needs_otel
def test_chain_llm_and_tool_spans_share_the_run_id_as_trace_id():
    exporter = _memory()
    trace = build_otel_trace(RUN, env=ENV, exporter=exporter)
    handler = trace.handler
    root, node, llm, tool = (uuid4() for _ in range(4))

    handler.on_chain_start(
        {}, {}, run_id=root, parent_run_id=None, name="LangGraph", metadata={}
    )
    handler.on_chain_start(
        {},
        {},
        run_id=node,
        parent_run_id=root,
        name="coder",
        metadata={"k0_agent": "coder"},
    )
    handler.on_chat_model_start(
        {},
        [[]],
        run_id=llm,
        parent_run_id=node,
        invocation_params={"model": "claude-x"},
        metadata={"ls_provider": "anthropic"},
    )
    message = AIMessage(
        content="hi", usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    )
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=llm)
    handler.on_tool_start({"name": "write_file"}, "{}", run_id=tool, parent_run_id=node)
    handler.on_tool_end("ok", run_id=tool)
    handler.on_chain_end({}, run_id=node)
    handler.on_chain_end({}, run_id=root)

    spans = _spans(trace, exporter)
    assert set(spans) == {
        "LangGraph",
        "coder",
        "chat claude-x",
        "execute_tool write_file",
    }
    assert {s.context.trace_id for s in spans.values()} == {int(RUN, 16)}
    assert spans["coder"].parent.span_id == spans["LangGraph"].context.span_id
    assert spans["chat claude-x"].parent.span_id == spans["coder"].context.span_id
    assert spans["execute_tool write_file"].parent.span_id == spans["coder"].context.span_id
    assert spans["coder"].attributes["k0.agent"] == "coder"
    llm_attrs = spans["chat claude-x"].attributes
    assert llm_attrs["gen_ai.operation.name"] == "chat"
    assert llm_attrs["gen_ai.request.model"] == "claude-x"
    assert llm_attrs["gen_ai.provider.name"] == "anthropic"
    assert llm_attrs["gen_ai.usage.input_tokens"] == 7
    assert llm_attrs["gen_ai.usage.output_tokens"] == 3
    assert spans["execute_tool write_file"].attributes["gen_ai.tool.name"] == "write_file"


@needs_otel
def test_internal_langgraph_chains_are_skipped():
    exporter = _memory()
    trace = build_otel_trace(RUN, env=ENV, exporter=exporter)
    hidden = uuid4()
    trace.handler.on_chain_start(
        {}, {}, run_id=hidden, parent_run_id=None, name="__start__", tags=["langsmith:hidden"]
    )
    trace.handler.on_chain_end({}, run_id=hidden)
    assert _spans(trace, exporter) == {}


@needs_otel
def test_errors_set_the_status_and_mask_secrets():
    exporter = _memory()
    secret = "sk-very-secret-value-123"
    env = {**ENV, "SOME_API_KEY": secret}
    trace = build_otel_trace(RUN, env=env, exporter=exporter)
    llm, tool = uuid4(), uuid4()
    trace.handler.on_llm_start({}, ["p"], run_id=llm, parent_run_id=None, invocation_params={"model": "m"})
    trace.handler.on_llm_error(RuntimeError(f"401 for {secret}"), run_id=llm)
    trace.handler.on_tool_start({"name": "t"}, "x", run_id=tool, parent_run_id=None)
    trace.handler.on_tool_error(ValueError("bad"), run_id=tool)

    spans = _spans(trace, exporter)
    failed = spans["chat m"]
    assert failed.status.status_code.name == "ERROR"
    assert MASKED in failed.status.description
    assert secret not in failed.status.description
    assert secret not in str(failed.events)
    assert spans["execute_tool t"].status.status_code.name == "ERROR"


@needs_otel
def test_no_prompt_or_tool_content_is_exported():
    exporter = _memory()
    trace = build_otel_trace(RUN, env=ENV, exporter=exporter)
    tool = uuid4()
    trace.handler.on_tool_start({"name": "t"}, "SECRET-INPUT", run_id=tool, parent_run_id=None)
    trace.handler.on_tool_end("SECRET-OUTPUT", run_id=tool)
    attrs = _spans(trace, exporter)["execute_tool t"].attributes
    assert "SECRET" not in str(dict(attrs))


@needs_otel
def test_a_sdk_error_inside_a_callback_never_raises(monkeypatch):
    trace = build_otel_trace(RUN, env=ENV, exporter=_memory())
    handler = trace.handler
    monkeypatch.setattr(handler, "_tracer", None)  # any use now raises
    handler.on_chain_start({}, {}, run_id=uuid4(), parent_run_id=None, name="x")
    handler.on_llm_end(LLMResult(generations=[[]]), run_id=uuid4())
    handler.on_tool_error(RuntimeError("x"), run_id=uuid4())


@needs_otel
def test_finish_emits_the_run_outcome_span_and_flushes():
    exporter = _memory()
    trace = build_otel_trace(RUN, env=ENV, exporter=exporter)
    trace.finish(
        kind="fast",
        outcome="done",
        session_id="acme/widgets#4",
        metadata={"branch": "k0/4", "k0_version": "1.2.3"},
    )
    # no explicit flush(): finish already pushed the span to the exporter
    span = {s.name: s for s in exporter.get_finished_spans()}["k0 fast end"]
    assert span.context.trace_id == int(RUN, 16)
    assert span.attributes["k0.kind"] == "fast"
    assert span.attributes["k0.outcome"] == "done"
    assert span.attributes["k0.session_id"] == "acme/widgets#4"
    assert span.attributes["k0.branch"] == "k0/4"
    assert span.attributes["k0.version"] == "1.2.3"


@needs_otel
def test_flush_and_finish_never_raise(monkeypatch):
    trace = build_otel_trace(RUN, env=ENV, exporter=_memory())

    def boom(*args, **kwargs):
        raise RuntimeError("collector down")

    monkeypatch.setattr(trace._provider, "force_flush", boom)
    trace.flush()
    trace.finish(kind="full", outcome="stuck")


@needs_otel
def test_the_provider_is_private_not_the_global_one():
    from opentelemetry import trace as otel_api

    before = otel_api.get_tracer_provider()
    build_otel_trace(RUN, env=ENV, exporter=_memory())
    assert otel_api.get_tracer_provider() is before


@needs_otel
def test_the_service_name_defaults_to_k0_and_yields_to_the_env(monkeypatch):
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    exporter = _memory()
    trace = build_otel_trace(RUN, env=ENV, exporter=exporter)
    trace.finish(kind="fast")
    resource = exporter.get_finished_spans()[0].resource
    assert resource.attributes["service.name"] == "k0"

    monkeypatch.setenv("OTEL_SERVICE_NAME", "mine")  # the SDK reads os.environ
    exporter = _memory()
    trace = build_otel_trace(RUN, env={**ENV, "OTEL_SERVICE_NAME": "mine"}, exporter=exporter)
    trace.finish(kind="fast")
    assert exporter.get_finished_spans()[0].resource.attributes["service.name"] == "mine"


@needs_otel
def test_the_real_otlp_exporter_is_built_from_the_standard_env(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-key=abc")
    trace = build_otel_trace(RUN, env=ENV)
    assert trace.handler is not None
    assert trace._provider is not None
    trace._provider.shutdown()


@needs_otel
def test_one_provider_is_reused_across_runs_for_the_same_endpoint(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
    first = build_otel_trace(RUN, env=ENV)
    second = build_otel_trace("f" * 32, env=ENV)
    assert first._provider is second._provider
    first._provider.shutdown()


@needs_otel
def test_run_trace_fans_out_to_langfuse_and_otel(monkeypatch):
    calls = []
    lf = SimpleNamespace(
        handler="lf-handler",
        finish=lambda **kw: calls.append(("lf.finish", kw["kind"])),
        flush=lambda: calls.append(("lf.flush",)),
    )
    monkeypatch.setattr("k0ntrol.otel.build_trace", lambda *a, **k: lf)
    exporter = _memory()
    trace = build_run_trace(RUN, env=ENV, exporter=exporter)

    assert trace.handler == "lf-handler"
    assert isinstance(trace.otel_handler, BaseCallbackHandler)
    assert otel_callbacks(trace) == [trace.otel_handler]
    trace.finish(kind="fast", outcome="done")
    trace.flush()
    assert ("lf.finish", "fast") in calls and ("lf.flush",) in calls
    assert "k0 fast end" in {s.name for s in exporter.get_finished_spans()}


@needs_otel
def test_one_side_failing_does_not_stop_the_other(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("langfuse down")

    lf = SimpleNamespace(handler=None, finish=boom, flush=lambda: boom())
    monkeypatch.setattr("k0ntrol.otel.build_trace", lambda *a, **k: lf)
    exporter = _memory()
    trace = build_run_trace(RUN, env=ENV, exporter=exporter)
    trace.finish(kind="fast", outcome="done")
    trace.flush()
    assert "k0 fast end" in {s.name for s in exporter.get_finished_spans()}


@needs_otel
def test_an_ask_exports_spans_under_the_chat_id(tmp_path):
    (tmp_path / ".k0-mem").mkdir(exist_ok=True)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    exporter = _memory()
    made = []

    def tracing(trace_id):
        trace = build_run_trace(trace_id, env=ENV, exporter=exporter)
        made.append(trace)
        return trace

    cfg = cfg_with(
        {"chatty": {"provider": "anthropic", "model": "m", "model_calls": 15}},
        helpers={"bulk_read": {"provider": "anthropic", "model": "m"}},
    )
    result = answer_ask(
        "what is x?",
        root=tmp_path,
        config=cfg,
        write=lambda line: None,
        on_text=lambda delta: None,
        init=lambda *a, **k: FakeChatModel(messages=[AIMessage(content="the answer")]),
        check=lambda *a, **k: [Check("config", "ok", "valid")],
        refresh=lambda *a, **k: "scanned",
        tracing=tracing,
    )

    assert result is not None
    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert "k0 ask end" in names
    assert any(name.startswith("chat") for name in names)
    assert {s.context.trace_id for s in spans} == {int(made[0].trace_id, 16)}


def _repo(root):
    def sh(*args):
        subprocess.run(args, cwd=root, check=True, capture_output=True)

    sh("git", "init", "-q", "-b", "main")
    sh("git", "config", "user.email", "t@t")
    sh("git", "config", "user.name", "t")
    (root / "README.md").write_text("tmp repo\n", encoding="utf-8")
    sh("git", "add", "-A")
    sh("git", "commit", "-qm", "init")


def _write(path, content):
    call = {"name": "write_file", "args": {"path": path, "content": content}, "id": "t1"}
    return AIMessage(content="", tool_calls=[call])


@needs_otel
def test_a_fast_run_exports_agent_model_and_tool_spans(tmp_path):
    _repo(tmp_path)
    exporter = _memory()
    test_src = "from calc import answer\n\n\ndef test_answer():\n    assert answer() == 42\n"
    tester = FakeChatModel(
        messages=[_write("tests/test_calc.py", test_src), AIMessage(content="tests")]
    )
    coder = FakeChatModel(
        messages=[_write("calc.py", "def answer():\n    return 42\n"), AIMessage(content="code")]
    )
    queue = [tester, coder]
    cfg = cfg_with(
        {
            "tester": {"provider": "anthropic", "model": "m", "model_calls": 40},
            "coder": {"provider": "anthropic", "model": "m", "model_calls": 60},
        }
    )
    cfg.update({"test_cmd": STUB_TEST_CMD, "gates": [], "max_attempts": 2, "test_timeout": 60})

    result = start(
        "fast",
        "add a calc module that answers 42",
        root=tmp_path,
        cfg=cfg,
        init=lambda *a, **k: queue.pop(0),
        check=lambda *a, **k: [],
        scan=lambda r: "scanned",
        tracing=lambda trace_id: build_run_trace(trace_id, env=ENV, exporter=exporter),
    )

    assert isinstance(result, Pending)
    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert {s.context.trace_id for s in spans} == {int(result.run, 16)}
    # the run paused at a gate: its spans were flushed there, no end marker yet
    assert any(n.startswith("chat") for n in names)
    assert "execute_tool write_file" in names
    assert {s.attributes.get("k0.agent") for s in spans} >= {"tester", "coder"}
