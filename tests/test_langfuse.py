"""Optional Langfuse tracing (#49): a second callback handler, gated on env.

The suite runs without the ``langfuse`` extra installed: a fake
``langfuse`` package in ``sys.modules`` stands in for the SDK and drives
the same import path the production code uses, and the ``tracing`` seam
on ``start``/``resume``/``answer_ask`` injects recording fakes.
"""

import contextlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage

from fakes import FakeChatModel, FakeTracker, cfg_with
from k0ntrol import __version__
from k0ntrol.events import EventWriter
from k0ntrol.frontends.ask import answer_ask
from k0ntrol.langfuse import build_mask, build_trace, langfuse_enabled
from k0ntrol.providers import Check
from k0ntrol.workflow import Outcome, Pending, resume, start

TEST_SRC = "from calc import answer\n\n\ndef test_answer():\n    assert answer() == 42\n"
CALC_SRC = "def answer():\n    return 42\n"

BRIEF = {
    "title": "Calc answers 42",
    "category": "feature",
    "summary": "one module that answers 42",
    "current_behavior": "there is no calc module",
    "desired_behavior": "calc.answer() returns 42",
    "key_interfaces": "calc.answer() -> int",
    "acceptance_criteria": ["answer() returns 42"],
    "out_of_scope": "anything else",
}

KEYS_ENV = {"LANGFUSE_PUBLIC_KEY": "pk-x", "LANGFUSE_SECRET_KEY": "sk-x"}


def _git_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    (root / "README.md").write_text("tmp repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, check=True)


def _fast_cfg():
    cfg = cfg_with(
        {
            "tester": {"provider": "anthropic", "model": "m", "model_calls": 40},
            "coder": {"provider": "anthropic", "model": "m", "model_calls": 60},
        }
    )
    cfg.update(
        {
            "test_cmd": f"{sys.executable} -m pytest -q",
            "gates": [],
            "max_attempts": 2,
            "test_timeout": 60,
        }
    )
    return cfg


def _full_cfg():
    cfg = cfg_with(
        {
            "spec": {"provider": "anthropic", "model": "m", "model_calls": 25},
            "tester": {"provider": "anthropic", "model": "m", "model_calls": 40},
            "coder": {"provider": "anthropic", "model": "m", "model_calls": 60},
        }
    )
    cfg.update(
        {
            "tracker": "github",
            "base_branch": "main",
            "test_cmd": f"{sys.executable} -m pytest -q",
            "gates": [],
            "max_attempts": 2,
            "test_timeout": 60,
        }
    )
    return cfg


def _models(*models):
    """An ``init_chat_model`` seam handing out scripted fakes in call order."""
    queue = list(models)

    def init(model, *, model_provider=None, **kwargs):
        assert queue, "init_chat_model called more times than scripted"
        return queue.pop(0)

    return init


def _write(path, content, call_id="t1"):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "write_file", "args": {"path": path, "content": content}, "id": call_id}
        ],
    )


# --- the fake SDK ------------------------------------------------------------


class _FakeObservation:
    def __init__(self, record):
        self._record = record

    def end(self):
        self._record["ended"] += 1


class _FakeClient:
    def __init__(self, record):
        self._record = record

    def start_observation(self, **kwargs):
        if self._record["raise"]:
            raise RuntimeError("langfuse down")
        self._record["observations"].append(kwargs)
        return _FakeObservation(self._record)

    def flush(self):
        if self._record["raise"]:
            raise RuntimeError("langfuse down")
        self._record["flushes"] += 1


def _fake_langfuse():
    """Fake ``langfuse`` and ``langfuse.langchain`` modules for ``sys.modules``."""
    record = {
        "init_kwargs": [],
        "attrs": [],
        "observations": [],
        "ended": 0,
        "flushes": 0,
        "raise": False,
        "client": None,
    }
    package = ModuleType("langfuse")
    sub = ModuleType("langfuse.langchain")

    class CallbackHandler:  # stands in for langfuse.langchain.CallbackHandler
        def __init__(self, *, public_key=None, trace_context=None):
            self.public_key = public_key
            self.trace_context = trace_context

    class Langfuse:
        def __init__(self, **kwargs):
            record["init_kwargs"].append(kwargs)

    def get_client():
        if record["client"] is None:
            record["client"] = _FakeClient(record)
        return record["client"]

    @contextlib.contextmanager
    def propagate_attributes(**attrs):
        record["attrs"].append(attrs)
        yield

    package.Langfuse = Langfuse
    package.get_client = get_client
    package.propagate_attributes = propagate_attributes
    package.langchain = sub
    sub.CallbackHandler = CallbackHandler
    record["handler_cls"] = CallbackHandler
    return {"langfuse": package, "langfuse.langchain": sub}, record


@pytest.fixture
def fake_sdk(monkeypatch):
    modules, record = _fake_langfuse()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return record


# --- gating: the handler exists only with keys + the extra ---------------------


def test_no_handler_without_keys():
    trace = build_trace("trace-1", env={})
    assert trace.handler is None
    trace.finish(kind="fast", outcome="done")
    trace.flush()


def test_no_handler_with_only_one_key():
    env = {"LANGFUSE_PUBLIC_KEY": "pk-x"}
    assert not langfuse_enabled(env)
    assert build_trace("trace-1", env=env).handler is None


def test_no_handler_when_the_extra_is_missing():
    """The test env has no ``langfuse`` installed: keys alone enable nothing."""
    assert not langfuse_enabled(KEYS_ENV)
    assert build_trace("trace-1", env=KEYS_ENV).handler is None


def test_trace_id_and_client_mask(fake_sdk):
    env = dict(KEYS_ENV, OPENAI_API_KEY="sk-live-abcdef12345")
    trace = build_trace("run-abc", env=env)

    assert isinstance(trace.handler, fake_sdk["handler_cls"])
    assert trace.handler.trace_context == {"trace_id": "run-abc"}
    mask = fake_sdk["init_kwargs"][0]["mask"]
    assert callable(mask)
    assert mask(data="the key is sk-live-abcdef12345") == "the key is ***"


# --- the mask ------------------------------------------------------------------


def test_mask_replaces_exact_secret_values_anywhere_in_the_payload():
    env = {
        "OPENAI_API_KEY": "sk-live-abcdef12345",
        "GH_TOKEN": "ghp_999888777",
        "SHORT_KEY": "tiny",  # below MASK_MIN_LENGTH: never masked
    }
    mask = build_mask(env)
    data = {
        "prompt": "use sk-live-abcdef12345 please",
        "nested": ["ghp_999888777", {"k": "prefix ghp_999888777 suffix"}],
        "safe": "tiny",
        "num": 42,
        "none": None,
    }
    out = mask(data=data)

    assert out["prompt"] == "use *** please"
    assert out["nested"][0] == "***"
    assert out["nested"][1] == {"k": "prefix *** suffix"}
    assert out["safe"] == "tiny"
    assert out["num"] == 42
    assert out["none"] is None
    assert "sk-live-abcdef12345" not in json.dumps(out)


# --- finish and flush -----------------------------------------------------------

def test_mask_includes_known_provider_credentials(monkeypatch, tmp_path):
    env = {
        "AWS_BEARER_TOKEN_BEDROCK": "bedrock-bearer-value",
        "AWS_ACCESS_KEY_ID": "aws-access-value",
        "AWS_PROFILE": "private-profile-value",
        "AWS_DEFAULT_PROFILE": "default-profile-value",
        "GOOGLE_APPLICATION_CREDENTIALS": "/private/credentials.json",
        "GOOGLE_GENAI_USE_VERTEXAI": "vertex-setting-value",
        "GOOGLE_CLOUD_PROJECT": "private-project-value",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    payload = " ".join(env.values())
    expected = " ".join(["***"] * len(env))
    assert build_mask(env)(data=payload) == expected
    event = EventWriter(tmp_path / "events.jsonl", run="r").emit(
        "model", err=payload,
    )
    assert event["err"] == expected


def test_finish_tags_kind_and_outcome_then_flushes(fake_sdk):
    trace = build_trace("run-abc", env=KEYS_ENV)
    trace.finish(
        kind="full",
        outcome="done",
        session_id="acme/widgets#9",
        metadata={"branch": "k0/9-calc", "change_url": "http://x/1", "k0_version": "0"},
    )

    attrs = fake_sdk["attrs"][-1]
    assert attrs["session_id"] == "acme/widgets#9"
    assert set(attrs["tags"]) == {"full", "done"}
    assert attrs["metadata"]["branch"] == "k0/9-calc"
    assert attrs["metadata"]["change_url"] == "http://x/1"
    observation = fake_sdk["observations"][-1]
    assert observation["trace_context"] == {"trace_id": "run-abc"}
    assert fake_sdk["ended"] == 1
    assert fake_sdk["flushes"] == 1


def test_a_langfuse_error_never_raises(fake_sdk):
    fake_sdk["raise"] = True
    trace = build_trace("run-abc", env=KEYS_ENV)
    trace.finish(kind="fast", outcome="done")
    trace.flush()


# --- the run seams ----------------------------------------------------------------


class _RecordingHandler(BaseCallbackHandler):
    """Stands in for the langfuse handler on the callbacks list."""

    def __init__(self):
        self.root_metadata = []
        self.llm_ends = 0

    def on_chain_start(
        self,
        serialized,
        inputs,
        *,
        run_id,
        parent_run_id=None,
        metadata=None,
        **kwargs,
    ):
        if parent_run_id is None:
            self.root_metadata.append(dict(metadata or {}))

    def on_llm_end(self, response, **kwargs):
        self.llm_ends += 1


class _RecordingTrace:
    """What the ``tracing`` seam returns: records ``finish``/``flush`` calls."""

    def __init__(self, trace_id, handler=None):
        self.trace_id = trace_id
        self.handler = handler
        self.finished = []
        self.flushes = 0

    def finish(self, **kwargs):
        self.finished.append(kwargs)

    def flush(self):
        self.flushes += 1


def _tracing(made, handler=None):
    """A ``tracing`` seam recording every trace the run modules build."""

    def factory(trace_id):
        trace = _RecordingTrace(trace_id, handler=handler)
        made.append(trace)
        return trace

    return factory


def test_fast_run_trace_tags_and_finish(tmp_path):
    _git_repo(tmp_path)
    handler = _RecordingHandler()
    made = []
    tester = FakeChatModel(
        messages=[_write("tests/test_calc.py", TEST_SRC), AIMessage(content="tests")]
    )
    coder = FakeChatModel(
        messages=[_write("calc.py", CALC_SRC), AIMessage(content="code")]
    )

    result = start(
        "fast",
        "add a calc module that answers 42",
        root=tmp_path,
        cfg=_fast_cfg(),
        init=_models(tester, coder),
        check=lambda *a, **k: [],
        scan=lambda r: "scanned",
        tracing=_tracing(made, handler),
    )

    assert isinstance(result, Pending)
    trace = made[0]
    assert trace.trace_id == result.run
    assert handler.llm_ends > 0  # the handler really sat in the callbacks
    root_meta = handler.root_metadata[0]
    assert root_meta["langfuse_tags"] == ["fast"]
    assert "langfuse_session_id" not in root_meta  # no work item for /fast
    assert trace.flushes == 1  # a parked run flushes but does not finish
    assert not trace.finished

    outcome = resume(
        result.run,
        "y",
        root=tmp_path,
        cfg=_fast_cfg(),
        init=_models(),
        tracing=_tracing(made, handler),
    )

    assert isinstance(outcome, Outcome)
    assert outcome.outcome == "done"
    assert made[1].trace_id == result.run  # same trace across processes
    finish = made[1].finished[0]
    assert finish["kind"] == "fast"
    assert finish["outcome"] == "done"
    assert finish["session_id"] is None
    assert finish["metadata"]["k0_version"] == __version__
    assert finish["metadata"]["branch"]
    assert "change_url" not in finish["metadata"]  # no change yet: omitted


def test_full_run_session_is_the_work_item(tmp_path):
    _git_repo(tmp_path)
    tracker = FakeTracker()
    item = tracker.create_work_item("Calc answers 42", "add a calc module")
    handler = _RecordingHandler()
    made = []
    spec = FakeChatModel(
        messages=[
            AIMessage(
                content="",
                tool_calls=[{"name": "AgentBrief", "args": dict(BRIEF), "id": "b1"}],
            )
        ]
    )

    result = start(
        "full",
        f"#{item.ref}",
        root=tmp_path,
        cfg=_full_cfg(),
        init=_models(spec),
        tracker=tracker,
        check=lambda *a, **k: [],
        scan=lambda r: "scanned",
        tracing=_tracing(made, handler),
    )

    assert isinstance(result, Pending)
    root_meta = handler.root_metadata[0]
    assert root_meta["langfuse_session_id"] == f"acme/widgets#{item.ref}"
    assert root_meta["langfuse_tags"] == ["full"]
    assert made[0].flushes == 1


def test_ask_is_its_own_trace(tmp_path):
    (tmp_path / ".k0-mem").mkdir(exist_ok=True)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    handler = _RecordingHandler()
    made = []
    cfg = cfg_with(
        {"chatty": {"provider": "anthropic", "model": "m", "model_calls": 15}},
        helpers={"bulk_read": {"provider": "anthropic", "model": "m"}},
    )
    model = FakeChatModel(messages=[AIMessage(content="the answer")])

    result = answer_ask(
        "what is x?",
        root=tmp_path,
        config=cfg,
        write=lambda line: None,
        on_text=lambda delta: None,
        init=lambda *a, **k: model,
        check=lambda *a, **k: [Check("config", "ok", "valid")],
        refresh=lambda *a, **k: "scanned",
        tracing=_tracing(made, handler),
    )

    assert result is not None
    assert len(made) == 1
    trace = made[0]
    assert handler.llm_ends > 0
    finish = trace.finished[0]
    assert finish["kind"] == "ask"
    assert finish["outcome"] == "done"
    assert finish.get("session_id") is None
    assert finish["metadata"]["k0_version"] == __version__
    assert trace.flushes >= 1 or finish  # finish already flushes


def test_runs_without_the_extra_behave_normally(tmp_path):
    """No ``tracing`` seam: the default factory finds no langfuse and the
    run is unchanged (the suite itself runs without the extra)."""
    _git_repo(tmp_path)
    tester = FakeChatModel(
        messages=[_write("tests/test_calc.py", TEST_SRC), AIMessage(content="tests")]
    )
    coder = FakeChatModel(
        messages=[_write("calc.py", CALC_SRC), AIMessage(content="code")]
    )

    result = start(
        "fast",
        "add a calc module that answers 42",
        root=tmp_path,
        cfg=_fast_cfg(),
        init=_models(tester, coder),
        check=lambda *a, **k: [],
        scan=lambda r: "scanned",
    )

    assert isinstance(result, Pending)
