"""/ask answers questions with CHATTY: doctor gate, streamed answer,
@path mentions as relative paths, the model-call limit line."""

import json
import subprocess

import pytest
from langchain_core.messages import AIMessage

from fakes import FakeChatModel, cfg_with
from k0ntrol.frontends.ask import answer_ask
from k0ntrol.providers import Check


def _repo(root, files=None):
    (root / ".k0-mem").mkdir(exist_ok=True)
    for name, text in (files or {}).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)


def _cfg(model_calls=15):
    return cfg_with(
        {"chatty": {"provider": "anthropic", "model": "m", "model_calls": model_calls}}
    )


def _init(model):
    return lambda _model, *, model_provider=None, **kwargs: model


def _ok_checks(cfg, **_kw):
    return [Check("config", "ok", "valid")]


def _events(root):
    path = root / ".k0-mem" / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _read_event_shown(shown):
    return "\n".join(shown)


# --- doctor gate --------------------------------------------------------------


def test_ask_refuses_when_doctor_checks_error(tmp_path):
    _repo(tmp_path)
    shown, deltas = [], []

    def bad_checks(cfg, **_kw):
        return [Check("provider anthropic", "error", "missing credential")]

    result = answer_ask(
        "what is x?",
        root=tmp_path,
        config=_cfg(),
        write=shown.append,
        on_text=deltas.append,
        init=_init(FakeChatModel(messages=[])),
        check=bad_checks,
    )

    assert result is None
    assert not deltas
    text = _read_event_shown(shown)
    assert "missing credential" in text
    assert not _events(tmp_path)


# --- answering -----------------------------------------------------------------


def test_ask_streams_the_answer(tmp_path):
    _repo(tmp_path, {"note.txt": "the sky is blue\n"})
    model = FakeChatModel(messages=[AIMessage(content="The note says the sky is blue.")])
    shown, deltas = [], []

    result = answer_ask(
        "what does the note say?",
        root=tmp_path,
        config=_cfg(),
        write=shown.append,
        on_text=deltas.append,
        init=_init(model),
        check=_ok_checks,
    )

    assert result is not None and not result.limit_hit
    assert "".join(deltas) == "The note says the sky is blue."
    assert "The note says the sky is blue." == result.text


def test_ask_writes_chat_scoped_events_and_progress(tmp_path):
    _repo(tmp_path, {"note.txt": "hi\n"})
    model = FakeChatModel(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_file", "args": {"path": "note.txt"}, "id": "t1"}
                ],
            ),
            AIMessage(content="hi is the note"),
        ]
    )
    shown, deltas = [], []

    answer_ask(
        "what?",
        root=tmp_path,
        config=_cfg(),
        write=shown.append,
        on_text=deltas.append,
        init=_init(model),
        check=_ok_checks,
    )

    events = _events(tmp_path)
    assert events and all("chat" in e and "run" not in e for e in events)
    assert [e["kind"] for e in events] == ["model", "tool", "model"]
    # one progress line per event reaches the user
    progress = [line for line in shown if line.startswith(("model ", "tool "))]
    assert len(progress) == len(events)


# --- @path mentions --------------------------------------------------------------


def test_ask_mentions_pass_relative_paths_only(tmp_path):
    secret = "TOP SECRET FILE CONTENTS"
    _repo(tmp_path, {"src/a.py": f"{secret}\n"})
    model = FakeChatModel(messages=[AIMessage(content="read it myself")])
    shown, deltas = [], []

    answer_ask(
        "what does @src/a.py do?",
        root=tmp_path,
        config=_cfg(),
        write=shown.append,
        on_text=deltas.append,
        init=_init(model),
        check=_ok_checks,
    )

    prompt_messages = model.invocations[0]
    prompt_text = " ".join(str(m.content) for m in prompt_messages)
    assert "src/a.py" in prompt_text
    # contents are never inlined: CHATTY reads the file itself
    assert secret not in prompt_text


def test_ask_bad_mention_refuses(tmp_path):
    _repo(tmp_path)
    shown, deltas = [], []

    result = answer_ask(
        "what does @missing.py do?",
        root=tmp_path,
        config=_cfg(),
        write=shown.append,
        on_text=deltas.append,
        init=_init(FakeChatModel(messages=[])),
        check=_ok_checks,
    )

    assert result is None
    assert any("missing.py" in line for line in shown)
    assert not deltas


# --- the model-call limit ---------------------------------------------------------


class _BrokenChatModel(FakeChatModel):
    def _next(self):
        raise ConnectionError("provider unreachable")


def test_ask_model_failure_shows_the_error(tmp_path):
    _repo(tmp_path)
    shown, deltas = [], []

    result = answer_ask(
        "q?",
        root=tmp_path,
        config=_cfg(),
        write=shown.append,
        on_text=deltas.append,
        init=_init(_BrokenChatModel(messages=[])),
        check=_ok_checks,
    )

    assert result is None
    assert any("provider unreachable" in line for line in shown)


def test_ask_at_limit_prints_partial_answer_and_limit_line(tmp_path):
    _repo(tmp_path, {"note.txt": "x\n"})
    model = FakeChatModel(
        messages=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_file", "args": {"path": "note.txt"}, "id": "t1"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_file", "args": {"path": "note.txt"}, "id": "t2"}
                ],
            ),
        ]
    )
    shown, deltas = [], []

    result = answer_ask(
        "q?",
        root=tmp_path,
        config=_cfg(model_calls=2),
        write=shown.append,
        on_text=deltas.append,
        init=_init(model),
        check=_ok_checks,
    )

    assert result is not None and result.limit_hit
    assert any(
        "CHATTY hit its 2 model-call limit; ask again or narrow the question."
        == line
        for line in shown
    )


def test_ask_partial_answer_streams_before_the_limit_line(tmp_path):
    _repo(tmp_path, {"note.txt": "x\n"})
    model = FakeChatModel(
        messages=[
            AIMessage(
                content="Working on it... ",
                tool_calls=[
                    {"name": "read_file", "args": {"path": "note.txt"}, "id": "t1"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_file", "args": {"path": "note.txt"}, "id": "t2"}
                ],
            ),
        ]
    )
    shown, deltas = [], []

    answer_ask(
        "q?",
        root=tmp_path,
        config=_cfg(model_calls=2),
        write=shown.append,
        on_text=deltas.append,
        init=_init(model),
        check=_ok_checks,
    )

    assert "Working on it" in "".join(deltas)
    limit_index = next(
        i for i, line in enumerate(shown) if "model-call limit" in line
    )
    assert limit_index >= 0
