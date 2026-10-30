"""Shared test fakes: no network, no installed extras needed.

Provider/doctor fakes: ``FakeChatModel`` is a scripted ``BaseChatModel``
stand-in and ``recording_init`` a fake ``init_chat_model``; ``cfg_with``
builds a 0.1.0 config dict. Agent tests script ``FakeChatModel`` with
``messages=[AIMessage(...), ...]`` — one entry per model call — which
works through ``create_agent`` (invoke and stream).

``FakeTracker`` is the in-memory ``Tracker`` implementation graph and E2E
tests use (spec #33, "Testing Decisions"). It records every public call
in ``self.calls`` so tests can assert what the harness did on the tracker.

Pass ``git_runner=argv -> subprocess.CompletedProcess`` (e.g. one that runs
real git in a tmp repo) when a test wants create_branch/commit/push to
act on real git state; otherwise those calls are only recorded.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any, Iterator

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict, Field

from k0ntrol.tracker import (
    Change,
    Comment,
    TrackerError,
    WorkItem,
    comment_marker,
)


class FakeChatModel(GenericFakeChatModel):
    """Scripted stand-in for a ``BaseChatModel``.

    Two roles:

    - The ``init_chat_model`` seam (provider/doctor tests): built by
      ``recording_init`` with ``(model, model_provider, init_kwargs)``;
      ``profile`` mirrors ``model.profile`` and ``reply`` is the answer
      every invoke gives when no script is left.
    - The scripted tool-calling fake for agent tests: ``messages`` is the
      script — ``AIMessage`` (with ``tool_calls``), ``FakeReply`` or ``str``
      entries consumed in order, one per model call. It works through
      ``create_agent`` on both ``invoke`` and ``stream``: ``_stream`` emits
      text in word chunks and tool calls as ``tool_call_chunks``, like a
      real provider.
    """

    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True)

    messages: Iterator = Field(default_factory=iter)
    model: str | None = None
    model_provider: str | None = None
    init_kwargs: dict = Field(default_factory=dict)
    reply: Any = None
    bound_tools: list | None = None
    bind_kwargs: dict | None = None
    invocations: list = Field(default_factory=list)

    def __init__(
        self,
        model=None,
        model_provider=None,
        init_kwargs=None,
        *,
        profile=None,
        reply=None,
        messages=None,
        **fields,
    ):
        super().__init__(
            messages=iter(messages or []),
            profile=profile,
            model=model,
            model_provider=model_provider,
            init_kwargs=init_kwargs or {},
            reply=reply,
            **fields,
        )

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.bound_tools = list(tools)
        self.bind_kwargs = kwargs
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.invocations.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.invocations.append(list(messages))
        yield from _reply_chunks(self._next())

    def _next(self) -> AIMessage:
        item = next(self.messages, self.reply)
        if item is None:
            raise AssertionError(
                "FakeChatModel script exhausted; script one message per model call"
            )
        return _to_ai_message(item)


def _to_ai_message(item) -> AIMessage:
    if isinstance(item, AIMessage):
        return item
    if isinstance(item, str):
        return AIMessage(content=item)
    tool_calls = [
        {
            "name": call["name"],
            "args": call.get("args", {}),
            "id": call.get("id") or f"call_{index}",
        }
        for index, call in enumerate(getattr(item, "tool_calls", None) or [])
    ]
    return AIMessage(content=getattr(item, "content", ""), tool_calls=tool_calls)


def _reply_chunks(message: AIMessage) -> Iterator[ChatGenerationChunk]:
    if isinstance(message.content, str):
        for token in re.split(r"(\s)", message.content):
            if token:
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
    for index, call in enumerate(message.tool_calls or []):
        chunk = {"name": call["name"], "args": json.dumps(call["args"]),
                 "id": call["id"], "index": index}
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="", tool_call_chunks=[chunk])
        )


class FakeReply:
    """A scripted reply shorthand; ``tool_calls`` like ``AIMessage``."""

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


class FakeRunner:
    """Scripted `argv -> CompletedProcess` runner for GhTracker contract tests.

    `script` records a response matched by argv prefix; calls are stored in
    `self.calls`. An unscripted call fails the test.
    """

    def __init__(self):
        self.calls: list[list[str]] = []
        self._scripts: list[tuple[list[str], subprocess.CompletedProcess]] = []

    def script(
        self,
        argv_prefix: list[str],
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> "FakeRunner":
        """Record a response for an argv prefix; re-scripting a prefix replaces it."""
        prefix = list(argv_prefix)
        self._scripts = [s for s in self._scripts if s[0] != prefix]
        self._scripts.append(
            (prefix, subprocess.CompletedProcess(prefix, returncode, stdout, stderr))
        )
        return self

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        for prefix, result in self._scripts:
            if argv[: len(prefix)] == prefix:
                return result
        raise AssertionError(f"unscripted call: {argv}")


class FakeTracker:
    """In-memory Tracker for tests; see module docstring."""

    def __init__(self, *, repo: str = "acme/widgets", git_runner=None):
        self.repo = repo
        self.git_runner = git_runner
        self.calls: list[tuple] = []
        self._items: dict[str, dict] = {}
        self._labels: set[str] = set()
        self._item_labels: dict[str, set[str]] = {}
        self._next_item = 1
        self._changes: dict[str, dict] = {}
        self._next_change = 101  # pull refs share the issue namespace; offset so they never collide with _items

    # -- work items ------------------------------------------------------

    def read_work_item(self, ref: str) -> WorkItem:
        try:
            item = self._item(ref)
        except TrackerError:
            try:
                return self._change_item(ref)
            except TrackerError:
                raise TrackerError(f"no work item: {ref}") from None
        comments = tuple(
            Comment(c["author"], c["author_association"], c["body"])
            for c in item["comments"]
        )
        return WorkItem(
            ref=ref,
            title=item["title"],
            body=item["body"],
            state=item["state"],
            url=item["url"],
            author=item["author"],
            author_association=item["author_association"],
            comments=comments,
        )

    def create_work_item(
        self,
        title: str,
        body: str,
        *,
        author: str = "rubenjp",
        author_association: str = "OWNER",
    ) -> WorkItem:
        self.calls.append(("create_work_item", title))
        ref = str(self._next_item)
        self._next_item += 1
        self._items[ref] = {
            "title": title,
            "body": body,
            "state": "open",
            "url": f"https://github.com/{self.repo}/issues/{ref}",
            "author": author,
            "author_association": author_association,
            "comments": [],
        }
        self._item_labels[ref] = set()
        return self.read_work_item(ref)

    def ensure_work_item(self, title: str, body: str, *, run: str, node: str) -> WorkItem:
        marker = comment_marker(run, node)
        for ref, item in self._items.items():
            if marker in item["body"]:
                return self.read_work_item(ref)
        return self.create_work_item(title, f"{body}\n\n{marker}")

    def comment_work_item(self, ref: str, body: str, *, run: str, node: str) -> None:
        marker = comment_marker(run, node)
        self.calls.append(("comment_work_item", ref))
        if self.find_comment(ref, marker) is not None:
            return
        self._item(ref)["comments"].append(
            {
                "author": "k0",
                "author_association": "OWNER",
                "body": f"{marker}\n\n{body}",
            }
        )

    def add_label(self, ref: str, label: str) -> None:
        self.calls.append(("add_label", ref, label))
        self._item(ref)
        if label not in self._labels:
            raise TrackerError(f"label does not exist: {label}")
        self._item_labels[ref].add(label)

    def remove_label(self, ref: str, label: str) -> None:
        self.calls.append(("remove_label", ref, label))
        self._item(ref)
        self._item_labels[ref].discard(label)

    def ensure_label(self, name: str) -> None:
        self.calls.append(("ensure_label", name))
        self._labels.add(name)

    def find_comment(self, ref: str, marker: str) -> Comment | None:
        for comment in self._item(ref)["comments"]:
            if marker in comment["body"]:
                return Comment(
                    comment["author"], comment["author_association"], comment["body"]
                )
        return None

    # -- changes -----------------------------------------------------------

    def create_branch(self, branch: str, base: str) -> None:
        self.calls.append(("create_branch", branch))
        self._git(["git", "checkout", "-B", branch])

    def commit(self, message: str) -> bool:
        self.calls.append(("commit", message))
        if self.git_runner is None:
            return True
        if not self._git(["git", "status", "--porcelain"]).stdout.strip():
            return False
        self._git(["git", "add", "-A"])
        self._git(["git", "commit", "-m", message])
        return True

    def push(self, branch: str) -> None:
        self.calls.append(("push", branch))

    def open_change(self, branch: str, base: str, title: str, body: str) -> Change:
        self.calls.append(("open_change", branch))
        existing = self.read_change(branch)
        if existing is not None and existing.state == "open":
            return existing
        ref = str(self._next_change)
        self._next_change += 1
        self._changes[ref] = {
            "branch": branch,
            "title": title,
            "body": body,
            "state": "open",
            "is_draft": True,
            "url": f"https://github.com/{self.repo}/pull/{ref}",
            "diff": "",
            "reviews": [],
            "comments": [],
        }
        return self._to_change(ref)

    def mark_change_ready(self, ref: str) -> None:
        self.calls.append(("mark_change_ready", ref))
        self._change(ref)["is_draft"] = False

    def post_review(self, ref: str, body: str, *, run: str, node: str) -> None:
        marker = comment_marker(run, node)
        self.calls.append(("post_review", ref))
        if any(marker in r for r in self._change(ref)["reviews"]):
            return
        self._change(ref)["reviews"].append(f"{marker}\n\n{body}")

    def comment_change(self, ref: str, body: str, *, run: str, node: str) -> None:
        marker = comment_marker(run, node)
        self.calls.append(("comment_change", ref))
        if self.find_change_comment(ref, marker) is not None:
            return
        self._change(ref)["comments"].append(
            {"author": "k0", "author_association": "OWNER", "body": f"{marker}\n\n{body}"}
        )

    def find_change_comment(self, ref: str, marker: str) -> Comment | None:
        for comment in self._change(ref)["comments"]:
            if marker in comment["body"]:
                return Comment(
                    comment["author"], comment["author_association"], comment["body"]
                )
        return None

    def read_change(self, branch: str) -> Change | None:
        refs = [r for r, c in self._changes.items() if c["branch"] == branch]
        open_ = [r for r in refs if self._changes[r]["state"] == "open"]
        # like the gh adapter: an open PR wins, else the most recent one
        pick = open_ or refs
        return self._to_change(max(pick, key=int)) if pick else None

    def read_change_diff(self, ref: str) -> str:
        return self._change(ref)["diff"]

    # -- test affordances -------------------------------------------------

    def labels_on(self, ref: str) -> set[str]:
        return set(self._item_labels[ref])

    def reviews_on(self, ref: str) -> list[str]:
        return list(self._change(ref)["reviews"])

    def change_body(self, ref: str) -> str:
        return self._change(ref)["body"]

    def add_change_comment(
        self, ref: str, body: str, *, author: str = "rubenjp", author_association: str = "OWNER"
    ) -> None:
        """Seed a PR comment as if a human wrote it (no marker)."""
        self._change(ref)["comments"].append(
            {"author": author, "author_association": author_association, "body": body}
        )

    def change_comments_on(self, ref: str) -> list[str]:
        return [c["body"] for c in self._change(ref)["comments"]]

    def set_diff(self, ref: str, diff: str) -> None:
        self._change(ref)["diff"] = diff

    def merge_change(self, ref: str) -> None:
        self._change(ref)["state"] = "merged"

    def close_change(self, ref: str) -> None:
        self._change(ref)["state"] = "closed"

    def add_comment(
        self, ref: str, body: str, *, author: str = "rubenjp", author_association: str = "OWNER"
    ) -> None:
        """Seed a comment as if a human wrote it (no marker)."""
        self._item(ref)["comments"].append(
            {"author": author, "author_association": author_association, "body": body}
        )

    def close_work_item(self, ref: str) -> None:
        self._item(ref)["state"] = "closed"

    # -- internals ---------------------------------------------------------

    def _item(self, ref: str) -> dict:
        try:
            return self._items[str(ref)]
        except KeyError:
            raise TrackerError(f"no work item: {ref}") from None

    def _change(self, ref: str) -> dict:
        ref = str(ref)
        if ref in self._changes:
            return self._changes[ref]
        for change in self._changes.values():
            if change["branch"] == ref:
                return change
        raise TrackerError(f"no change: {ref}")

    def _change_item(self, ref: str) -> WorkItem:
        """A change (PR) viewed as a work item — pull refs live in the issue
        namespace on GitHub, so `read_work_item <pr_ref>` returns the PR's
        body and comments."""
        change = self._change(ref)
        return WorkItem(
            ref=str(ref),
            title=change["title"],
            body=change["body"],
            state=change["state"],
            url=change["url"],
            author="k0",
            author_association="OWNER",
            comments=tuple(
                Comment(c["author"], c["author_association"], c["body"])
                for c in change["comments"]
            ),
        )

    def _to_change(self, ref: str) -> Change:
        change = self._changes[ref]
        return Change(
            ref=ref,
            branch=change["branch"],
            title=change["title"],
            state=change["state"],
            is_draft=change["is_draft"],
            url=change["url"],
        )

    def _git(self, argv: list[str]):
        if self.git_runner is None:
            return None
        result = self.git_runner(argv)
        if result.returncode != 0:
            raise TrackerError(f"git failed: {result.stderr}")
        return result
