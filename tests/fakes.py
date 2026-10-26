"""Shared test fakes.

`FakeTracker` is the in-memory `Tracker` implementation graph and E2E
tests use (spec #33, "Testing Decisions"). It records every public call
in `self.calls` so tests can assert what the harness did on the tracker.

Pass `git_runner=argv -> subprocess.CompletedProcess` (e.g. one that runs
real git in a tmp repo) when a test wants create_branch/commit/push to
act on real git state; otherwise those calls are only recorded.
"""

from __future__ import annotations

import subprocess

from k0ntrol.tracker import (
    Change,
    Comment,
    TrackerError,
    WorkItem,
    comment_marker,
)


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
        self._next_change = 1

    # -- work items ------------------------------------------------------

    def read_work_item(self, ref: str) -> WorkItem:
        item = self._item(ref)
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
        for ref, change in self._changes.items():
            if change["branch"] == branch:
                return self._to_change(ref)
        return None

    def read_change_diff(self, ref: str) -> str:
        return self._change(ref)["diff"]

    # -- test affordances -------------------------------------------------

    def labels_on(self, ref: str) -> set[str]:
        return set(self._item_labels[ref])

    def reviews_on(self, ref: str) -> list[str]:
        return list(self._change(ref)["reviews"])

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
