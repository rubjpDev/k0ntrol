"""The GitHub tracker adapter: `gh … --json` / `gh api` plus git for changes.

Shells out through an injected command runner (`argv -> CompletedProcess`)
so contract tests replay recorded output. `gh` reuses the user's auth;
a non-zero exit raises TrackerError (exit 4 means auth, for `k0 doctor`).
"""

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from k0ntrol.tracker.protocol import (
    Change,
    Comment,
    TrackerError,
    WorkItem,
    comment_marker,
)

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess]

_ISSUE_FIELDS = "number,title,state,isDraft,url"


def subprocess_runner(cwd: Path) -> CommandRunner:
    """The real runner: argv in `cwd`, gh prompts disabled."""
    env = {**os.environ, "GH_PROMPT_DISABLED": "1"}

    def run(argv: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=120
        )

    return run


class GhTracker:
    """Tracker over the `gh` CLI and local git, bound to the checkout's repo."""

    def __init__(self, runner: CommandRunner):
        self._runner = runner
        self._login: str | None = None

    # -- work items ------------------------------------------------------

    def read_work_item(self, ref: str) -> WorkItem:
        issue = self._gh_json(["gh", "api", f"repos/{{owner}}/{{repo}}/issues/{ref}"])
        return WorkItem(
            ref=str(issue["number"]),
            title=issue["title"],
            body=issue["body"] or "",
            state=issue["state"].lower(),
            url=issue["html_url"],
            author=issue["user"]["login"],
            author_association=issue["author_association"],
            comments=self._comments(ref),
        )

    def create_work_item(self, title: str, body: str) -> WorkItem:
        out = self._run(["gh", "issue", "create", "--title", title, "--body", body])
        return self.read_work_item(_number_from_url(out.stdout))

    def ensure_work_item(self, title: str, body: str, *, run: str, node: str) -> WorkItem:
        marker = comment_marker(run, node)
        for issue in self._work_items():
            if self._has_own_marker(issue, marker):
                return self.read_work_item(str(issue["number"]))
        return self.create_work_item(title, f"{body}\n\n{marker}")

    def _work_items(self) -> list[dict]:
        pages = self._gh_json([
            "gh", "api", "--paginate", "--slurp",
            "repos/{owner}/{repo}/issues?state=all&per_page=100",
        ])
        return [issue for page in pages for issue in page]

    def _has_own_marker(self, issue: dict, marker: str) -> bool:
        """A body marker counts only on an issue authored by the gh user."""
        if "pull_request" in issue or marker not in (issue.get("body") or ""):
            return False
        return (issue.get("user") or {}).get("login") == self._me()

    def _me(self) -> str:
        """The login `gh` is authenticated as: the author of issues k0 opens."""
        if self._login is None:
            self._login = self._gh_json(["gh", "api", "user"])["login"]
        return self._login

    def comment_work_item(self, ref: str, body: str, *, run: str, node: str) -> None:
        marker = comment_marker(run, node)
        if self.find_comment(ref, marker) is not None:
            return
        self._run(["gh", "issue", "comment", ref, "--body", f"{marker}\n\n{body}"])

    def add_label(self, ref: str, label: str) -> None:
        self._run(["gh", "issue", "edit", ref, "--add-label", label])

    def remove_label(self, ref: str, label: str) -> None:
        if label not in self._label_names(ref):
            return
        self._run(["gh", "issue", "edit", ref, "--remove-label", label])

    def ensure_label(self, name: str) -> None:
        existing = self._gh_json(["gh", "label", "list", "--json", "name"])
        if name in {label["name"] for label in existing}:
            return
        self._run(["gh", "label", "create", name, "--color", "d93f0b"])

    def find_comment(self, ref: str, marker: str) -> Comment | None:
        return _first_with_marker(self._comments(ref), marker)

    # -- changes -----------------------------------------------------------

    def create_branch(self, branch: str, base: str) -> None:
        self._run(["git", "fetch", "origin", base])
        if self._has_ref(f"refs/heads/{branch}"):
            self._run(["git", "checkout", branch])
        elif self._has_ref(f"refs/remotes/origin/{branch}"):
            self._run(["git", "checkout", "-b", branch, f"origin/{branch}"])
        else:
            self._run(["git", "checkout", "-b", branch, f"origin/{base}"])

    def commit(self, message: str) -> bool:
        status = self._run(["git", "status", "--porcelain"])
        if not status.stdout.strip():
            return False
        self._run(["git", "add", "-A"])
        self._run(["git", "commit", "-m", message])
        return True

    def push(self, branch: str) -> None:
        self._run(["git", "push", "-u", "origin", branch])

    def open_change(self, branch: str, base: str, title: str, body: str) -> Change:
        existing = self.read_change(branch)
        if existing is not None and existing.state == "open":
            return existing
        out = self._run(
            [
                "gh", "pr", "create", "--draft",
                "--base", base,
                "--head", branch,
                "--title", title,
                "--body", body,
            ]
        )
        url = out.stdout.strip()
        return Change(
            ref=_number_from_url(url),
            branch=branch,
            title=title,
            state="open",
            is_draft=True,
            url=url,
        )

    def mark_change_ready(self, ref: str) -> None:
        self._run(["gh", "pr", "ready", ref])

    def post_review(self, ref: str, body: str, *, run: str, node: str) -> None:
        marker = comment_marker(run, node)
        reviews = self._gh_json(
            ["gh", "api", f"repos/{{owner}}/{{repo}}/pulls/{ref}/reviews"]
        )
        if any(marker in (review.get("body") or "") for review in reviews):
            return
        self._run(
            ["gh", "pr", "review", ref, "--comment", "--body", f"{marker}\n\n{body}"]
        )

    def comment_change(self, ref: str, body: str, *, run: str, node: str) -> None:
        marker = comment_marker(run, node)
        if self.find_change_comment(ref, marker) is not None:
            return
        self._run(["gh", "pr", "comment", ref, "--body", f"{marker}\n\n{body}"])

    def find_change_comment(self, ref: str, marker: str) -> Comment | None:
        return _first_with_marker(self._comments(ref), marker)

    def read_change(self, branch: str) -> Change | None:
        prs = self._gh_json(
            [
                "gh", "pr", "list",
                "--head", branch,
                "--state", "all",
                "--json", _ISSUE_FIELDS,
            ]
        )
        if not prs:
            return None
        pr = _preferred_pr(prs)
        return Change(
            ref=str(pr["number"]),
            branch=branch,
            title=pr["title"],
            state=pr["state"].lower(),
            is_draft=pr["isDraft"],
            url=pr["url"],
        )

    def read_change_diff(self, ref: str) -> str:
        return self._run(["gh", "pr", "diff", ref]).stdout

    # -- internals ---------------------------------------------------------

    def _comments(self, ref: str) -> tuple[Comment, ...]:
        raw = self._gh_json(
            ["gh", "api", "--paginate", f"repos/{{owner}}/{{repo}}/issues/{ref}/comments"]
        )
        return tuple(
            Comment(c["user"]["login"], c["author_association"], c["body"])
            for c in raw
        )

    def _has_ref(self, ref: str) -> bool:
        probe = self._runner(["git", "rev-parse", "--verify", "--quiet", ref])
        return probe.returncode == 0

    def _label_names(self, ref: str) -> set[str]:
        issue = self._gh_json(["gh", "api", f"repos/{{owner}}/{{repo}}/issues/{ref}"])
        return {label["name"] for label in issue["labels"]}

    def _run(self, argv: list[str]) -> subprocess.CompletedProcess:
        result = self._runner(argv)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise TrackerError(
                f"{argv[0]} {argv[1]} failed (exit {result.returncode}): {detail}"
            )
        return result

    def _gh_json(self, argv: list[str]):
        return json.loads(self._run(argv).stdout)


def _preferred_pr(prs: list[dict]) -> dict:
    """The branch's open PR, else its most recent one (highest number).

    ``gh pr list`` order is not a contract, so the choice never leans on it.
    """
    open_prs = [pr for pr in prs if pr["state"].lower() == "open"]
    return max(open_prs or prs, key=lambda pr: pr["number"])


def _number_from_url(url: str) -> str:
    return url.strip().rstrip("/").rsplit("/", 1)[-1]


def _first_with_marker(comments: tuple[Comment, ...], marker: str) -> Comment | None:
    for comment in comments:
        if marker in comment.body:
            return comment
    return None
