"""The k0-owned Tracker protocol: work items plus code-host changes.

The harness does every tracker write in code through this protocol.
Agents never see it directly; they get read-only tools built on it.
There is no merge method: the human merges.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

#: Tracker authors whose content may reach an agent (ADR 0006).
TRUSTED_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})


def comment_marker(run: str, node: str) -> str:
    """The idempotency marker every k0 comment carries."""
    return f"<!-- k0:run={run} node={node} -->"


class TrackerError(Exception):
    """A tracker operation failed; the run stops resumable, never stuck."""


@dataclass(frozen=True)
class Comment:
    author: str
    author_association: str
    body: str


@dataclass(frozen=True)
class WorkItem:
    ref: str
    title: str
    body: str
    state: str  # "open" | "closed"
    url: str
    author: str
    author_association: str
    comments: tuple[Comment, ...] = field(default_factory=tuple)

    @property
    def qualified_ref(self) -> str:
        """`<owner>/<repo>#<n>` for the `work_item` event field."""
        parts = self.url.rstrip("/").split("/")
        if len(parts) >= 4 and parts[-2] == "issues":
            return f"{parts[-4]}/{parts[-3]}#{parts[-1]}"
        return self.ref


@dataclass(frozen=True)
class Change:
    ref: str
    branch: str
    title: str
    state: str  # "open" | "merged" | "closed"
    is_draft: bool
    url: str


class WorkItems(Protocol):
    """Work-item operations: GitHub Issues, later GitLab/Linear/Jira."""

    def read_work_item(self, ref: str) -> WorkItem:
        """Read a work item's body and comments."""
        ...

    def create_work_item(self, title: str, body: str) -> WorkItem:
        """Create a work item and return it."""
        ...

    def comment_work_item(self, ref: str, body: str, *, run: str, node: str) -> None:
        """Post a marked comment; a no-op when the marker already exists."""
        ...

    def add_label(self, ref: str, label: str) -> None:
        """Add an existing label to a work item."""
        ...

    def remove_label(self, ref: str, label: str) -> None:
        """Remove a label; a no-op when the label is absent."""
        ...

    def ensure_label(self, name: str) -> None:
        """Create the label in the tracker unless it already exists."""
        ...

    def find_comment(self, ref: str, marker: str) -> Comment | None:
        """Return the first work-item comment containing the marker."""
        ...


class Changes(Protocol):
    """Change operations: branches and pull requests on the code host."""

    def create_branch(self, branch: str, base: str) -> None:
        """Create `branch` from `origin/<base>` and check it out."""
        ...

    def commit(self, message: str) -> bool:
        """Commit the whole tree; False when there is nothing to commit."""
        ...

    def push(self, branch: str) -> None:
        """Push `branch` to the code host."""
        ...

    def open_change(self, branch: str, base: str, title: str, body: str) -> Change:
        """Open the draft change once; return the open one if it exists."""
        ...

    def mark_change_ready(self, ref: str) -> None:
        """Take the change out of draft."""
        ...

    def post_review(self, ref: str, body: str, *, run: str, node: str) -> None:
        """Post a marked COMMENT review; a no-op when the marker exists."""
        ...

    def comment_change(self, ref: str, body: str, *, run: str, node: str) -> None:
        """Post a marked comment on the change; no-op when the marker exists."""
        ...

    def find_change_comment(self, ref: str, marker: str) -> Comment | None:
        """Return the first change comment containing the marker."""
        ...

    def read_change(self, branch: str) -> Change | None:
        """Read the change for a branch, or None when there is none."""
        ...

    def read_change_diff(self, ref: str) -> str:
        """Read the change's diff."""
        ...


@runtime_checkable
class Tracker(WorkItems, Changes, Protocol):
    """The full tracker: work items plus, for code hosts, changes."""
