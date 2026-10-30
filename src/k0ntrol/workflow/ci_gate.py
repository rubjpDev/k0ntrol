"""The CI gate: no push changes ``.github/`` without the human (#47).

ADR 0006: ``.github/`` stays writable, so the risk is the push, not the
write. Every ``/full`` push goes through this module: ``run_ci_gate`` is
the node body that decides, ``gated_push`` is the only place the harness
calls ``tracker.push`` and it refuses to push a HEAD the gate has not
cleared. The gate looks at the ``.github/`` diff since the last approved
push (or since the base branch before the first one), so it fires again
on any later push that changes ``.github/``.
"""

from collections.abc import Callable

from langgraph.types import interrupt

from k0ntrol.workflow.change import diff_since, head_sha, merge_base
from k0ntrol.workflow.context import RunContext

GITHUB_PATH = ".github/"


class CiGateError(RuntimeError):
    """A push was attempted without the CI gate clearing HEAD."""


def ci_diff(ctx: RunContext, state: dict) -> str:
    """The ``.github/`` diff since the last approved push, "" if none."""
    since = state.get("ci_approved_sha") or merge_base(ctx.root, ctx.base_branch)
    return diff_since(ctx.root, since, GITHUB_PATH)


def run_ci_gate(ctx: RunContext, state: dict) -> dict:
    """Clear HEAD for a push: pass through, or interrupt on a ``.github/`` diff.

    ``ci_cleared_sha`` names the HEAD the push may use; ``None`` means the
    human rejected and the feedback goes to CODER. ``ci_approved`` is True
    only when the human approved on this pass (the gate comment is due).
    """
    head = head_sha(ctx.root)
    diff = ci_diff(ctx, state)
    if not diff:
        return {"ci_cleared_sha": head, "ci_approved": False, "ci_feedback": None}
    decision = interrupt(
        {
            "gate": "ci",
            "question": "approve pushing these .github/ changes?",
            "answers": ["ok", "redo"],
            "detail": diff,
        }
    )
    word, _, feedback = str(decision).strip().partition(" ")
    if word.lower() == "ok":
        return {
            "ci_cleared_sha": head,
            "ci_approved": True,
            "ci_feedback": None,
            "ci_round": state.get("ci_round", 0) + 1,
        }
    return {
        "ci_cleared_sha": None,
        "ci_approved": False,
        "coder_attempts": 0,
        "ci_feedback": feedback.strip(),
        "ci_rejections": [*state.get("ci_rejections", []), feedback.strip()],
    }


def make_ci_gate(ctx: RunContext) -> Callable[[dict], dict]:
    """The ``ci_gate`` graph node."""

    def node(state: dict) -> dict:
        return run_ci_gate(ctx, state)

    return node


def gated_push(ctx: RunContext, state: dict) -> dict:
    """Push the run's branch, only at a HEAD the CI gate cleared."""
    head = head_sha(ctx.root)
    if state.get("ci_cleared_sha") != head:
        raise CiGateError("push refused: the CI gate has not cleared HEAD")
    ctx.tracker.push(state["branch"])
    return {"ci_approved_sha": head}
