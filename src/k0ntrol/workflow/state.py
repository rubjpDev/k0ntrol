"""The workflow graph's state.

Strict msgpack is on (``LANGGRAPH_STRICT_MSGPACK``), so state stays plain
types: strings, numbers, bools, lists, dicts and ``None``. Agent message
lists never enter the outer state — each agent run starts fresh from a
brief built out of these fields (ADR 0002).
"""

from typing import Any, TypedDict


class RunState(TypedDict, total=False):
    """One run's state; every field is a plain msgpack-safe value."""

    kind: str  # "implement" | "plan" | "research" | "triage"
    task: str  # the run's task text (issue body for ``#<n>`` runs)
    start_ms: int  # epoch ms when the run started (run_end.ms)
    max_attempts: int  # agent runs allowed per loop (default 2)
    redone: bool  # the human answered "redo" at a gate, possibly in an earlier sitting

    validator_ok: bool  # last VALIDATOR run: test_cmd + gates all green
    validator_output: str  # last VALIDATOR raw output (IMPLEMENTER retry input)

    # The work item a ``#<n>`` run carries, its branch and gates.
    item_ref: str | None  # the issue number (None until a gate publishes)
    item_title: str
    branch: str  # k0/<n>-<slug>
    spec_attempts: int  # spec-drafting runs so far (/plan's draft)
    spec_approved: bool  # the human ok'd the draft at the spec gate
    spec_rejections: list[str]  # earlier spec-gate feedback, for the comment
    spec_feedback: str | None  # the feedback of the last spec-gate reject

    # The change (draft pull request) and its review gate.
    change_ref: str | None  # the PR ref once the draft is open
    change_url: str
    review_approved: bool  # the human ok'd the review
    review_route: str  # the review gate's decision: a node name
    review_feedback: str | None  # review + feedback, sent back on redo
    review_rejections: list[str]  # earlier review-gate feedback

    # The CI gate for pushes that change .github/.
    ci_cleared_sha: str | None  # the HEAD the CI gate cleared for a push
    ci_approved: bool  # the human approved on the last pass (comment due)
    ci_approved_sha: str | None  # HEAD at the last push; the next diff base
    ci_round: int  # approvals so far (gate-comment marker suffix)
    ci_feedback: str | None  # the feedback of the last CI-gate reject
    ci_rejections: list[str]  # earlier CI-gate feedback, for the comment

    # /implement <task> (#116): the working-tree path — seam gate, then
    # IMPLEMENTER's red-to-green slices under the guard. /implement #<n>
    # (#117) shares files and implementer_attempts and reuses the
    # work-item, branch, change and ci fields above.
    files: list[str]  # paths IMPLEMENTER wrote (tests and code)
    seam_attempts: int  # IMPLEMENTER proposal runs so far
    seam_proposal: str  # the seams IMPLEMENTER proposed (gate detail, brief)
    seam_approved: bool  # the human ok'd the seams
    seam_feedback: str | None  # the feedback of the last seam-gate reject
    implementer_attempts: int  # IMPLEMENTER build runs so far (VALIDATOR loop)

    # /implement #<n> (#118): once the draft change is open, REVIEWER runs
    # twice with fresh context — a Conventions review and a Contract
    # review — each posted on the change as its own labelled review.
    conventions_review: dict | None  # the Conventions Review as a plain dict
    contract_review: dict | None  # the Contract Review as a plain dict
    conventions_review_attempts: int  # Conventions REVIEWER runs so far
    contract_review_attempts: int  # Contract REVIEWER runs so far
    conventions_review_round: int  # Conventions reviews produced (marker suffix)
    contract_review_round: int  # Contract reviews produced (marker suffix)

    # /implement #<n> (#119): the review gate over both posted reviews —
    # ``ok`` marks the change ready, ``redo`` reruns IMPLEMENTER in the
    # guard's refactor state, ``plan`` returns the item to planning.
    refactor: bool  # IMPLEMENTER's guard starts in refactor (a review redo)
    plan_feedback: str | None  # the human's `plan` feedback, commented on the item

    # /plan (#120): the Q&A definition — PLANNER's rounds of numbered
    # questions, each round answered by the human in one line. #121:
    # once a round holds no questions PLANNER drafts the Spec under its
    # own brief, and the spec gate decides ok / redo. #127: the
    # question-limit gate at max_questions — end as is, hand off to
    # /research, or continue for 10 more.
    rounds: list[dict]  # answered rounds: {"questions": [...], "answer": str}
    questions: list[dict]  # the open round's questions (the gate's detail)
    planner_attempts: int  # PLANNER runs so far
    max_round_questions: int  # how many questions one round may hold
    max_questions: int  # how many questions the whole Q&A may ask
    question_limit_route: str  # the limit gate's decision: spec | publish_map | round_gate
    spec: dict | None  # the Spec draft at the spec gate (plain dict)
    decided_map: bool  # /plan #<n> on a decided Map (#126): no Q&A, straight to the draft

    # /plan #122: the approved spec cut into tickets — the first, made
    # in code, records the glossary and ADR changes and blocks the rest.
    ticket_size: str  # small | medium | large — a ticket's criteria cap
    tickets: list[dict] | None  # the breakdown at the ticket gate
    ticket_refs: list[str]  # the published child refs, ticket order (#123)
    tickets_approved: bool  # the ticket gate's ok
    tickets_attempts: int  # breakdown runs so far
    ticket_rejections: list[str]  # the feedback of each redo so far
    ticket_feedback: str | None  # the feedback the next breakdown gets

    # /research (#124): PLANNER's charted map, then the published refs.
    map: dict | None  # the ChartedMap as a plain dict
    decision_refs: list[str]  # child work-item refs, one per decision

    # /research #<n> (#125): the map's next unblocked decision ticket.
    decision: dict | None  # the picked child: ref, question, type, url
    finding: dict | None  # RESEARCHER's ResearchFinding as a plain dict
    proposal: str  # the proposed decision an ``ok`` at the gate takes
    decided: str  # the confirmed decision the ticket closes with
    researcher_attempts: int  # RESEARCHER runs so far

    # /triage (#128): the outcome proposal, its gate and the applied result.
    triage_outcome: dict | None  # the TriageOutcome proposal as a plain dict
    outcome_attempts: int  # outcome-drafting PLANNER runs so far
    triage_approved: bool  # the human ok'd the outcome at its gate
    triage_feedback: str | None  # the feedback of the last outcome-gate redo
    triage_rejections: list[str]  # earlier outcome-gate feedback

    limit: dict[str, Any] | None  # set when the last node can hit a limit
    retry_node: str | None  # the node a limit_hit retry sends back to
    retries_granted: int  # limit_hit retries the human granted (stuck comment)
    retry_context: str | None  # limit note + output tail + human feedback
    outcome: str  # "done" | "stuck" | "rejected", set at the run's end
    reason: str | None  # "model_call_limit" | "max_attempts" on stuck
    node: str | None  # the node a stuck run came from
