"""The workflow graph's state.

Strict msgpack is on (``LANGGRAPH_STRICT_MSGPACK``), so state stays plain
types: strings, numbers, bools, lists, dicts and ``None``. Agent message
lists never enter the outer state — each agent run starts fresh from a
brief built out of these fields (ADR 0002).
"""

from typing import Any, TypedDict


class RunState(TypedDict, total=False):
    """One run's state; every field is a plain msgpack-safe value."""

    kind: str  # "fast" | "full" | "implement" | "plan" | "research"
    task: str  # the run's task text (issue body for /full #<n>)
    start_ms: int  # epoch ms when the run started (run_end.ms)
    max_attempts: int  # agent runs allowed per loop (default 2)

    test_files: list[str]  # test paths TESTER wrote
    code_files: list[str]  # non-test paths CODER wrote
    tester_attempts: int  # TESTER agent runs so far (red-check loop)
    coder_attempts: int  # CODER agent runs so far (VALIDATOR loop)

    tests_red: bool  # last red check: test_cmd failed on the test files
    validator_ok: bool  # last VALIDATOR run: test_cmd + gates all green
    red_check_output: str  # last red-check output (TESTER retry context)
    validator_output: str  # last VALIDATOR raw output (CODER retry input)

    # /full (#44): the work item the run carries, its branch and gates.
    item_ref: str | None  # the issue number (None until the spec gate passes)
    refining: bool  # /full #<n>: the issue existed before the run (re-runs)
    item_title: str
    branch: str  # k0/<n>-<slug>
    brief: dict | None  # SPEC's structured AgentBrief as a plain dict
    spec_attempts: int  # spec-drafting runs so far (/full's SPEC, /plan's draft)
    spec_approved: bool  # the human ok'd the brief or draft at the spec gate
    spec_rejections: list[str]  # earlier spec-gate feedback, for the comment
    spec_feedback: str | None  # the feedback of the last spec-gate reject
    test_approved: bool  # the human ok'd the tests
    test_rejections: list[str]  # earlier test-gate feedback
    test_feedback: str | None  # the feedback of the last test-gate reject

    # /full (#45): the change, REVIEWER's structured review and its gate.
    change_ref: str | None  # the PR ref once the draft is open
    change_url: str
    review: dict | None  # REVIEWER's structured Review as a plain dict
    review_round: int  # REVIEWER runs that produced a review (marker suffix)
    review_approved: bool  # the human ok'd the review
    review_route: str  # the review gate's decision: a node name
    review_feedback: str | None  # review + feedback, sent to CODER on redo
    review_rejections: list[str]  # earlier review-gate feedback
    reviewer_attempts: int  # REVIEWER agent runs so far
    spec_round: int  # approved briefs so far (marker suffix on republish)

    # /full (#47): the CI gate for pushes that change .github/.
    ci_cleared_sha: str | None  # the HEAD the CI gate cleared for a push
    ci_approved: bool  # the human approved on the last pass (comment due)
    ci_approved_sha: str | None  # HEAD at the last push; the next diff base
    ci_round: int  # approvals so far (gate-comment marker suffix)
    ci_feedback: str | None  # the feedback of the last CI-gate reject
    ci_rejections: list[str]  # earlier CI-gate feedback, for the comment

    # /implement <task> (#116): the working-tree path — seam gate, then
    # IMPLEMENTER's red-to-green slices under the guard. /implement #<n>
    # (#117) shares files and implementer_attempts and reuses the /full
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

    # /plan (#120): the Q&A definition — PLANNER's rounds of numbered
    # questions, each round answered by the human in one line. #121:
    # once a round holds no questions PLANNER drafts the Spec under its
    # own brief, and the spec gate decides ok / redo.
    rounds: list[dict]  # answered rounds: {"questions": [...], "answer": str}
    questions: list[dict]  # the open round's questions (the gate's detail)
    planner_attempts: int  # PLANNER runs so far
    max_round_questions: int  # how many questions one round may hold
    spec: dict | None  # the Spec draft at the spec gate (plain dict)

    # /research (#124): PLANNER's charted map, then the published refs.
    map: dict | None  # the ChartedMap as a plain dict
    decision_refs: list[str]  # child work-item refs, one per decision

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
