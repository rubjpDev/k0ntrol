# AGENTS.md — rules for any coding agent working on this repo

k0ntrol (`k0`) is a local harness for coding agents (Python). The 0.1.0 pivot to LangGraph is charted in the wayfinder map [rubjpDev/k0ntrol#2](https://github.com/rubjpDev/k0ntrol/issues/2). Domain language lives in `GLOSSARY.md`, decisions in `docs/adr/`, research notes in `docs/research/`. If code contradicts an ADR, either the code is wrong or a new ADR supersedes it: never silently diverge.

## How to work

- Work from a GitHub issue on `rubjpDev/k0ntrol`. Use one issue per branch and PR, and commit with `#<issue>: <what>`.
- TDD, always: write the failing test, run it and see it fail (RED), minimal code to pass (GREEN), then refactor. No production code without a failing test first.
- A task is done only when `uv run pytest -q` passes **in full** (whole suite, not just the new tests) and you have shown the output.
- Stuck after 2 attempts on the same task: stop, write what failed and why, and hand back to the human. Do not improvise a workaround.

## Git safety and human approval

- Work on your own feature branch, never directly on `main` or `develop`.
- When a task is verified, commit it on your branch, push the branch, and open a PR to `develop`. That PR is the approval request: the human reviews and merges it.
- Never merge, force-push, reset, rebase, or rewrite history. Never push to `main` or `develop`.
- In an interactive local session, show the diff, test output and proposed commit message, and wait for approval before committing.
- Never discard work. Prefer non-destructive operations that preserve the working tree.

### Exception: `/implement-spec` (2026-10-08)

A `/implement-spec` run works from one spec issue and replaces "one issue per branch and PR" with these rules:

- Create one integration branch per spec, `spec/<spec-issue>-<slug>`, from `develop`. Open one draft PR from it to `develop` that closes the spec issue and every ticket in scope.
- Build only the tickets the spec issue lists as in scope. Do not start any other ticket, even if it becomes unblocked.
- Give each ticket its own worktree and branch, `ticket/<issue>-<slug>`, based on the integration branch. Run `uv sync` in the worktree before running tests. Commit with `#<ticket>: <what>`.
- An implementer may reset its fresh worktree onto the integration branch before it starts work, and may merge the integration branch tip into its own ticket branch. The merger subagent may merge ticket branches into the integration branch only.
- Everything else in this file still applies: never touch `main` or `develop`, never force-push or rebase, and the human merges the PR.

## Invariants (never break these)

From the charting decision [rubjpDev/k0ntrol#3](https://github.com/rubjpDev/k0ntrol/issues/3):

- Edges between agents are code, never a supervisor LLM. The agent decides how to do its node; the workflow graph decides which node runs.
- Each agent's tool allowlist and path guards are enforced in code.
- Gates are code or the human, never a prompt. VALIDATOR is deterministic (`test_cmd` + `gates`).
- Each agent run starts with fresh context.
- k0 never merges.
- Every model invocation emits one JSONL event.

## Code rules

- Python ≥ 3.11, stdlib first. Runtime deps are a closed list: today `pyyaml` and `rich`; the 0.1.0 additions are recorded in an ADR. Ask before adding anything not on it.
- Cognitive complexity ≤ 6 per function and coverage ≥ 90 %, both enforced in CI. Small functions, explicit names, type hints on public functions.
- Everything is English: code, comments, commits, tests, docs, ADRs and issues.

## Commands

Run from the repo root:

- Setup: `uv sync`
- Tests: `uv run pytest -q`
- Complexity: `uv run complexipy src --max-complexity-allowed 6`
- Coverage: `uv run pytest --cov=k0ntrol --cov-fail-under=90`
- CLI: `uv run k0 --help`

## Agent skills

Skills are vendored per repo in `.agents/skills/` (from `mattpocock/skills`, pinned in `skills-lock.json`); `.claude/skills` is a symlink to it. Update them from upstream in this repo; do not install them globally.

### Subagent routing

In Devin, use the profiles in `.devin/agents/`; in Claude Code, the same-named profiles in `.claude/agents/`. Clients without custom subagents run the step inline.

- `/code-review`: two sub-agents in one turn — Standards `standards-reviewer`, Spec `spec-reviewer`. The parent aggregates after both return.
- `/research` background agent, including wayfinder research tickets: `researcher`.
- `/implement-spec`: use `swe-worker` for implementer and exploration subagents, and `merger` for the merger subagent.
- `codebase-design` design-it-twice: use `design-variant` for Agent 3 ("optimise for the most common caller"). Use `swe-worker` for the other designers and include the read-only constraint in each brief.
- Other Matt Pocock skill subagents that previously inherited the parent model use `swe-worker`.

A custom Devin profile without `model` uses the subagent router, not the parent's model, so every profile in `.devin/agents/` names a SWE-2 model: `swe-2-medium` for mechanical work (`merger`), `swe-2-high` for implementation, review and research, `swe-2-max` for interface design (`design-variant`). The `.claude/agents/` profiles mirror those tiers on `sonnet` with `effort: medium`, `high` and `max`; the parent session is the Opus orchestrator.

Never let a reviewer, merger, implementer or exploration subagent run on the orchestrator's model. This holds in Devin cloud too, where a step may run as a child session instead of a profile subagent: name the model explicitly, the profile's model or `swe-2-high` when unsure. If no profile or explicit model is available, run the step inline in the parent instead of spawning a copy of it; for `/code-review`, that means one inline pass per axis, not two spawned sessions.

### Issue tracker

GitHub Issues on `rubjpDev/k0ntrol`. Apply the title prefix in `docs/agents/issue-tracker.md`.

### Triage labels

Default role names: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
