# The done gate is the CI gate, as one command

ADR 0009 made `uv run pytest -q` the done gate and the merger's merge gate. In the #102 implement-spec run that proved too narrow: merge composition pushed `_tracker_for`, `_checked_input`, `_initial_state` and `make_question_limit` over the ≤6 complexity cap while the suite stayed green, and nobody's gate caught it because neither the merger profile nor the "done" rule ran the same checks as CI. We now define the gate once, as `scripts/check`: `uv run mypy src`, `uv run pytest -q --cov=k0ntrol --cov-fail-under=90`, and `uv run complexipy src --max-complexity-allowed 6`. The CI workflow calls the script instead of repeating its steps, and every doc or profile that names a gate — AGENTS.md's done gate, the merger profile — points at the same command, so a change to the gate cannot drift between them. The inner loop is unchanged: during red/green an agent still runs only the tests in play.

## Considered Options

- Keep `pytest -q` as the gate and add a separate complexity step to briefs: rejected, because two commands named in N places is exactly what drifted.
- Repeat the three CI steps inline in the workflow and the docs: rejected for the same reason; the workflow now calls the script so CI and local runs cannot diverge.
- A `Makefile` or task runner: rejected, because a shell script needs no new dependency and reads in one screen.

## Consequences

- This supersedes ADR 0009's "the done gate is unchanged: `uv run pytest -q` passes in full" and "`uv run pytest -q` stays the literal done-gate command". The inner-loop rule from that ADR stands.
- A merge or a done check now also runs typecheck and coverage locally, so the gate takes tens of seconds, not the suite's ~25 s alone.
