# Changelog

Weekly summary of pull requests merged into `develop`.

## Week of 2026-10-26 to 2026-11-02

### Breaking changes

- Foundation slice of the 0.1.0 LangGraph pivot: the old pipeline is torn down and the new config schema (`agents`/`helpers`/`providers`) rejects retired keys; adds the provider registry, `k0 doctor`, event log, tools with path guard, GitHub tracker, `/ask`, `/fast` and `k0 init` ([#53](https://github.com/rubjpDev/k0ntrol/pull/53))

### Features

- `/full` runs from the SPEC brief through approved tests, `limit_hit` pauses with retry/stop, and reflect runs after each run outcome ([#56](https://github.com/rubjpDev/k0ntrol/pull/56))
- `/full` continues from code to a draft PR and REVIEWER review gate, with optional Langfuse tracing and marker-spoofing and grep-symlink guard fixes ([#63](https://github.com/rubjpDev/k0ntrol/pull/63))
- `/full` gains a CI gate on `.github/` changes before every push, plus stuck handling, re-runs and resume ([#66](https://github.com/rubjpDev/k0ntrol/pull/66))

### Bug fixes

- Review bugfixes: failed reflect ranges are retried in order, missing structured output stops the run resumably, test gate shows red output, `bulk_read` preflight ([#64](https://github.com/rubjpDev/k0ntrol/pull/64))
- CI gate fails closed when the `.github/` diff cannot be computed instead of clearing HEAD for push ([#72](https://github.com/rubjpDev/k0ntrol/pull/72))
- `run_command` no longer reads bytecode cached before the call, so edited code is always recompiled ([#73](https://github.com/rubjpDev/k0ntrol/pull/73))
- Abandoning a `/full` run with a dirty working tree keeps the run instead of deleting its checkpoint ([#74](https://github.com/rubjpDev/k0ntrol/pull/74))
- Tool errors such as path-guard refusals are returned to the agent instead of ending the run ([#80](https://github.com/rubjpDev/k0ntrol/pull/80))
- Tracker reads accept `#1`-style work item refs, and their errors reach the agent ([#82](https://github.com/rubjpDev/k0ntrol/pull/82))

### Improvements

- Every skill subagent is routed to an explicit SWE-2 / Sonnet tier instead of the orchestrator's model ([#65](https://github.com/rubjpDev/k0ntrol/pull/65))
- Faster test suite: parallel runs with `pytest-xdist`, a stub `test_cmd` and a single CI run (ADR 0009) ([#77](https://github.com/rubjpDev/k0ntrol/pull/77))
- `run_command` shares one bytecode cache per process and only purges the repo's part before each command ([#78](https://github.com/rubjpDev/k0ntrol/pull/78))
