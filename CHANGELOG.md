# Changelog

All notable changes to k0ntrol are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versioning: a stable release is `x.y.0`, and a patch release `x.y.z` with `z > 0` is unstable.

## [Unreleased]

### Added

- **PyPI package** `k0ntrol-cli`: `uv tool install "k0ntrol-cli[openai]"` installs k0 without cloning the repo. PyPI refuses the name `k0ntrol` as too close to `kontrol`. Pushing a release tag publishes it through PyPI Trusted Publishing ([#224](https://github.com/rubjpDev/k0ntrol/issues/224)).

## [0.1.0] - 2026-11-24

The first stable release. k0 moves from a single-pass pipeline to a LangGraph workflow: plan with a human, build one ticket at a time test-first, review twice, and let the human merge. Release evidence: [docs/releases/0.1.0.md](docs/releases/0.1.0.md).

### Added

- **Workflow graph** on LangGraph ([ADR 0002](docs/adr/0002-agents-run-as-a-langgraph-workflow-graph.md)). Code routes between agents, gates are code or the human, every agent run starts with fresh context, and k0 never merges.
- **Agent roster.** PLANNER, RESEARCHER, IMPLEMENTER, REVIEWER and CHATTY each have a tool allowlist enforced in code, and VALIDATOR is a code gate ([ADR 0001](docs/adr/0001-validator-is-a-code-gate-and-agents-get-run-tests-not-a-shell.md)) ([#135](https://github.com/rubjpDev/k0ntrol/pull/135), [#145](https://github.com/rubjpDev/k0ntrol/pull/145)).
- **`/plan`:** Q&A rounds with recommended answers, a question-limit gate, spec and ticket gates, and specs and tickets published on GitHub with blocking links ([ADR 0012](docs/adr/0012-work-is-planned-on-the-tracker-as-a-spec-and-its-tickets.md)) ([#153](https://github.com/rubjpDev/k0ntrol/pull/153), [#156](https://github.com/rubjpDev/k0ntrol/pull/156), [#161](https://github.com/rubjpDev/k0ntrol/pull/161), [#163](https://github.com/rubjpDev/k0ntrol/pull/163)).
- **`/research`:** charts a Map of decision tickets and works them through RESEARCHER, PLANNER or the human, and `/plan #<map>` turns a decided Map into a spec ([#154](https://github.com/rubjpDev/k0ntrol/pull/154), [#157](https://github.com/rubjpDev/k0ntrol/pull/157), [#162](https://github.com/rubjpDev/k0ntrol/pull/162)).
- **`/implement #<ticket>`:** builds on a ticket branch to a draft PR, and REVIEWER posts separate Conventions and Contract reviews before the review gate ([#152](https://github.com/rubjpDev/k0ntrol/pull/152), [#155](https://github.com/rubjpDev/k0ntrol/pull/155), [#160](https://github.com/rubjpDev/k0ntrol/pull/160)).
- **`/implement <task>`:** works in the working tree with a seam gate and a diff gate ([#146](https://github.com/rubjpDev/k0ntrol/pull/146), [#170](https://github.com/rubjpDev/k0ntrol/pull/170), [#171](https://github.com/rubjpDev/k0ntrol/pull/171)).
- **Red-to-green slice guard:** IMPLEMENTER's write tools refuse code before a failing test and tests while one is failing, and a red slice escalates to a human recovery gate ([ADR 0011](docs/adr/0011-one-implementer-builds-red-to-green-slices-under-a-code-guard.md), [ADR 0015](docs/adr/0015-a-red-slice-escalates-to-a-human-recovery-gate.md)) ([#136](https://github.com/rubjpDev/k0ntrol/pull/136), [#184](https://github.com/rubjpDev/k0ntrol/pull/184)).
- **`/triage #<issue>`:** labels a raw issue and posts an agent brief, and untrusted issues are refused with a next step ([#157](https://github.com/rubjpDev/k0ntrol/pull/157), [#186](https://github.com/rubjpDev/k0ntrol/pull/186)).
- **REPL modes:** Chat, Plan and Implement, cycled with Shift+Tab, with a bare `#42` running `/implement #42` ([#164](https://github.com/rubjpDev/k0ntrol/pull/164)).
- **Providers** through LangChain: Anthropic, OpenAI, Amazon Bedrock, Ollama, OpenRouter and Google Vertex, each an optional install extra ([ADR 0004](docs/adr/0004-providers-replace-the-cursor-backend.md), [ADR 0008](docs/adr/0008-the-0-1-0-runtime-dependencies-and-provider-extras.md)).
- **Network tools for PLANNER and RESEARCHER only:** `fetch_url` and `web_search` through Brave ([ADR 0013](docs/adr/0013-planner-and-researcher-get-read-only-network-tools.md)) ([#143](https://github.com/rubjpDev/k0ntrol/pull/143)).
- **Provider-native web search,** opt-in with `search.native: true`: OpenAI's built-in search and Anthropic's server search, capped per call. A model that can't search warns and runs without search, never falling back to Brave ([ADR 0017](docs/adr/0017-planner-and-researcher-can-use-the-providers-own-web-search.md)) ([#195](https://github.com/rubjpDev/k0ntrol/pull/195), [#210](https://github.com/rubjpDev/k0ntrol/pull/210)).
- **Run accounting:** one JSONL event per model call, a per-run cost line, read-only Reflect proposals after rough runs, and the `/klog` HTML report ([#169](https://github.com/rubjpDev/k0ntrol/pull/169), [#172](https://github.com/rubjpDev/k0ntrol/pull/172)).
- **Trusted App logins:** a `trusted_authors` list lets GitHub App identities round-trip their own artifacts ([ADR 0016](docs/adr/0016-repo-configured-app-logins-join-the-trusted-authors.md)) ([#188](https://github.com/rubjpDev/k0ntrol/pull/188)).

### Changed

- **OpenAI is the default provider.** The shipped defaults use `gpt-6.1-terra` for PLANNER, RESEARCHER and REVIEWER, and `gpt-6-luna` for IMPLEMENTER, CHATTY and the helpers. Setup offers OpenAI first.
- **Setup and `k0 doctor`:** OpenAI defaults to the Responses API, effort options keep a fixed order, a missing key prints one line, and a fresh home gets a set-up hint ([#206](https://github.com/rubjpDev/k0ntrol/pull/206)).
- **RESEARCHER** shares the strong model with PLANNER and REVIEWER, and IMPLEMENTER gets its own setup question ([#212](https://github.com/rubjpDev/k0ntrol/pull/212)).
- **Preflight** checks only the services a command uses ([#185](https://github.com/rubjpDev/k0ntrol/pull/185)).
- **Done gate:** the CI gate, `./scripts/check`, is the done gate, and the suite runs in parallel ([ADR 0009](docs/adr/0009-the-suite-runs-in-parallel-and-the-tdd-loop-runs-only-the-tests-in-play.md), [ADR 0014](docs/adr/0014-the-done-gate-is-the-ci-gate.md)).

### Removed

- **`/fast` and `/full`,** replaced by `/plan` and `/implement` ([#165](https://github.com/rubjpDev/k0ntrol/pull/165)).
- **The `spec`, `tester` and `coder` agent keys.** Config that still uses them fails with a message naming `planner` and `implementer` ([#166](https://github.com/rubjpDev/k0ntrol/pull/166)).
- **LM Studio** from the setup menu. A hand-written LM Studio config still loads.

### Fixed

- **Run safety:** pickers ignore keys typed before they open, the review gate lists every file that will be pushed (the refreshed knowledge graph in its own commit), and Ctrl-C pauses a run instead of failing it ([#207](https://github.com/rubjpDev/k0ntrol/pull/207)).
- **Output:** `/ask` renders its answer as one block, `/research` stops at its Map gate, `NO_COLOR` is honoured everywhere, and klog shows run status and prices correctly ([#208](https://github.com/rubjpDev/k0ntrol/pull/208)).
- **Private run files** stay out of k0's commits and pushes, a malformed test can recover under a human decision, and a missing or refused review can't mark a PR ready ([#184](https://github.com/rubjpDev/k0ntrol/pull/184)).
- **Resume:** a reviewed change resumes at its reviews and is never re-implemented ([#189](https://github.com/rubjpDev/k0ntrol/pull/189)).
- **Retries:** Bedrock timeouts and transport breaks retry as transient errors ([#187](https://github.com/rubjpDev/k0ntrol/pull/187)).
- **Config:** output-token caps such as `max_tokens` are no longer refused as secrets ([#198](https://github.com/rubjpDev/k0ntrol/pull/198)).
- **Tests** never read the developer's real home config or credentials ([#201](https://github.com/rubjpDev/k0ntrol/pull/201)).

### Security

- **Tool safety** relies on process hygiene and code guards ([ADR 0006](docs/adr/0006-tool-safety-is-process-hygiene-and-code-guards-not-a-sandbox.md)). Test runs get a scrubbed environment, path guards resolve symlinks and skip ignored files, tracker content is trusted only from repo members, and a CI gate holds back any push that changes `.github/`. Agent-written code runs as you, with no OS sandbox.

### Known limitations

- **GitHub is the only tracker.** GitLab, Linear and Jira are marked "coming soon" in setup.
- **UI polish** from the 0.1.0 pass is tracked for 0.1.1 in [#215](https://github.com/rubjpDev/k0ntrol/issues/215).

[Unreleased]: https://github.com/rubjpDev/k0ntrol/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rubjpDev/k0ntrol/releases/tag/v0.1.0
