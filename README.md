# k0ntrol

k0ntrol (`k0`) is a local harness for coding agents. Plan work with a human, build one ticket at a time, and review the result before merging. The workflow graph decides which agent runs next; code and the human decide whether gates pass.

## Get started

From this checkout, install `k0` with the extra for your model provider. For example, for the default Anthropic setup:

```sh
uv tool install --editable ".[anthropic]"
```

From the Git repository you want to work on, authenticate the GitHub CLI (`gh auth login`) for tracker-backed work, then run:

```sh
k0 init . --tracker github
k0 doctor
k0
```

`k0 init` creates the project configuration and memory under `.k0-mem/`. GitHub is the supported tracker in 0.1.0. You can also run `k0` in a terminal to use the first-run setup screens.

## Workflow commands

| Command | What it does |
| --- | --- |
| `/plan <idea>` or `/plan #<issue>` | PLANNER asks numbered questions in rounds with recommended answers. The human approves the spec and the ticket breakdown before k0 publishes them on the configured tracker. For a decided Map, PLANNER drafts the spec from its recorded decisions instead of starting a new Q&A. |
| `/research <question>` or `/research #<map>` | PLANNER charts a Map of decision tickets from a question. Use the Map number to work its next unblocked decision: RESEARCHER brings cited findings, PLANNER runs a Q&A, or the human answers directly. Once all decisions are recorded, use `/plan #<map>`. |
| `/implement #<ticket>` | Build a ready work item from a clean working tree. IMPLEMENTER works test-first in red-to-green slices on a ticket branch. When VALIDATOR passes, k0 opens a draft pull request and posts separate Conventions and Contract reviews from REVIEWER. The human can mark it ready, request changes, or send it back to planning. k0 never merges it. |
| `/implement <task>` | Build a small task in the current working tree without a tracker, a new branch, or a pull request. Approve the proposed test seams before the build and accept or reject the final diff. |
| `/triage #<issue>` | PLANNER runs a short Q&A on a raw work item. Approve the category, state, and comment before k0 applies them. A `ready-for-agent` outcome posts the agent brief used by `/implement`. |
| `/ask <question>` | Ask CHATTY a question about the repo without starting a workflow run or changing project code. |

In the terminal editor, **Shift+Tab** cycles through Chat, Plan, and Implement for plain input. In Chat, plain text asks CHATTY and a bare issue number like `#42` runs `/implement #42`. Explicit slash commands work in any mode. Type `/help` to see the available commands.

## Contributing

Read [AGENTS.md](AGENTS.md) for the repository rules and [GLOSSARY.md](GLOSSARY.md) for workflow terms. Run `uv sync` to set up development dependencies and `./scripts/check` to run the full CI gate.
