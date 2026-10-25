# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues on `rubjpDev/k0ntrol`. Use the `gh` CLI for all operations; it infers the repo from `git remote -v` inside a clone.

Work starts from a GitHub issue. Use one issue per branch and pull request, and commit with `#<issue>: <what>`.

## Issue titles

Every issue created for product work carries exactly one title prefix. The prefix says what kind of issue it is. The rest of the title is the outcome, in English.

### Spec

The spec is the parent issue that holds the sub-issues. Title it:

`[MVP - <version>] <title>`

Example: `[MVP - 0.1.0] Agents on LangGraph`

`<version>` is `major.minor.patch` (`0.1.0`). A release file named `0.1.0-unstable.md` still titles the spec `[MVP - 0.1.0]`. Take the version from the human, or from the release already named in the conversation (`docs/releases/`). Keep the spaces around the hyphen, and close the bracket against the version: `[MVP - 0.1.0]`.

### Tickets and small tasks

A sub-issue of a spec, and a small task filed on its own, starts with one of these. Choose the prefix when you create the issue.

- `[FEATURE] <title>` — new behaviour a user of k0ntrol can observe.
- `[CHORE] <title>` — work that keeps the repo healthy and leaves observable behaviour unchanged: tooling, docs, refactor, dependencies, process.
- `[BUG] <title>` — something that should already work and does not.

### Proof of concept

A proof of concept is its own issue, separate from a spec and from a ticket:

`[PoC] <title>`

Use it for an experiment whose job is to learn whether an approach holds, before that approach becomes a spec or a ticket.

### Shape

- One prefix per title. A spec is only `[MVP - x.y.z]`. A ticket is only `[FEATURE]`, `[CHORE]`, or `[BUG]`. A proof of concept is only `[PoC]`.
- Capitalisation is fixed: `MVP`, `FEATURE`, `CHORE`, `BUG`, `PoC`.
- Wayfinder maps and their child tickets keep the titles and labels in [Wayfinding operations](#wayfinding-operations).

The title is done when the `gh issue create --title` value matches one of these shapes.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Set the title from [Issue titles](#issue-titles). Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature requests; `/triage` reads this flag.)_

When set to `yes`, PRs run through the same labels and states as issues, using the `gh pr` equivalents:

- **Read a PR**: `gh pr view <number> --comments` and `gh pr diff <number>` for the diff.
- **List external PRs for triage**: `gh pr list --state open --json number,title,body,labels,author,authorAssociation,comments` then keep only `authorAssociation` of `CONTRIBUTOR`, `FIRST_TIME_CONTRIBUTOR`, or `NONE` (drop `OWNER`/`MEMBER`/`COLLABORATOR`).
- **Comment / label / close**: `gh pr comment`, `gh pr edit --add-label`/`--remove-label`, `gh pr close`.

GitHub shares one number space across issues and PRs, so a bare `#42` may be either: resolve with `gh pr view 42` and fall back to `gh issue view 42`.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body. `gh issue create --label wayfinder:map`.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue (`gh api` on the sub-issues endpoint). Labels: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, the ticket is assigned to the driving dev.
- **Blocking**: GitHub's **native issue dependencies**, the canonical, UI-visible representation. Add an edge with `gh api --method POST repos/rubjpDev/k0ntrol/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`, where `<blocker-db-id>` is the blocker's numeric **database id** (`gh api repos/rubjpDev/k0ntrol/issues/<n> --jq .id`, _not_ the `#number` or `node_id`). GitHub reports `issue_dependencies_summary.blocked_by` (open blockers only, the live gate). A ticket is unblocked when every blocker is closed.
- **Frontier query**: list the map's open children (`gh issue list --state open`, scoped to the map's sub-issues), drop any with an open blocker (`issue_dependencies_summary.blocked_by > 0`) or an assignee; first in map order wins.
- **Claim**: `gh issue edit <n> --add-assignee @me`, the session's first write.
- **Resolve**: `gh issue comment <n> --body "<answer>"`, then `gh issue close <n>`, then append a context pointer (gist + link) to the map's Decisions-so-far.
- **Research notes**: research tickets save their findings in `docs/research/<topic>.md` and link it from the resolution comment.
