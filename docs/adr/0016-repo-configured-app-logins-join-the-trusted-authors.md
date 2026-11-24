# Repo-configured App logins join the trusted authors

ADR 0006 lets tracker content reach agents only from `OWNER`, `MEMBER` and `COLLABORATOR` authors. A run authenticated as a GitHub App publishes everything — maps, decision tickets, spec tickets, triage briefs, the PR — under `authorAssociation: NONE`, so a later run (or resume) refuses or drops its own artifacts (#176).

`trusted_authors`, a top-level config list of GitHub App logins (`<app-slug>[bot]`), is now a second way an author counts as trusted: content is trusted when `author_association` is a trusted association **or** the author login is in the list. The rule applies everywhere the old one did — a work item's body, its comments, and the `## Agent Brief` contract — with the tracker carrying the list the config set.

We chose the author login, not a marker: GitHub assigns `…[bot]` logins to Apps only, so the login is server-verified provenance, while the `<!-- k0:run=… -->` marker is plain text anyone can copy. The list lives in the repo's `.k0-mem/config.yaml`, so the trust decision is repo-scoped, travels with fresh clones, and behaves identically on resume — a run resumed without the entry (or under a different one) fails closed rather than trusting a different author. Entries must end in `[bot]`; a human login there is a config error, since trusting contributors is a different feature.

## Considered Options

- Trust `authorAssociation: BOT` or every bot: rejected — GitHub reports App posts as `NONE`, and trusting every bot lets any App or action inject content.
- Trust a k0 marker in the body or comment: rejected — the marker is copyable text; it grants provenance to whoever pastes it.
- Trust only artifacts carrying a marker *from* an allowlisted login: rejected for 0.1.0 — the formats are public, so a trusted App could mimic them anyway; the real decision is whether the App's identity is trusted at all.
- Trust the App only while the run itself is authenticated as it: rejected — a human resuming the run could no longer read its artifacts.

## Consequences

- ADR 0006's boundary is unchanged by default: with no `trusted_authors` configured, behavior is exactly as before.
- A repo that lists an App trusts whatever that App posts — everyone holding the App's credentials can write trusted content into the repo's runs.
- This amends ADR 0006's injected-content rule; its other guards stand.
