# Does #104 require retiring the `spec`/`tester`/`coder` config keys now?

**Question.** Is ticket #104's "old spec, tester and coder keys still work in this ticket" an explicit incremental-compatibility requirement, or must the PR already implement ADR 0012's retired-key failure?

**Verdict.** The reviewer is wrong. Keeping the old keys is an explicit acceptance criterion of #104, and retirement is the dedicated scope of a later ticket (#132) that is deliberately sequenced last, after the old workflow itself is removed (#131). ADR 0012 states the end state, not the timing.

## Evidence

### 1. Ticket #104 makes old-key compatibility an explicit, testable requirement

[rubjpDev/k0ntrol#104](https://github.com/rubjpDev/k0ntrol/issues/104), "What to build":

> The config accepts three new agent entries **beside the existing ones**: `planner` … `researcher` … `implementer` … The shipped defaults include them. **The old `spec`, `tester` and `coder` keys still work in this ticket.**

Acceptance criterion 3:

> - [ ] Configs that still use `spec`, `tester` and `coder` load exactly as before.

Per ADR 0012 (`docs/adr/0012-...md:17`), a ticket's body is the contract IMPLEMENTER and Contract review judge against: "A ticket and an agent brief share one format: what to build, seams, acceptance criteria, blocked by, and the parent. IMPLEMENTER reads the body." A PR that made `spec`/`tester`/`coder` fail validation would fail #104's own AC.

### 2. ADR 0012 describes the end state, not this ticket's scope

`docs/adr/0012-work-is-planned-on-the-tracker-as-a-spec-and-its-tickets.md:19`:

> - The `agents:` config keys become `planner`, `researcher`, `implementer`, `reviewer` and `chatty`. The old `spec`, `tester` and `coder` keys fail with a message naming their replacements.

This is a Consequences bullet — the target state of the whole 0.1.0 redesign. Nothing in it assigns the failure behaviour to #104. #102's "Config" implementation decision says the same thing ("Retired keys `spec`, `tester`, `coder` fail validation naming their replacements"), again as spec-level end state. Spec story 46 phrases it as the user's desired final behaviour, which the spec then allocates to a ticket — #132.

### 3. The spec's stated delivery strategy is "old beside new, delete last"

[rubjpDev/k0ntrol#102](https://github.com/rubjpDev/k0ntrol/issues/102), "Further Notes":

> New commands land beside the old ones and the old code is deleted last, so `develop` stays green and usable between tickets.

And:

> 0.1.0 is not released, so retired commands and config keys fail with a clear message instead of being mapped silently.

i.e. the retired-key failure is required by the time 0.1.0 ships — not by each intermediate ticket. #104 sits at the front of the chain (blocked only by #103, the docs ticket); retiring keys there would break the still-live `/fast`/`/full` workflow that builds SPEC, TESTER and CODER from those very config entries.

### 4. Retirement already has its own ticket, sequenced last

[rubjpDev/k0ntrol#132](https://github.com/rubjpDev/k0ntrol/issues/132) "[CHORE] Retire the spec, tester and coder config keys":

> The old agent keys are removed from the defaults, and a config that still sets `spec`, `tester` or `coder` fails validation with a message naming its replacement (planner, implementer), like the other retired keys.

> Blocked by: #131

#131 ("[CHORE] Retire /fast and /full", which deletes the SPEC node, TESTER node, batch red check and test gate) is itself blocked by #119, #123, #126, #127, #129 — the graph/agent tickets. So the dependency chain is: #104 (add new keys beside old) → … → #131 (delete old workflow) → #132 (retire old keys). Implementing #132's acceptance criteria inside #104 would invert the planned order and break develop between tickets.

### 5. The codebase already has the mechanism #132 will extend

`src/k0ntrol/config.py:27-36` (`_RETIRED_KEYS`) already maps gone-in-0.1.0 keys to replacement messages (e.g. `"backend": "set provider on each agents:/helpers: entry"`), enforced at `config.py:177-178` (`f"{path}: retired in 0.1.0; {_RETIRED_KEYS[lowered]}"`). "Like the other retired keys" in #132 refers to exactly this table. `spec`, `tester` and `coder` are currently entries in `AGENTS` (`config.py:13`), not retired keys — correctly so, for this ticket.

## Secondary scope observation

The uncommitted diff also touches `src/k0ntrol/init_flow.py`'s `_TIER_ROSTER` (adding planner/researcher/implementer alongside the old entries) and adds a tier-mapping test in `tests/test_init.py`. Mapping provider-setup tiers onto the new agents is ticket #106's scope ("The provider wizard's tiers map onto the new agents … Blocked by: #104"), and #106's AC says the quick path writes *only* the five new keys. Whether an interim both-old-and-new tier roster is acceptable is a separate contract-review question, but it is not required by #104's body, which covers config loading and the shipped defaults only.

## Recommended scope for the #104 PR

1. Keep the old `spec`, `tester`, `coder` entries in `AGENTS` and in `defaults/config.yaml`, beside the three new entries — as the current uncommitted diff does.
2. Keep the new test asserting old configs load unchanged (`test_legacy_agent_keys_load_exactly_as_before`) — it is AC 3.
3. Do **not** add `spec`/`tester`/`coder` to `_RETIRED_KEYS` and do not remove them from the shipped defaults: that is #132's acceptance criteria, after #131.
4. Consider whether the `init_flow.py` tier-roster change belongs here or should be deferred to #106.

## Sources

- https://github.com/rubjpDev/k0ntrol/issues/102 (spec body: Config decision, story 46, Further Notes)
- https://github.com/rubjpDev/k0ntrol/issues/104 (ticket body and ACs)
- https://github.com/rubjpDev/k0ntrol/issues/103, /issues/105, /issues/106, /issues/114, /issues/115, /issues/131, /issues/132 (dependency chain)
- `docs/adr/0012-work-is-planned-on-the-tracker-as-a-spec-and-its-tickets.md` (lines 3, 14, 17, 19)
- `docs/adr/0011-one-implementer-builds-red-to-green-slices-under-a-code-guard.md`
- `src/k0ntrol/config.py` (`AGENTS`, `_RETIRED_KEYS`, `_check_key`)
- `src/k0ntrol/defaults/config.yaml`, `src/k0ntrol/init_flow.py`, `tests/test_config.py`, `tests/test_config_schema.py`, `tests/test_init.py` (uncommitted #104 diff)
