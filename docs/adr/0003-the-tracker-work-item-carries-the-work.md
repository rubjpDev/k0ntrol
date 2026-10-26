# The tracker's work item carries the work, and k0 never merges

From 0.1.0, the work item in the repo's tracker carries what an agent is asked to do, in place of the `spec.md` and `review.md` files in the repo root. In `/full`:

- SPEC's output becomes the issue, and the spec gate is recorded on it.
- TESTER, CODER and VALIDATOR work on a branch.
- REVIEWER's review is posted on the pull request.
- The human merges. k0 never does.

`/fast` runs with no work item. The tracker is picked at `k0 init`. 0.1.0 implements GitHub only, behind a tracker interface. We chose this because code, tests, memory and the tracker already tell the story of a change, so a second carrier in the working tree duplicated it and was lost when the run ended. Merging stays with the human because it is the one step that cannot be taken back.

## Consequences

- k0 needs tracker credentials for `/full`.
- The tracker interface has no merge method.
- GitLab, Linear and Jira adapters are later work.
- How the harness writes to the tracker is in the tracker-access decision on the 0.1.0 map.
