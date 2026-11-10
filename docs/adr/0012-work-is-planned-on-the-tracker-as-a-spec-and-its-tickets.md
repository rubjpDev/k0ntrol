# Work is planned on the tracker as a spec and its tickets, through a Q&A definition

From 0.1.0, k0 splits planning from building. `/plan` runs a **Q&A definition**: PLANNER asks the human numbered questions in rounds, each with its recommended answer, and every round is a human gate run on fresh context with the questions and answers so far. Rounds hold at most `max_round_questions` (7) questions, and at `max_questions` (25) the run pauses and offers to end the plan as it stands, hand the open questions to `/research`, or continue. A spec gate and a ticket gate follow, then the harness publishes a **Spec** as the parent work item and its **Tickets** as children, blockers first, with native blocking links where the tracker has them, labelled `ready-for-agent` and sized by `ticket_size` (default `small`). The first ticket always records the plan's glossary and ADR changes, because PLANNER has no write tools. `/implement #N` then builds one ticket, or one triaged work item, to a ready pull request. Foggy questions go to `/research`, which charts a **Map** of decision tickets (answered by RESEARCHER, by PLANNER, or by the human) that `/plan #map` turns into a spec once they are all decided. Raw work items reach `ready-for-agent` through `/triage`, a short Q&A that posts an **agent brief**. The commands are `/plan`, `/research`, `/implement`, `/triage` and `/ask`, with `/fast` and `/full` gone. We chose this because one brief per work item asked SPEC to guess what the human meant in one pass, and one run per request could not hold work bigger than one context. An interview that settles each decision with the human, and tickets each small enough for one fresh run, fix both.

## Considered Options

- Keep SPEC's one-shot agent brief and add an "open questions" section: rejected, because the questions would reach the human after the brief was written, not before.
- Keep one work item per run and let SPEC write a bigger brief: rejected, because a run's context, not the brief, is the limit.
- Keep specs and tickets as local files, as the skills k0 borrows from do by default: rejected, per ADR 0003; the tracker already carries the work.
- An uncapped interview: rejected, because a loop of questions with no end costs the human more than an imperfect spec.

## Consequences

- This replaces ADR 0003's SPEC brief comment and spec gate. The rest of 0003 stands: the tracker carries the work, TESTER's and CODER's successor works on a branch, the review is posted on the pull request, and k0 never merges.
- The tracker interface gains five operations: create a child work item, link a blocker, close a work item, list a work item's children and their state, and read its labels. GitHub implements them in 0.1.0; GitLab, Linear and Jira stay later work behind the same interface.
- Only work labelled `ready-for-agent`, or created by k0 as a ticket, is runnable by `/implement #N`.
- A ticket and an agent brief share one format: what to build, seams, acceptance criteria, blocked by, and the parent. IMPLEMENTER reads the body, or the latest agent brief.
- Building a whole spec in parallel across worktrees is 0.2.0 work. ADR 0007's one run per checkout still holds.
- The `agents:` config keys become `planner`, `researcher`, `implementer`, `reviewer` and `chatty`. The old `spec`, `tester` and `coder` keys fail with a message naming their replacements.
