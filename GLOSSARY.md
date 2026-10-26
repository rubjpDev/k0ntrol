# k0ntrol

k0ntrol (`k0`) is a local harness for coding agents: a human defines work and approves gates, and a workflow of agents writes tests, code and reviews against the repo's tracker and memory.

## Language

### Workflow

**Agent**:
A model with its own prompt and tool allowlist that the harness runs with fresh context: SPEC, TESTER, CODER and REVIEWER as nodes of the workflow graph, CHATTY on its own for `/ask`.
_Avoid_: Mode, worker, bot

**Workflow graph**:
The graph of agents and the code-defined edges that decide which agent runs next.
_Avoid_: Graph (alone), pipeline, loop

**Tool allowlist**:
The fixed set of tools an agent may call, enforced in code.
_Avoid_: Permissions (alone), capabilities

**Path guard**:
The code check that keeps a tool's path inside the files an agent may see and, for writes, the paths its role may change.
_Avoid_: Sandbox, jail

**Gate**:
A pass/fail decision taken by code or by the human, never by a prompt alone. VALIDATOR is the gate that runs the test command and the configured checks; it is not an agent.
_Avoid_: Check, approval step

**Helper**:
A one-shot model call made by the harness or by a tool, with no tools of its own and no place in the workflow graph (`bulk_read`, reflect).
_Avoid_: Mode, stage, agent

**Run**:
One execution of the workflow graph for one work item, or for one `/fast` task, from start to its outcome: `done`, `stuck` or `rejected`.
_Avoid_: Job, session, attempt

**Limit**:
A cap the harness enforces on how many model calls one agent run may make, or how many automatic attempts one loop of a run may take. Hitting one pauses the run until the human retries or stops it.
_Avoid_: Budget, quota

**Event**:
One line of k0's event log recording a model call, a tool call, a gate, or the start or end of a run.
_Avoid_: Log entry, span, trace

**Harness**:
The deterministic program that owns the workflow graph, its gates and its tool allowlists. It is not an agent.
_Avoid_: Orchestrator, supervisor

### Work

**Tracker**:
The issue system a repo picks at `k0 init` (GitHub, GitLab, Linear, Jira) where work items live.
_Avoid_: Backlog, board

**Work item**:
One issue or ticket in the tracker; the carrier of what an agent is asked to do.
_Avoid_: Spec file, task file, ticket (when meaning the tracker-agnostic concept)

**Trusted author**:
A tracker account with write access to the repo. Only trusted authors' content in a work item reaches agents.
_Avoid_: Maintainer, collaborator (alone)

**Change**:
The branch and its pull or merge request that propose the code for one work item. Trackers without a code host have none.
_Avoid_: PR (when meaning the tracker-neutral concept), patch

### Memory

**Code graph**:
The scanner's typed nodes and edges of the codebase, stored as `graph.json` in `.k0-mem/`.
_Avoid_: Graph (alone), knowledge graph

**Project memory**:
The committed memory every agent can read: the code graph, file notes, `Mem.md`, `Rules.md` and topology. Only humans and the scanner change it.
_Avoid_: Knowledge base, context (alone)

**Session memory**:
The facts reflect distills from transcripts. It stays local and is never committed.
_Avoid_: History, chat log

**Reflect**:
The background pass after each run that turns its transcript into session memory and proposes additions to `Mem.md`.
_Avoid_: Summarize, consolidate

**Provider**:
Where a model runs: a local runtime (Ollama, LM Studio), a lab API, an aggregator or a cloud.
_Avoid_: Backend, vendor

**Effort**:
How hard an agent's or helper's model reasons on each call: low, medium or high, or the provider's default when unset or when the model offers no such setting.
_Avoid_: Tier, thinking budget, reasoning level
