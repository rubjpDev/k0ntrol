# k0ntrol

k0ntrol (`k0`) is a local harness for coding agents: a human defines work and approves gates, and a workflow of agents writes tests, code and reviews against the repo's tracker and memory.

## Language

### Workflow

**Agent**:
A node of the workflow graph with its own prompt, model and tool allowlist (SPEC, TESTER, CODER, VALIDATOR, REVIEWER, CHATTY).
_Avoid_: Mode, worker, bot

**Workflow graph**:
The graph of agents and the code-defined edges that decide which agent runs next.
_Avoid_: Graph (alone), pipeline, loop

**Tool allowlist**:
The fixed set of tools an agent may call, enforced in code.
_Avoid_: Permissions (alone), capabilities

**Gate**:
A pass/fail decision taken by code or by the human, never by a prompt alone.
_Avoid_: Check, approval step

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

### Memory

**Code graph**:
The scanner's typed nodes and edges of the codebase, stored as `graph.json` in `.k0-mem/`.
_Avoid_: Graph (alone), knowledge graph

**Reflect**:
The background pass that turns session transcripts into session memory and proposes additions to project memory.
_Avoid_: Summarize, consolidate

**Provider**:
Where a model runs: a local runtime (Ollama, LM Studio), a lab API, an aggregator or a cloud.
_Avoid_: Backend, vendor
