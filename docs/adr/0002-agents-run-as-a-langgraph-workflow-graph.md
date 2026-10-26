# Agents run as a LangGraph workflow graph with edges in code

From 0.1.0, k0's agents are real tool-calling agents run as nodes of a LangGraph `StateGraph`. Up to 0.0.3 they were one-shot model calls chained by the hand-written loop in `harness/loop.py`. The edges between agents are Python routers, never a supervisor LLM: the agent decides how to do its node, and the workflow graph decides which node runs next. Each agent's tool allowlist and path guards are enforced in code. Human gates are `interrupt()` nodes. Each agent run starts with fresh context. We chose this because agents that read and write the repo themselves do better work than one-shot calls fed by the harness, and LangGraph gives checkpoints, resumable gates and streaming without giving up deterministic control over the workflow.

## Considered Options

- Keep the hand-written loop with one-shot calls: rejected, because every new capability (reading more files, retrying a write) meant more harness code, and agents could not look anything up themselves.
- A supervisor LLM or a multi-agent framework that routes between agents: rejected, because routing must be deterministic and testable, and a prompt is not a gate.

## Consequences

- The old rule that no agent calls a stage no longer applies. `bulk_read` becomes a tool backed by a cheap model, and `code_write` becomes a guarded write tool.
- D24 dies: an agent reads its target files itself instead of having the harness inline them.
- The roster, allowlists and edges are in ADR 0001 and in the agent roster decision on the 0.1.0 map.
