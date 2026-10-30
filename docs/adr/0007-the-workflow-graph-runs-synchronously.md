# The workflow graph runs synchronously

In 0.1.0, k0's own code is synchronous: the workflow graph runs with `graph.stream`, agents with `agent.invoke`, and every node, tool, tracker call, git call and the REPL are plain functions. Reflect keeps its background thread. Nothing is `async`. We chose this because async would spread through every module and every test (and add `pytest-asyncio`) to buy concurrency that k0 does not use: one run per checkout, one agent at a time. LangGraph's sync `ToolNode` still runs the tool calls of one model turn in parallel on a thread pool.

## Considered Options

- Async throughout (`ainvoke`, `astream`, async tools and handlers): rejected for 0.1.0. Its only gain k0 would use is LangGraph's `TimeoutPolicy`, which works only on async nodes.

## Consequences

- No agent node has a wall-clock cap. Agents are stopped by their model-call limits, a stalled model call by the provider's HTTP timeout, and a hung test command by `test_timeout` (ADR 0006). A wall-clock limit per agent, if one is ever needed, is the budgets decision's call: async can return for it, since LangGraph runs sync nodes inside an async graph, or a small watchdog can do it.
- Parallel sessions do not need async: they are separate processes. What limits them is git. There is one run per checkout, so a second `/fast` in a checkout with an open run is refused with a hint to use another git worktree. A second `/full` instead offers `resume run <id> at <node>? [y/n]` (#46): `y` resumes the open run, `n` abandons it and starts a new one. `/ask` stays allowed alongside a run because it only reads. k0 does not create worktrees itself in 0.1.0.
