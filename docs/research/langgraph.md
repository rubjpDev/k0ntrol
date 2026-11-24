# LangGraph mechanics k0 needs

Research for [rubjpDev/k0ntrol#8](https://github.com/rubjpDev/k0ntrol/issues/8). Checked 2026-10-25 against PyPI, the pinned upstream source tags, and docs.langchain.com. Where the docs and the source disagree, the source wins, and the disagreement is flagged.

## Answer in one paragraph

Build the workflow graph as a plain `langgraph.graph.StateGraph` whose nodes are k0 Python functions. Routing between agents uses `add_conditional_edges` with Python routers, and there is no supervisor LLM. Each agent node is a thin wrapper. It renders a fresh brief from the run state, invokes a `langchain.agents.create_agent` graph built with that agent's fixed tool list and limit middleware, and maps the result back into the run state. `create_react_agent` is deprecated. Human gates are ordinary graph nodes that call `interrupt()`, compiled with `SqliteSaver`, with `thread_id` set to the k0 run id. Resume works after the process exits because every super-step is checkpointed to SQLite. Limits come from four layers: `recursion_limit` on the outer run, `recursion_limit` bound to each agent, `ModelCallLimitMiddleware`/`ToolCallLimitMiddleware` inside each agent, and an attempt counter in the run state. Progress lines should come from the stable `graph.stream(..., stream_mode=["updates", "custom"], subgraphs=True, version="v2")` API, not the experimental `stream_events(version="v3")`. Keep `.k0-mem/` session memory in k0's own SQLite. Do not adopt `BaseStore` in 0.1.0.

## Pinned versions (PyPI, 2026-10-25)

| Package | Version | Released | Requires | Notes |
| - | - | - | - | - |
| `langgraph` | 1.2.17 | 2026-10-24 | Python >=3.10 | deps: `langchain-core>=1.4.7,<2`, `langgraph-checkpoint>=4.3.0,<5`, `langgraph-prebuilt>=1.1.0,<1.2`, `langgraph-sdk>=0.4.6,<0.5`, `pydantic`, `xxhash` |
| `langgraph-prebuilt` | 1.1.1 | 2026-10-15 | | `ToolNode`, `tools_condition`, deprecated `create_react_agent` |
| `langgraph-checkpoint` | 4.3.0 | 2026-10-12 | | `BaseCheckpointSaver`, `InMemorySaver`, serializers |
| `langgraph-checkpoint-sqlite` | 3.1.2 | 2026-10-12 | | `SqliteSaver`, `AsyncSqliteSaver`, `SqliteStore`; deps `aiosqlite`, `sqlite-vec` (native extension) |
| `langchain` | 1.4.7 | 2026-10-21 | | `create_agent` + middleware; deps `langchain-core>=1.6.13,<2`, `langgraph>=1.2.12,<1.3` |
| `langchain-core` | 1.7.0 | 2026-10-22 | | hard dep on `langsmith` (inert unless its env vars are set) |

Sources: `https://pypi.org/pypi/<package>/json` for each package. Release tags: [langgraph 1.2.17](https://github.com/langchain-ai/langgraph/releases/tag/1.2.17), [langchain 1.4.7](https://github.com/langchain-ai/langchain/releases/tag/langchain%3D%3D1.4.7).

Proposed constraint for `pyproject.toml`: `langgraph>=1.2.17,<1.3`, `langchain>=1.4.7,<1.5`, `langgraph-checkpoint-sqlite>=3.1.2,<3.2`. 1.2.17 is the floor because it fixes an explicit `recursion_limit` being ignored when it equals the default ([PR #9417](https://github.com/langchain-ai/langgraph/pull/9417)). Watch out: `langchain-core` 1.7.0 changed how tool-call argument deltas stream (see Deprecations). Per AGENTS.md, every one of these packages is a new runtime dependency that needs the human's approval.

## 1. `StateGraph` with conditional edges in Python (no supervisor LLM)

- `StateGraph(State, context_schema=Ctx)`, `add_node`, `add_edge`, `add_conditional_edges(source, router, path_map=None)`. The router is a plain Python function of state and returns a node name, a list of names (fan-out in the next super-step), or a key of `path_map`. Source: [Graph API docs](https://docs.langchain.com/oss/python/langgraph/graph-api.md).
- `Command(update=..., goto=...)` is the alternative to routing from inside a node. It needs a `Command[Literal[...]]` return annotation. Static edges still fire alongside it, so use only one routing mechanism per node. Same source.
- Per-run immutable data, such as root path, config and provider, goes in `context_schema` and is read through `runtime.context`. It does not belong in state. Same source.
- Fault tolerance added in 1.2.0, all Python-only: `add_node(..., retry_policy=RetryPolicy(...))` (defaults `max_attempts=3`, `initial_interval=0.5`), `timeout=TimeoutPolicy(run_timeout=..., idle_timeout=...)` (async nodes only), and `error_handler=` returning a `Command`. Sources: [changelog, langgraph v1.2.0](https://docs.langchain.com/oss/python/releases/changelog.md), [`types.py` at 1.2.17](https://github.com/langchain-ai/langgraph/blob/1.2.17/libs/langgraph/langgraph/types.py) (`RetryPolicy` L427, `TimeoutPolicy` L461).

Fit for k0: this maps one-to-one onto the invariant that edges are code. Today's `run_fast`/`run_full` loop in `src/k0ntrol/harness/loop.py` (TESTER → CODER → VALIDATOR, plus `_validation_retry` bounded by `config["max_retries"]`) becomes nodes plus a router after VALIDATOR. The router returns `"coder"` while `attempt < max_retries`, then `"stuck"`. VALIDATOR stays a deterministic node that runs `test_cmd` and gates, and it calls no model.

## 2. One agent per node, fixed tool allowlist, fresh history per run

**Use `langchain.agents.create_agent` (langchain 1.4.7). Do not use `create_react_agent`.**

- `create_react_agent` in `langgraph.prebuilt` carries `@deprecated("create_react_agent has been moved to langchain.agents ...", category=LangGraphDeprecatedSinceV10)`. It is still exported in prebuilt 1.1.1, and no removal date is published. Sources: [`chat_agent_executor.py` L274](https://github.com/langchain-ai/langgraph/blob/prebuilt%3D%3D1.1.1/libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py), [LangGraph v1 migration guide](https://docs.langchain.com/oss/python/migrate/langgraph-v1).
- The `create_agent` signature is `create_agent(model, tools=None, *, system_prompt, middleware=(), response_format, state_schema, context_schema, checkpointer, store, interrupt_before, interrupt_after, debug, name, cache, transformers)`. It returns a `CompiledStateGraph`, so it can be invoked from inside a node or added as a subgraph. Source: [`factory.py` L892](https://github.com/langchain-ai/langchain/blob/langchain%3D%3D1.4.7/libs/langchain_v1/langchain/agents/factory.py).

**How the allowlist is enforced in code.** Tools run through a `ToolNode` that holds exactly the tools passed in. A tool call whose name is not registered gets back an error `ToolMessage` (`"Error: {requested_tool} is not a valid tool, try one of [...]"`) and nothing runs. Source: [`tool_node.py` L116, L1302](https://github.com/langchain-ai/langgraph/blob/prebuilt%3D%3D1.1.1/libs/prebuilt/langgraph/prebuilt/tool_node.py). Two escape hatches would weaken the allowlist, and k0 should forbid both in code:

1. Middleware that defines `wrap_tool_call` can run an unregistered tool through `handler(request.override(tool=...))`. When any `wrap_tool_call` exists, `create_agent` also skips its check for unknown tools added in `wrap_model_call` ([`factory.py` L127-150, L1366-1398](https://github.com/langchain-ai/langchain/blob/langchain%3D%3D1.4.7/libs/langchain_v1/langchain/agents/factory.py)). Rule: k0 builds each agent's middleware list itself, and no k0 middleware overrides `request.tool`.
2. Tools passed as `dict` (provider built-ins such as server-side web search) are not checked against the `ToolNode`. Rule: pass only `BaseTool` instances or callables.

A k0-side unit test can lock the allowlist down: build each agent and assert the set of names in `tools_by_name`.

**Fresh context per agent run.** Use the "call a subgraph inside a node" pattern. The wrapper node turns run state into a new `{"messages": [HumanMessage(brief)]}`, invokes the agent, and returns only the fields k0 needs, such as a report or a list of written files. The agent's message list never goes into the outer state. The agent is compiled without its own checkpointer (`checkpointer=None`), which is the default "per-invocation" subgraph mode: "Each call starts fresh, but within a single call the subgraph inherits the parent's checkpointer and can use `interrupt()`". `checkpointer=True` (per-thread memory) would break the fresh-context invariant. Source: [Subgraphs, "Call a subgraph inside a node" and "Subgraph persistence"](https://docs.langchain.com/oss/python/langgraph/use-subgraphs.md).

**Hand-written `ToolNode` loop vs `create_agent`.** `ToolNode` and `tools_condition` are still supported and not deprecated in prebuilt 1.1.1 ([`__init__.py`](https://github.com/langchain-ai/langgraph/blob/prebuilt%3D%3D1.1.1/libs/prebuilt/langgraph/prebuilt/__init__.py)). A hand-written model→tools loop gives the same allowlist guarantee, because it uses the same `ToolNode`. It does not get the middleware layer: call limits, model retry and fallback, tool-error handling, summarization, and `wrap_model_call` for one JSONL event per model call. Recommendation: use `create_agent`, and keep a hand-written loop as a fallback only if a local-model quirk forces it.

Sketch (illustrative, not tested):

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware

def build_coder(model, limits):
    agent = create_agent(
        model,
        tools=CODER_TOOLS,                      # the allowlist, fixed in code
        system_prompt=CODER_PROMPT,
        middleware=[
            ModelCallLimitMiddleware(run_limit=limits.model_calls, exit_behavior="end"),
            ToolCallLimitMiddleware(run_limit=limits.tool_calls, exit_behavior="continue"),
        ],
        name="coder",
    )
    return agent.with_config(recursion_limit=limits.agent_steps)

def coder_node(state: RunState, runtime: Runtime[K0Context]) -> dict:
    result = build_coder(runtime.context.model, runtime.context.limits).invoke(
        {"messages": [HumanMessage(render_coder_brief(state))]}
    )
    return {"coder_report": final_text(result)}
```

## 3. Human gates via `interrupt()` + SQLite checkpointer, resumable after exit

- Signature: `interrupt(value, *, response_schema=None)`. The first call raises `GraphInterrupt`, and the value surfaces to the caller. Resume with `graph.invoke(Command(resume=answer), config)` using the same `thread_id`. The node re-runs from its start, and resume values match `interrupt()` calls by order within the node. A checkpointer is required. Source: [`types.py` L903](https://github.com/langchain-ai/langgraph/blob/1.2.17/libs/langgraph/langgraph/types.py), [Interrupts docs](https://docs.langchain.com/oss/python/langgraph/interrupts.md).
- Rules from the docs that k0 must follow:
  - Code before `interrupt()` re-runs on resume, so it must be idempotent.
  - Do not wrap `interrupt()` in a bare `try/except Exception`.
  - Do not skip interrupts conditionally, and do not loop `interrupt()` inside one node. Route back with a conditional edge instead.
  - Parallel interrupts are answered with `Command(resume={interrupt_id: value, ...})`.
- **Gate placement.** Make each gate its own small graph node between agents, for example `gate_plan → coder`, with exactly one `interrupt()` and no side effects before it. Gate placement then stays in code, and resume is trivially idempotent. An interrupt inside an agent's tool also works: the parent node re-runs and the agent subgraph resumes at its own interrupted node. That path re-runs the wrapper's pre-invoke code, so use it only for tool-level approvals (see `HumanInTheLoopMiddleware` below).
- **Checkpointer.** `SqliteSaver(sqlite3.connect(path, check_same_thread=False))` or `SqliteSaver.from_conn_string(path)` (a context manager). It sets `PRAGMA journal_mode=WAL` and takes an internal lock, so sharing a connection across threads is safe. `AsyncSqliteSaver` is the async variant. `delete_thread(thread_id)` exists for cleanup. Source: [`checkpoint/sqlite/__init__.py` L49-146, L498](https://github.com/langchain-ai/langgraph/blob/checkpointsqlite%3D%3D3.1.2/libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/__init__.py).
- **Durability.** The default is `durability="async"`: each step is checkpointed while the next one runs, with "a small risk" of losing the last write if the process crashes. `"sync"` writes before every step, and `"exit"` writes only when execution ends, including on an interrupt. Sources: [Checkpointers, "Durability modes"](https://docs.langchain.com/oss/python/langgraph/checkpointers.md), [`pregel/main.py` L2696](https://github.com/langchain-ai/langgraph/blob/1.2.17/libs/langgraph/langgraph/pregel/main.py). Recommendation: `durability="sync"`. k0 runs are a few super-steps of slow model calls, so the cost is negligible, and a crash mid-CODER then resumes from the last finished node. `checkpoint_during=` is deprecated in favor of `durability=`.
- **Resume after the process exits.** Open the same SQLite file and compile the same graph. Then read `graph.get_state({"configurable": {"thread_id": run_id}})`. `StateSnapshot.interrupts` lists the interrupts still waiting for an answer, and `.next` lists the pending nodes ([`types.py`, `StateSnapshot`](https://github.com/langchain-ai/langgraph/blob/1.2.17/libs/langgraph/langgraph/types.py)). Answer with `Command(resume=...)`. A crash between steps resumes by invoking with `None` input on the same thread.
- **Thread id.** Use the run id k0 already generates for JSONL (`_begin_event_run` → `uuid4().hex`), so one JSONL `run` maps to one LangGraph thread.
- **Where the DB lives.** Use a gitignored runtime file such as `.k0/checkpoints.sqlite`, kept separate from `.k0-mem/session.sqlite`. Checkpoints hold full message blobs and are throwaway. Memory is curated.
- **Serializer safety.** `JsonPlusSerializer` (ormsgpack) by default deserializes any type and logs a warning. The upstream docstring warns that write access to the checkpoint DB can lead to code execution. Set `LANGGRAPH_STRICT_MSGPACK=true` or pass `allowed_msgpack_modules`. Source: [`serde/jsonplus.py` L104-135](https://github.com/langchain-ai/langgraph/blob/checkpoint%3D%3D4.3.0/libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py). k0 state should therefore stay plain types (str/int/list/dict/TypedDict), and strict mode should be on.
- **`HumanInTheLoopMiddleware`** (langchain) is the prebuilt per-tool approval mechanism (`interrupt_on={tool_name: ...}`). Since langchain 1.4.6 it has `interrupt_mode="per_call"`, typed `response_schema` on each interrupt, and answers validated before they are saved. Batched mode is still the default, and per-call is planned as the default in the next major release. Source: [changelog, langchain v1.4.6](https://docs.langchain.com/oss/python/releases/changelog.md). It is optional for 0.1.0, for example to approve shell commands. The core gates stay graph nodes.

## 4. Recursion and iteration limits per node and per run

Four layers, from coarse to fine:

1. **Per run.** Set `config={"recursion_limit": N}` (a top-level config key, not inside `configurable`) on the outer `invoke`/`stream`. Going over it raises `langgraph.errors.GraphRecursionError`. A `remaining_steps: RemainingSteps` state field lets a router degrade gracefully before the limit. Source: [Graph API, "Recursion limit"](https://docs.langchain.com/oss/python/langgraph/graph-api.md).
   - **Docs/source drift:** the docs say the default is 1000 "starting in version 1.0.6". The source at 1.2.17 has `DEFAULT_RECURSION_LIMIT = getenv("LANGGRAPH_DEFAULT_RECURSION_LIMIT", "10007")` ([`_internal/_config.py` L42](https://github.com/langchain-ai/langgraph/blob/1.2.17/libs/langgraph/langgraph/_internal/_config.py)). k0 must always pass the limit explicitly.
2. **Per agent.** `create_agent` binds `{"recursion_limit": 9_999}` to every agent it builds ([`factory.py` L1897-1899](https://github.com/langchain-ai/langchain/blob/langchain%3D%3D1.4.7/libs/langchain_v1/langchain/agents/factory.py)). Override it with `agent.with_config(recursion_limit=M)`. Since 1.2.17, a limit bound to a subgraph is kept when the parent runs at the default, and an explicit limit applies wherever configs merge ([PR #9417](https://github.com/langchain-ai/langgraph/pull/9417)). Before that fix, a looping agent could run to 9999 steps. One model→tools round costs about 2 super-steps.
3. **Per agent, in calls.** `ModelCallLimitMiddleware(run_limit=..., thread_limit=..., exit_behavior="end"|"error")` and `ToolCallLimitMiddleware(tool_name=None|"x", run_limit=..., thread_limit=..., exit_behavior="continue"|"error"|"end")`. They raise `ModelCallLimitExceededError` / `ToolCallLimitExceededError` on `"error"`. `run_limit` is per invocation, so it fits fresh-context agents. `thread_limit` needs a checkpointer. Sources: [Prebuilt middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in.md), [`model_call_limit.py`](https://github.com/langchain-ai/langchain/blob/langchain%3D%3D1.4.7/libs/langchain_v1/langchain/agents/middleware/model_call_limit.py), [`tool_call_limit.py`](https://github.com/langchain-ai/langchain/blob/langchain%3D%3D1.4.7/libs/langchain_v1/langchain/agents/middleware/tool_call_limit.py). These are the natural hooks for the cost caps that the map lists under "not yet specified".
4. **Per workflow loop.** k0's own `attempt` counter lives in run state and is checked by the VALIDATOR router. It replaces `config["max_retries"]` in today's loop. This is the limit the human reasons about ("CODER gets 3 tries"). The recursion limits are only a safety net.

Also available: per-node `timeout=TimeoutPolicy(...)` (async only) and graceful drain through `RunControl.request_drain()`, which raises `GraphDrained` and leaves a resumable checkpoint. That could back a Ctrl-C in the REPL. Source: [changelog, langgraph v1.2.0](https://docs.langchain.com/oss/python/releases/changelog.md).

## 5. Streaming events for one progress line per agent step

Two APIs exist:

- **Stable: `graph.stream(..., stream_mode=[...], subgraphs=True, version="v2")`** (since 1.1). Every chunk is `{"type": mode, "ns": tuple, "data": ...}`. `ns == ()` is the outer graph, and `("coder:<task_id>",)` is inside the CODER agent. Modes: `values`, `updates`, `messages`, `custom`, `checkpoints`, `tasks`, `debug`. Without `subgraphs=True`, nothing from inside a compiled agent is emitted. `get_stream_writer()` (from `langgraph.config`) emits `custom` events from nodes and tools, and on Python 3.11+ it works in async too. Source: [Streaming docs](https://docs.langchain.com/oss/python/langgraph/streaming.md).
- **Experimental: `graph.stream_events(..., version="v3")`** (since 1.2.0), with typed projections (`stream.messages`, `stream.values`, `stream.subgraphs`, `stream.interrupts`, `stream.interrupted`) and `tools`/`lifecycle` channels (`tool-started`, `tool-finished`, `started`, `completed`, `interrupted`, ...). The docs call it "the recommended in-process streaming model". The source still marks it `@beta("The v3 streaming protocol on Pregel is experimental.")`, and the docstring says "experimental and may change". It already had a breaking change in `langchain-core` 1.6.13/1.7.0, when tool-call args moved to `tool-call-delta` and became incremental. Sources: [Event streaming docs](https://docs.langchain.com/oss/python/langgraph/event-streaming.md), [`pregel/main.py` L3593, L3727](https://github.com/langchain-ai/langgraph/blob/1.2.17/libs/langgraph/langgraph/pregel/main.py), [changelog, langchain-core v1.7.0](https://docs.langchain.com/oss/python/releases/changelog.md).

Recommendation: use stable v2 with `stream_mode=["updates", "custom"]` and `subgraphs=True`.

- An `updates` chunk with `ns=()` means an outer node (agent or gate) finished: one progress line per agent step.
- `updates` chunks with `ns` inside an agent mean a `model` or `tools` node finished. Use them for "coder: model call 3 / tool write_file" lines.
- `custom` carries k0's own events, such as "running test_cmd".

Hide the stream behind a k0 adapter so moving to v3 later is one module. After the stream ends, read `graph.get_state(cfg).interrupts` to detect a gate deterministically, instead of parsing interrupts out of the stream. For one JSONL event per model invocation, a k0 `AgentMiddleware.wrap_model_call` is the cleanest hook (it sees the request, the response and usage). That detail belongs to the tracing ticket.

## 6. Long-term `BaseStore` vs k0's own SQLite for session memory

- `BaseStore` gives cross-thread key/value memory under tuple namespaces (`put`, `get`, `search` with prefix matching, `list_namespaces`), plus optional semantic search when configured with embeddings. It is passed through `compile(store=...)` / `create_agent(store=...)` and reached in tools via `InjectedStore`/`ToolRuntime`. Source: [Stores docs](https://docs.langchain.com/oss/python/langgraph/stores.md).
- A local SQLite implementation exists: `langgraph.store.sqlite.SqliteStore`, in `langgraph-checkpoint-sqlite` 3.1.2. It loads `sqlite-vec` for vector search. Source: [`store/sqlite/base.py` L789](https://github.com/langchain-ai/langgraph/blob/checkpointsqlite%3D%3D3.1.2/libs/checkpoint-sqlite/langgraph/store/sqlite/base.py). The docs list only Postgres, Mongo, Redis and Upstash as production stores. They call `InMemoryStore` dev-only and do not mention `SqliteStore` there.

Recommendation: **keep `.k0-mem/session.sqlite` (`src/k0ntrol/harness/session_memory.py`) as it is, and do not adopt `BaseStore` in 0.1.0.**

- k0's memory is relational and human-curated: facts, episodes, and `mem_proposals` with a pending/approved/rejected status feeding `Mem.md` and reflect. A JSON-value namespace store adds nothing for that.
- With `BaseStore`, any agent given `InjectedStore` can write memory directly, which works against code-enforced allowlists.
- k0 has no embedding model requirement today.

Expose memory to agents as explicit read-only k0 tools on the agents that need them, and leave writes to deterministic harness code (reflect). Revisit `SqliteStore` only if semantic recall becomes a requirement.

## Deprecations and drift to know about

| Item | Status | Replacement | Source |
| - | - | - | - |
| `langgraph.prebuilt.create_react_agent` | Deprecated since v1.0 (`LangGraphDeprecatedSinceV10`), still exported | `langchain.agents.create_agent` | [chat_agent_executor.py L274](https://github.com/langchain-ai/langgraph/blob/prebuilt%3D%3D1.1.1/libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py) |
| `create_react_agent(prompt=, pre_model_hook=, post_model_hook=)` | Goes with the function | `system_prompt=`, middleware | [migration guide](https://docs.langchain.com/oss/python/migrate/langgraph-v1) |
| `config_schema` kwarg | Deprecated, removal in v2.0 | `context_schema` | chat_agent_executor.py L471-521 |
| `langgraph.prebuilt` `AgentState*` | Deprecated | `langchain.agents.AgentState` | migration guide |
| `HumanInterrupt`, `HumanInterruptConfig`, `ActionRequest` | Deprecated | `langchain.agents.middleware.human_in_the_loop` `HITLRequest` / `InterruptOnConfig` | migration guide |
| `ValidationNode`, `MessageGraph` | Deprecated | none needed / `StateGraph` with `messages` | migration guide |
| `checkpoint_during=` | Deprecated | `durability=` | [pregel/main.py L2818](https://github.com/langchain-ai/langgraph/blob/1.2.17/libs/langgraph/langgraph/pregel/main.py) |
| Dict-style access on v2 `GraphOutput` (`result["__interrupt__"]`) | Deprecated | `.value`, `.interrupts` | [Streaming docs](https://docs.langchain.com/oss/python/langgraph/streaming.md) |
| `stream_events(version="v3")` | Experimental (`@beta`), docs call it recommended | use `stream(version="v2")` for now | pregel/main.py L3593 |
| `HumanInTheLoopMiddleware` batched mode | Still the default, `per_call` to become the default in the next major | `interrupt_mode="per_call"` | changelog v1.4.6 |
| `langchain-mcp-adapters` | Replaced | `langchain.mcp` (beta) | changelog v1.4.0 |
| Default recursion limit | Docs say 1000; source says 10007; `create_agent` binds 9999 | Always set it explicitly | `_config.py` L42, `factory.py` L1899 |

## Open risks

- **Dependency weight.** `langchain` + `langchain-core` + `langsmith` + `langgraph` + `langgraph-checkpoint-sqlite` + `aiosqlite` + `sqlite-vec` (a native wheel) + provider packages. Each needs the human's approval under AGENTS.md "ask before adding". `sqlite-vec` comes in only through the store module, but it is a hard dependency of the package.
- **Release churn.** langgraph shipped 6 patch releases between 2026-09-21 and 2026-10-24, and langchain-core changed streamed tool-call semantics in a minor release. Pin to minor ranges and run the suite on upgrade.
- **Local models and tool calling.** `create_agent` relies on `bind_tools`. Weak tool calling shows up as invalid-tool error messages and call-limit exits, not crashes. Measuring this is the provider ticket's job.
- **Interrupt idempotency.** Any side effect placed before `interrupt()` in a node, or before `agent.invoke` in a wrapper whose agent interrupts, runs twice on resume.
- **Checkpoint DB as attack surface.** Use strict msgpack and plain-typed state (see §3).
- **Unverified here.** Nothing was executed for this research. The sketch and the per-agent `with_config(recursion_limit=...)` behavior come from source and the PR description. A short spike test should confirm them when 0.1.0 starts: build the graph with `SqliteSaver`, gate, kill the process, resume, and test the agent limit. Testing fakes: `GenericFakeChatModel` needs a `bind_tools` subclass to work with `create_agent` ([Unit testing](https://docs.langchain.com/oss/python/langchain/test/unit-testing.md)).
