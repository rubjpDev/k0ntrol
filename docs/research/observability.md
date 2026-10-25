# Research: Langfuse and JSONL from one LangGraph run

Ticket: rubjpDev/k0ntrol#9 (map: #2). Researched 2026-10-25.

**Question.** How does a LangGraph run reach Langfuse and an always-on local JSONL log from the same events?

**Short answer.** Use two LangChain callback handlers on the same `config["callbacks"]` list. LangChain fans every run event out to all of them. k0 owns a small JSONL handler that is always attached. The Langfuse `CallbackHandler` is appended only when both Langfuse keys are set, so with no keys nothing is imported or constructed. The k0 run id (already `uuid4().hex`, 32 lowercase hex) becomes the Langfuse trace id, so the JSONL line and the Langfuse trace join on one key. A later plain OTel exporter is added as a second span processor on the OTel `TracerProvider`. JSONL does not change.

## Pinned versions (PyPI, 2026-10-25)

| Package | Version | Released | Notes |
|---|---|---|---|
| `langfuse` | 5.0.4 | 2026-10-21 | `requires_python >=3.11,<4.0`; v5.0.x line started 2026-10-14 |
| `langchain-core` | 1.7.0 | 2026-10-22 | Langfuse v5 requires `langchain-core>=1` |
| `langgraph` | 1.2.17 | 2026-10-24 | |
| `langchain` | 1.4.7 | 2026-10-21 | no longer needed by the Langfuse handler in v5 |
| `opentelemetry-api` / `-sdk` / `-exporter-otlp-proto-http` | 1.45.1 | 2026-10-06 | hard deps of `langfuse` 5.x (`>=1.45.0,<2`) |
| `langchain-openrouter` | 0.2.9 | | puts provider-reported cost in `response_metadata["cost"]` |
| `openinference-instrumentation-langchain` | 0.1.80 | 2026-10-23 | only relevant to the later OTel option |
| `opentelemetry-instrumentation-langchain` (OpenLLMetry) | 0.62.4 | 2026-09-29 | same |
| Langfuse server | v4.70.0 | 2026-10-24 | SDK v5 needs Cloud or self-hosted >= v4.59.0 |

Sources: `https://pypi.org/pypi/<name>/json`; `gh release list --repo langfuse/langfuse-python`; `gh release list --repo langfuse/langfuse`.

`langfuse` 5.0.4 `requires_dist`: `httpx`, `pydantic>=2`, `backoff`, `wrapt`, `packaging`, `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`, `typing-extensions`. That is a large tree next to k0's current `pyyaml` + `rich`, so it belongs in an optional extra (`k0ntrol[langfuse]`). AGENTS.md requires asking the human before adding it.

## 1. Langfuse's LangChain/LangGraph handler on the current SDK

### It is OTel-based

- "The Langfuse SDKs are built on top of OpenTelemetry." "The Python SDK automatically sets up OpenTelemetry when initializing the client." (https://langfuse.com/docs/observability/sdk/overview)
- In source (`langfuse/_client/resource_manager.py`, v5.0.4, `_init_tracer_provider`): if no global `TracerProvider` is registered (the global is still a `ProxyTracerProvider`), Langfuse creates one and calls `otel_trace_api.set_tracer_provider(provider)`. That makes it the **global** provider. If one is already registered, Langfuse reuses it and adds its `LangfuseSpanProcessor` to it. You can pass `Langfuse(tracer_provider=...)` to keep Langfuse isolated.
- The handler (`langfuse/langchain/CallbackHandler.py`, class `LangchainCallbackHandler`, exported as `langfuse.langchain.CallbackHandler`) is a `langchain_core` `BaseCallbackHandler`. It turns LangChain run events into Langfuse observations (OTel spans with `langfuse.*` attributes). Types are span, generation, agent, chain, tool and retriever.
- Constructor in v5: `CallbackHandler(*, public_key=None, trace_context=None)`. It calls `get_client(public_key=...)`. All other config (keys, base URL, `tracing_enabled`, `sample_rate`, flush settings) lives on the `Langfuse` client or in env vars.
- v5 breaking changes that matter to k0: the handler imports only `langchain_core`; Python >=3.11; server >= v4.59.0, and self-hosted v3 is not supported; metadata values are JSON-encoded on spans; `blocked_instrumentation_scopes` is removed in favour of `should_export_span`. (https://langfuse.com/docs/observability/sdk/upgrade-path/python-v4-to-v5)
- v4 changes still in force: `CallbackHandler(update_trace=...)` is removed. Trace attributes go through `propagate_attributes()` or the `langfuse_*` metadata keys. The default span filter exports only Langfuse spans, `gen_ai.*` spans and known LLM instrumentation scopes. (https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4)

### Wiring into LangGraph

From https://langfuse.com/integrations/frameworks/langchain:

```python
from langfuse.langchain import CallbackHandler
handler = CallbackHandler()
graph.invoke(inputs, config={"callbacks": [handler]})
```

- "Each invocation becomes its own trace." To group invocations, wrap them in an enclosing observation, or pass the same `trace_context`.
- The handler has LangGraph resume handling (`_get_langgraph_resume_key`, `_PendingResumeTraceContextStore`). A resume after `interrupt()` can continue the original trace.
- Session, user and tags per invocation go in `config["metadata"]` keys: `langfuse_session_id`, `langfuse_user_id`, `langfuse_tags`, `langfuse_trace_name`. The handler strips them before storing the remaining metadata (`_strip_langfuse_keys_from_dict`).
- Deterministic trace id: `Langfuse.create_trace_id(seed=...)` returns 32 lowercase hex. `CallbackHandler(trace_context={"trace_id": ...})` makes the root LangChain run use it. k0's run id is already `uuid4().hex` (`src/k0ntrol/harness/loop.py:38`), which is a valid trace id as is.

### Self-hosted vs cloud

- Only `LANGFUSE_BASE_URL` changes. In v5.0.4 source the resolution order is `base_url` arg, `LANGFUSE_BASE_URL`, `host` arg, then `LANGFUSE_HOST`, with a default of `https://cloud.langfuse.com` (EU). US is `https://us.cloud.langfuse.com`. A local docker install is `http://localhost:3000`.
- Self-hosting (https://langfuse.com/self-hosting, "Version: v4") needs Web and Worker containers plus PostgreSQL, Redis/Valkey, ClickHouse and S3/blob storage. Docker Compose is for local use ("Single VM without high availability, scaling, or backups"). Helm/Kubernetes is for production. The core is open source, and some features need a license key (EE).
- Pin a floor: SDK 5.x needs server >= v4.59.0. The upgrade guide says self-hosted v3 is unsupported; how it fails was not tested.

### Staying off with no keys

From source (`langfuse/_client/client.py`, `Langfuse.__init__`, v5.0.4):

- If `public_key` / `LANGFUSE_PUBLIC_KEY` is missing (or the secret key is), it logs a **warning**, "Authentication error: Langfuse client initialized without public_key. Client will be disabled.", and uses an OTel `NoOpTracer`. Nothing is sent and no `TracerProvider` is created.
- `tracing_enabled=False` or `LANGFUSE_TRACING_ENABLED=false` logs at info level, "Langfuse tracing is explicitly disabled", and skips the span processor.
- `OTEL_SDK_DISABLED=true` also disables export.
- The docs do not describe the no-key behaviour (overview and FAQ pages). This comes from source only.

**Implication for k0.** Do not construct `CallbackHandler()` without keys. It would emit a warning on every run and import OTel, httpx and pydantic for nothing. Gate it in code: build the handler only if `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are both set and the `langfuse` extra is importable. That gate is deterministic code, consistent with "gates in code".

### Flushing in a CLI

`Langfuse` registers `atexit.register(self.shutdown)` (`resource_manager.py:278`). Spans are batched and exported on a background thread. For the CLI, call `get_client().flush()` at the end of `k0 run` / before the REPL exits. The docs advise this for short-lived processes: "Call `langfuse.shutdown()` (or `langfuse.flush()` in short-lived jobs)" (https://langfuse.com/docs/observability/sdk/troubleshooting-and-faq). That way an `interrupt()` gate waiting on the human does not hold unsent spans.

## 2. Per-agent token usage, cost and latency

### Where the data comes from (LangChain side)

- Every chat model run fires `on_chat_model_start(serialized, messages, *, run_id, parent_run_id, tags, metadata)` and `on_llm_end(response: LLMResult, *, run_id, parent_run_id)` (`langchain_core/callbacks/base.py`, 1.7.0).
- `RunnableConfig.callbacks` is "Callbacks for this call and any sub-calls ... Tags are passed to all callbacks, metadata is passed to handle*Start callbacks." (`langchain_core/runnables/config.py:92`). Handlers passed at `graph.invoke` therefore reach every model and tool call inside every node.
- LangGraph injects node metadata into each task's config: `langgraph_step`, `langgraph_node`, `langgraph_triggers`, `langgraph_path`, `langgraph_checkpoint_ns` (`langgraph/pregel/_algo.py:654`, 1.2.17). So the "agent" (node) is known in `on_chat_model_start`.
- Chat models add `ls_provider`, `ls_model_name` and `ls_model_type` to the start metadata (`langchain_core/language_models/base.py:46`).
- Token usage is standardised as `AIMessage.usage_metadata: UsageMetadata` with `input_tokens`, `output_tokens`, `total_tokens` and optional `input_token_details` / `output_token_details` (cache reads, reasoning). See `langchain_core/messages/ai.py`. `langchain_core.callbacks.UsageMetadataCallbackHandler` is a ready reference: it reads `response.generations[0][0].message.usage_metadata` in `on_llm_end`.
- Ollama fills `usage_metadata` (`langchain_ollama/chat_models.py`).
- **Streaming caveat.** `ChatOpenAI.stream_usage` is "enabled unless `openai_api_base` is set or the model is initialized with a custom client" (`langchain_openai/chat_models/base.py:898`). Pointing `ChatOpenAI` at LM Studio, vLLM or another OpenAI-compatible base URL and streaming gives **no tokens** unless k0 sets `stream_usage=True`.
- **Cost from the provider.** `langchain-openrouter` copies OpenRouter's `usage.cost` / `cost_details` into `response_metadata["cost"]` (`langchain_openrouter/chat_models.py:622`). Most other providers report tokens only.

### What Langfuse does with it

- On `on_llm_end` the handler parses usage from `llm_output["token_usage"|"usage"]`, `generation_info["usage_metadata"]`, Bedrock `response_metadata` or `message.usage_metadata`. It converts inclusive counts into exclusive buckets (`input`, `output`, `input_cache_read`, ...) and sets `usage_details` and `model` on the generation (`CallbackHandler.py:1293`, `_parse_usage`, `_parse_usage_model`). It does **not** read `response_metadata["cost"]`.
- Cost is computed by Langfuse at ingestion: "ingested values take priority over inferred ones". Without ingested cost, it multiplies usage by the model definition's price. Model definitions match on a regex of the model name. Built-ins cover OpenAI, Anthropic and Google. Local or self-hosted models need a custom model definition (UI or `/api/public/models`), otherwise "no cost is calculated". (https://langfuse.com/docs/observability/features/token-and-cost-tracking)
- Latency is the observation's span duration. Time-to-first-token comes from `on_llm_new_token` setting `completion_start_time` (`CallbackHandler.py:169`).
- Per-agent views in Langfuse come from the observation tree (node chain span, then the generation). Filtering by agent is easiest if k0 also puts `k0_agent` in the observation metadata.

### What the JSONL handler should capture (proposal)

One line per model invocation (the existing invariant), written in `on_llm_end` / `on_llm_error`, keyed by `run_id` from `on_chat_model_start`:

`ts, run (= trace id), step, agent (langgraph_node or k0_agent), provider (ls_provider), model (ls_model_name or response model_name), ms (monotonic start to end), ok, err, tokens_in, tokens_out, tokens (total), cost_usd (only when the provider reports it)`.

The current fields (`ts, run, step, mode, model, ms, chars, ok, err, tokens`, from `src/k0ntrol/backends/invoke.py:_record_event`) map onto this almost one to one. `mode` becomes `agent`.

Cost in JSONL: do not infer it in 0.1.0. Record provider-reported cost when present. Leave pricing tables to Langfuse until the budgets work decides whether k0 needs its own (see risks).

## 3. One callback for both, or a separate JSONL handler?

**Separate handler, same callback list.** LangChain's callback manager dispatches each event to every handler in the list, so "the same events" happen by construction. Each sink stays independent.

Options considered:

| Option | How | Verdict |
|---|---|---|
| A. Two handlers | `callbacks=[K0JsonlHandler(...), *([CallbackHandler(trace_context=...)] if keys else [])]` | **Recommended.** JSONL needs only `langchain_core`, which LangGraph already brings. It works with Langfuse absent. Each handler is testable alone. |
| B. Subclass Langfuse's handler and also write JSONL | override `on_llm_end` etc. | Rejected. It makes always-on JSONL depend on the optional extra, ties k0 to Langfuse private internals (`_detach_observation`, run memo), and breaks on SDK majors (v4 and v5 both landed in 2026). |
| C. OTel-first: JSONL as a `SpanExporter` on Langfuse's `TracerProvider` | Langfuse handler produces spans, and a k0 exporter writes JSONL lines from `langfuse.observation.usage_details` | Rejected for 0.1.0. JSONL would only exist when OTel and Langfuse are installed and enabled, and it would parse `langfuse.*` attributes whose encoding changed in v5 (metadata JSON-encoded). It could return later if k0 owns the provider (see section 4). |

Handler notes, from langchain-core 1.7.0 source:

- `BaseCallbackHandler.raise_error = False` by default. A failing handler is logged and swallowed, which is fine for Langfuse. For the k0 handler, consider `raise_error=True` in tests so a broken JSONL write fails the suite.
- Under async execution, sync handlers run in a thread pool unless `run_inline = True` (`callbacks/manager.py:427`). The JSONL handler should set `run_inline = True`, or guard appends with a lock, to keep line order and avoid interleaved writes.
- LangGraph 1.2.x adds `langgraph.callbacks.GraphCallbackHandler` with `on_interrupt(GraphInterruptEvent)` and `on_resume(GraphResumeEvent)` (checkpoint id, namespace, interrupt payloads). If the k0 handler subclasses it, the JSONL log can also record human gates (`interrupt()`). Only handlers that inherit from `GraphCallbackHandler` receive these events.
- Correlation: pass `config={"callbacks": [...], "metadata": {"k0_run": run, "k0_agent": ..., "langfuse_session_id": <work item id>, "langfuse_tags": [...]}, "run_name": ...}`. Use `trace_context={"trace_id": run}`, and JSONL `run` holds the same 32-hex value. A JSONL line can then be looked up in Langfuse by trace id.
- `update_last_event` (`invoke.py:31`) rewrites the last JSONL line after reflect (`loop.py:823`). That keeps working unchanged because the file format stays the same.

LangGraph streaming (https://docs.langchain.com/oss/python/langgraph/streaming) is for the TUI, not tracing. The modes are `values`, `updates`, `messages` (token, metadata with `langgraph_node`), `custom`, `checkpoints`, `tasks` and `debug`, and LangGraph >= 1.1 has `version="v2"` `StreamPart` dicts. Streaming does not replace callbacks for usage capture. The docs do not promise usage in streamed events, while `on_llm_end` always receives the final `LLMResult`.

## 4. What changes when k0 adds a plain OTel exporter

- **Shared provider.** k0 creates and registers its own `TracerProvider` **before** constructing the Langfuse client, then adds `BatchSpanProcessor(OTLPSpanExporter(...))`. Langfuse detects the existing global provider and attaches its own processor to it (`_init_tracer_provider`). Langfuse-handler spans then go to both Langfuse and the OTLP backend, and no k0 callback changes.
  - The spans carry `langfuse.*` attributes (`langfuse/_client/attributes.py`: `langfuse.observation.type`, `...usage_details`, `...model.name`), not OTel GenAI `gen_ai.*` semconv. A generic backend shows the tree but no usage fields.
- **Isolation.** Langfuse documents `Langfuse(tracer_provider=TracerProvider())` without registering it globally, so Langfuse and other backends don't see each other's spans. It warns that providers "share the same OpenTelemetry context", which can orphan spans. (https://langfuse.com/docs/observability/sdk/advanced-features)
- **Semconv-native instrumentation.** Use `openinference-instrumentation-langchain` or `opentelemetry-instrumentation-langchain` (OpenLLMetry) instead of the Langfuse handler. Both scopes are on Langfuse's default export allowlist (`KNOWN_LLM_INSTRUMENTATION_SCOPE_PREFIXES` in `langfuse/_client/span_filter.py`: `openinference`, `opentelemetry.instrumentation.langchain`). Langfuse can also ingest plain OTLP/HTTP at `<base_url>/api/public/otel` with Basic auth (`pk:sk`) and header `x-langfuse-ingestion-version=4`. gRPC is not supported. (https://langfuse.com/integrations/native/opentelemetry)
  - At that point k0 could drop the Langfuse handler entirely and treat Langfuse as one OTLP destination.
  - Do not run both the Langfuse handler and a LangChain OTel instrumentor, or every model call is double-counted.
- **Semconv is not stable.** GenAI semconv moved to `open-telemetry/semantic-conventions-genai`. Span attributes such as `gen_ai.operation.name` and `gen_ai.usage.input_tokens` are still **Development** status (`docs/gen-ai/gen-ai-spans.md`). Do not make JSONL field names follow them yet.
- **JSONL is unaffected** in every variant, because it is a separate callback. The one thing to lock now is the id format. `run` = 32 lowercase hex = a W3C/OTel trace id, so JSONL lines join with any OTel backend later.

## Recommendation

1. **Two handlers on one callback list.** Use a k0 `JsonlEventsHandler` (subclass of `langgraph.callbacks.GraphCallbackHandler`, `run_inline = True`) that is always attached. Append Langfuse's `CallbackHandler(trace_context={"trace_id": run})` only when both keys are set and the `langfuse` extra imports. Pass both at the graph's `invoke` / `stream` call.
2. **The run id is the trace id.** Keep `uuid4().hex` and carry it in config metadata (`k0_run`, `k0_agent`). JSONL `run` = Langfuse trace id = future OTel trace id.
3. **JSONL schema:** one line per model invocation (invariant kept), with `agent`, `provider`, `model`, `ms`, `tokens_in`, `tokens_out`, `tokens`, `ok`, `err`, and `cost_usd` only when the provider reports it. Optional extra event kinds for `interrupt` / `resume` gates.
4. **Langfuse as an optional extra** `k0ntrol[langfuse]` pinned `langfuse>=5.0.4,<6`, which needs human approval per AGENTS.md. Document the server floor (>= v4.59.0) and `LANGFUSE_BASE_URL` for self-hosting. Call `get_client().flush()` at run end.
5. **Defer plain OTel.** When it comes, k0 owns and registers the `TracerProvider` first and adds an OTLP processor. Pick semconv instrumentation over the Langfuse handler at that point. JSONL does not change.

## Open risks

- **Cost caps cannot lean on Langfuse.** Langfuse cost is computed asynchronously, server-side and only when keys are set. Local or OpenRouter model names may not match its price table. If the budgets work needs in-run cost caps, k0 needs its own price table, or must use provider-reported cost (OpenRouter only today).
- **Missing tokens on OpenAI-compatible local servers when streaming.** k0's provider factory must set `stream_usage=True` for `ChatOpenAI` with a custom base URL. `k0 doctor` could check that a test call returns `usage_metadata`.
- **SDK churn.** Langfuse shipped v4 (2026-03) and v5 (2026-10) with breaking changes. Pin `<6` and keep the k0 integration to the public `CallbackHandler` + `get_client().flush()` only.
- **Server floor.** SDK v5 does not support self-hosted v3 servers. Users on older self-hosts must upgrade or pin `langfuse<5`.
- **Global TracerProvider side effect.** The first `Langfuse()` with keys registers a global provider. If another library in the user's process sets one later, that call is ignored and spans go to Langfuse's provider. This matters when k0 adds OTel: construct the k0 provider first.
- **Payload privacy.** The Langfuse handler sends full prompts, outputs and tool inputs, including repo source code under D24, to the Langfuse server. Cloud users should know this. `mask_otel_spans` exists if redaction is needed.
- **Docs gaps.** No-key behaviour and global-provider registration come from SDK source (v5.0.4), not docs. Re-check on SDK upgrades.

## Sources

- Langfuse Python SDK v5.0.4 source: https://github.com/langfuse/langfuse-python/tree/v5.0.4 (`langfuse/_client/client.py`, `resource_manager.py`, `span_filter.py`, `attributes.py`, `get_client.py`, `langfuse/langchain/CallbackHandler.py`)
- Langfuse release notes: https://github.com/langfuse/langfuse-python/releases (v5.0.1 to v5.0.4, v4.0.0)
- https://langfuse.com/docs/observability/sdk/upgrade-path/python-v4-to-v5
- https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4
- https://langfuse.com/integrations/frameworks/langchain
- https://langfuse.com/docs/observability/sdk/overview
- https://langfuse.com/docs/observability/sdk/advanced-features
- https://langfuse.com/docs/observability/sdk/troubleshooting-and-faq
- https://langfuse.com/docs/observability/features/token-and-cost-tracking
- https://langfuse.com/integrations/native/opentelemetry
- https://langfuse.com/self-hosting
- langchain-core 1.7.0 source: https://github.com/langchain-ai/langchain/tree/langchain-core==1.7.0 (`libs/core/langchain_core/callbacks/{base,manager,usage}.py`, `runnables/config.py`, `messages/ai.py`, `language_models/base.py`; `libs/partners/openai/.../chat_models/base.py`; `libs/partners/ollama/.../chat_models.py`; `libs/partners/openrouter/.../chat_models.py`)
- langgraph 1.2.17 source: https://github.com/langchain-ai/langgraph/tree/1.2.17 (`libs/langgraph/langgraph/callbacks.py`, `pregel/_algo.py`)
- https://docs.langchain.com/oss/python/langgraph/streaming
- OTel GenAI semconv: https://github.com/open-telemetry/semantic-conventions-genai (`docs/gen-ai/gen-ai-spans.md`)
- PyPI JSON API for all version pins
