# Plain OpenTelemetry export is an optional `otel` extra

ADR 0008 listed the optional extras and kept Langfuse as one of them. This ADR adds one more, `otel` (`opentelemetry-api>=1.45.0,<2`, `opentelemetry-sdk>=1.45.0,<2`, `opentelemetry-exporter-otlp-proto-http>=1.45.0,<2`), so a run can also export spans over OTLP/HTTP to any backend (Jaeger, Tempo, Honeycomb, a collector) without Langfuse. The pins are exactly the ones `langfuse` 5.x already requires, so installing `k0ntrol[langfuse]` already satisfies the extra and no new package enters the install. The core dependency list from ADR 0005 and 0008 does not change.

It is off unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set (the standard variable; `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_SERVICE_NAME` and the other standard ones are read by the SDK; `OTEL_SDK_DISABLED=true` switches it off). With the variable unset, or the packages missing, nothing is imported or constructed. The endpoint is OTLP over HTTP, for example `http://localhost:4318`; gRPC is not supported. The trace id is the k0 run id (the chat id for `/ask`), spans are flushed at every gate and at run end, and an export error never changes a run's outcome.

We chose a small callback-to-span bridge in `src/k0ntrol/otel.py`, written on the OTel API, over an instrumentation package. The research (`docs/research/observability.md`, section 4) names `openinference-instrumentation-langchain` and OpenLLMetry for that job; each is a new dependency that would need its own ADR, and the bridge is one small module. It runs on its own `TracerProvider`, never the global one. The research suggested k0 register a shared provider before Langfuse and add an OTLP span processor to it; we did not, because Langfuse's default span filter also exports `gen_ai.*` spans, so k0's own model-call spans on a shared provider would reach Langfuse a second time, and Langfuse's own spans would reach the OTLP backend with `langfuse.*` attributes it cannot read. Two private pipelines give each backend one clean copy and make start-up order irrelevant.

## Considered Options

- Share the global `TracerProvider` with Langfuse and add an OTLP span processor, as the research suggested: rejected, per the duplicate-span and attribute-format reasons above, and because the first library to register a provider would decide where spans go.
- Add `openinference-instrumentation-langchain` or `opentelemetry-instrumentation-langchain`: rejected for now, because it adds a runtime dependency and would double-count model calls whenever the Langfuse handler is on.
- Put the OTLP exporter in the `langfuse` extra: rejected, because OTel export has nothing to do with Langfuse and should work without it.

## Consequences

- `uv sync --extra otel` (or `--extra langfuse`) installs the packages; the test suite passes with or without them, and the on-path tests skip when they are missing.
- Spans carry names, `gen_ai.*` model and token attributes, the agent name and error text, masked with the same rule as the event log (`secret_env_values`). They never carry prompts, tool inputs or tool outputs, so repo source does not leave the machine through this exporter. Adding opt-in content capture is a later decision.
- GenAI semantic conventions are still in Development status, so attribute names may change with a minor release.
- All spans of one run hang under a remote parent that is never exported, so Jaeger and similar UIs may flag the root span as having a missing parent. The trace id is correct and every span joins on it.
- Langfuse's own handler spans are not copied to the OTLP backend; the backend gets k0's spans only. Helper-model calls outside the agent callbacks are not traced by this bridge yet.
