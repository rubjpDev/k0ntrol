# Provider layer: reaching local runtimes, lab APIs, OpenRouter and clouds

Research for [rubjpDev/k0ntrol#7](https://github.com/rubjpDev/k0ntrol/issues/7), part of the 0.1.0 LangGraph pivot map (#2). Date: 2026-10-25.

**Question.** What is the smallest way for a LangGraph app to reach Ollama, LM Studio, Anthropic, OpenAI, OpenRouter, AWS Bedrock and Google Vertex with tool calling, with the provider chosen per agent from config?

**Answer.** Use LangChain's `init_chat_model` with one partner package per provider, each behind an optional extra. LM Studio has no partner package, so it goes through `ChatOpenAI` with a `base_url`. Vertex goes through `langchain-google-genai` with `vertexai=True`. Do not use LiteLLM as the main path, and use plain OpenAI-compatible endpoints only for LM Studio and as an escape hatch for other servers.

## Pinned versions (PyPI, 2026-10-25)

| Package | Version | Released | Notes |
|---|---|---|---|
| `langgraph` | 1.2.17 | 2026-10-24 | needs `langchain-core>=1.4.7,<2` |
| `langchain` | 1.4.7 | 2026-10-21 | ships `init_chat_model`; deps: `langchain-core`, `langgraph`, `pydantic` |
| `langchain-core` | 1.7.0 | 2026-10-22 | `UsageMetadata`, `ModelProfile` |
| `langchain-openai` | 1.7.2 | 2026-10-14 | `openai>=2.45,<4`, `tiktoken` |
| `langchain-anthropic` | 1.7.6 | 2026-10-12 | `anthropic>=0.120,<2`, so it resolves `anthropic` 1.17.0, not 2.3.0 |
| `langchain-ollama` | 1.1.0 | 2026-04-07 | `ollama>=0.6.1,<1`; only release in 2026 |
| `langchain-openrouter` | 0.2.9 | 2026-09-22 | `openrouter>=0.9.2,<1`; beta |
| `langchain-aws` | 1.9.0 | 2026-10-15 | `boto3>=1.43.64`, `numpy` |
| `langchain-google-genai` | 4.4.2 | 2026-10-13 | `google-genai>=2.20,<3` |
| `langchain-google-vertexai` | 3.2.5 | 2026-10-23 | `google-cloud-aiplatform`, `google-cloud-storage`, `numexpr`, `bottleneck`, … |
| `langchain-litellm` | 0.11.1 | 2026-10-14 | `litellm>=1.101,<2`; requires Python ≥ 3.11 |
| `litellm` | 1.106.4 | 2026-10-25 | 258 releases in 2026 so far |
| `openai` | 3.37.0 | 2026-10-22 | extra `bedrock` (SigV4 provider) |

Source: `https://pypi.org/pypi/<name>/json`, fetched 2026-10-25.

## Option A: `init_chat_model` + partner packages

`init_chat_model(model, model_provider=..., **kwargs)` imports the partner class for that provider and forwards `**kwargs` to its constructor (`base_url`, `temperature`, `timeout`, `max_retries`, …). The `"provider:model"` form splits on the first colon only, so `ollama:qwen3:8b` resolves to provider `ollama` and model `qwen3:8b` (`langchain/chat_models/base.py::_parse_model`). `configurable_fields="any"` lets callers change `api_key` and `base_url` at runtime, and the reference warns against it for untrusted config. k0 should build each agent's model once from config, with no `configurable_fields`.
([init_chat_model reference](https://reference.langchain.com/python/langchain/chat_models/base/init_chat_model))

The provider-to-package map comes from the same reference and was checked in the installed `base.py`:

| k0 target | `model_provider` | Class | Package |
|---|---|---|---|
| Ollama | `ollama` | `ChatOllama` | `langchain-ollama` |
| LM Studio | `openai` + `base_url` | `ChatOpenAI` | `langchain-openai` (no LM Studio package exists) |
| Anthropic | `anthropic` | `ChatAnthropic` | `langchain-anthropic` |
| OpenAI | `openai` | `ChatOpenAI` | `langchain-openai` |
| OpenRouter | `openrouter` | `ChatOpenRouter` | `langchain-openrouter` |
| Bedrock | `bedrock_converse` (or `anthropic_bedrock`) | `ChatBedrockConverse` / `ChatAnthropicBedrock` | `langchain-aws` |
| Vertex (Gemini) | `google_genai` + `vertexai=True` | `ChatGoogleGenerativeAI` | `langchain-google-genai` |
| Vertex (Claude) | `google_anthropic_vertex` | `ChatAnthropicVertex` | `langchain-google-vertexai` (heavy) |
| anything LiteLLM covers | `litellm` | `ChatLiteLLM` | `langchain-litellm` |

**Offline smoke test.** With all six partner packages installed, `init_chat_model(...)` followed by `.bind_tools([tool])` built a model for all seven targets. Keys were fake and no network was used. Default `stream_usage` came out as `True` for `ChatAnthropic`, `ChatOpenAI` (OpenAI) and `ChatOpenRouter`. For `ChatOpenAI` pointed at a `base_url`, it came out as `None` (off). `model.profile` (`langchain_core.language_models.model_profile.ModelProfile`) returned `tool_calling`, `tool_call_streaming` and `max_input_tokens` for the Anthropic and OpenAI models, and `None` for `ChatOllama`.

### Tool calling per provider

- **Ollama.** `ChatOllama.bind_tools` uses native `/api/chat` tool calling, with parallel and streamed tool calls ([Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling); [LangChain ChatOllama](https://docs.langchain.com/oss/python/integrations/chat/ollama)). Only models with the `tools` capability can call tools. `ollama.ShowResponse.capabilities` (ollama-python) exposes that, so `k0 doctor` can check it. `ChatOllama` exposes `num_ctx` and `reasoning`.
- **LM Studio.** Every model gets some form of tool use. "Native" support needs a supported chat template, shown by a hammer badge. Other models fall back to a "default" mode driven by the system prompt, where "results will vary by model" and badly formatted calls come back as plain `content` ([LM Studio tool use](https://lmstudio.ai/docs/developer/openai-compat/tools)).
- **Anthropic, OpenAI, OpenRouter.** Tool calling, structured output and token-level streaming are all supported. `ChatOpenRouter` also supports `strict=True` tools, a `reasoning` setting and provider routing ([ChatOpenRouter](https://docs.langchain.com/oss/python/integrations/chat/openrouter)).
- **Bedrock.** `ChatBedrockConverse` supports tool calling, streaming and token usage, but not native async. LangChain recommends Converse "for users who do not need to use custom models" ([ChatBedrock](https://docs.langchain.com/oss/python/integrations/chat/bedrock)).
- **Vertex.** `ChatGoogleGenerativeAI` supports tool calling, streaming and `usage_metadata`. Version 4.0 moved it to `google-genai` and supersedes `ChatVertexAI` for Gemini ([ChatGoogleGenerativeAI](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai)).

### Auth and secrets

Every partner reads the vendor's own env vars or credential chain. k0 does not need its own secrets store.

| Target | Credentials |
|---|---|
| Ollama | none locally. Host from `base_url` or `OLLAMA_HOST` (ollama-python `_client.py`) |
| LM Studio | none by default ("does not require authentication"). If the token switch is on, a `Bearer` token ([LM Studio auth](https://lmstudio.ai/docs/developer/core/authentication)) |
| OpenAI | `OPENAI_API_KEY`. Base URL precedence: `base_url` kwarg, then `OPENAI_API_BASE`, then `OPENAI_BASE_URL` ([ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai)) |
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Bedrock | boto3 default chain (env keys, `credentials_profile_name`/`AWS_PROFILE`, SSO, instance roles). Region from `AWS_REGION`/`AWS_DEFAULT_REGION`. Bedrock API key via `AWS_BEARER_TOKEN_BEDROCK` (`bedrock_api_key` field, `langchain_aws/chat_models/bedrock_converse.py`) |
| Vertex | ADC (`gcloud auth application-default login`, `GOOGLE_APPLICATION_CREDENTIALS`, workload identity), plus `GOOGLE_CLOUD_PROJECT`/`GOOGLE_CLOUD_LOCATION` or `GOOGLE_GENAI_USE_VERTEXAI` |

### Streaming and token usage

Each partner fills the standard `AIMessage.usage_metadata` (`input_tokens`, `output_tokens`, `total_tokens`, plus details). That gives k0's JSONL events and the Langfuse LangChain callback a single field to read.

- `ChatOllama` reads usage from `prompt_eval_count` and `eval_count` (`_get_usage_metadata_from_generation_info`). LangChain's feature table marks token usage ❌ for ChatOllama, but the source shows it is reported.
- `ChatOpenAI` turns `stream_usage` off whenever `OPENAI_BASE_URL` is set, "because many non-OpenAI endpoints don't support streaming token usage." With a custom `base_url` the smoke test also showed `None`. k0 must set `stream_usage=True` explicitly for LM Studio and Ollama /v1, then check that the server sends usage.

### Footprint

Measured on Python 3.11 with `uv pip compile` (transitive package count) and `uv pip install` (`site-packages` size):

| Stack | Packages | Size |
|---|---|---|
| `langgraph` alone | 38 | 31 MB |
| + `langchain-openai` | 43 | — |
| + openai, anthropic, ollama | 49 | — |
| + all six partners (openai, anthropic, ollama, openrouter, aws, google-genai) | 69 | 144 MB, of which botocore is 25 MB and numpy 23 MB |
| `langgraph` + `langchain-google-vertexai` | 70 | 220 MB |
| `langgraph` + `langchain-litellm` | 79 | 214 MB, of which `litellm` alone is 107 MB |

Because each partner sits behind an extra, an Ollama-only user pays about 6 packages beyond `langgraph`. The `langchain` package already ships `ollama`, `openai`, `anthropic`, `aws`, `google-genai` and `google-vertexai` extras that k0 can mirror.

### Maintenance

All partners except `langchain-litellm` live in `langchain-ai` repos and move with `langchain-core` 1.x (`langchain-aws` had 36 releases in 2026, `langchain-google-genai` 20). `langchain-ollama` is the slow one: 1.1.0 (2026-04-07) is its only 2026 release. `langchain-openrouter` is marked beta.

## Option B: LiteLLM (direct, or via `ChatLiteLLM`)

- **Coverage.** One model-string grammar covers every target: `ollama_chat/<m>` ("We recommend using ollama_chat"), `lm_studio/<m>` (`LM_STUDIO_API_BASE`, optional `LM_STUDIO_API_KEY`), `openrouter/…`, `bedrock/converse/<m>`, `vertex_ai/<m>` ([Ollama](https://docs.litellm.ai/docs/providers/ollama), [LM Studio](https://docs.litellm.ai/docs/providers/lm_studio), [Bedrock](https://docs.litellm.ai/docs/providers/bedrock), [Vertex](https://docs.litellm.ai/docs/providers/vertex)).
- **Tool calling.** Tools use the OpenAI `tools` format. `litellm.supports_function_calling(model=…)` reports support. For Ollama models without native tools, LiteLLM "defaults to json mode tool calls", and `litellm.add_function_to_prompt` injects tools into the prompt ([function calling](https://docs.litellm.ai/docs/completion/function_call)). That fallback works silently, which hides weak models when k0 would rather reject them.
- **Auth.** Bedrock goes through boto3 ("LiteLLM uses boto3 to handle authentication"): `AWS_PROFILE`, STS role, `AWS_BEARER_TOKEN_BEDROCK`. Vertex needs `google-cloud-aiplatform` installed on top, plus ADC or `vertex_credentials`, with `VERTEXAI_PROJECT`/`VERTEXAI_LOCATION`.
- **LangGraph fit.** Calling `litellm.completion` directly gives no `BaseChatModel`. k0 would have to convert messages and tool calls itself and would lose `bind_tools`, `create_agent`/`ToolNode` and the LangChain callbacks Langfuse uses. `ChatLiteLLM` (in `langchain-ai/langchain-litellm`, maintained by Akshay Dongare) restores those, and its table lists tool calling, streaming, async and token usage ([ChatLiteLLM](https://docs.langchain.com/oss/python/integrations/chat/litellm)). It is also reachable as `init_chat_model(..., model_provider="litellm")`.
- **Footprint.** `litellm` 1.106.4 makes `boto3`, `tokenizers`, `huggingface-hub`, `aiohttp`, `jinja2`, `click` and others mandatory. That totals 79 packages and 214 MB, even for an Ollama-only user.
- **Maintenance and supply chain.** LiteLLM is very active (258 releases in 2026), which also means heavy churn. On 2026-03-24, PyPI releases 1.82.7 and 1.82.8 carried a credential stealer. LiteLLM says the attack most likely came through the Trivy dependency in its CI, and it shipped a clean 1.83.0 from a rebuilt pipeline ([LiteLLM security update](https://docs.litellm.ai/blog/security-update-march-2026)). k0 runs on machines that hold API keys, so this is a real cost for a hard dependency.
- **Upside.** Cost tables (`completion_cost`), a `Router` with fallbacks, and a proxy. These could matter for the budgets ticket. All three are reachable later by adding the `litellm` provider as an optional extra.

## Option C: plain OpenAI-compatible endpoints (`ChatOpenAI(base_url=…)`)

This is one dependency (`langchain-openai`, 43 packages with `langgraph`). Coverage varies by provider:

| Target | OpenAI-compatible surface | Caveat |
|---|---|---|
| Ollama | `http://localhost:11434/v1/`, key "required but ignored". Tools, streaming and `stream_options.include_usage` supported ([Ollama OpenAI compat](https://docs.ollama.com/api/openai-compatibility)) | "The OpenAI API does not have a way of setting the context size". You need a Modelfile or `OLLAMA_CONTEXT_LENGTH`. The default context is 4k on machines with < 24 GiB VRAM, and Ollama recommends ≥ 64000 for agents and coding ([context length](https://docs.ollama.com/context-length)) |
| LM Studio | `http://localhost:1234/v1`: `/v1/chat/completions`, `/v1/responses`, `/v1/models` ([LM Studio OpenAI compat](https://lmstudio.ai/docs/developer/openai-compat)) | This is the only route; no partner package exists. The chat-completions parameter list does not mention `stream_options`, so streamed usage is **unverified** and k0 should probe it |
| OpenAI | native | none |
| Anthropic | `https://api.anthropic.com/v1/` | "not considered a long-term or production-ready solution". `strict` is ignored, there is no prompt caching, thinking is not returned, and usage details are always empty ([Anthropic OpenAI SDK compat](https://platform.claude.com/docs/en/api/openai-sdk)) |
| OpenRouter | OpenAI-compatible | `ChatOpenAI` "targets official OpenAI API specifications only" and drops `reasoning`/`reasoning_details`. LangChain recommends a provider package for OpenRouter, LiteLLM, vLLM and DeepSeek ([ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai)) |
| Bedrock | Chat Completions and Responses on `bedrock-runtime…/openai/v1` and `bedrock-mantle.{region}.api.aws` ([Bedrock endpoints](https://docs.aws.amazon.com/bedrock/latest/userguide/endpoints.html)) | A plain key works only as a Bedrock API key (bearer). SigV4 and the AWS credential chain need `openai[bedrock]`'s `OpenAI(provider=bedrock(...))` (openai README). Wiring that into `ChatOpenAI` is not verified |
| Vertex | `https://aiplatform.googleapis.com/v1/projects/{p}/locations/{l}/endpoints/openapi`. Tools, streaming and usage supported ([Vertex OpenAI compat](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/openai)) | The `api_key` is an ADC OAuth token, and "Access tokens live for 1 hour by default". A static config value expires mid-run |

This is the smallest install, but it degrades three of the seven targets (Anthropic, OpenRouter reasoning, Vertex auth), and Bedrock loses the credential chain.

## Recommendation

1. **Provider layer = `init_chat_model` + partner packages behind extras.** The core dependencies become `langgraph` + `langchain` (or `langchain-core` alone, if k0 copies the ~20-line provider map instead of importing `init_chat_model`). Extras: `k0ntrol[ollama]`, `[openai]` (also covers LM Studio), `[anthropic]`, `[openrouter]`, `[aws]`, `[google]`. Per the AGENTS.md dependency rule, the human must approve these in the spec.
2. **A k0 provider registry in code, not prefix guessing.** Config per agent is `provider:` + `model:` + optional `base_url`/`params`. A small table maps k0 provider names to `(model_provider, defaults)`:
   - `lmstudio` → `openai`, with `base_url=http://localhost:1234/v1`, a placeholder `api_key` and `stream_usage=True`.
   - `ollama` → `ollama`, with `num_ctx` at 64k by default.
   - `vertex` → `google_genai`, with `vertexai=True`.
   - `bedrock` → `bedrock_converse`.
   - `openai_compatible` → `openai` + `base_url`, as the escape hatch for vLLM, llama.cpp and similar servers.

   Secrets stay in env vars and the cloud credential chains. Config never stores keys, and k0 never sets `configurable_fields`.
3. **Usage and capability come from LangChain primitives.** JSONL events read `AIMessage.usage_metadata`. `k0 doctor` reads `model.profile.tool_calling` for hosted models and Ollama `/api/show` `capabilities` for local ones, and rejects agents whose model cannot call tools. That beats LiteLLM's silent JSON-mode fallback.
4. **LiteLLM is optional only.** Expose it later as `provider: litellm` through the existing `init_chat_model` mapping, if the budgets ticket needs its cost tables or router. Keep it out of the default install: it adds 107 MB, makes boto3 mandatory, and had a March 2026 supply-chain incident.
5. **Claude on Vertex is out of 0.1.0 by default.** It needs `langchain-google-vertexai` (70 packages, 220 MB). Anthropic direct, Bedrock and OpenRouter already reach Claude. Add a `[vertex-anthropic]` extra only if someone asks for it.

## Open risks

- **Local tool-calling quality.** LM Studio's default mode and small Ollama models can emit badly formatted calls that arrive as plain text. k0 needs a model-capability check and a clear failure rather than a silent loop. This feeds the map's "Model capability by provider" gap.
- **Ollama context.** With < 24 GiB VRAM the default context is 4k, which silently truncates agent prompts. k0 should set `num_ctx` (native path) and have `doctor` warn about it.
- **Streamed usage on LM Studio.** Not confirmed by the docs. Probe it in `doctor` and fall back to non-streamed usage.
- **`langchain-ollama` cadence.** One release in 2026. If it falls behind Ollama features, the fallback is Ollama's OpenAI-compatible `/v1` through `ChatOpenAI`, at the cost of no `num_ctx`.
- **`langchain-openrouter` is beta.** If it breaks, the fallback is `ChatOpenAI` with the OpenRouter base URL, at the cost of reasoning fields.
- **Claude 4.7+ parameter rules.** Any `temperature` below 1 and any `top_p` other than the default return 400 errors. Claude Opus 5.5 and Sonnet 5.5 reject forced `tool_choice` ([Anthropic compat field table](https://platform.claude.com/docs/en/api/openai-sdk)). k0 should not set those parameters by default.
- **Not tested live.** No local Ollama or LM Studio server was running and no real keys were used. Verification covered model construction and `bind_tools` only. The spec should include a live smoke test per provider, run in `doctor` or the E2E.
