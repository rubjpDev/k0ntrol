# The 0.1.0 runtime dependencies and provider extras

Following ADR 0005's closed list, the 0.1.0 runtime dependencies are `pyyaml`, `rich`, `langgraph>=1.2.17,<1.3`, `langchain>=1.4.7,<1.5` and `langgraph-checkpoint-sqlite>=3.1.2,<3.2`. LangGraph runs the workflow graph, LangChain's `init_chat_model` builds each agent's model, and the SQLite checkpointer resumes runs and gates. The provider partner packages and Langfuse stay optional extras: `ollama` (`langchain-ollama`), `openai` (`langchain-openai`, also covers LM Studio), `anthropic` (`langchain-anthropic`), `openrouter` (`langchain-openrouter`), `aws` (`langchain-aws`, Bedrock Converse), `google` (`langchain-google-genai`, Vertex via `vertexai=True`) and `langfuse` (`langfuse>=5.0.4,<6`). We chose this because the human approved exactly this list on [Write the 0.1.0 spec and its tickets](https://github.com/rubjpDev/k0ntrol/issues/19), and each provider package behind an extra keeps the default install small (ADR 0004, provider research in `docs/research/providers.md`). Any further runtime dependency needs the human's approval and its own ADR.

## Considered Options

- Put the provider packages in the core dependencies: rejected, because an Ollama-only user would pay for boto3, numpy and the Google stack.
- `langchain-core` alone instead of `langchain`: rejected, because `init_chat_model` ships in `langchain`, and copying its provider map would fork upstream maintenance.
- LiteLLM (`langchain-litellm`) as one package covering every provider: rejected, per the provider research — heavy footprint, silent tool-calling fallbacks, and a 2026 supply-chain incident.

## Consequences

- `uv sync` and the test suite install only the core list; provider extras install with `k0ntrol[<extra>]` and `k0 doctor` reports a missing extra, per the provider-config decision.
- Versions in the pins are the floors recorded in `docs/research/providers.md` (PyPI, 2026-10-25); raising a floor is a maintenance change, adding a package is an ADR.
