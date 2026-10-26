# Providers replace the Cursor subscription backend

From 0.1.0, models run on a provider: a local runtime (Ollama, LM Studio), a lab API, an aggregator such as OpenRouter, or a cloud such as Bedrock or Vertex. The `cursor-agent` backend that 0.0.x used through a Cursor subscription is dropped. We chose this because a subscription CLI runs its own agent loop and tools, so k0 cannot enforce tool allowlists in code, choose the model per agent, or emit one event per model call through it. The subscription has also ended.

## Consequences

- Users bring their own API keys, or run models locally.
- Local models are weaker at tool calling, so `k0 doctor` has to check what a configured model can do.
