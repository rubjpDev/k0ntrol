# PLANNER and RESEARCHER can use the provider's own web search

From 0.1.0, `search.native: true` in config lets PLANNER and RESEARCHER search with their model provider's own web search tool instead of k0's `web_search`. It is off by default. 0.1.0 wires two providers. An agent on `openai` gets OpenAI's built-in `{"type": "web_search"}` tool, which routes its model calls through the Responses API (#182). An agent on `anthropic` gets Anthropic's basic `web_search_20250305` server tool (#209), which runs each search directly. Dynamic filtering (`web_search_20260209`) runs searches from code execution, and its paused turns need a `caller` and a container that langchain-anthropic does not replay. Every other case keeps today's path: an agent on any other provider, or any agent while `search.native` is off, uses k0's `web_search` through `search.provider` (Brave), or has no search when that is unset. `fetch_url` is unchanged. The provider bills each search to the user's own key on top of tokens, and switching search on is the user's choice to pay for it. We chose this because the user already holds the key, the provider's search needs no second account or server, and the risk ADR 0013 contained is unchanged: the only agents that search have no write tools, and what they produce reaches a human at a gate before anything is built.

OpenRouter and Vertex also offer their own search. Each can be wired later by its own ticket under the rules below, without a new ADR. Bedrock through Converse and the local providers (`ollama`, `lmstudio`, `openai_compatible`) have no provider search and always use `search.provider`.

The rules:

- **Code adds the tool, not config.** k0 puts the provider tool in the `web_search` slot of PLANNER's and RESEARCHER's allowlists, and nowhere else. Config only switches it on: `params` still may not carry tool definitions, and the allowlist test still asserts each agent's tools. The provider tool is the one exception to ADR 0006's "tools are only `BaseTool` instances": a constant k0 owns, accepted for those two agents only.
- **It follows the k0 provider name.** Native search keys on the k0 providers `openai` and `anthropic`, not on LangChain's `model_provider` "openai", which `lmstudio` and `openai_compatible` share.
- **Never both.** An agent gets either the provider's `web_search` or k0's, never both: the tool names would clash.
- **Capped per call.** Searches per model call are capped by a fixed k0 constant: OpenAI's `max_tool_calls` on each call, Anthropic's `max_uses` on the tool itself. Searches per agent run stay within that cap times the agent's `model_calls` limit, and the cap is not configurable.
- **A paused turn is resumed, not dropped.** When Anthropic pauses a long server-search turn (`pause_turn`), k0 sends the paused content back unchanged, with the same tools, up to a fixed number of times per call. Each continuation is its own model invocation with its own `model` event. The paused content and the answer are kept as one assistant message, so the history still shows one message per model call.
- **A warning, never a fallback.** If the provider rejects the search tool for the configured model, or the organization has web search off, k0 retries that call once without the tool, shows a warning that names the model, and runs the rest of that agent run with no search. It never switches to Brave.
- **Content-free accounting.** The call still emits one `model` event. That event also records how many searches the call ran (OpenAI's `web_search_call` items, Anthropic's billed `web_search_requests`, or its `web_search_tool_result` blocks when streaming leaves out that usage), never the queries, URLs or results. Cost estimates stay token-only: the provider's bill is the truth.

## Considered Options

- k0's `web_search` calling OpenAI's Responses API as one more search backend (the first design for #182): rejected, because it hides a second model call inside a tool, which then needs its own model event, model choice and request body, all to rebuild what binding the provider's tool gives directly.
- Native search for every provider in 0.1.0: deferred. Each provider's LangChain package handles its search tool differently, and OpenRouter's is still in beta. OpenAI ships first, and the rest follow one ticket each.
- On by default for OpenAI users: rejected, because choosing a model provider must not start search charges.
- Falling back to Brave when the model cannot search: rejected, because a silent switch hides a misconfiguration and can bill a second account.
- Stopping the run when the model cannot search: rejected, because search is an extra. A warning tells the user what to fix without losing the run.
- A self-hosted SearXNG server as a free search backend: not chosen for 0.1.0, because a server must be running whenever an agent searches. It can still join `SEARCH_PROVIDERS` later.

## Consequences

- This amends ADR 0006, which allowed no provider built-ins, and reverses ADR 0013's rejection of the provider's built-in search. Their other guards stand.
- k0 can no longer mark these results untrusted or cap their size, because the provider puts them straight into the model's context. PLANNER's and RESEARCHER's prompts already say web results are untrusted data, and that rule must cover results that carry no marker. In return the provider fetches pages from its own servers, so a search cannot reach the user's local network.
- Searches run inside the model call, so they emit no `web_search` tool event. The search count in the `model` event takes its place.
- Each agent run still starts with fresh context. The agent graph has no checkpointer of its own and runs in a fresh checkpoint namespace, so search items never carry into a later run or a resumed one.
- `k0 doctor` says which agents use the provider's search and which use `search.provider`.
- Live search stays opt-in and outside the release benchmark budget.
- OpenAI documents `max_tool_calls` as a hard cap, but users have reported overruns. The cap bounds the normal case; the bill is still the truth.
