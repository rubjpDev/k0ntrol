# PLANNER and RESEARCHER get read-only network tools

From 0.1.0, two agents can read the web: PLANNER, to ground its questions, and RESEARCHER, to answer research decision tickets from primary sources. They get two tools. `fetch_url` reads one page over HTTP(S) with the standard library and returns its text. It refuses any host that resolves to a loopback, private or link-local address (checked again after each redirect), caps the size it reads, and times out. `web_search` queries one search API the user configures, whose key is saved like the provider keys and called with the standard library, so no new package enters the install. Everything either tool returns is marked untrusted, the way untrusted tracker content already is. IMPLEMENTER, REVIEWER and CHATTY get no network tool. We chose this because planning and research without the web leave agents to answer from model memory, which is stale and cannot be cited, and the risk that kept the network out (ADR 0006: injected text steering an agent) is contained when the only agents that read the web have no write tools.

## Considered Options

- The model provider's built-in search: rejected, because ADR 0006 allows only k0's own `BaseTool` instances, and not every provider offers one.
- `fetch_url` alone, with no search: rejected, because the model would have to know the right URL already, which is the stale-memory problem again.
- An MCP fetch or search server: rejected for 0.1.0, because it is a new runtime dependency and a second tool path to guard.
- Network for every agent: rejected, because fetched text could then steer an agent that writes code.

## Consequences

- This changes ADR 0006's "agents have no network tool". Its other guards stand, and agent-written code still runs with the user's network access, as before.
- A search API key is a new credential. The provider setup asks for it, and `k0 doctor` checks it.
- The event log stays content-free: a network tool's event records the host, not the full URL or the page.
- Fetched content can still try to steer PLANNER's questions or RESEARCHER's findings. Both reach the human at a gate before anything is built.
