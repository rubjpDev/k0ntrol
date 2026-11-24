# Tracker access: GitHub MCP server vs `gh` vs SDK

Research for [rubjpDev/k0ntrol#10](https://github.com/rubjpDev/k0ntrol/issues/10) (map: #2). Checked 2026-10-25.

**Question.** How should k0 read and write GitHub issues, branches and PRs, and can the same path later cover GitLab, Linear and Jira?

**Short answer.** The harness does every tracker write in code, through a small k0-owned `Tracker` protocol. The GitHub adapter shells out to `gh`. Agents never get a raw tracker write tool. MCP can be an optional read-only extra, filtered by name in code. A single interface does hold across GitHub, GitLab, Linear and Jira, but only as k0's own Python protocol. The vendors' MCP tool sets differ too much in names, argument shapes and auth to share one interface. Linear and Jira also have no branches or PRs, so the protocol has to split work items from code review.

## Pinned versions (2026-10-25)

| Thing | Version | Source |
|---|---|---|
| `github/github-mcp-server` | v2.0.3 (2026-10-21) | [releases](https://github.com/github/github-mcp-server/releases) |
| `langchain-mcp-adapters` | 0.3.2 (2026-08-06), **repo archived 2026-09-17** | [repo](https://github.com/langchain-ai/langchain-mcp-adapters), [PyPI](https://pypi.org/project/langchain-mcp-adapters/) |
| `langchain` (`langchain.mcp`, extra `[mcp]`) | 1.4.7. `langchain.mcp` needs `>=1.4.0` and is **beta** | [MCP docs](https://docs.langchain.com/oss/python/langchain/mcp/index), [migration](https://docs.langchain.com/oss/python/migrate/langchain-mcp-adapters) |
| `fastmcp` (pulled in by `langchain[mcp]`) | 4.1.3 (`langchain[mcp]` pins `fastmcp>=4.0.11,<5`) | [PyPI](https://pypi.org/project/fastmcp/) |
| `mcp` (official SDK) | 2.3.0. The archived adapters pin `mcp<2.0.0` | [PyPI](https://pypi.org/project/mcp/) |
| `gh` CLI | 2.98.0 (2026-08-20) | [releases](https://github.com/cli/cli/releases) |
| PyGithub | 2.10.0 (deps: requests, urllib3, pynacl, pyjwt[crypto], typing-extensions) | [PyPI](https://pypi.org/project/PyGithub/) |
| githubkit (typed, async alternative) | 0.16.1 (deps: httpx, pydantic, hishel, anyio) | [PyPI](https://pypi.org/project/githubkit/) |
| langgraph | 1.2.17 | [PyPI](https://pypi.org/project/langgraph/) |

**`langchain-mcp-adapters` is dead.** The issue names it, but its README says "Please migrate to `langchain[mcp]`" and the repo was archived on 2026-09-17. It also caps `mcp<2.0.0`, while `mcp` is now at 2.3.0. Any MCP path in 0.1.0 means `langchain.mcp.MCPAdapter`, which is still beta: importing it raises a `LangChainBetaWarning`, and its docs say the API may change ([source](https://docs.langchain.com/oss/python/langchain/mcp/index)).

## Option 1: official GitHub MCP server via `langchain.mcp`

**How it runs**
- **Remote:** `https://api.githubcopilot.com/mcp/` over HTTP, with OAuth or a PAT sent as `Authorization: Bearer`. GitHub Enterprise Server is not supported, but `ghe.com` is ([README](https://github.com/github/github-mcp-server)). The server is available to all GitHub users. For Copilot Business and Enterprise orgs, the "MCP servers in Copilot" policy can gate it ([GitHub docs](https://docs.github.com/en/copilot/how-tos/context/model-context-protocol/using-the-github-mcp-server)).
- **Local:** stdio, using a prebuilt Go binary (per-OS tarballs on the v2.0.3 release) or `ghcr.io/github/github-mcp-server`. Takes `GITHUB_PERSONAL_ACCESS_TOKEN`, and `--gh-host`/`GITHUB_HOST` for GHES ([README](https://github.com/github/github-mcp-server)).

**Server-side tool filters** ([server-configuration.md](https://github.com/github/github-mcp-server/blob/main/docs/server-configuration.md))
- `--toolsets` and `--tools`. Default toolsets: `context, repos, issues, pull_requests, users`.
- `--exclude-tools`, which wins over the include flags.
- `--read-only`, which "takes precedence over any other configuration".
- `--lockdown-mode`, which on public repos hides content from users without push access. The docs call it "a best-effort content filter … not an authorization boundary".
- On the remote server, the same filters are headers: `X-MCP-Toolsets`, `X-MCP-Tools`, `X-MCP-Exclude-Tools` and `X-MCP-Readonly`.
- With a classic PAT, tools are also hidden by token scope. Fine-grained PATs get no scope filtering ([scope-filtering.md](https://github.com/github/github-mcp-server/blob/main/docs/scope-filtering.md)).

**Tools the agent would see**

| Need | Tool |
|---|---|
| Write an issue | `issue_write` with `method: create \| update`. The same tool also edits labels, assignees and state |
| Read an issue | `issue_read` |
| Comment on an issue | `add_issue_comment` |
| Branches | `create_branch`, `list_branches` |
| Files | `create_or_update_file`, `push_files` |
| PRs | `create_pull_request`, `update_pull_request`, `pull_request_read` (methods such as `get_diff`, `get_files`, `get_reviews`, `get_review_comments`) |
| Review a PR | `pull_request_review_write` with `method` plus `event` |
| Merge | `merge_pull_request` |

Source: [README tool list](https://github.com/github/github-mcp-server).

**What this means for allowlists**
- **Names are too coarse.** One write tool covers several operations. An allowlist entry for `issue_write` allows both create and update. An entry for `pull_request_review_write` allows APPROVE as well as COMMENT. Enforcing "SPEC may only refine issue N" or "REVIEWER may only comment" would need argument checks in `@wrap_tool_call` middleware, which is code that understands GitHub's argument schema.
- **Names drift.** The server renames tools and keeps deprecation aliases (`get_issue` → `issue_read`, see [tool-renaming.md](https://github.com/github/github-mcp-server/blob/main/docs/tool-renaming.md)). An allowlist pinned by name has to follow upstream renames.
- **`merge_pull_request` is in the default `pull_requests` toolset.** Because k0 never merges, k0 would have to start the server with `--exclude-tools merge_pull_request` or `--read-only`, and also filter the name in code.

**`langchain.mcp` mechanics** ([tools](https://docs.langchain.com/oss/python/langchain/mcp/tools), [migration](https://docs.langchain.com/oss/python/migrate/langchain-mcp-adapters))
- `async with MCPAdapter(target) as a: tools = await a.list_tools()`. There is no include or exclude parameter, so k0 filters the list by `tool.name` itself.
- Each returned tool opens its own session per call.
- Server-reported errors reach the model as a `ToolMessage` with `status="error"`. Transport errors raise.
- Structured results are in `ToolMessage.artifact` (`MCPToolArtifact`).
- Interceptors are now `@wrap_tool_call` middleware.
- Elicitation becomes a LangGraph `interrupt()`. On resume, the tool reruns from the start, so side effects before the prompt can happen twice.

**Offline fake.** `MCPAdapter` accepts an in-process FastMCP server as its target ([source](https://docs.langchain.com/oss/python/langchain/mcp/index)). A test can register fake `issue_read` and similar tools on a FastMCP instance without network. The fake has to copy GitHub's schemas, which change upstream: the server's own tests snapshot schemas with `toolsnaps` for this reason ([testing.md](https://github.com/github/github-mcp-server/blob/main/docs/testing.md)).

**Footprint**
- A Go binary or Docker for local use, or network access to the Copilot endpoint for remote use.
- `langchain[mcp]` brings in `fastmcp` 4.x and its dependency tree.
- Every exposed tool schema goes into the prompt. Local models handle tool calling worse, so a long tool list hurts them (inference, not measured here).

## Option 2: `gh` CLI from Python (`subprocess`)

**Auth**
- `gh auth login` stores credentials. `GH_TOKEN`/`GITHUB_TOKEN` override them for github.com and ghe.com, and `GH_ENTERPRISE_TOKEN` with `GH_HOST` covers GHES ([gh environment](https://cli.github.com/manual/gh_help_environment)).
- `GH_PROMPT_DISABLED` turns off interactive prompts, which a harness needs.
- `gh auth token` prints the stored token, so another client can reuse it ([manual](https://cli.github.com/manual/gh_auth_token)).

**Operations**
- Issues and PRs: `gh issue view/create/edit/comment --json …`, `gh pr create`, `gh pr view --json`, `gh pr diff`.
- `gh pr review --comment | --approve | --request-changes -b/-F` ([manual](https://cli.github.com/manual/gh_pr_review)).
- `gh api` reaches any REST or GraphQL endpoint ([manual](https://cli.github.com/manual/gh_api)).
- Exit codes: 0 ok, 1 failure, 2 cancelled, 4 auth required ([manual](https://cli.github.com/manual/gh_help_exit-codes)). A missing login therefore shows up as a distinct code, which `k0 doctor` can check.

**Agent tools vs harness calls.** `gh` gives the agent nothing automatically. k0 decides which operations exist and exposes only the ones it wraps itself.

**Offline fake.** Inject a command runner (argv → stdout JSON) and replay recorded `--json` output. This works the same way k0 already fakes `cursor-agent` and `git`.

**Footprint.** No Python dependencies. It needs the `gh` binary on PATH, which `k0 init`/`k0 doctor` must check. Each call is a process spawn. The harness makes only a handful of writes per run, so that cost does not matter.

## Option 3: Python SDK or plain REST

**PyGithub 2.10.0** ([constructor reference](https://pygithub.readthedocs.io/en/stable/github.html))
- Synchronous.
- `Auth.Token`, App auth, and `base_url="https://{host}/api/v3"` for GHES ([intro](https://pygithub.readthedocs.io/en/stable/introduction.html)).
- Retries by default with `GithubRetry(total=10)`, and spaces calls with `seconds_between_requests=0.25` and `seconds_between_writes=1.0`. This built-in throttling is the main advantage over `gh`.
- Adds 5 runtime dependencies, which AGENTS.md requires asking about.
- Gets its token from k0 config or from `gh auth token`.

**githubkit 0.16.1.** Typed and async-capable on httpx and pydantic. It is heavier than PyGithub.

**Plain REST on stdlib `urllib` or httpx.** httpx already comes in through the LangChain stack (inference: langsmith and langgraph-sdk depend on it; verify against the lockfile when 0.1.0 dependencies land). No new dependency, but pagination, retries and `retry-after` handling would be k0 code.

**Offline fake.** Mock HTTP, or better, fake at the `Tracker` protocol level (below).

## Rate limits (same for every option, since all use the user's token)

From [GitHub REST rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api):
- Primary: 5,000 requests per hour for an authenticated user, shared across every client that uses that user's identity.
- Secondary:
  - at most 100 concurrent requests
  - 900 points per minute for REST (a write costs 5 points)
  - 2,000 points per minute for GraphQL
  - about 80 content-creating requests per minute and 500 per hour
- When `retry-after` is present, honor it.

The harness writes only a few times per run, which is far below these limits. The real risk is an agent with read tools looping, which the map's budgets and loop limits should cover.

## Comparison

| | GitHub MCP (`langchain.mcp`) | `gh` CLI | PyGithub / REST |
|---|---|---|---|
| Auth | PAT or OAuth. Remote needs the Copilot endpoint. GHES local only | `gh auth login` (keyring), `GH_TOKEN`, GHES | token from config or `gh auth token`, GHES via `base_url` |
| What the agent sees | Upstream tool schemas, coarse (`issue_write` = create + update) | Only what k0 wraps | Only what k0 wraps |
| Code enforcement | Name filter plus argument checks in middleware | k0 picks the operation in code | k0 picks the operation in code |
| Offline fake | In-memory FastMCP server copying upstream schemas | Fake runner with recorded JSON | Mocked HTTP |
| Throttling and retry | Server side, not documented | none built in (not verified) | built in (`GithubRetry`, write spacing) |
| New deps | `langchain[mcp]` → fastmcp 4 (beta API), plus a Go binary or Docker | none (needs the `gh` binary) | PyGithub + 5, or none on httpx |
| Churn risk | High: adapter package archived, `langchain.mcp` beta, tool renames | Low: stable CLI, `--json` fields | Low |

## GitLab, Linear and Jira: what MCP servers exist

| Tracker | Server | Status | Endpoint and transport | Auth | Relevant tools |
|---|---|---|---|---|---|
| **GitLab** | Built-in GitLab MCP server ([docs](https://docs.gitlab.com/user/model_context_protocol/mcp_server/)) | GA since GitLab 19.5. Free, Premium and Ultimate on .com, Self-Managed and Dedicated. An admin or group owner must allow it | `https://<host>/api/v4/mcp` over HTTP. stdio only through `mcp-remote` (Node 20+) | OAuth with the `mcp` scope (dynamic client registration), or a **fine-grained** PAT. Legacy PATs get 403 | `save_work_item` (create and update), `get_work_item`, `save_note`, `save_merge_request`, `get_merge_request` (include diffs and notes), `save_merge_request_review` (methods `create_note`, `submit_review`, `approve`…), `add_branch`, `accept_merge_request` ([tools](https://docs.gitlab.com/user/model_context_protocol/mcp_server_tools/)) |
| **Linear** | Official remote server ([docs](https://linear.app/docs/mcp)) | Live. Docs say more functionality is "on the way" | `https://mcp.linear.app/mcp` (Streamable HTTP), `/mcp/readonly`. SSE is deprecated | OAuth 2.1 with dynamic client registration, or `Authorization: Bearer` with an OAuth token or **API key** | Find, create and update issues, projects and comments. No branches or PRs: Linear is not a code host |
| **Jira** | Atlassian Rovo MCP server ([repo](https://github.com/atlassian/atlassian-mcp-server), [auth](https://support.atlassian.com/atlassian-rovo-mcp-server/docs/configuring-authentication-via-api-token)) | GA, v2 recommended. **Cloud only** | `https://mcp.atlassian.com/v2/mcp`. `/v1/sse` ended 2026-06-30 | OAuth 2.1. API token (Basic `email:token`) or service-account key (Bearer) **only if an org admin enables it** | Jira permission groups `read_jira`, `write_jira`, `search_jira`. `delete_jira` and `manage_jira` are off by default. No PRs, except Bitbucket tools on the same server |
| Jira Server/DC | `sooperset/mcp-atlassian` (community, MIT, v0.23.1, [repo](https://github.com/sooperset/mcp-atlassian)) | Community, "Not an official Atlassian product" | stdio via `uvx`, SSE or streamable-http | Cloud API token, DC PAT, OAuth 2.0 | `jira_get_issue`, `jira_create_issue`, `jira_update_issue`, `jira_transition_issue`, … (98 tools) |

### Does one tracker interface over MCP hold?

**At the MCP tool level, no.**
- **Names and shapes differ per vendor.** GitHub uses `issue_write{method}`, GitLab `save_work_item`, Atlassian has its own set, and Linear's names are not documented on its page. A role prompt or allowlist written against one vendor's tools does not carry to another.
- **Auth differs per vendor.** GitLab rejects legacy PATs. Atlassian API tokens need an admin opt-in. Linear and GitLab expect OAuth with dynamic client registration, a browser flow that a headless harness run handles poorly.
- **Linear and Jira are not code hosts.** The TESTER/CODER branch and the REVIEWER's PR review always live on GitHub, GitLab or Bitbucket. A "Jira tracker" setup is really Jira for work items plus a code host for changes.

**At the code level, yes.** A small k0-owned protocol that the harness calls covers all four, because each backend can implement it, through its REST API or its MCP server:

```python
class WorkItems(Protocol):   # GitHub Issues, GitLab work items, Linear, Jira
    def get(self, ref: str) -> WorkItem: ...
    def create(self, title: str, body: str) -> WorkItem: ...
    def update_body(self, ref: str, body: str) -> None: ...
    def comment(self, ref: str, body: str) -> None: ...

class CodeHost(Protocol):    # GitHub, GitLab (Bitbucket later)
    def open_change(self, branch: str, base: str, title: str, body: str, work_item: str) -> Change: ...
    def get_change(self, ref: str) -> Change: ...          # diff, files, reviews
    def post_review(self, ref: str, body: str, comments: list[ReviewComment]) -> None: ...  # always COMMENT
```

Sketch only. The real seam belongs to the codebase-design and tracker tickets (#12, #13).

## Recommendation

1. **The harness does tracker writes in code, through k0's own `WorkItems` and `CodeHost` protocols.**
   - SPEC returns a structured issue (title and body), and the harness creates or updates the issue.
   - The harness creates the branch with local `git`, and opens the PR after the VALIDATOR gate passes.
   - REVIEWER returns a structured review, and the harness posts it with event `COMMENT`.
   - No agent gets a tracker write tool. Writes then stay deterministic, gated and loggable, in line with the invariant that gates and routing live in code. Merge stays out of reach, because the protocol has no merge method.
2. **0.1.0's GitHub adapter uses `gh`** (`gh issue|pr … --json`, `gh pr review --comment`, `gh api` for the rest). It adds no Python dependency, it reuses the user's existing `gh auth login` (keyring, GHES, SSO), and exit code 4 maps cleanly to a `k0 doctor` check. If throttling or retries become a problem, swap in a REST or PyGithub adapter behind the same protocol. Only the adapter changes.
3. **Agents get only k0-defined read tools** backed by the same protocol, for example `read_work_item` and `read_change_diff`, and only where their allowlist needs them. The REVIEWER might get the PR diff and the existing reviews. These tools are tracker-neutral, so role prompts do not change when GitLab, Linear or Jira arrive.
4. **Raw MCP stays optional.** When a user configures it, k0 still starts the server with `--read-only` (or `X-MCP-Readonly`), an explicit `--tools` list, and `--lockdown-mode` on public repos. Each agent's list is then filtered by `tool.name` in code.
5. **Tests use an in-memory `FakeTracker`** that implements both protocols, for graph and E2E tests without network. Contract tests for the `gh` adapter use a fake command runner with recorded `--json` output.

## Open risks

- **The REVIEWER cannot approve or request changes on the user's own PR.** "Pull request authors cannot approve their own pull requests" ([GitHub docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/approving-a-pull-request-with-required-reviews)). k0 acts with the user's token, so the PR author is the user. Requesting changes on your own PR is refused too (expected from the same rule, not verified on a live PR). Reviews should therefore always use `COMMENT`. A real approve or block would need a separate bot identity, such as a GitHub App, which is out of scope for 0.1.0.
- **`langchain.mcp` is beta and its predecessor is archived.** Any MCP code in 0.1.0 should stay behind one small k0 module, and the version should be pinned.
- **Prompt injection through tracker content.** Issue bodies, comments and PR reviews from third parties reach the agents. GitHub's lockdown mode is best-effort and only works for the MCP path. With `gh`, k0 needs its own policy, for example reading only the issue author and collaborators. This falls under the map's "tool safety" gap.
- **Not verified here:** whether `gh` retries on secondary rate limits; Linear's exact MCP tool names; Atlassian Data Center for the official server (the docs describe Cloud only).
- **OAuth-only backends** (GitLab with OAuth, Atlassian without API tokens) will need a browser login step in `k0 init` when those adapters arrive. That work is out of scope for 0.1.0.
