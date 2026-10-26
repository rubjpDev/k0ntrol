# Tool safety in 0.1.0 is process hygiene and code guards, not a sandbox

In 0.1.0, agent-written code runs unsandboxed, with network access, as the user. k0 limits what that code and the agents can reach with checks in code instead of an OS sandbox:

- **Test commands.** Every run of `test_cmd` or `gates` (`run_tests`, the red check, VALIDATOR) gets a scrubbed environment, with provider and tracker vendor variables and secret-shaped names (`*_KEY`, `*_TOKEN`, `*_SECRET`) removed. It also gets a configurable `test_timeout` (default 600 s; a timeout counts as red), and its output is truncated before an agent sees it.
- **Allowlist bypasses.** One k0 function builds every agent from a closed set of k0 middleware. No middleware overrides `request.tool`, tools are only `BaseTool` instances (never dicts or provider built-ins), and config `params` may not carry tool definitions. A unit test per agent asserts that its tools equal its allowlist.
- **Path guards.** Every read and write tool resolves symlinks first, then checks the resolved path. Reads see only files git does not ignore (`git ls-files --cached --others --exclude-standard`), so `.env` and similar files never reach a provider. The test/non-test check runs on the resolved path, and `conftest.py` counts as a test path anywhere.
- **Workflow changes.** `.github/` stays writable. If a change touches `.github/`, the harness holds the push behind a human CI gate that shows that diff. The gate fires again on any later push that changes `.github/` since the last approved one.
- **Injected content.** Tracker content reaches agents only from trusted authors (`OWNER`, `MEMBER`, `COLLABORATOR`). Other comments are dropped, with a count line saying how many. An issue body from anyone else stops `/full #<n>` until the human restates it. Repo files are not filtered: the human chose to run k0 on the repo, and agents have no network tool, shell or tracker write that injected text could steer.

`k0 doctor` and the docs state that agent code runs unsandboxed. We chose this because the agents have no shell (ADR 0001), so the remaining risk is the code they write and what they can read. A scrubbed environment and the guards cover the cheap, likely leaks: secrets in env vars, `.env` files, symlinks, CI secrets on push. An OS sandbox would be platform-specific work that 0.1.0 does not need yet.

## Considered Options

- An OS sandbox with no network (macOS `sandbox-exec`, which is deprecated; Linux namespaces or containers): deferred until a general shell tool returns.
- No guards beyond documentation: rejected, because `run_tests` inherited the provider and tracker tokens of the user's shell.
- Blocking writes to `.github/` outright: rejected, because CI work is a legitimate task. The CI gate holds back the push, which is the real risk, not the write.

## Consequences

- This settles the open tool-safety consequence of ADR 0001.
- CI runs CODER's unreviewed code against tests the human approved, as it would for any branch pushed to the same repo. Draft PRs still open at first green.
- A legitimate test that needs a secret from the environment must get it some other way. The scrub list is not configurable in 0.1.0.
