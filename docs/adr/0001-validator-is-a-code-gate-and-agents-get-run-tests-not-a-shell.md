# VALIDATOR is a code gate, and coding agents get `run_tests`, not a shell

In 0.1.0, VALIDATOR has no model. It is a deterministic gate node that runs `test_cmd` and the configured `gates`, and on a red result CODER gets the raw output and finds the files to fix with its own read tools, so the old failure summary and `redo:` file list are gone. No agent gets a general shell: TESTER and CODER get one `run_tests` tool, which runs the configured `test_cmd` as an argv list, optionally narrowed to existing test paths, so they can see red and work toward green without choosing a command. SPEC, REVIEWER and CHATTY run nothing. We chose this because the invariant says gates are code, and a shell tool would need the sandboxing and allowlist work that 0.1.0 has not specified yet; agents that read code themselves no longer need a model to summarize failures for them.

## Considered Options

- Keep a cheap model that summarizes VALIDATOR's failure for CODER: rejected, because CODER can read the files itself, and a second model call between verdict and retry adds cost and a place for the summary to be wrong.
- A general shell tool for TESTER and CODER, like most coding agents have: rejected for 0.1.0, until tool safety (sandboxing, command policy) is decided.
- No command tool at all: rejected, because TESTER could not check that its tests start red and CODER would have to wait for VALIDATOR to learn anything.

## Consequences

- `run_tests` still executes code the agents wrote, so it is not a sandbox. Whether it needs one is a separate tool-safety decision.
- Agents cannot install packages, run linters or start servers. The configured `gates` are the only other commands, and only VALIDATOR runs them.
