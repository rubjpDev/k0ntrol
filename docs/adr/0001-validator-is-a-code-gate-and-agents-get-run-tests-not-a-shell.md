# VALIDATOR is a code gate, and coding agents get `run_tests`, not a shell

> Partly superseded by [ADR 0011](0011-one-implementer-builds-red-to-green-slices-under-a-code-guard.md): IMPLEMENTER replaces TESTER and CODER. `run_tests`, no shell, and VALIDATOR as a code gate stand.

In 0.1.0, VALIDATOR has no model. It is a deterministic gate node that runs `test_cmd` and the configured `gates`. On a red result IMPLEMENTER gets the raw output and uses its read tools to find the files to fix. No agent gets a general shell: only IMPLEMENTER gets `run_tests`, which runs the configured `test_cmd` as an argv list, optionally narrowed to existing test paths, so it can see red and work toward green without choosing a command. We chose this because the invariant says gates are code, and a shell tool would need separate sandboxing and allowlist work; agents that read code themselves do not need a model to summarize failures for them.

## Considered Options (original agent roster)

- Keep a cheap model that summarizes VALIDATOR's failure for CODER: rejected, because CODER can read the files itself, and a second model call between verdict and retry adds cost and a place for the summary to be wrong.
- A general shell tool for TESTER and CODER, like most coding agents have: rejected for 0.1.0, until tool safety (sandboxing, command policy) is decided.
- No command tool at all: rejected, because TESTER could not check that its tests start red and CODER would have to wait for VALIDATOR to learn anything.

## Consequences

- `run_tests` still executes code the agents wrote, so it is not a sandbox. Whether it needs one is a separate tool-safety decision.
- Agents cannot install packages, run linters or start servers. The configured `gates` are the only other commands, and only VALIDATOR runs them.
