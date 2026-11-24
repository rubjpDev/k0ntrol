# Devin child sessions (headless)

The canonical recipe for a headless child session — an implementer, reviewer, or any step that must run on its own model rather than the parent's:

```bash
env -i HOME="$HOME" PATH="$PATH" TERM=xterm-256color \
  devin --model <model> --export <log.jsonl> \
  --respect-workspace-trust false -p "$(cat brief.md)"
```

- `env -i` gives the child a minimal environment: an embedded parent session's inherited environment can hide CLI auth. Keep `HOME` (CLI config and logs), `PATH` (`devin`, `uv`, `git`) and a real `TERM` (Rich renders dumb at `TERM=dumb`).
- Non-interactive mode cannot show the workspace-trust prompt and fails in an untrusted directory; `--respect-workspace-trust false` skips the check.
- `--export` writes the transcript only after each turn — it does not stream mid-run. For progress, poll the CLI log instead.

## Verifying the model

Do not trust the requested model — check the server-reported one. The child's CLI log at `~/.local/share/devin/cli/logs/devin_*_<pid>.log` records `resolved_model_uid`; grep it before letting a model-sensitive session do real work.

## Sandboxing

`devin --sandbox` (research preview) sandboxes the child's exec commands — macOS seatbelt / Linux bwrap+seccomp — so writes land only inside the workspace and granted `Write` scopes, and it forces the autonomous permission mode (`--permission-mode` is ignored). It is the safer replacement for `--permission-mode dangerous` on worktree-confined implementers — with one caveat proven by smoke test (2026-11-18, devin 3000.14.30821, `swe-2-flash-medium`, worktree on `develop`):

- `uv sync` and `uv run pytest` **fail**: uv writes its cache to `~/.cache/uv`, outside the workspace.
- `git` works.
- `uv` works once its cache is relocated inside the workspace: `env -i … UV_CACHE_DIR="$PWD/.uv-cache" devin --sandbox …` passed `uv sync`, `uv run pytest` and `git status` — and `.uv-cache` self-ignores via the `.gitignore` uv writes inside it.

The smoke test as specced fails — `dangerous` stays the canonical recipe for worktree implementers. For a run that needs nothing beyond `uv`/`pytest`/`git`, `--sandbox` with `UV_CACHE_DIR` inside the workspace is a proven safer option; anything else that writes `$HOME` (caches, tool state, credentials) still fails there, so those runs keep `dangerous`.
