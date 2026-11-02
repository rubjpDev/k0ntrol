# The suite runs in parallel, and the TDD loop runs only the tests in play

The full suite took 14 minutes locally and ran twice in CI, and agents ran all of it on every red/green cycle because nothing said otherwise. Most of the time was the workflow tests' `test_cmd`: a nested `python -m pytest -q` that, under the fresh bytecode cache `run_command` gives every command (#70), recompiled pytest and its autoloaded plugins on every call (about 3.9 s each). We decided four things:

- **Inner loop vs. done gate.** During red/green, run only the tests in play: the test file or node id being changed, or `--lf -x`. The done gate is unchanged: `uv run pytest -q` passes in full. The merger still runs the full suite after every merge.
- **Parallel by default.** `pytest-xdist` joins the dev dependencies (now `complexipy`, `pytest`, `pytest-cov`, `pytest-xdist`), and `addopts = "-n auto"`, so `uv run pytest -q` stays the literal done-gate command everywhere it already appears. Use `-n0` for serial debugging. CI runs the suite once, with coverage, and caches uv.
- **A stub runner as the fixture `test_cmd`.** The workflow tests use a small stdlib test runner checked into `tests/` as their `test_cmd`, not pytest. It is still a real subprocess over the real test files the fake TESTER wrote, in a real temp git repo, so the red check and VALIDATOR stay what spec #33 agreed: `test_cmd` runs for real. k0 reads only a command's exit code and output, so pytest is not the seam under test. A few named tests keep real pytest: those that assert on pytest's output, the #70 bytecode tests, and one `/fast` and one `/full` happy path.
- **Nested pytest without plugin autoload.** Wherever a test still runs a nested pytest, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is set per test (`monkeypatch.setenv` in an autouse fixture). It is never set in `os.environ` at import time, because xdist workers would inherit it and start without xdist and pytest-cov.

Targets: the CI test job under 3 minutes, including checkout and `uv sync`; the full local suite under 90 s on 4 cores; one workflow test node under 5 s.

## Considered Options

- `pytest-testmon` for affected-test selection: rejected. It traces coverage in-process, so it cannot see what the slow tests depend on through subprocesses (real git, real `test_cmd`). It also adds a state file and conflicts with pytest-cov.
- Plain `python -c` red/green commands per test: rejected. The command would stop running TESTER's files, which hollows out the red check.
- A session-scoped template git repo copied per test: rejected. `git init` plus the first commit costs about 0.12 s per test, about 12 s across the suite.
- A `slow` marker and a fast/slow CI split: rejected while the suite meets the targets above. A marker is upkeep with no payoff, and a second job doubles setup. This is not a glossary term. "Tier" is already an avoided word there, and "E2E" means the manual release check.
- A cheaper bytecode cache in `run_command` itself: out of scope here. It changes product behaviour and must keep #70 fixed, so it is tracked as its own issue.

## Consequences

- Measured on 2026-11-02 on 4 cores: 853 s serial before; 297 s with plugin autoload off alone; 225 s with xdist alone. The stub runner removes most of what remains.
- Tests must stay safe to run in parallel: per-test state lives under `tmp_path` or `monkeypatch`, and no test writes to a fixed shared path.
