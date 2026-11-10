# One IMPLEMENTER builds red-to-green slices, under a guard in its write tools

From 0.1.0, one agent, IMPLEMENTER, builds a piece of work test-first, one slice at a time, in place of TESTER writing every test and then CODER writing every line. Red before green stays a gate in code, not a prompt: IMPLEMENTER's write tools carry a guard with two states that `run_tests` flips. In **green** (where every run starts), it may write test files and its writes to code are refused. A `run_tests` result that comes back red after a test write moves it to **red**, where it may write code and its writes to test files are refused, so a test cannot be weakened to go green. The next green result moves it back. Writes to Markdown under `docs/` and to `GLOSSARY.md` are not code and pass in either state. We chose this because writing every test before any code is the horizontal slicing the test-first loop warns against (tests of imagined behaviour, written before the implementation has taught anything), and the TESTER/CODER split only kept the tests honest by keeping the roles apart. The guard keeps them honest per slice.

## Considered Options

- Keep TESTER and CODER, but alternate them one slice at a time: rejected, because every slice would cost two fresh-context agent runs re-reading the same code, and the guard already gives the separation the split was for.
- Keep today's two batch stages (all tests, the batch red check, a test gate, then all code): rejected, per the slicing problem above.
- One agent with the rule only in its prompt: rejected, because the invariant says gates are code.

## Consequences

- This replaces the TESTER/CODER part of ADR 0001. `run_tests` and "no agent gets a shell" stand unchanged, and VALIDATOR is still the final, deterministic gate on the whole change.
- The batch red check and the human test gate are gone. The human sees tests and code together at the review gate.
- A run that starts from a review redo begins in a third state, **refactor**: code may change and test files may not, so review findings can be fixed without inventing a failing test. Its first test write returns it to the normal cycle.
- "Red" is the test command's exit code after a test write, which works for any test runner. The guard does not prove the new test is the one failing; the Contract review and VALIDATOR still judge the result.
- `/implement <task>` in the working tree runs under the same guard.
