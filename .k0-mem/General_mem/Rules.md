# Rules

Global rules for every agent.

- Python ≥ 3.11, stdlib first. Runtime dependencies are a closed list (`pyyaml`, `rich` and the ADR 0008 additions): never add one.
- Cognitive complexity ≤ 6 per function, coverage ≥ 90 %. VALIDATOR enforces both.
- Small functions, explicit names, type hints on public functions.
- Everything is English: code, comments, tests and docs.
- Use the words in `GLOSSARY.md`. If code contradicts an ADR in `docs/adr/`, the code is wrong.
