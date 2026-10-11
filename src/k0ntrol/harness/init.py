from pathlib import Path


MEM = "# Mem\n\nProject facts. One fact per bullet.\n"
RULES = "# Rules\n\nGlobal rules for every agent.\n"
ARCHITECTURE = "# Architecture\n\nSystem shape. Layers, boundaries, key modules.\n"
STRATEGY = "# Strategy\n\nHow we change this codebase.\n"
CODE_RULES = "# Code rules\n\nLocal coding constraints.\n"
MEMORY_GITIGNORE = "sessions/\nevents.jsonl\n*.sqlite\n"
PROJECT_CONFIG = (
    "# Project overrides. An empty mapping inherits the package defaults.\n"
    "#\n"
    "# Uncomment to enable validator gates after a green test_cmd:\n"
    "# gates:\n"
    "#   - name: complexity\n"
    "#     cmd: complexipy src --max-complexity-allowed 6\n"
    "#   - name: coverage\n"
    "#     cmd: pytest --cov=src --cov-fail-under=90\n"
    "{}\n"
)


def init_project(root: Path) -> list[str]:
    """Create the initial project memory layout."""
    root = root.resolve()
    memory_root = root / ".k0-mem"
    memory_root.mkdir(parents=True, exist_ok=True)
    topology = memory_root / "Project_topology" / root.name
    entries = (
        (memory_root / "Project_context", None),
        (memory_root / "Project_topology", None),
        (topology / "architecture.md", ARCHITECTURE),
        (topology / "strategy.md", STRATEGY),
        (topology / "code_rules.md", CODE_RULES),
        (memory_root / ".gitignore", MEMORY_GITIGNORE),
        (memory_root / "General_mem" / "Mem.md", MEM),
        (memory_root / "General_mem" / "Rules.md", RULES),
        (memory_root / "config.yaml", PROJECT_CONFIG),
        (memory_root / "modes", None),
    )
    lines = []
    for path, content in entries:
        if path.exists():
            action = "exists"
        else:
            if content is None:
                path.mkdir()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            action = "created"
        lines.append(f"{action} {path.relative_to(root).as_posix()}")
    return lines
