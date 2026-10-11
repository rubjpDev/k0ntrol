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
    return [
        _init_entry(path, content, root)
        for path, content in _init_entries(memory_root, topology)
    ]


def _init_entries(
    memory_root: Path, topology: Path
) -> tuple[tuple[Path, str | None], ...]:
    return (
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


def _init_entry(path: Path, content: str | None, root: Path) -> str:
    if path.exists():
        action = "exists"
    else:
        _create_init_path(path, content)
        action = "created"
    return f"{action} {path.relative_to(root).as_posix()}"


def _create_init_path(path: Path, content: str | None) -> None:
    if content is None:
        path.mkdir()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
