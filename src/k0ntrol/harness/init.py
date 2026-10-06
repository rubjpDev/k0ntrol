from pathlib import Path


MEM = "# Mem\n\nProject facts. One fact per bullet.\n"
RULES = "# Rules\n\nGlobal rules for every agent.\n"
PROJECT_CONFIG = (
    "# Project overrides. An empty mapping inherits the package defaults.\n{}\n"
)


def init_project(root: Path) -> list[str]:
    """Create the initial project memory layout."""
    memory_root = root / ".k0-mem"
    memory_root.mkdir(parents=True, exist_ok=True)
    entries = (
        (memory_root / "Project_context", None),
        (memory_root / "Project_topology", None),
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
