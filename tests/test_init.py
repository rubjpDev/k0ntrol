from k0ntrol.cli import main
from k0ntrol.config import load_config
from k0ntrol.harness.init import init_project


MEM = "# Mem\n\nProject facts. One fact per bullet.\n"
RULES = "# Rules\n\nGlobal rules for every agent.\n"
PROJECT_CONFIG = (
    "# Project overrides. An empty mapping inherits the package defaults.\n{}\n"
)


def test_init_creates_the_soul_layout(tmp_path):
    lines = init_project(tmp_path)
    root = tmp_path / ".k0-mem"
    assert (root / "Project_context").is_dir()
    assert (root / "Project_topology").is_dir()
    assert (root / "modes").is_dir()
    assert (root / "General_mem" / "Mem.md").read_text(encoding="utf-8") == MEM
    assert (root / "General_mem" / "Rules.md").read_text(encoding="utf-8") == RULES
    assert (root / "config.yaml").read_text(encoding="utf-8") == PROJECT_CONFIG
    assert lines == [
        "created .k0-mem/Project_context",
        "created .k0-mem/Project_topology",
        "created .k0-mem/General_mem/Mem.md",
        "created .k0-mem/General_mem/Rules.md",
        "created .k0-mem/config.yaml",
        "created .k0-mem/modes",
    ]
    assert load_config(tmp_path, tmp_path / "home")["threshold_lines"] == 350
    assert not (root / "modes").joinpath("tester.yaml").exists()


def test_cli_init_prints_the_report(tmp_path, capsys):
    assert main(["init", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "created .k0-mem/General_mem/Mem.md" in out
    assert (tmp_path / ".k0-mem" / "General_mem" / "Mem.md").is_file()
