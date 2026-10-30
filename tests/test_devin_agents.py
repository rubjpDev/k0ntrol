import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEVIN_AGENTS = REPO_ROOT / ".devin" / "agents"
LEGACY_AGENTS = REPO_ROOT / ".claude" / "agents"


def parse_frontmatter(path: Path) -> tuple[dict, str]:
    assert path.exists(), f"missing file: {path}"
    match = re.fullmatch(r"---\n(.*?)\n---\n(.*)", path.read_text(), re.DOTALL)
    assert match, f"{path} has no YAML frontmatter block"
    return yaml.safe_load(match.group(1)), match.group(2)


@pytest.mark.parametrize(
    ("profile", "model"),
    [
        ("standards-reviewer", "swe-2-high"),
        ("spec-reviewer", "swe-2-high"),
        ("researcher", "swe-2-high"),
        ("merger", "swe-2-medium"),
        ("design-variant", "swe-2-max"),
    ],
)
def test_migrated_profile_matches_legacy(profile: str, model: str) -> None:
    devin_fm, devin_body = parse_frontmatter(DEVIN_AGENTS / f"{profile}.md")
    legacy_fm, legacy_body = parse_frontmatter(LEGACY_AGENTS / f"{profile}.md")

    assert devin_fm["name"] == profile
    assert devin_fm["model"] == model
    assert devin_fm["description"] == legacy_fm["description"]
    assert devin_body == legacy_body
    assert "tools" not in devin_fm

    if profile in ("standards-reviewer", "spec-reviewer", "design-variant"):
        assert devin_fm["allowed-tools"] == ["read", "grep", "glob", "exec"]
    else:
        assert "allowed-tools" not in devin_fm


def test_swe_worker_profile() -> None:
    fm, _body = parse_frontmatter(DEVIN_AGENTS / "swe-worker.md")
    assert fm["name"] == "swe-worker"
    assert fm["model"] == "swe-2-high"


def test_every_devin_profile_uses_swe_2() -> None:
    for path in DEVIN_AGENTS.glob("*.md"):
        fm, _body = parse_frontmatter(path)
        assert fm["model"] in ("swe-2-medium", "swe-2-high", "swe-2-max"), path.name


def routing_section() -> str:
    text = (REPO_ROOT / "AGENTS.md").read_text()
    match = re.search(r"### Subagent routing\n(.*?)\n### Issue tracker", text, re.DOTALL)
    assert match, "AGENTS.md has no Subagent routing section"
    return match.group(1)


def test_agents_routing_section() -> None:
    section = routing_section()

    assert ".devin/agents/" in section
    for name in (
        "standards-reviewer",
        "spec-reviewer",
        "researcher",
        "merger",
        "swe-worker",
        "design-variant",
    ):
        assert name in section
    assert "implementer" in section
    assert "exploration" in section
    assert "other designers" in section
    assert "The other agents inherit." not in section


def test_code_review_skill_uses_named_profiles() -> None:
    text = (REPO_ROOT / ".agents" / "skills" / "code-review" / "SKILL.md").read_text()

    assert "Task spawns" not in text
    assert "two Task calls" not in text
    assert "Spec on inherit" not in text
    assert "same model as the parent" not in text
    assert "standards-reviewer" in text
    assert "spec-reviewer" in text
