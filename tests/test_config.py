import re
from pathlib import Path

import pytest

from k0ntrol.config import ConfigError, deep_merge, load_config


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_deep_merge_nested_override_keeps_siblings_and_does_not_mutate():
    base = {"agents": {"coder": {"model": "a", "temperature": 0.2}}, "threshold_lines": 350}
    out = deep_merge(base, {"agents": {"coder": {"model": "b"}}})
    assert out == {"agents": {"coder": {"model": "b", "temperature": 0.2}}, "threshold_lines": 350}
    assert base["agents"]["coder"]["model"] == "a"


def test_defaults_only(tmp_path):
    cfg = load_config(tmp_path, tmp_path / "home")
    assert cfg["threshold_lines"] == 350


def test_default_agent_models_match_selected_cursor_models(tmp_path):
    cfg = load_config(tmp_path, tmp_path / "home")
    assert cfg["agents"]["spec"]["model"] == "claude-opus-5-5-high"
    for name in ("tester", "coder", "validator", "bulk_reader"):
        assert cfg["agents"][name]["model"] == "gpt-5.6-luna-max"


def test_home_overrides_defaults(tmp_path):
    home = tmp_path / "home"
    write(home / ".k0ntrol/config.yaml", "threshold_lines: 200\n")
    assert load_config(tmp_path, home)["threshold_lines"] == 200


def test_project_overrides_home(tmp_path):
    home = tmp_path / "home"
    write(home / ".k0ntrol/config.yaml", "threshold_lines: 200\n")
    write(tmp_path / ".k0-mem/config.yaml", "threshold_lines: 100\n")
    assert load_config(tmp_path, home)["threshold_lines"] == 100


def test_empty_override_file_is_ignored(tmp_path):
    write(tmp_path / ".k0-mem/config.yaml", "")
    assert load_config(tmp_path, tmp_path / "home")["threshold_lines"] == 350


def test_non_mapping_file_raises_with_its_path(tmp_path):
    f = tmp_path / ".k0-mem/config.yaml"
    write(f, "- just\n- a list\n")
    with pytest.raises(ConfigError, match=re.escape(str(f))):
        load_config(tmp_path, tmp_path / "home")


def test_broken_yaml_raises_with_its_path(tmp_path):
    f = tmp_path / ".k0-mem/config.yaml"
    write(f, "threshold_lines: [unclosed\n")
    with pytest.raises(ConfigError, match=re.escape(str(f))):
        load_config(tmp_path, tmp_path / "home")
