import re
from pathlib import Path

import pytest

from k0ntrol.config import ConfigError, deep_merge, load_config


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_deep_merge_nested_override_keeps_siblings_and_does_not_mutate():
    base = {"agents": {"coder": {"model": "a", "effort": "medium"}}, "threshold_lines": 350}
    out = deep_merge(base, {"agents": {"coder": {"model": "b"}}})
    assert out == {"agents": {"coder": {"model": "b", "effort": "medium"}}, "threshold_lines": 350}
    assert base["agents"]["coder"]["model"] == "a"


def test_defaults_only(tmp_path):
    cfg = load_config(tmp_path, tmp_path / "home")
    assert cfg["threshold_lines"] == 350
    assert cfg["test_cmd"] == "pytest -q"
    assert cfg["gates"] == []


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


def test_home_config_overrides_each_new_agent(tmp_path):
    home = tmp_path / "home"
    write(
        home / ".k0ntrol/config.yaml",
        "agents:\n"
        "  planner: {provider: openai, model: gpt-planner}\n"
        "  researcher: {effort: low, model_calls: 12}\n"
        "  implementer: {model: qwen-implementer, model_calls: 200}\n",
    )
    agents = load_config(tmp_path, home)["agents"]
    assert agents["planner"] == {
        "provider": "openai",
        "model": "gpt-planner",
        "effort": "high",
        "model_calls": 25,
    }
    assert agents["researcher"] == {
        "provider": "anthropic",
        "model": "claude-sonnet-5-5",
        "effort": "low",
        "model_calls": 12,
    }
    assert agents["implementer"] == {
        "provider": "anthropic",
        "model": "qwen-implementer",
        "effort": "medium",
        "model_calls": 200,
    }


def test_project_config_overrides_each_new_agent(tmp_path):
    write(
        tmp_path / ".k0-mem/config.yaml",
        "agents:\n"
        "  planner: {model_calls: 30}\n"
        "  researcher: {provider: openai, model: gpt-researcher, effort: high}\n"
        "  implementer: {model: qwen-implementer}\n",
    )
    agents = load_config(tmp_path, tmp_path / "home")["agents"]
    assert agents["planner"] == {
        "provider": "anthropic",
        "model": "claude-opus-5-5",
        "effort": "high",
        "model_calls": 30,
    }
    assert agents["researcher"] == {
        "provider": "openai",
        "model": "gpt-researcher",
        "effort": "high",
        "model_calls": 40,
    }
    assert agents["implementer"] == {
        "provider": "anthropic",
        "model": "qwen-implementer",
        "effort": "medium",
        "model_calls": 100,
    }


def test_new_agent_model_calls_keep_the_minimum(tmp_path):
    write(
        tmp_path / ".k0-mem/config.yaml",
        "agents:\n  implementer: {model_calls: 0}\n",
    )
    with pytest.raises(ConfigError, match="agents.implementer.model_calls"):
        load_config(tmp_path, tmp_path / "home")


def test_legacy_agent_keys_load_exactly_as_before(tmp_path):
    home = tmp_path / "home"
    write(
        home / ".k0ntrol/config.yaml",
        "agents:\n"
        "  spec: {model: claude-old-spec}\n"
        "  tester: {provider: openai, model: gpt-tester, effort: low, model_calls: 7}\n"
        "  coder: {model_calls: 99}\n",
    )
    write(tmp_path / ".k0-mem/config.yaml", "agents:\n  coder: {model: claude-coder-proj}\n")
    agents = load_config(tmp_path, home)["agents"]
    assert agents["spec"] == {
        "provider": "anthropic",
        "model": "claude-old-spec",
        "effort": "high",
        "model_calls": 25,
    }
    assert agents["tester"] == {
        "provider": "openai",
        "model": "gpt-tester",
        "effort": "low",
        "model_calls": 7,
    }
    assert agents["coder"] == {
        "provider": "anthropic",
        "model": "claude-coder-proj",
        "effort": "medium",
        "model_calls": 99,
    }
