import re
from pathlib import Path

import pytest

from k0ntrol.config import ConfigError, deep_merge, load_config, validate_config
from k0ntrol.providers import model_kwargs


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_deep_merge_nested_override_keeps_siblings_and_does_not_mutate():
    base = {"agents": {"coder": {"model": "a", "effort": "medium"}}, "threshold_lines": 350}
    out = deep_merge(base, {"agents": {"coder": {"model": "b"}}})
    assert out == {"agents": {"coder": {"model": "b", "effort": "medium"}}, "threshold_lines": 350}
    assert base["agents"]["coder"]["model"] == "a"


def test_researcher_defaults_to_the_planners_strong_model(tmp_path):
    cfg = load_config(tmp_path, tmp_path / "home")
    for key in ("provider", "model", "effort"):
        assert cfg["agents"]["researcher"][key] == cfg["agents"]["planner"][key]


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
        "model": "claude-opus-5-5",
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


def test_planning_settings_have_defaults(tmp_path):
    cfg = load_config(tmp_path, tmp_path / "home")
    assert cfg["max_questions"] == 25
    assert cfg["max_round_questions"] == 7
    assert cfg["ticket_size"] == "small"


def test_project_config_overrides_planning_settings(tmp_path):
    write(
        tmp_path / ".k0-mem/config.yaml",
        "max_questions: 40\nmax_round_questions: 3\nticket_size: large\n",
    )
    cfg = load_config(tmp_path, tmp_path / "home")
    assert cfg["max_questions"] == 40
    assert cfg["max_round_questions"] == 3
    assert cfg["ticket_size"] == "large"


@pytest.mark.parametrize(
    "line",
    [
        "max_questions: 0",
        "max_questions: -5",
        'max_questions: "25"',
        "max_round_questions: 0",
        "max_round_questions: -2",
        "max_round_questions: 7.5",
    ],
)
def test_invalid_question_limits_are_rejected(tmp_path, line):
    key = line.split(":", 1)[0]
    write(tmp_path / ".k0-mem/config.yaml", line + "\n")
    with pytest.raises(ConfigError, match=key):
        load_config(tmp_path, tmp_path / "home")


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_ticket_size_accepts_the_three_sizes(tmp_path, size):
    write(tmp_path / ".k0-mem/config.yaml", f"ticket_size: {size}\n")
    assert load_config(tmp_path, tmp_path / "home")["ticket_size"] == size


@pytest.mark.parametrize("line", ["ticket_size: huge", "ticket_size: Small", "ticket_size: 42", "ticket_size: true"])
def test_invalid_ticket_size_is_rejected(tmp_path, line):
    write(tmp_path / ".k0-mem/config.yaml", line + "\n")
    with pytest.raises(ConfigError, match="ticket_size"):
        load_config(tmp_path, tmp_path / "home")


@pytest.mark.parametrize(
    "name,replacement", [("spec", "planner"), ("tester", "implementer"), ("coder", "implementer")]
)
@pytest.mark.parametrize("location", [".k0-mem/config.yaml", "home/.k0ntrol/config.yaml"])
def test_retired_agent_keys_name_their_replacement(tmp_path, name, replacement, location):
    home = tmp_path / "home"
    write(tmp_path / location, f"agents:\n  {name}: {{model: old-model}}\n")
    message = f"agents.{name}: retired in 0.1.0; use agents.{replacement}"
    with pytest.raises(ConfigError) as error:
        load_config(tmp_path, home)
    assert str(error.value) == message


def test_validate_config_allows_output_token_count_params(tmp_path):
    cfg = load_config(tmp_path, tmp_path / "home")
    cfg["providers"] = {
        "openai": {
            "params": {
                "max_tokens": 4000,
                "max_output_tokens": 4000,
                "max_completion_tokens": 4000,
            }
        }
    }

    validate_config(cfg)


@pytest.mark.parametrize("secret_key", ["token", "access_token", "auth_token"])
def test_validate_config_still_rejects_token_secrets(tmp_path, secret_key):
    cfg = load_config(tmp_path, tmp_path / "home")
    cfg["providers"] = {"openai": {"params": {secret_key: "secret-value"}}}

    with pytest.raises(ConfigError, match="secrets are not allowed"):
        validate_config(cfg)


def test_a_project_max_tokens_param_reaches_the_model_on_start(tmp_path):
    write(tmp_path / ".k0-mem/config.yaml", "providers:\n  openai:\n    params:\n      max_tokens: 4000\n")

    cfg = load_config(tmp_path, tmp_path / "home")

    entry = {"provider": "openai", "model": "gpt-6-luna"}
    assert model_kwargs(cfg, entry, environ={})["max_tokens"] == 4000
