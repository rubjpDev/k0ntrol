import re

import pytest

from k0ntrol.cli import main
from k0ntrol.config import ConfigError, load_config, resolved_params, validate_config


@pytest.fixture
def cfg(tmp_path):
    return load_config(tmp_path, tmp_path / "home")


def test_defaults_are_valid(cfg):
    validate_config(cfg)


def test_default_gates_are_empty(cfg):
    assert cfg["gates"] == []


def test_shipped_agent_defaults_match_the_spec(cfg):
    assert cfg["agents"] == {
        "spec": {"provider": "anthropic", "model": "claude-opus-5-5", "effort": "high", "model_calls": 25},
        "tester": {"provider": "anthropic", "model": "claude-sonnet-5-5", "effort": "medium", "model_calls": 40},
        "coder": {"provider": "anthropic", "model": "claude-sonnet-5-5", "effort": "medium", "model_calls": 60},
        "reviewer": {"provider": "anthropic", "model": "claude-opus-5-5", "effort": "high", "model_calls": 25},
        "chatty": {"provider": "anthropic", "model": "claude-haiku-5-5", "model_calls": 15},
    }


def test_shipped_helper_defaults_carry_no_effort_or_model_calls(cfg):
    assert cfg["helpers"] == {
        "bulk_read": {"provider": "anthropic", "model": "claude-haiku-5-5"},
        "reflect": {"provider": "anthropic", "model": "claude-haiku-5-5"},
    }


def test_shipped_project_key_defaults(cfg):
    assert cfg["test_cmd"] == "pytest -q"
    assert cfg["max_attempts"] == 2
    assert cfg["test_timeout"] == 600


@pytest.mark.parametrize("kind", ["agents", "helpers"])
def test_roster_must_be_a_mapping(cfg, kind):
    cfg[kind] = "not-a-mapping"
    with pytest.raises(ConfigError, match=kind):
        validate_config(cfg)


@pytest.mark.parametrize(
    "key,value",
    [
        ("provider", ""),
        ("provider", 42),
        ("model", ""),
        ("model", None),
        ("effort", "extreme"),
        ("effort", 3),
        ("params", "num_ctx=65536"),
    ],
)
def test_agent_entry_fields_validate(cfg, key, value):
    cfg["agents"]["coder"][key] = value
    with pytest.raises(ConfigError, match=f"agents.coder.{key}"):
        validate_config(cfg)


def test_agent_entry_must_be_a_mapping(cfg):
    cfg["agents"]["coder"] = "claude-sonnet-5-5"
    with pytest.raises(ConfigError, match="agents.coder"):
        validate_config(cfg)


@pytest.mark.parametrize("value", [None, "60", 0, -1, True])
def test_agents_require_model_calls(cfg, value):
    if value is None:
        del cfg["agents"]["coder"]["model_calls"]
    else:
        cfg["agents"]["coder"]["model_calls"] = value
    with pytest.raises(ConfigError, match="agents.coder.model_calls"):
        validate_config(cfg)


def test_helpers_take_no_model_calls(cfg):
    cfg["helpers"]["reflect"]["model_calls"] = 10
    with pytest.raises(ConfigError, match="helpers.reflect.model_calls"):
        validate_config(cfg)


@pytest.mark.parametrize("kind,name", [("agents", "codder"), ("helpers", "reflectt")])
def test_unknown_roster_names_are_rejected(cfg, kind, name):
    cfg[kind][name] = {"provider": "anthropic", "model": "m", "model_calls": 5}
    with pytest.raises(ConfigError, match=f"{kind}.{name}"):
        validate_config(cfg)


def test_unknown_keys_in_an_entry_are_rejected(cfg):
    cfg["agents"]["coder"]["temperature"] = 0.2
    with pytest.raises(ConfigError, match="agents.coder.temperature"):
        validate_config(cfg)


def test_effort_levels_are_low_medium_high(cfg):
    for agent in cfg["agents"].values():
        agent.pop("effort", None)
    cfg["agents"]["spec"]["effort"] = "low"
    cfg["helpers"]["bulk_read"]["effort"] = "high"
    validate_config(cfg)


def test_providers_map_is_optional_and_validates(cfg):
    cfg["providers"] = {"ollama": {"base_url": "http://gpu-box:11434", "params": {"num_ctx": 65536}}}
    validate_config(cfg)


def test_providers_must_be_a_mapping(cfg):
    cfg["providers"] = ["ollama"]
    with pytest.raises(ConfigError, match="providers"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "entry,match",
    [
        ("http://gpu-box", "providers.ollama"),
        ({"base_url": ""}, "providers.ollama.base_url"),
        ({"params": "num_ctx=65536"}, "providers.ollama.params"),
        ({"region": "us-east-1"}, "providers.bedrock.region"),
    ],
)
def test_provider_entries_validate(cfg, entry, match):
    cfg["providers"] = {match.split(".")[1]: entry}
    with pytest.raises(ConfigError, match=re.escape(match)):
        validate_config(cfg)


def test_agent_params_override_their_provider_params(cfg):
    cfg["providers"] = {"ollama": {"params": {"num_ctx": 65536, "keep_alive": "5m"}}}
    entry = {"provider": "ollama", "model": "m", "params": {"num_ctx": 32768}}
    assert resolved_params(cfg, entry) == {"num_ctx": 32768, "keep_alive": "5m"}


def test_resolved_params_fall_back_to_provider_and_defaults(cfg):
    cfg["providers"] = {"ollama": {"params": {"num_ctx": 65536}}}
    assert resolved_params(cfg, {"provider": "ollama", "model": "m"}) == {"num_ctx": 65536}
    assert resolved_params(cfg, {"provider": "anthropic", "model": "m"}) == {}


@pytest.mark.parametrize(
    "key,replacement",
    [
        ("backend", "provider"),
        ("max_retries", "max_attempts"),
        ("retrieval_top_k", "search_memory"),
        ("reflect_every", "reflect"),
        ("validator", "test_cmd"),
        ("retrieval_gate", "search_memory"),
        ("bulk_reader", "helpers.bulk_read"),
    ],
)
def test_retired_top_level_keys_name_the_replacement(cfg, key, replacement):
    cfg[key] = "anything"
    with pytest.raises(ConfigError, match=re.escape(replacement)):
        validate_config(cfg)


@pytest.mark.parametrize(
    "name,replacement",
    [
        ("validator", "test_cmd"),
        ("retrieval_gate", "search_memory"),
        ("bulk_reader", "helpers.bulk_read"),
    ],
)
def test_retired_agent_names_name_the_replacement(cfg, name, replacement):
    cfg["agents"][name] = {"provider": "anthropic", "model": "m", "model_calls": 5}
    with pytest.raises(ConfigError, match=re.escape(replacement)):
        validate_config(cfg)


@pytest.mark.parametrize("key", ["api_key", "auth_token", "client_secret", "db_password"])
def test_secret_keys_are_rejected_at_the_top_level(cfg, key):
    cfg[key] = "x"
    with pytest.raises(ConfigError, match="secrets"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "patch",
    [
        lambda cfg: cfg["agents"]["coder"].update({"api_key": "x"}),
        lambda cfg: cfg["agents"]["coder"].update({"params": {"openai_api_key": "x"}}),
        lambda cfg: cfg["agents"]["coder"].update({"params": {"auth": {"bearer_token": "x"}}}),
        lambda cfg: cfg["helpers"]["reflect"].update({"params": {"aws_secret": "x"}}),
        lambda cfg: cfg.update({"providers": {"ollama": {"params": {"password_file": "x"}}}}),
        lambda cfg: cfg.update({"gates": [{"name": "g", "cmd": "c", "token": "x"}]}),
    ],
)
def test_secret_keys_are_rejected_at_any_level(cfg, patch):
    patch(cfg)
    with pytest.raises(ConfigError, match="secrets"):
        validate_config(cfg)


@pytest.mark.parametrize("key", ["tools", "tool_choice", "functions"])
def test_tool_definitions_in_params_are_rejected(cfg, key):
    cfg["agents"]["coder"]["params"] = {key: [{"name": "shell"}]}
    with pytest.raises(ConfigError, match="agents.coder.params"):
        validate_config(cfg)


def test_tool_definitions_in_provider_params_are_rejected(cfg):
    cfg["providers"] = {"ollama": {"params": {"tools": []}}}
    with pytest.raises(ConfigError, match="providers.ollama.params"):
        validate_config(cfg)


def test_several_errors_are_collected_and_reported_together(cfg):
    cfg["agents"]["coder"]["model"] = ""
    cfg["helpers"]["bulk_read"]["model_calls"] = 5
    cfg["max_retries"] = 3
    with pytest.raises(ConfigError) as exc:
        validate_config(cfg)
    message = str(exc.value)
    assert "agents.coder.model" in message
    assert "helpers.bulk_read.model_calls" in message
    assert "max_attempts" in message


def test_tracker_and_base_branch_are_optional_strings(cfg):
    cfg["tracker"] = "github"
    cfg["base_branch"] = "main"
    validate_config(cfg)


@pytest.mark.parametrize(
    "key,value",
    [
        ("tracker", ""),
        ("tracker", 42),
        ("base_branch", " "),
        ("base_branch", ["main"]),
        ("max_attempts", 0),
        ("max_attempts", "2"),
        ("test_timeout", 0),
        ("test_timeout", 600.5),
    ],
)
def test_project_keys_validate(cfg, key, value):
    cfg[key] = value
    with pytest.raises(ConfigError, match=key):
        validate_config(cfg)


@pytest.mark.parametrize(
    "gate",
    [
        {"cmd": "complexipy src"},
        {"name": "complexity"},
        {"name": "", "cmd": "complexipy src"},
        {"name": "complexity", "cmd": ""},
        "not-a-mapping",
    ],
)
def test_gate_without_name_or_cmd_is_rejected(cfg, gate):
    cfg["gates"] = [gate]
    with pytest.raises(ConfigError, match="gates"):
        validate_config(cfg)


def test_gates_must_be_a_list(cfg):
    cfg["gates"] = {"name": "x", "cmd": "y"}
    with pytest.raises(ConfigError, match="gates"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "key,value",
    [
        ("threshold_lines", 0),
        ("threshold_lines", "350"),
        ("threshold_lines", True),
        ("test_cmd", ""),
        ("test_cmd", 42),
    ],
)
def test_invalid_top_level_value_names_the_key(cfg, key, value):
    cfg[key] = value
    with pytest.raises(ConfigError, match=key):
        validate_config(cfg)


def test_all_errors_are_reported_not_just_the_first(cfg):
    cfg["threshold_lines"] = 0
    cfg["test_cmd"] = ""
    with pytest.raises(ConfigError) as exc:
        validate_config(cfg)
    assert "threshold_lines" in str(exc.value)
    assert "test_cmd" in str(exc.value)


def test_load_config_validates_the_merged_result(tmp_path):
    (tmp_path / ".k0-mem").mkdir()
    (tmp_path / ".k0-mem/config.yaml").write_text("threshold_lines: -5\n")
    with pytest.raises(ConfigError, match="threshold_lines"):
        load_config(tmp_path, tmp_path / "home")


def test_doctor_prints_effective_config(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "threshold_lines: 350" in out
    assert "test_cmd: pytest -q" in out
    assert "active gates: none" in out


def test_doctor_lists_active_gates(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / ".k0-mem").mkdir()
    (tmp_path / ".k0-mem/config.yaml").write_text(
        "gates:\n"
        "  - name: complexity\n"
        "    cmd: complexipy src --max-complexity-allowed 6\n"
        "  - name: coverage\n"
        "    cmd: pytest --cov=src --cov-fail-under=90\n"
    )
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "active gates:" in out
    assert "complexity: complexipy src --max-complexity-allowed 6" in out
    assert "coverage: pytest --cov=src --cov-fail-under=90" in out
    assert "active gates: none" not in out


def test_doctor_reports_invalid_config_and_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / ".k0-mem").mkdir()
    (tmp_path / ".k0-mem/config.yaml").write_text("threshold_lines: 0\n")
    assert main(["doctor"]) == 1
    assert "threshold_lines" in capsys.readouterr().err
