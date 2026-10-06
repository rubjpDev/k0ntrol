import pytest

from k0ntrol.cli import main
from k0ntrol.config import ConfigError, load_config, validate_config


@pytest.fixture
def cfg(tmp_path):
    return load_config(tmp_path, tmp_path / "home")


def test_defaults_are_valid(cfg):
    validate_config(cfg)


@pytest.mark.parametrize(
    "key,value",
    [
        ("backend", "openai"),
        ("threshold_lines", 0),
        ("threshold_lines", "350"),
        ("threshold_lines", True),
        ("test_cmd", ""),
        ("max_retries", -1),
    ],
)
def test_invalid_top_level_value_names_the_key(cfg, key, value):
    cfg[key] = value
    with pytest.raises(ConfigError, match=key):
        validate_config(cfg)


def test_missing_agent_is_named(cfg):
    del cfg["agents"]["coder"]
    with pytest.raises(ConfigError, match="coder"):
        validate_config(cfg)


def test_agent_without_model_is_named(cfg):
    cfg["agents"]["tester"] = {"temperature": 0.2}
    with pytest.raises(ConfigError, match="tester"):
        validate_config(cfg)


def test_temperature_is_optional(cfg):
    cfg["agents"]["spec"] = {"model": "x"}
    validate_config(cfg)


def test_all_errors_are_reported_not_just_the_first(cfg):
    cfg["threshold_lines"] = 0
    cfg["max_retries"] = -1
    with pytest.raises(ConfigError) as exc:
        validate_config(cfg)
    assert "threshold_lines" in str(exc.value)
    assert "max_retries" in str(exc.value)


def test_load_config_validates_the_merged_result(tmp_path):
    (tmp_path / ".k0-mem").mkdir()
    (tmp_path / ".k0-mem/config.yaml").write_text("threshold_lines: -5\n")
    with pytest.raises(ConfigError, match="threshold_lines"):
        load_config(tmp_path, tmp_path / "home")


def test_doctor_prints_effective_config(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("k0ntrol.backends.cursor.cursor_problems", lambda cfg: [])
    assert main(["doctor"]) == 0
    assert "threshold_lines: 350" in capsys.readouterr().out


def test_doctor_reports_invalid_config_and_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / ".k0-mem").mkdir()
    (tmp_path / ".k0-mem/config.yaml").write_text("threshold_lines: 0\n")
    assert main(["doctor"]) == 1
    assert "threshold_lines" in capsys.readouterr().err
