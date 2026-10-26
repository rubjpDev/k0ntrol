import pytest

from k0ntrol.cli import main
from k0ntrol.config import ConfigError, load_config, validate_config


@pytest.fixture
def cfg(tmp_path):
    return load_config(tmp_path, tmp_path / "home")


def test_defaults_are_valid(cfg):
    validate_config(cfg)


def test_default_gates_are_empty(cfg):
    assert cfg["gates"] == []


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
