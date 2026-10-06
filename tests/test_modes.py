import pytest

from k0ntrol.config import AGENTS, load_config
from k0ntrol.modespec import ModeError, ModeSpec, load_all_modes, load_mode


@pytest.fixture
def cfg(tmp_path):
    return load_config(tmp_path, tmp_path / "home")


def test_all_five_modes_load(cfg):
    modes = load_all_modes(cfg)
    assert set(modes) == set(AGENTS)
    assert all(isinstance(m, ModeSpec) and m.instructions.strip() for m in modes.values())


def test_model_and_temperature_come_from_config(cfg):
    cfg["agents"]["coder"] = {"model": "composer-2.5", "temperature": 0.5}
    mode = load_mode("coder", cfg)
    assert (mode.model, mode.temperature) == ("composer-2.5", 0.5)


def test_missing_temperature_is_none(cfg):
    cfg["agents"]["spec"] = {"model": "x"}
    assert load_mode("spec", cfg).temperature is None


def test_coder_prompt_is_the_freeform_desc(cfg):
    assert load_mode("coder", cfg).instructions.startswith(
        "You generate code files based on a spec and reference files."
    )


def test_bulk_reader_prompt_is_the_freeform_desc(cfg):
    assert "Output structured bullets only." in load_mode("bulk_reader", cfg).instructions


def test_all_modes_are_one_shot(cfg):
    assert all(m.one_shot for m in load_all_modes(cfg).values())


def test_unknown_mode_raises_and_names_it(cfg):
    with pytest.raises(ModeError, match="reviewer"):
        load_mode("reviewer", cfg)
