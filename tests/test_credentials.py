import stat

import pytest

from k0ntrol.credentials import (
    credentials_path,
    known_credential_names,
    load_credentials,
    save_credential,
)


def test_credentials_path_is_under_the_k0ntrol_dir(tmp_path):
    assert credentials_path(tmp_path) == tmp_path / ".k0ntrol" / "credentials"


def test_known_names_cover_every_provider_variable():
    names = known_credential_names()
    assert {
        "ANTHROPIC_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "BRAVE_SEARCH_API_KEY",
        "OPENAI_API_KEY",
    } <= names


def test_save_creates_the_dir_and_a_private_file(tmp_path):
    save_credential(tmp_path, "ANTHROPIC_API_KEY", "sk-1")

    path = credentials_path(tmp_path)
    assert path.read_text() == "ANTHROPIC_API_KEY=sk-1\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_save_replaces_an_existing_key_and_keeps_the_others(tmp_path):
    save_credential(tmp_path, "OPENAI_API_KEY", "old")
    save_credential(tmp_path, "ANTHROPIC_API_KEY", "a")
    save_credential(tmp_path, "OPENAI_API_KEY", "new")

    lines = credentials_path(tmp_path).read_text().splitlines()
    assert lines == ["ANTHROPIC_API_KEY=a", "OPENAI_API_KEY=new"]


def test_save_tightens_the_mode_of_a_loose_file(tmp_path):
    path = credentials_path(tmp_path)
    path.parent.mkdir()
    path.write_text("OPENAI_API_KEY=x\n")
    path.chmod(0o644)

    save_credential(tmp_path, "ANTHROPIC_API_KEY", "a")

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("name", ["", "BAD NAME", "A=B", "1KEY"])
def test_save_rejects_an_invalid_name(tmp_path, name):
    with pytest.raises(ValueError, match="invalid credential name"):
        save_credential(tmp_path, name, "v")


@pytest.mark.parametrize("value", ["a\nb", "a\rb"])
def test_save_rejects_a_multiline_value(tmp_path, value):
    with pytest.raises(ValueError, match="span lines"):
        save_credential(tmp_path, "ANTHROPIC_API_KEY", value)


def test_load_sets_known_names_missing_from_the_environment(tmp_path):
    save_credential(tmp_path, "ANTHROPIC_API_KEY", "sk-1")
    environ: dict[str, str] = {}

    assert load_credentials(tmp_path, environ) == ["ANTHROPIC_API_KEY"]
    assert environ == {"ANTHROPIC_API_KEY": "sk-1"}


def test_load_never_overrides_what_the_user_set(tmp_path):
    save_credential(tmp_path, "ANTHROPIC_API_KEY", "saved")
    environ = {"ANTHROPIC_API_KEY": "mine"}

    assert load_credentials(tmp_path, environ) == []
    assert environ["ANTHROPIC_API_KEY"] == "mine"


def test_load_treats_an_empty_variable_as_unset(tmp_path):
    save_credential(tmp_path, "ANTHROPIC_API_KEY", "saved")
    environ = {"ANTHROPIC_API_KEY": ""}

    assert load_credentials(tmp_path, environ) == ["ANTHROPIC_API_KEY"]
    assert environ["ANTHROPIC_API_KEY"] == "saved"


def test_load_ignores_unknown_keys_comments_and_blank_lines(tmp_path):
    path = credentials_path(tmp_path)
    path.parent.mkdir()
    path.write_text("# note\n\nPATH=/evil\nnot a pair\nOPENAI_API_KEY = sk-2 \n")
    environ: dict[str, str] = {}

    assert load_credentials(tmp_path, environ) == ["OPENAI_API_KEY"]
    assert environ == {"OPENAI_API_KEY": "sk-2"}


def test_load_without_a_file_returns_nothing(tmp_path):
    environ: dict[str, str] = {}

    assert load_credentials(tmp_path, environ) == []
    assert environ == {}
