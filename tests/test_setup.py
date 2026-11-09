import io
import stat
import subprocess

import pytest
import yaml
from rich.console import Console

from k0ntrol.config import load_config
from k0ntrol.credentials import credentials_path
from k0ntrol.frontends import setup as screens
from k0ntrol.frontends.prompt import Prompt
from k0ntrol.frontends.setup import (
    run_project_setup,
    detect_vcs,
    run_provider_setup,
    suggest_test_command,
)


def make_console():
    return Console(file=io.StringIO(), record=True, width=100, color_system=None)


def make_prompt(console, *answers):
    pending = list(answers)

    def read(_prompt):
        return pending.pop(0)

    prompt = Prompt(console, commands={}, tty=False, read_input=read)
    prompt.pending = pending
    return prompt


def screen(console):
    return console.export_text()


def fake_secret(value):
    asked = []

    def read(label):
        asked.append(label)
        return value

    read.asked = asked
    return read


def no_secret(_label):
    raise AssertionError("no secret expected")


def user_config(home):
    return yaml.safe_load((home / ".k0ntrol" / "config.yaml").read_text())


# --- project setup -------------------------------------------------------------


@pytest.mark.parametrize(
    "marker, command",
    [
        ("uv.lock", "uv run pytest -q"),
        ("pyproject.toml", "pytest -q"),
        ("setup.cfg", "pytest -q"),
        ("pytest.ini", "pytest -q"),
        ("package.json", "npm test"),
        ("go.mod", "go test ./..."),
        ("Cargo.toml", "cargo test"),
    ],
)
def test_suggest_test_command_by_marker_file(tmp_path, marker, command):
    (tmp_path / marker).write_text("")

    assert suggest_test_command(tmp_path) == command


def test_suggest_test_command_prefers_uv_lock(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "uv.lock").write_text("")

    assert suggest_test_command(tmp_path) == "uv run pytest -q"


def test_suggest_test_command_without_a_marker(tmp_path):
    assert suggest_test_command(tmp_path) is None


def test_project_setup_writes_the_answers(tmp_path):
    root = tmp_path / "shop"
    root.mkdir()
    (root / "uv.lock").write_text("")
    console = make_console()

    path = run_project_setup(root, make_prompt(console, "1", "1", "1", "1"), console)

    assert path == root / ".k0-mem" / "config.yaml"
    text = path.read_text()
    assert text.startswith("# Project overrides.")
    assert yaml.safe_load(text) == {
        "vcs": "github",
        "tracker": "github",
        "base_branch": "main",
        "test_cmd": "uv run pytest -q",
    }
    assert (root / ".k0-mem" / "General_mem" / "Mem.md").exists()
    assert load_config(root, home_dir=tmp_path / "home")["test_cmd"] == "uv run pytest -q"
    out = screen(console)
    assert "Set up k0 for shop" in out
    assert "Saved .k0-mem/config.yaml" in out


def saved(root):
    return yaml.safe_load((root / ".k0-mem/config.yaml").read_text())


def test_a_coming_soon_tracker_is_saved_with_a_warning(tmp_path):
    console = make_console()

    run_project_setup(tmp_path, make_prompt(console, "github", "jira", "1", "1", "pytest"), console)

    assert screen(console).count("/full needs GitHub in 0.1.0") == 1
    assert saved(tmp_path)["tracker"] == "jira"


def test_a_coming_soon_host_is_saved_and_offers_only_its_trackers(tmp_path):
    console = make_console()
    seen = []
    prompt = make_prompt(console, "bitbucket", "jira", "1", "1", "pytest")
    choose = prompt.choose
    prompt.choose = lambda question, options: seen.append(options) or choose(question, options)

    run_project_setup(tmp_path, prompt, console)

    assert [key for key, _ in seen[1]] == ["jira", "linear"]
    assert screen(console).count("/full needs GitHub in 0.1.0") == 1
    assert (saved(tmp_path)["vcs"], saved(tmp_path)["tracker"]) == ("bitbucket", "jira")


def git_repo(path, remote=None):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    if remote:
        subprocess.run(["git", "remote", "add", "origin", remote], cwd=path, check=True)
    return path


@pytest.mark.parametrize(
    "remote, host",
    [
        ("https://github.com/me/repo.git", "github"),
        ("git@github.com:me/repo.git", "github"),
        ("https://user@bitbucket.org/me/repo.git", "bitbucket"),
        ("ssh://git@bitbucket.org:22/me/repo.git", "bitbucket"),
        ("https://gitlab.com/me/repo.git", "gitlab"),
        ("git@gitlab.mycorp.io:me/repo.git", "gitlab"),
        ("https://git.example.com/me/repo.git", None),
        (None, None),
    ],
)
def test_detect_vcs_from_the_origin_remote(tmp_path, remote, host):
    assert detect_vcs(git_repo(tmp_path, remote)) == host


def test_the_detected_host_comes_first_and_is_labelled(tmp_path):
    seen = []

    class Spy:
        def choose(self, question, options):
            seen.append(options)
            return "github"

    git_repo(tmp_path, "git@bitbucket.org:me/repo.git")

    screens._ask_vcs(tmp_path, Spy())

    assert seen[0] == [
        ("bitbucket", "Bitbucket (detected, coming soon)"),
        ("github", "GitHub"),
        ("gitlab", "GitLab (coming soon)"),
    ]


def test_without_a_remote_github_comes_first_and_nothing_is_detected(tmp_path):
    options = []

    class Spy:
        def choose(self, question, choices):
            options.extend(choices)
            return "github"

    screens._ask_vcs(git_repo(tmp_path), Spy())

    assert options[0] == ("github", "GitHub")
    assert all("(detected)" not in label for _, label in options)


def test_a_detected_gitlab_is_saved_as_detected(tmp_path):
    console = make_console()
    git_repo(tmp_path, "https://gitlab.com/me/repo.git")

    run_project_setup(tmp_path, make_prompt(console, "1", "gitlab", "1", "1", "pytest"), console)

    assert saved(tmp_path)["vcs"] == "gitlab"
    assert saved(tmp_path)["tracker"] == "gitlab"


def test_trackers_are_filtered_by_the_code_host(tmp_path):
    seen = []

    class Spy:
        def choose(self, question, options):
            seen.append([key for key, _ in options])
            return "github"

    for host in ("github", "gitlab", "bitbucket"):
        screens._ask_tracker(host, Spy())

    assert seen == [["github", "linear", "jira"], ["gitlab", "linear", "jira"], ["jira", "linear"]]


def test_project_setup_takes_another_branch_and_command(tmp_path):
    console = make_console()
    prompt = make_prompt(console, "1", "1", "2", "", "develop", "1", "", "make check")

    path = run_project_setup(tmp_path, prompt, console)

    data = yaml.safe_load(path.read_text())
    assert data["base_branch"] == "develop"
    assert data["test_cmd"] == "make check"
    assert prompt.pending == []


def test_project_setup_without_a_test_suggestion_offers_only_another(tmp_path):
    console = make_console()

    run_project_setup(tmp_path, make_prompt(console, "1", "1", "1", "1", "pytest -x"), console)

    assert yaml.safe_load((tmp_path / ".k0-mem/config.yaml").read_text())["test_cmd"] == "pytest -x"


# --- provider setup ------------------------------------------------------------


def lister(*models):
    calls = []

    def list_models(provider, base_url):
        calls.append((provider, base_url))
        return list(models)

    list_models.calls = calls
    return list_models


def test_ollama_offers_installed_models_and_writes_the_config(tmp_path):
    home, root = tmp_path / "home", tmp_path / "repo"
    root.mkdir()
    console = make_console()
    models = lister("qwen3:8b", "llama3.1")
    # provider; strong: model 2, effort default; middle: model 1, effort low; light: model 1, none
    prompt = make_prompt(console, "ollama", "2", "1", "1", "3", "1", "1")

    path = run_provider_setup(home, prompt, console, root=root, list_models=models)

    assert path == home / ".k0ntrol" / "config.yaml"
    assert models.calls[0] == ("ollama", "http://localhost:11434")
    assert len(models.calls) == 1
    data = user_config(home)
    assert "providers" not in data
    assert data["agents"]["spec"] == {"provider": "ollama", "model": "llama3.1", "effort": "high"}
    assert data["agents"]["reviewer"] == data["agents"]["spec"]
    assert data["agents"]["coder"] == {"provider": "ollama", "model": "qwen3:8b", "effort": "low"}
    assert data["agents"]["tester"] == data["agents"]["coder"]
    assert data["agents"]["chatty"] == {"provider": "ollama", "model": "qwen3:8b"}
    assert data["helpers"]["bulk_read"] == data["agents"]["chatty"]
    assert data["helpers"]["reflect"] == data["agents"]["chatty"]
    out = screen(console)
    assert "Choose the models k0 uses" in out
    assert "SPEC" in out and "llama3.1 · high" in out
    assert "check it with /status · k0 doctor --live" in out


def test_ollama_uses_the_base_url_from_the_project_config(tmp_path):
    home, root = tmp_path / "home", tmp_path / "repo"
    (root / ".k0-mem").mkdir(parents=True)
    (root / ".k0-mem/config.yaml").write_text("providers:\n  ollama:\n    base_url: http://box:1\n")
    console = make_console()
    models = lister("m")

    run_provider_setup(home, make_prompt(console, "ollama", *["1"] * 6), console,
                       root=root, list_models=models)

    assert models.calls == [("ollama", "http://box:1")]


def test_ollama_unreachable_warns_and_asks_for_model_names(tmp_path):
    home = tmp_path / "home"
    console = make_console()

    def down(provider, base_url):
        raise ConnectionError("refused")

    answers = ["ollama", "qwen3:8b", "1", "qwen3:8b", "1", "", "qwen3:4b", "1"]
    path = run_provider_setup(home, make_prompt(console, *answers), console,
                              root=tmp_path, list_models=down)

    out = screen(console)
    assert "Ollama: cannot reach the server (refused)" in out
    assert "pull a model that can call tools" in out
    assert out.count("Start Ollama") == 1
    assert yaml.safe_load(path.read_text())["helpers"]["reflect"]["model"] == "qwen3:4b"


def test_lmstudio_without_tool_models_warns_once(tmp_path):
    home = tmp_path / "home"
    console = make_console()
    answers = ["lmstudio", "m", "1", "m", "1", "m", "1", "2"]

    run_provider_setup(home, make_prompt(console, *answers), console,
                       root=tmp_path, list_models=lister(), read_secret=no_secret)

    out = screen(console)
    assert out.count("LM Studio: no installed model can call tools") == 1
    assert user_config(home)["agents"]["coder"]["provider"] == "lmstudio"


def test_default_lister_wraps_tool_calling_models(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(screens, "tool_calling_models",
                        lambda provider, url: seen.append((provider, url)) or ["m1"])
    console = make_console()

    run_provider_setup(tmp_path / "home", make_prompt(console, "ollama", *["1"] * 6),
                       console, root=tmp_path)

    assert seen == [("ollama", "http://localhost:11434")]


def test_bedrock_presets_region_and_saved_key(tmp_path):
    home = tmp_path / "home"
    console = make_console()
    secret = fake_secret("  bearer-123 ")
    # strong: Qwen3 Next (1); middle: Qwen3 Coder Next (1); light: Nova Lite (4), no effort (1)
    answers = ["bedrock", "1", "1", "1", "1", "4", "1", "3", "save"]  # region 3: eu-west-1

    path = run_provider_setup(home, make_prompt(console, *answers), console,
                              root=tmp_path, read_secret=secret)

    data = yaml.safe_load(path.read_text())
    assert data["providers"] == {"bedrock": {"params": {"region_name": "eu-west-1"}}}
    assert data["agents"]["spec"]["model"] == "qwen.qwen3-next-80b-a3b"
    assert data["agents"]["coder"]["model"] == "qwen.qwen3-coder-next"
    assert data["helpers"]["reflect"]["model"] == "us.amazon.nova-lite-v1:0"
    assert credentials_path(home).read_text() == "AWS_BEARER_TOKEN_BEDROCK=bearer-123\n"
    assert stat.S_IMODE(credentials_path(home).stat().st_mode) == 0o600
    assert secret.asked == ["AWS_BEARER_TOKEN_BEDROCK: "]
    out = screen(console)
    assert "Saved AWS_BEARER_TOKEN_BEDROCK" in out
    assert "bearer-123" not in out
    assert "bearer-123" not in path.read_text()


def test_bedrock_tested_default_comes_first_per_tier():
    assert [key for key, _ in screens._bedrock_options("strong")][:2] == [
        "qwen.qwen3-next-80b-a3b", "qwen.qwen3-coder-next"]
    assert screens._bedrock_options("middle")[0] == (
        "qwen.qwen3-coder-next", "Qwen3 Coder Next (tested)")
    assert screens._bedrock_options("light")[0][0] == "qwen.qwen3-next-80b-a3b"
    keys = [key for key, _ in screens._bedrock_options("light")]
    assert len(keys) == 7 and len(set(keys)) == 7
    assert "us.anthropic.claude-haiku-5-5-v1:0" in keys


def test_bedrock_another_region_and_another_model(tmp_path):
    home = tmp_path / "home"
    console = make_console()
    answers = ["bedrock", "8", "my.model", "1", "1", "1", "1", "1", "4", "ap-south-1", "env"]

    run_provider_setup(home, make_prompt(console, *answers), console,
                       root=tmp_path, read_secret=no_secret)

    data = user_config(home)
    assert data["providers"]["bedrock"]["params"]["region_name"] == "ap-south-1"
    assert data["agents"]["spec"]["model"] == "my.model"
    assert not credentials_path(home).exists()


def test_effort_options_put_the_tier_default_first():
    seen = []

    class Spy:
        def choose(self, question, options):
            seen.append((question, [key for key, _ in options]))
            return options[0][0]

    assert screens._ask_effort(Spy(), "strong") == "high"
    assert screens._ask_effort(Spy(), "middle") == "medium"
    assert screens._ask_effort(Spy(), "light") is None
    assert seen[0] == ("Effort for SPEC and REVIEWER", ["high", "medium", "low", "none"])
    assert seen[1][1] == ["medium", "high", "low", "none"]
    assert seen[2][1] == ["none", "high", "medium", "low"]


def test_choosing_no_effort_cancels_the_shipped_default(tmp_path):
    home = tmp_path / "home"
    console = make_console()
    # strong: model 1, effort "none" (4th); middle: 1, medium; light: 1, none
    answers = ["anthropic", "1", "4", "1", "1", "1", "1", "2"]

    run_provider_setup(home, make_prompt(console, *answers), console, root=tmp_path,
                       read_secret=no_secret)

    cfg = load_config(tmp_path, home_dir=home)
    assert cfg["agents"]["spec"]["effort"] is None
    assert cfg["agents"]["coder"]["effort"] == "medium"
    assert "effort" not in cfg["agents"]["chatty"]
    assert "effort" not in user_config(home)["helpers"]["reflect"]


def test_anthropic_offers_its_claude_ids_tier_first(tmp_path):
    seen = []

    class Spy:
        def choose(self, question, options):
            seen.append((question, [key for key, _ in options]))
            return options[0][0]

    models = screens._model_options("anthropic", "light", None)
    assert [key for key, _ in models][0] == "claude-haiku-5-5"
    assert [label for _, label in models] == ["Claude Haiku 5.5", "Claude Opus 5.5", "Claude Sonnet 5.5"]
    assert screens._ask_model(Spy(), "middle", models) == "claude-haiku-5-5"
    assert seen[0][0] == "Model for TESTER and CODER"
    assert seen[0][1][-1] == screens._OTHER
    openrouter = screens._model_options("openrouter", "strong", None)
    assert openrouter[0][0] == "anthropic/claude-opus-5.5"


def test_anthropic_saves_the_first_credential_name(tmp_path):
    home = tmp_path / "home"
    console = make_console()
    secret = fake_secret("sk-ant")
    answers = ["anthropic"] + ["1"] * 6 + ["1"]

    run_provider_setup(home, make_prompt(console, *answers), console, root=tmp_path,
                       read_secret=secret)

    assert credentials_path(home).read_text() == "ANTHROPIC_API_KEY=sk-ant\n"
    assert "ANTHROPIC_API_KEY" not in (home / ".k0ntrol" / "config.yaml").read_text().split("\n", 2)[2]


def test_an_empty_key_saves_nothing(tmp_path):
    home = tmp_path / "home"
    console = make_console()
    answers = ["openrouter"] + ["1"] * 6 + ["1"]

    run_provider_setup(home, make_prompt(console, *answers), console, root=tmp_path,
                       read_secret=fake_secret("  "))

    assert not credentials_path(home).exists()
    assert "No key entered; nothing saved." in screen(console)


def test_openai_and_vertex_offer_labelled_placeholders():
    openai = screens._model_options("openai", "strong", None)
    assert openai == [("gpt-6.1-terra", "gpt-6.1-terra (placeholder)"),
                      ("gpt-6-luna", "gpt-6-luna (placeholder)")]
    vertex = screens._model_options("vertex", "light", None)
    assert [key for key, _ in vertex] == ["gemini-3-pro", "gemini-3-flash"]
    assert screens._model_options("openai_compatible", "light", None)
    assert screens._model_options("unknown", "light", None) == []


def test_openai_runs_end_to_end_with_another_model(tmp_path):
    home = tmp_path / "home"
    console = make_console()
    answers = ["openai", "3", "gpt-x", "1", "1", "1", "1", "1", "2"]

    run_provider_setup(home, make_prompt(console, *answers), console, root=tmp_path,
                       read_secret=no_secret)

    assert user_config(home)["agents"]["spec"]["model"] == "gpt-x"


def test_a_symlinked_user_config_is_replaced_by_a_regular_file(tmp_path):
    home = tmp_path / "home"
    (home / ".k0ntrol").mkdir(parents=True)
    target = tmp_path / "dotfiles" / "k0.yaml"
    target.parent.mkdir()
    target.write_text("threshold_lines: 500\n")
    link = home / ".k0ntrol" / "config.yaml"
    link.symlink_to(target)
    console = make_console()

    run_provider_setup(home, make_prompt(console, "ollama", *["1"] * 6), console,
                       root=tmp_path, list_models=lister("m"))

    assert not link.is_symlink()
    assert target.read_text() == "threshold_lines: 500\n"
    data = yaml.safe_load(link.read_text())
    assert data["threshold_lines"] == 500
    assert data["agents"]["spec"]["model"] == "m"


def test_the_rewrite_keeps_other_settings_and_drops_old_providers(tmp_path):
    home = tmp_path / "home"
    (home / ".k0ntrol").mkdir(parents=True)
    (home / ".k0ntrol" / "config.yaml").write_text(
        "max_attempts: 3\nproviders:\n  ollama:\n    base_url: http://old:1\n"
        "agents:\n  spec: {provider: ollama, model: old}\n")
    console = make_console()

    run_provider_setup(home, make_prompt(console, "ollama", *["1"] * 6), console,
                       root=tmp_path, list_models=lister("m"))

    data = user_config(home)
    assert data["max_attempts"] == 3
    assert "providers" not in data
    assert data["agents"]["spec"]["model"] == "m"
    assert (home / ".k0ntrol" / "config.yaml").read_text().startswith("# k0 user config")


@pytest.mark.parametrize("content", ["{{ not yaml", "- a\n- b\n"])
def test_an_unreadable_user_config_is_simply_replaced(tmp_path, content):
    home = tmp_path / "home"
    (home / ".k0ntrol").mkdir(parents=True)
    (home / ".k0ntrol" / "config.yaml").write_text(content)
    console = make_console()

    run_provider_setup(home, make_prompt(console, "ollama", *["1"] * 6), console,
                       root=tmp_path, list_models=lister("m"))

    assert set(user_config(home)) == {"agents", "helpers"}


def test_no_secret_is_ever_written_to_the_config(tmp_path):
    home = tmp_path / "home"
    console = make_console()
    answers = ["bedrock"] + ["1"] * 6 + ["1", "save"]

    path = run_provider_setup(home, make_prompt(console, *answers), console, root=tmp_path,
                              read_secret=fake_secret("topsecret"))

    assert "topsecret" not in path.read_text()
    assert load_config(tmp_path, home_dir=home)["providers"]["bedrock"]["params"] == {
        "region_name": "us-east-1"}
