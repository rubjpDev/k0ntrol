import yaml

from k0ntrol.cli import main
from k0ntrol.config import load_config
from k0ntrol.init_flow import run_init
from k0ntrol.memory.layout import init_project


MEM = "# Mem\n\nProject facts. One fact per bullet.\n"
RULES = "# Rules\n\nGlobal rules for every agent.\n"
ARCHITECTURE = "# Architecture\n\nSystem shape. Layers, boundaries, key modules.\n"
STRATEGY = "# Strategy\n\nHow we change this codebase.\n"
CODE_RULES = "# Code rules\n\nLocal coding constraints.\n"
PROJECT_CONFIG = (
    "# Project overrides. An empty mapping inherits the package defaults.\n"
    "#\n"
    "# Uncomment to enable validator gates after a green test_cmd:\n"
    "# gates:\n"
    "#   - name: complexity\n"
    "#     cmd: complexipy src --max-complexity-allowed 6\n"
    "#   - name: coverage\n"
    "#     cmd: pytest --cov=src --cov-fail-under=90\n"
    "{}\n"
)
MEMORY_GITIGNORE = "sessions/\nevents.jsonl\n*.sqlite\n"


def test_init_creates_the_project_memory_layout(tmp_path):
    lines = init_project(tmp_path)
    root = tmp_path / ".k0-mem"
    topo = root / "Project_topology" / tmp_path.name
    assert (root / "Project_context").is_dir()
    assert (root / "Project_topology").is_dir()
    assert (topo / "architecture.md").read_text(encoding="utf-8") == ARCHITECTURE
    assert (topo / "strategy.md").read_text(encoding="utf-8") == STRATEGY
    assert (topo / "code_rules.md").read_text(encoding="utf-8") == CODE_RULES
    assert (root / ".gitignore").read_text(encoding="utf-8") == MEMORY_GITIGNORE
    assert (root / "General_mem" / "Mem.md").read_text(encoding="utf-8") == MEM
    assert (root / "General_mem" / "Rules.md").read_text(encoding="utf-8") == RULES
    assert (root / "config.yaml").read_text(encoding="utf-8") == PROJECT_CONFIG
    repo = tmp_path.name
    assert lines == [
        "created .k0-mem/Project_context",
        "created .k0-mem/Project_topology",
        f"created .k0-mem/Project_topology/{repo}/architecture.md",
        f"created .k0-mem/Project_topology/{repo}/strategy.md",
        f"created .k0-mem/Project_topology/{repo}/code_rules.md",
        "created .k0-mem/.gitignore",
        "created .k0-mem/General_mem/Mem.md",
        "created .k0-mem/General_mem/Rules.md",
        "created .k0-mem/config.yaml",
    ]
    assert not (root / "modes").exists()
    assert load_config(tmp_path, tmp_path / "home")["gates"] == []


def test_init_gate_examples_are_commented(tmp_path):
    init_project(tmp_path)
    text = (tmp_path / ".k0-mem" / "config.yaml").read_text(encoding="utf-8")
    assert "complexipy src --max-complexity-allowed 6" in text
    assert "pytest --cov=src --cov-fail-under=90" in text
    for line in text.splitlines():
        if "complexipy" in line or "cov-fail-under" in line or line.strip() == "gates:":
            assert line.lstrip().startswith("#")


def test_cli_init_prints_the_report(tmp_path, capsys, monkeypatch):
    answers = iter(["github", "anthropic"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path / "home"))
    assert main(["init", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "created .k0-mem/General_mem/Mem.md" in out
    assert (tmp_path / ".k0-mem" / "General_mem" / "Mem.md").is_file()


def test_cli_init_dot_writes_topology_under_resolved_folder_name(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    answers = iter(["github", "anthropic"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path / "home"))
    assert main(["init", "."]) == 0
    repo = tmp_path.name
    topo = tmp_path / ".k0-mem" / "Project_topology" / repo
    assert (topo / "architecture.md").read_text(encoding="utf-8") == ARCHITECTURE
    assert (topo / "strategy.md").read_text(encoding="utf-8") == STRATEGY
    assert (topo / "code_rules.md").read_text(encoding="utf-8") == CODE_RULES
    out = capsys.readouterr().out
    assert f"created .k0-mem/Project_topology/{repo}/architecture.md" in out
    assert not (tmp_path / ".k0-mem" / "Project_topology" / "architecture.md").exists()


def test_second_init_keeps_existing_files(tmp_path, capsys, monkeypatch):
    answers = iter(["github", "anthropic", "github", "anthropic"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path / "home"))
    assert main(["init", str(tmp_path)]) == 0
    mem = tmp_path / ".k0-mem" / "General_mem" / "Mem.md"
    mem.write_text("KEEP\n", encoding="utf-8")
    note = tmp_path / ".k0-mem" / "Project_context" / "note.md"
    note.write_text("note\n", encoding="utf-8")
    arch = (
        tmp_path
        / ".k0-mem"
        / "Project_topology"
        / tmp_path.name
        / "architecture.md"
    )
    arch.write_text("KEEP ARCH\n", encoding="utf-8")
    assert main(["init", str(tmp_path)]) == 0
    assert mem.read_text(encoding="utf-8") == "KEEP\n"
    assert note.read_text(encoding="utf-8") == "note\n"
    assert arch.read_text(encoding="utf-8") == "KEEP ARCH\n"
    assert (tmp_path / ".k0-mem" / ".gitignore").read_text(encoding="utf-8") == (
        MEMORY_GITIGNORE
    )
    out = capsys.readouterr().out
    assert "exists .k0-mem/General_mem/Mem.md" in out
    assert "exists .k0-mem/.gitignore" in out
    assert "exists .k0-mem/Project_context" in out
    assert f"exists .k0-mem/Project_topology/{tmp_path.name}/architecture.md" in out
    assert "created .k0-mem/General_mem/Mem.md" not in out.split("exists .k0-mem/General_mem/Mem.md")[-1]


# --- tracker and provider questions (#50) ------------------------------------


def refuse_input(prompt=""):
    raise AssertionError(f"unexpected prompt: {prompt}")


def scripted(*replies):
    """A read_line seam answering from the script, then refusing."""
    it = iter(replies)
    prompts = []

    def read(prompt=""):
        prompts.append(prompt)
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(f"unscripted prompt: {prompt}")

    read.prompts = prompts
    return read


def init_lines(tmp_path, **kw):
    """Run init with scripted seams; return (exit code, printed lines)."""
    lines = []
    kw.setdefault("read_line", refuse_input)
    kw.setdefault("home", tmp_path / "home")
    code = run_init(tmp_path, write=lines.append, **kw)
    return code, lines


def read_user_config(home):
    path = home / ".k0ntrol" / "config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_init_tracker_flag_writes_the_project_config(tmp_path):
    code, lines = init_lines(
        tmp_path, tracker="github", provider="anthropic"
    )
    assert code == 0
    text = (tmp_path / ".k0-mem" / "config.yaml").read_text(encoding="utf-8")
    assert "tracker: github" in text
    assert "complexipy" in text  # the commented gate examples survive
    assert load_config(tmp_path, tmp_path / "home")["tracker"] == "github"


def test_init_prompts_for_the_tracker_when_no_flag(tmp_path):
    code, lines = init_lines(
        tmp_path, provider="anthropic", read_line=scripted("github")
    )
    assert code == 0
    out = "\n".join(lines)
    assert "github" in out
    assert "gitlab" in out and "linear" in out and "jira" in out
    assert "not in 0.1.0" in out
    assert load_config(tmp_path, tmp_path / "home")["tracker"] == "github"


def test_init_rejects_a_tracker_that_is_not_in_0_1_0(tmp_path):
    code, lines = init_lines(tmp_path, tracker="gitlab", provider="anthropic")
    assert code == 1
    assert "not in 0.1.0" in "\n".join(lines)
    assert not (tmp_path / ".k0-mem").exists()  # failed before writing layout


def test_anthropic_provider_flag_writes_claude_defaults_to_user_config(tmp_path):
    home = tmp_path / "home"
    code, lines = init_lines(tmp_path, tracker="github", provider="anthropic")
    assert code == 0
    cfg = read_user_config(home)
    assert cfg["agents"]["spec"] == {
        "provider": "anthropic",
        "model": "claude-opus-5-5",
        "effort": "high",
    }
    assert cfg["agents"]["coder"] == {
        "provider": "anthropic",
        "model": "claude-sonnet-5-5",
        "effort": "medium",
    }
    assert cfg["agents"]["reviewer"]["model"] == "claude-opus-5-5"
    for name in ("chatty",):
        assert "effort" not in cfg["agents"][name]
    for name in ("bulk_read", "reflect"):
        entry = cfg["helpers"][name]
        assert entry == {"provider": "anthropic", "model": "claude-haiku-5-5"}
    # model_calls stay in the package defaults
    assert all("model_calls" not in e for e in cfg["agents"].values())


def test_openrouter_and_bedrock_get_claude_defaults_in_their_model_ids(tmp_path):
    for provider, prefix in (
        ("openrouter", "anthropic/claude-"),
        ("bedrock", "us.anthropic.claude-"),
    ):
        home = tmp_path / provider / "home"
        code, _ = init_lines(
            tmp_path / provider, tracker="github", provider=provider, home=home
        )
        assert code == 0
        cfg = read_user_config(home)
        assert cfg["agents"]["spec"]["model"].startswith(prefix)
        assert cfg["agents"]["spec"]["effort"] == "high"
        assert cfg["helpers"]["reflect"]["provider"] == provider


def test_openai_init_asks_one_model_per_tier(tmp_path):
    answers = scripted("gpt-x-strong", "gpt-x-mid", "gpt-x-light")
    code, lines = init_lines(
        tmp_path, tracker="github", provider="openai", read_line=answers
    )
    assert code == 0
    prompts = " ".join(answers.prompts)
    assert "strong" in prompts and "middle" in prompts and "light" in prompts
    cfg = read_user_config(tmp_path / "home")
    for name in ("spec", "reviewer"):
        assert cfg["agents"][name]["model"] == "gpt-x-strong"
    for name in ("tester", "coder"):
        assert cfg["agents"][name]["model"] == "gpt-x-mid"
    assert cfg["agents"]["chatty"]["model"] == "gpt-x-light"
    assert cfg["helpers"]["bulk_read"]["model"] == "gpt-x-light"
    assert cfg["helpers"]["reflect"]["model"] == "gpt-x-light"


def test_ollama_init_offers_only_installed_tool_capable_models(tmp_path):
    def fetch(url, payload=None):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3:8b"}, {"name": "llama3"},
                               {"name": "mistral"}]}
        if url.endswith("/api/show"):
            caps = {"qwen3:8b": ["completion", "tools"],
                    "mistral": ["tools"]}.get(payload["model"], ["completion"])
            return {"capabilities": caps}
        raise AssertionError(url)

    answers = scripted("1", "mistral", "1")
    code, lines = init_lines(
        tmp_path, tracker="github", provider="ollama",
        read_line=answers, fetch=fetch,
    )
    assert code == 0
    out = "\n".join(lines)
    assert "qwen3:8b" in out and "mistral" in out
    assert "llama3" not in out  # cannot call tools
    cfg = read_user_config(tmp_path / "home")
    assert cfg["agents"]["spec"]["model"] == "qwen3:8b"
    assert cfg["agents"]["coder"]["model"] == "mistral"
    assert cfg["agents"]["coder"]["provider"] == "ollama"


def test_ollama_init_fails_when_the_server_cannot_be_reached(tmp_path):
    def fetch(url, payload=None):
        raise ConnectionError("refused")

    code, lines = init_lines(
        tmp_path, tracker="github", provider="ollama", fetch=fetch
    )
    assert code == 1
    assert "ollama" in "\n".join(lines)
    assert not (tmp_path / "home" / ".k0ntrol" / "config.yaml").exists()


def test_lmstudio_init_offers_only_tool_capable_models(tmp_path):
    def fetch(url, payload=None):
        assert url.endswith("/models")
        return {"data": [
            {"id": "qwen3-8b", "capabilities": ["tool_use"]},
            {"id": "embed-x", "capabilities": []},
            {"id": "plain-y"},
        ]}

    answers = scripted("1", "1", "1")
    code, lines = init_lines(
        tmp_path, tracker="github", provider="lmstudio",
        read_line=answers, fetch=fetch,
    )
    assert code == 0
    out = "\n".join(lines)
    assert "qwen3-8b" in out
    assert "embed-x" not in out and "plain-y" not in out
    cfg = read_user_config(tmp_path / "home")
    assert cfg["agents"]["spec"]["model"] == "qwen3-8b"


def test_provider_already_in_effect_is_reported_and_kept(tmp_path):
    home = tmp_path / "home"
    user_dir = home / ".k0ntrol"
    user_dir.mkdir(parents=True)
    original = (
        "agents:\n  coder: {provider: ollama, model: qwen3:8b}\n"
        "providers:\n  ollama: {base_url: 'http://gpu-box:11434'}\n"
    )
    path = user_dir / "config.yaml"
    path.write_text(original, encoding="utf-8")

    code, lines = init_lines(
        tmp_path, tracker="github", provider="anthropic"
    )
    assert code == 0
    out = "\n".join(lines)
    assert "ollama" in out and "already" in out
    assert path.read_text(encoding="utf-8") == original


def test_provider_question_runs_when_no_flag(tmp_path):
    answers = scripted("2")  # sorted registry: 1. anthropic, 2. bedrock, ...
    code, lines = init_lines(tmp_path, tracker="github", read_line=answers)
    assert code == 0
    out = "\n".join(lines)
    for name in ("anthropic", "ollama", "openrouter", "vertex"):
        assert name in out
    cfg = read_user_config(tmp_path / "home")
    assert cfg["agents"]["coder"]["provider"] == "bedrock"


def test_unknown_provider_flag_is_an_error_listing_the_registry(tmp_path):
    code, lines = init_lines(tmp_path, tracker="github", provider="litellm")
    assert code == 1
    out = "\n".join(lines)
    assert "litellm" in out and "anthropic" in out
    assert not (tmp_path / ".k0-mem").exists()


def test_init_prints_extra_env_var_doctor_and_langfuse(tmp_path):
    code, lines = init_lines(tmp_path, tracker="github", provider="anthropic")
    assert code == 0
    out = "\n".join(lines)
    assert "k0ntrol[anthropic]" in out
    assert "ANTHROPIC_API_KEY" in out
    assert "run `k0 doctor`" in out
    assert "Langfuse" in out and "source code" in out


def test_ollama_init_prints_the_extra_and_no_required_env_var(tmp_path):
    def fetch(url, payload=None):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3:8b"}]}
        return {"capabilities": ["tools"]}

    answers = scripted("1", "1", "1")
    code, lines = init_lines(
        tmp_path, tracker="github", provider="ollama",
        read_line=answers, fetch=fetch,
    )
    assert code == 0
    out = "\n".join(lines)
    assert "k0ntrol[ollama]" in out
    assert "run `k0 doctor`" in out
    assert "API_KEY" not in out


def test_existing_tracker_is_kept_without_asking(tmp_path):
    code, _ = init_lines(tmp_path, tracker="github", provider="anthropic")
    assert code == 0
    # second init: no tracker question even without the flag
    code, lines = init_lines(tmp_path, provider="anthropic",
                             read_line=refuse_input,
                             home=tmp_path / "other-home")
    assert code == 0
    assert "already" in "\n".join(lines)
    assert load_config(tmp_path, tmp_path / "other-home")["tracker"] == "github"


def test_tracker_prompt_retries_a_non_0_1_0_answer(tmp_path):
    answers = scripted("jira", "github")
    code, lines = init_lines(
        tmp_path, provider="anthropic", read_line=answers
    )
    assert code == 0
    assert "not in 0.1.0" in "\n".join(lines)
    assert load_config(tmp_path, tmp_path / "home")["tracker"] == "github"


def test_init_eof_during_questions_fails_cleanly(tmp_path):
    def eof(prompt=""):
        raise EOFError

    code, lines = init_lines(tmp_path, read_line=eof)
    assert code == 1
    assert "--tracker" in "\n".join(lines)


def test_provider_write_preserves_other_user_config_keys(tmp_path):
    home = tmp_path / "home"
    user_dir = home / ".k0ntrol"
    user_dir.mkdir(parents=True)
    (user_dir / "config.yaml").write_text(
        "providers:\n  ollama: {params: {num_ctx: 131072}}\n",
        encoding="utf-8",
    )
    code, _ = init_lines(tmp_path, tracker="github", provider="anthropic")
    assert code == 0
    cfg = read_user_config(home)
    assert cfg["providers"]["ollama"]["params"]["num_ctx"] == 131072
    assert cfg["agents"]["coder"]["provider"] == "anthropic"


def test_init_reports_a_broken_project_config(tmp_path):
    mem = tmp_path / ".k0-mem"
    mem.mkdir()
    (mem / "config.yaml").write_text("- a\n- b\n", encoding="utf-8")
    code, lines = init_lines(tmp_path, tracker="github", provider="anthropic")
    assert code == 1
    assert "config error" in "\n".join(lines)


def test_init_reports_a_broken_user_config(tmp_path):
    user_dir = tmp_path / "home" / ".k0ntrol"
    user_dir.mkdir(parents=True)
    (user_dir / "config.yaml").write_text("agents: [", encoding="utf-8")
    code, lines = init_lines(tmp_path, tracker="github", provider="anthropic")
    assert code == 1
    assert "config error" in "\n".join(lines)


def test_init_eof_during_the_provider_question_fails(tmp_path):
    answers = scripted("github")

    def eof(prompt=""):
        raise EOFError

    calls = {"n": 0}

    def read(prompt=""):
        calls["n"] += 1
        return answers(prompt) if calls["n"] == 1 else eof(prompt)

    code, lines = init_lines(tmp_path, read_line=read)
    assert code == 1
    assert "--provider" in "\n".join(lines)


def test_provider_prompt_retries_an_unknown_name(tmp_path):
    answers = scripted("not-a-provider", "anthropic")
    code, lines = init_lines(tmp_path, tracker="github", read_line=answers)
    assert code == 0
    assert "unknown provider" in "\n".join(lines)
    cfg = read_user_config(tmp_path / "home")
    assert cfg["agents"]["coder"]["provider"] == "anthropic"


def test_provider_prompt_empty_answer_picks_the_first(tmp_path):
    answers = scripted("")
    code, _ = init_lines(tmp_path, tracker="github", read_line=answers)
    assert code == 0
    cfg = read_user_config(tmp_path / "home")
    assert cfg["agents"]["coder"]["provider"] == "anthropic"


def test_ollama_init_errors_when_no_tool_capable_models(tmp_path):
    def fetch(url, payload=None):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "llama3"}]}
        return {"capabilities": ["completion"]}

    code, lines = init_lines(
        tmp_path, tracker="github", provider="ollama", fetch=fetch
    )
    assert code == 1
    assert "no tool-capable" in "\n".join(lines)


def test_model_menu_retries_an_unlisted_answer(tmp_path):
    def fetch(url, payload=None):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3:8b"}]}
        return {"capabilities": ["tools"]}

    answers = scripted("nope", "9", "1", "1", "1")
    code, lines = init_lines(
        tmp_path, tracker="github", provider="ollama",
        read_line=answers, fetch=fetch,
    )
    assert code == 0
    assert "pick a listed model" in "\n".join(lines)


def test_text_model_question_requires_a_name(tmp_path):
    answers = scripted("", "gpt-x", "gpt-y", "gpt-z")
    code, lines = init_lines(
        tmp_path, tracker="github", provider="vertex", read_line=answers
    )
    assert code == 0
    assert "a model name is required" in "\n".join(lines)
    cfg = read_user_config(tmp_path / "home")
    assert cfg["agents"]["spec"]["model"] == "gpt-x"


def test_init_eof_during_model_questions_fails(tmp_path):
    def eof(prompt=""):
        raise EOFError

    code, lines = init_lines(
        tmp_path, tracker="github", provider="openai", read_line=eof
    )
    assert code == 1
    assert "--provider" in "\n".join(lines)


def test_an_empty_user_config_file_counts_as_unset(tmp_path):
    user_dir = tmp_path / "home" / ".k0ntrol"
    user_dir.mkdir(parents=True)
    (user_dir / "config.yaml").write_text("", encoding="utf-8")
    code, _ = init_lines(tmp_path, tracker="github", provider="anthropic")
    assert code == 0
    cfg = read_user_config(tmp_path / "home")
    assert cfg["agents"]["coder"]["provider"] == "anthropic"


def test_in_effect_provider_outside_the_registry_is_reported(tmp_path):
    user_dir = tmp_path / "home" / ".k0ntrol"
    user_dir.mkdir(parents=True)
    (user_dir / "config.yaml").write_text(
        "agents:\n  coder: {provider: litellm, model: m}\n", encoding="utf-8"
    )
    code, lines = init_lines(tmp_path, tracker="github", provider="anthropic")
    assert code == 0
    assert "litellm" in "\n".join(lines)


def test_cli_init_accepts_tracker_and_provider_flags(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: tmp_path / "home"))

    def refuse(prompt=""):
        raise AssertionError("prompted despite flags")

    monkeypatch.setattr("builtins.input", refuse)
    assert main(["init", str(tmp_path), "--tracker", "github",
                 "--provider", "anthropic"]) == 0
    out = capsys.readouterr().out
    assert "tracker 'github' written" in out
    assert "ANTHROPIC_API_KEY" in out
    cfg = yaml.safe_load(
        (tmp_path / "home" / ".k0ntrol" / "config.yaml").read_text()
    )
    assert cfg["agents"]["reviewer"]["model"] == "claude-opus-5-5"
