"""`k0 doctor`: free checks (no tokens spent) plus the --live round trip."""

from fakes import FakeChatModel, cfg_with, recording_init


def ok_run(argv, root):
    """Every shell check succeeds."""
    return (0, "")


def ok_fetch(url, payload=None):
    """Every local server answers; models exist and can call tools."""
    if url.endswith("/api/show"):
        return {"capabilities": ["completion", "tools"]}
    if url.endswith("/models"):
        return {"data": []}
    return {"version": "0.0.0"}


def ok_importable(module):
    return True


def free_checks(cfg, **overrides):
    from k0ntrol.providers import free_checks as real

    seams = {
        "init": recording_init([], profiles={"any": {"tool_calling": True}}),
        "importable": ok_importable,
        "fetch": ok_fetch,
        "run": ok_run,
        "environ": {"ANTHROPIC_API_KEY": "x", "OPENAI_API_KEY": "x",
                    "OPENROUTER_API_KEY": "x"},
        "root": None,
    }
    seams.update(overrides)
    return real(cfg, **seams)


def errors(checks):
    return [c for c in checks if c.status == "error"]


def warnings(checks):
    return [c for c in checks if c.status == "warning"]


def anthropic_cfg(**overrides):
    return cfg_with(
        agent_entries={"coder": {"provider": "anthropic", "model": "claude-x"}},
        helpers={"reflect": {"provider": "anthropic", "model": "claude-x"}},
        providers=overrides.pop("providers", None),
    )


def test_all_green_config_reports_only_ok():
    checks = free_checks(anthropic_cfg())
    assert errors(checks) == []
    assert warnings(checks) == []
    assert any(c.name == "config" and c.status == "ok" for c in checks)


def test_unknown_provider_is_an_error_naming_it():
    cfg = cfg_with(agent_entries={"coder": {"provider": "litellm", "model": "m"}})
    checks = free_checks(cfg)
    assert any(
        c.status == "error" and "litellm" in c.detail and "coder" in c.name
        for c in checks
    )


def test_unknown_provider_in_the_providers_map_is_a_warning():
    cfg = cfg_with(
        agent_entries={"coder": {"provider": "anthropic", "model": "m"}},
        providers={"ollmaa": {"base_url": "http://x"}},
    )
    checks = free_checks(cfg)
    assert any(
        c.status == "warning" and "ollmaa" in c.detail for c in checks
    )


def test_missing_extra_is_an_error_naming_the_extra():
    checks = free_checks(anthropic_cfg(), importable=lambda module: False)
    assert any(
        c.status == "error" and "k0ntrol[anthropic]" in c.detail for c in checks
    )


def test_missing_required_credential_is_an_error_naming_only_the_var():
    checks = free_checks(anthropic_cfg(), environ={})
    cred = [c for c in errors(checks) if "ANTHROPIC_API_KEY" in c.detail]
    assert cred, "expected a missing-credential error naming the env var"


def test_cloud_chain_without_env_hints_is_a_warning_not_an_error():
    cfg = cfg_with(agent_entries={"coder": {"provider": "bedrock", "model": "m"}})
    checks = free_checks(cfg, environ={})
    assert any(
        c.status == "warning" and "bedrock" in c.name for c in checks
    )
    assert not any(
        "credential" in c.detail and c.status == "error" for c in errors(checks)
    )


def test_profile_without_tool_calling_is_an_error():
    init = recording_init([], profiles={"any": {"tool_calling": False}})
    checks = free_checks(anthropic_cfg(), init=init)
    assert any(
        c.status == "error" and "tool" in c.detail for c in checks
    )


def test_profile_unknown_is_a_warning():
    init = recording_init([], profiles={"any": None})
    checks = free_checks(anthropic_cfg(), init=init)
    assert any(
        c.status == "warning" and "tool" in c.detail.lower() for c in checks
    )


def test_ollama_num_ctx_below_64k_is_a_warning():
    cfg = cfg_with(
        agent_entries={"coder": {"provider": "ollama", "model": "m"}},
        providers={"ollama": {"params": {"num_ctx": 8192}}},
    )
    checks = free_checks(cfg)
    assert any(
        c.status == "warning" and "num_ctx" in c.detail for c in checks
    )


def test_dropped_effort_is_a_warning():
    init = recording_init(
        [], profiles={"any": {"tool_calling": True, "reasoning_effort_levels": ["low"]}}
    )
    cfg = cfg_with(
        agent_entries={
            "coder": {"provider": "anthropic", "model": "m", "effort": "high"}
        }
    )
    checks = free_checks(cfg, init=init)
    assert any(
        c.status == "warning" and "effort" in c.detail and "high" in c.detail
        for c in checks
    )


def test_subset_checks_only_the_named_entries():
    cfg = cfg_with(
        agent_entries={
            "coder": {"provider": "anthropic", "model": "m"},
            "reviewer": {"provider": "openai", "model": "m"},
        },
    )
    checks = free_checks(cfg, names=["agents.coder"], environ={})
    detail = " ".join(c.detail for c in checks)
    assert "OPENAI_API_KEY" not in detail
    assert "ANTHROPIC_API_KEY" in detail


def test_names_from_both_rosters_select_across_kinds():
    cfg = cfg_with(
        agent_entries={"coder": {"provider": "anthropic", "model": "m"}},
        helpers={"reflect": {"provider": "openai", "model": "m"}},
    )
    checks = free_checks(cfg, names=["helpers.reflect"], environ={})
    detail = " ".join(c.detail for c in checks)
    assert "OPENAI_API_KEY" in detail
    assert "ANTHROPIC_API_KEY" not in detail


def ollama_cfg(**providers):
    return cfg_with(
        agent_entries={"coder": {"provider": "ollama", "model": "qwen3:8b"}},
        providers=providers or None,
    )


def test_gh_missing_from_path_is_an_error():
    def run(argv, root):
        return (None, "cannot run gh")

    checks = free_checks(anthropic_cfg(), run=run)
    assert any(c.name == "gh" and c.status == "error" for c in checks)


def test_gh_auth_failure_is_an_error():
    def run(argv, root):
        if argv[:2] == ["gh", "auth"]:
            return (4, "You are not logged into any GitHub hosts")
        return (0, "git@github.com:x/y.git\n")

    checks = free_checks(anthropic_cfg(), run=run)
    assert any(c.name == "gh" and c.status == "error" for c in checks)


def test_no_git_remote_is_an_error():
    def run(argv, root):
        if argv[0] == "git":
            return (2, "")
        return (0, "")

    checks = free_checks(anthropic_cfg(), run=run)
    assert any(c.name == "remote" and c.status == "error" for c in checks)


def test_remote_url_is_never_printed():
    def run(argv, root):
        if argv[0] == "git":
            return (0, "https://user:ghp_secret@github.com/x/y.git\n")
        return (0, "")

    checks = free_checks(anthropic_cfg(), run=run)
    assert all("ghp_secret" not in c.detail for c in checks)


def test_ollama_unreachable_is_an_error():
    def fetch(url, payload=None):
        raise ConnectionError("refused")

    checks = free_checks(ollama_cfg(), fetch=fetch)
    assert any(
        c.name == "provider ollama" and c.status == "error"
        and "not reachable" in c.detail for c in checks
    )


def test_ollama_model_not_pulled_is_an_error():
    import urllib.error

    def fetch(url, payload=None):
        if url.endswith("/api/show"):
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        return {"version": "0"}

    checks = free_checks(ollama_cfg(), fetch=fetch)
    assert any(
        c.status == "error" and "qwen3:8b" in c.detail and "not pulled" in c.detail
        for c in checks
    )


def test_ollama_model_without_tools_capability_is_an_error():
    def fetch(url, payload=None):
        if url.endswith("/api/show"):
            return {"capabilities": ["completion"]}
        return {"version": "0"}

    checks = free_checks(ollama_cfg(), fetch=fetch)
    assert any(
        c.status == "error" and "cannot call tools" in c.detail for c in checks
    )


def test_ollama_model_with_tools_capability_is_ok():
    checks = free_checks(ollama_cfg())
    assert any(
        c.name == "provider ollama" and c.status == "ok" and "tools" in c.detail
        for c in checks
    )
    assert not any(
        c.status == "warning" and "tool" in c.detail.lower() and "ollama" in c.name
        for c in checks
    )


def test_lmstudio_unreachable_is_an_error():
    cfg = cfg_with(agent_entries={"chatty": {"provider": "lmstudio", "model": "m"}})

    def fetch(url, payload=None):
        raise ConnectionError("refused")

    checks = free_checks(cfg, fetch=fetch)
    assert any(
        c.name == "provider lmstudio" and c.status == "error"
        and "not reachable" in c.detail for c in checks
    )


def test_lmstudio_model_not_loaded_is_an_error():
    cfg = cfg_with(agent_entries={"chatty": {"provider": "lmstudio", "model": "m1"}})

    def fetch(url, payload=None):
        return {"data": [{"id": "other-model"}]}

    checks = free_checks(cfg, fetch=fetch)
    assert any(
        c.name == "provider lmstudio" and c.status == "error"
        and "not loaded" in c.detail for c in checks
    )


def test_lmstudio_loaded_model_is_ok_but_tool_calling_warns():
    cfg = cfg_with(agent_entries={"chatty": {"provider": "lmstudio", "model": "m1"}})

    def fetch(url, payload=None):
        return {"data": [{"id": "m1"}]}

    checks = free_checks(cfg, fetch=fetch)
    assert any(
        c.name == "provider lmstudio" and c.status == "ok" for c in checks
    )
    assert any(c.status == "warning" and "--live" in c.detail for c in checks)


# --- --live: one tiny tool-call round trip per distinct (provider, model) ---


def live_checks(cfg, init, **overrides):
    from k0ntrol.providers import live_checks as real

    return real(cfg, init=init, **overrides)


def test_live_makes_one_call_per_distinct_provider_model():
    from fakes import FakeReply

    calls = []
    models = {}
    reply = FakeReply(tool_calls=[{"name": "ping"}])

    def init(model, *, model_provider=None, **kwargs):
        calls.append(model)
        models[model] = FakeChatModel(model, model_provider, kwargs, reply=reply)
        return models[model]

    cfg = cfg_with(
        agent_entries={
            "coder": {"provider": "anthropic", "model": "claude-x"},
            "reviewer": {"provider": "anthropic", "model": "claude-x"},
            "chatty": {"provider": "anthropic", "model": "claude-y"},
        },
    )
    checks = live_checks(cfg, init)

    assert calls == ["claude-x", "claude-y"]  # one build per distinct pair
    invokes = sum(len(m.invocations) for m in models.values())
    assert invokes == 2
    assert sum(c.status == "ok" for c in checks) == 2


def test_live_answer_without_a_tool_call_is_an_error():
    from fakes import FakeReply

    init = recording_init([], replies={"m": FakeReply(tool_calls=[])})
    cfg = cfg_with(agent_entries={"coder": {"provider": "anthropic", "model": "m"}})
    checks = live_checks(cfg, init)
    assert any(
        c.status == "error" and "tool call" in c.detail for c in checks
    )


def test_live_failure_is_an_error():
    def init(model, *, model_provider=None, **kwargs):
        raise RuntimeError("401 unauthorized")

    cfg = cfg_with(agent_entries={"coder": {"provider": "anthropic", "model": "m"}})
    checks = live_checks(cfg, init)
    assert any(
        c.status == "error" and "401" in c.detail for c in checks
    )


def test_live_binds_tools_without_forcing_tool_choice():
    from fakes import FakeReply

    calls = []
    reply = FakeReply(tool_calls=[{"name": "ping"}])
    init = recording_init(calls, replies={"m": reply})
    cfg = cfg_with(agent_entries={"coder": {"provider": "anthropic", "model": "m"}})
    live_checks(cfg, init)

    for call in calls:
        assert "tool_choice" not in call["kwargs"]


def test_live_respects_the_names_subset():
    from fakes import FakeReply

    calls = []
    reply = FakeReply(tool_calls=[{"name": "ping"}])
    init = recording_init(calls, replies={"m": reply})
    cfg = cfg_with(
        agent_entries={
            "coder": {"provider": "anthropic", "model": "m"},
            "reviewer": {"provider": "openai", "model": "n"},
        }
    )
    checks = live_checks(cfg, init, names=["agents.coder"])
    assert {c.name for c in checks} == {"agents.coder"}
    assert all(call["model"] == "m" for call in calls)


# --- the `k0 doctor` command ---


def test_doctor_exits_0_when_checks_pass(capsys, monkeypatch, tmp_path):
    from k0ntrol.cli import main
    from k0ntrol.providers import Check

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "k0ntrol.cli.free_checks",
        lambda cfg, **kw: [Check("config", "ok", "valid"),
                           Check("provider anthropic", "warning", "low num_ctx")],
    )
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "warning: provider anthropic: low num_ctx" in out
    assert "unsandboxed" in out
    assert "Langfuse" in out


def test_doctor_exits_1_on_an_error(capsys, monkeypatch, tmp_path):
    from k0ntrol.cli import main
    from k0ntrol.providers import Check

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "k0ntrol.cli.free_checks",
        lambda cfg, **kw: [Check("gh", "error", "gh is not on PATH")],
    )
    assert main(["doctor"]) == 1
    assert "error: gh: gh is not on PATH" in capsys.readouterr().out


def test_doctor_live_runs_the_round_trip(capsys, monkeypatch, tmp_path):
    from k0ntrol.cli import main
    from k0ntrol.providers import Check

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("k0ntrol.cli.free_checks", lambda cfg, **kw: [])
    monkeypatch.setattr(
        "k0ntrol.cli.live_checks",
        lambda cfg, **kw: [Check("agents.coder", "ok", "anthropic:m made a tool call")],
    )
    assert main(["doctor", "--live"]) == 0
    assert "made a tool call" in capsys.readouterr().out


def test_doctor_config_error_exits_1(capsys, monkeypatch, tmp_path):
    from k0ntrol.cli import main

    monkeypatch.chdir(tmp_path)
    mem = tmp_path / ".k0-mem"
    mem.mkdir()
    (mem / "config.yaml").write_text("backend: cursor\n")

    def refuse(**kw):
        raise AssertionError("checks ran on invalid config")

    monkeypatch.setattr("k0ntrol.cli.free_checks", refuse)
    assert main(["doctor"]) == 1
    assert "config error:" in capsys.readouterr().err
