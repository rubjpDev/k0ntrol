"""The provider registry maps one agents:/helpers: entry to a chat model."""

import pytest

from fakes import cfg_with, recording_init


@pytest.fixture
def calls():
    return []


@pytest.mark.parametrize(
    "provider, model_provider",
    [
        ("ollama", "ollama"),
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("openrouter", "openrouter"),
        ("bedrock", "bedrock_converse"),
        ("vertex", "google_genai"),
    ],
)
def test_each_provider_builds_through_init_chat_model(calls, provider, model_provider):
    from k0ntrol.providers import build_chat_model

    cfg = cfg_with(agent_entries={"coder": {"provider": provider, "model": "m-1"}})
    model = build_chat_model(cfg, cfg["agents"]["coder"], init=recording_init(calls))

    assert calls[0]["model"] == "m-1"
    assert calls[0]["model_provider"] == model_provider
    assert model.model_provider == model_provider


def test_lmstudio_goes_through_openai_with_localhost_and_stream_usage(calls):
    from k0ntrol.providers import build_chat_model

    cfg = cfg_with(agent_entries={"chatty": {"provider": "lmstudio", "model": "qwen"}})
    build_chat_model(cfg, cfg["agents"]["chatty"], init=recording_init(calls))

    assert calls[0]["model_provider"] == "openai"
    assert calls[0]["kwargs"]["base_url"] == "http://localhost:1234/v1"
    assert calls[0]["kwargs"]["stream_usage"] is True
    assert calls[0]["kwargs"]["api_key"]  # a placeholder; LM Studio ignores it


def test_openai_compatible_is_the_openai_escape_hatch(calls):
    from k0ntrol.providers import build_chat_model

    cfg = cfg_with(
        agent_entries={"coder": {"provider": "openai_compatible", "model": "m"}},
        providers={"openai_compatible": {"base_url": "http://vllm:8000/v1"}},
    )
    build_chat_model(cfg, cfg["agents"]["coder"], init=recording_init(calls))

    assert calls[0]["model_provider"] == "openai"
    assert calls[0]["kwargs"]["base_url"] == "http://vllm:8000/v1"


def test_ollama_defaults_num_ctx_to_64k(calls):
    from k0ntrol.providers import build_chat_model

    cfg = cfg_with(agent_entries={"coder": {"provider": "ollama", "model": "qwen3:8b"}})
    build_chat_model(cfg, cfg["agents"]["coder"], init=recording_init(calls))

    assert calls[0]["kwargs"]["num_ctx"] == 65536


def test_vertex_builds_google_genai_with_vertexai(calls):
    from k0ntrol.providers import build_chat_model

    cfg = cfg_with(agent_entries={"coder": {"provider": "vertex", "model": "gemini-3"}})
    build_chat_model(cfg, cfg["agents"]["coder"], init=recording_init(calls))

    assert calls[0]["model_provider"] == "google_genai"
    assert calls[0]["kwargs"]["vertexai"] is True


def test_no_temperature_and_no_forced_tool_choice(calls):
    from k0ntrol.providers import build_chat_model

    cfg = cfg_with(agent_entries={"coder": {"provider": "anthropic", "model": "claude-x"}})
    build_chat_model(cfg, cfg["agents"]["coder"], init=recording_init(calls))

    for call in calls:
        assert "temperature" not in call["kwargs"]
        assert "tool_choice" not in call["kwargs"]


def test_entry_params_override_provider_params(calls):
    from k0ntrol.providers import build_chat_model

    cfg = cfg_with(
        agent_entries={
            "coder": {"provider": "ollama", "model": "m", "params": {"num_ctx": 128000}}
        },
        providers={"ollama": {"params": {"num_ctx": 96000, "keep_alive": "1h"}}},
    )
    build_chat_model(cfg, cfg["agents"]["coder"], init=recording_init(calls))

    assert calls[0]["kwargs"]["num_ctx"] == 128000
    assert calls[0]["kwargs"]["keep_alive"] == "1h"


def test_provider_base_url_overrides_the_registry_default(calls):
    from k0ntrol.providers import build_chat_model

    cfg = cfg_with(
        agent_entries={"coder": {"provider": "lmstudio", "model": "m"}},
        providers={"lmstudio": {"base_url": "http://gpu-box:1234/v1"}},
    )
    build_chat_model(cfg, cfg["agents"]["coder"], init=recording_init(calls))

    assert calls[0]["kwargs"]["base_url"] == "http://gpu-box:1234/v1"


def test_lmstudio_env_bearer_token_becomes_the_api_key(calls):
    from k0ntrol.providers import build_chat_model

    cfg = cfg_with(agent_entries={"chatty": {"provider": "lmstudio", "model": "m"}})
    build_chat_model(
        cfg,
        cfg["agents"]["chatty"],
        init=recording_init(calls),
        environ={"LM_STUDIO_API_KEY": "tok-123"},
    )

    assert calls[0]["kwargs"]["api_key"] == "tok-123"


def test_unknown_provider_raises_naming_the_bad_value():
    from k0ntrol.providers import ProviderError, build_chat_model

    cfg = cfg_with(agent_entries={"coder": {"provider": "litellm", "model": "m"}})
    with pytest.raises(ProviderError, match="unknown provider 'litellm'"):
        build_chat_model(cfg, cfg["agents"]["coder"], init=recording_init([]))


@pytest.mark.parametrize(
    "provider, expected",
    [
        ("anthropic", {"effort": "high"}),
        ("openai", {"reasoning_effort": "high"}),
        ("lmstudio", {"reasoning_effort": "high"}),
        ("openai_compatible", {"reasoning_effort": "high"}),
        ("openrouter", {"reasoning": {"effort": "high"}}),
        ("vertex", {"thinking_level": "high"}),
        ("bedrock", {"reasoning_effort": "high"}),
        ("ollama", {"reasoning": "high"}),
    ],
)
def test_effort_uses_the_providers_setting_when_the_profile_lists_it(
    calls, provider, expected
):
    from k0ntrol.providers import build_chat_model

    profile = {"reasoning_effort_levels": ["low", "medium", "high"]}
    cfg = cfg_with(
        agent_entries={"coder": {"provider": provider, "model": "m", "effort": "high"}}
    )
    build_chat_model(
        cfg,
        cfg["agents"]["coder"],
        init=recording_init(calls, profiles={"any": profile}),
    )

    final_kwargs = calls[-1]["kwargs"]
    for key, value in expected.items():
        assert final_kwargs[key] == value


def test_effort_is_dropped_when_the_profile_lacks_the_level(calls):
    from k0ntrol.providers import build_chat_model, dropped_effort

    profile = {"reasoning_effort_levels": ["low"]}
    cfg = cfg_with(
        agent_entries={
            "coder": {"provider": "anthropic", "model": "m", "effort": "high"}
        }
    )
    model = build_chat_model(
        cfg,
        cfg["agents"]["coder"],
        init=recording_init(calls, profiles={"any": profile}),
    )

    assert "effort" not in calls[-1]["kwargs"]
    assert dropped_effort(cfg["agents"]["coder"], model.profile) == "high"


def test_effort_is_dropped_when_the_profile_is_none(calls):
    from k0ntrol.providers import build_chat_model, dropped_effort

    cfg = cfg_with(
        agent_entries={"coder": {"provider": "ollama", "model": "m", "effort": "low"}}
    )
    model = build_chat_model(
        cfg, cfg["agents"]["coder"], init=recording_init(calls)
    )

    assert "reasoning" not in calls[-1]["kwargs"]
    assert dropped_effort(cfg["agents"]["coder"], model.profile) == "low"


def test_no_effort_configured_means_nothing_is_dropped():
    from k0ntrol.providers import dropped_effort

    entry = {"provider": "anthropic", "model": "m"}
    profile = {"reasoning_effort_levels": ["low", "medium", "high"]}
    assert dropped_effort(entry, profile) is None


# --- tool_calling_models: installed models that can call tools (#50) ---------


def test_tool_calling_models_is_empty_for_hosted_providers():
    from k0ntrol.providers import tool_calling_models

    assert tool_calling_models("openai") == []


def test_tool_calling_models_ollama_filters_on_show_capabilities():
    from k0ntrol.providers import tool_calling_models

    def fetch(url, payload=None):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3:8b"}, {"name": "llama3"}]}
        if url.endswith("/api/show"):
            caps = {"qwen3:8b": ["completion", "tools"]}.get(
                payload["model"], ["completion"]
            )
            return {"capabilities": caps}
        raise AssertionError(url)

    models = tool_calling_models("ollama", fetch=fetch)
    assert models == ["qwen3:8b"]


def test_tool_calling_models_ollama_skips_a_model_when_show_fails():
    from k0ntrol.providers import tool_calling_models

    def fetch(url, payload=None):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "gone"}, {"name": "qwen3:8b"}]}
        if url.endswith("/api/show") and payload["model"] == "gone":
            raise ConnectionError("model vanished")
        return {"capabilities": ["tools"]}

    models = tool_calling_models("ollama", fetch=fetch)
    assert models == ["qwen3:8b"]


def test_tool_calling_models_lmstudio_filters_on_tool_use():
    from k0ntrol.providers import tool_calling_models

    def fetch(url, payload=None):
        return {"data": [
            {"id": "qwen3-8b", "capabilities": ["tool_use"]},
            {"id": "embed-x", "capabilities": ["embeddings"]},
            {"id": "plain-y"},
        ]}

    models = tool_calling_models("lmstudio", fetch=fetch)
    assert models == ["qwen3-8b"]


def test_tool_calling_models_uses_the_project_base_url():
    from k0ntrol.providers import tool_calling_models

    urls = []

    def fetch(url, payload=None):
        urls.append(url)
        return {"data": [{"id": "m", "capabilities": ["tool_use"]}]}

    models = tool_calling_models(
        "lmstudio", "http://gpu-box:1234/v1", fetch=fetch
    )
    assert models == ["m"]
    assert urls == ["http://gpu-box:1234/v1/models"]
