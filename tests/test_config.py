import pytest

from simple_agent.config import Settings


def clear_env(monkeypatch):
    for name in [
        "LLM_PROVIDER",
        "MIMO_MODEL",
        "MIMO_API_BASE",
        "MIMO_API_KEY",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_API_BASE",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_THINKING",
        "CUSTOM_MODEL",
        "CUSTOM_API_BASE",
        "CUSTOM_API_KEY",
        "LLM_MODEL",
        "LLM_API_BASE",
        "OPENAI_API_KEY",
        "AGENT_MAX_STEPS",
        "AGENT_MAX_INPUT_CHARS",
        "AGENT_DEBUG",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_mimo_defaults(monkeypatch):
    clear_env(monkeypatch)

    settings = Settings(_env_file=None)
    config = settings.provider_config("mimo")

    assert settings.provider == "mimo"
    assert config.model == "mimo-v2.5-pro"
    assert config.api_base == "https://token-plan-cn.xiaomimimo.com/v1"
    assert config.api_key == ""


def test_deepseek_defaults_and_thinking(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    monkeypatch.setenv("DEEPSEEK_THINKING", "true")

    config = Settings(_env_file=None).provider_config("deepseek")

    assert config.model == "deepseek-v4-flash"
    assert config.api_base == "https://api.deepseek.com"
    assert config.api_key == "ds-key"
    assert config.extra_body == {"thinking": {"type": "enabled"}}


def test_custom_provider(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("CUSTOM_MODEL", "custom-model")
    monkeypatch.setenv("CUSTOM_API_BASE", "https://example.com/v1")
    monkeypatch.setenv("CUSTOM_API_KEY", "custom-key")

    config = Settings(_env_file=None).provider_config("custom")

    assert config.model == "custom-model"
    assert config.api_base == "https://example.com/v1"
    assert config.api_key == "custom-key"


def test_model_override_and_validation(monkeypatch):
    clear_env(monkeypatch)
    settings = Settings(_env_file=None)

    config = settings.provider_config("deepseek", model_override="deepseek-v4-pro")

    assert config.model == "deepseek-v4-pro"
    with pytest.raises(ValueError, match="api_key"):
        config.validate_ready()


def test_legacy_mimo_key_fallback(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.setenv("LLM_API_BASE", "https://legacy.example/v1")

    config = Settings(_env_file=None).provider_config("mimo")

    assert config.api_key == "legacy-key"
    assert config.model == "mimo-v2.5-pro"
