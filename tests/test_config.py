from simple_agent.config import Settings


def test_environment_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "openai/environment-model")

    settings = Settings(_env_file=None)

    assert settings.model == "openai/environment-model"


def test_defaults_without_environment(monkeypatch):
    for name in [
        "LLM_MODEL",
        "LLM_API_BASE",
        "AGENT_MAX_STEPS",
        "AGENT_MAX_INPUT_CHARS",
        "AGENT_DEBUG",
    ]:
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.max_steps == 5
    assert settings.max_input_chars == 4000
    assert settings.debug is False
