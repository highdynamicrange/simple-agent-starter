from types import SimpleNamespace

import pytest

from simple_agent.llm import LiteLLMClient, LLMClientError


@pytest.mark.parametrize(
    ("error_name", "expected"),
    [
        ("AuthenticationError", "认证失败"),
        ("RateLimitError", "频率或额度"),
        ("APIConnectionError", "无法连接"),
        ("Timeout", "超时"),
        ("UnexpectedProviderError", "模型调用失败"),
    ],
)
def test_friendly_errors(error_name, expected):
    error_type = type(error_name, (Exception,), {})
    client = LiteLLMClient()

    assert expected in client._friendly_error(error_type("provider details"))


def test_debug_error_includes_details():
    client = LiteLLMClient(debug=True)

    assert "provider details" in client._friendly_error(ValueError("provider details"))


def test_complete_normalizes_response(monkeypatch):
    function = SimpleNamespace(name="calculator", arguments='{"expression":"2+2"}')
    message = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(id="call-1", function=function)],
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def fake_completion(**kwargs):
        assert kwargs["model"] == "test/model"
        return response

    monkeypatch.setattr("litellm.completion", fake_completion)
    reply = LiteLLMClient().complete(model="test/model", messages=[], tools=[])

    assert reply.tool_calls[0].name == "calculator"


def test_complete_converts_provider_error(monkeypatch):
    def fake_completion(**kwargs):
        error_type = type("AuthenticationError", (Exception,), {})
        raise error_type("secret provider detail")

    monkeypatch.setattr("litellm.completion", fake_completion)

    with pytest.raises(LLMClientError, match="认证失败") as exc_info:
        LiteLLMClient().complete(model="test/model", messages=[], tools=[])
    assert "secret provider detail" not in str(exc_info.value)
