from types import SimpleNamespace

import pytest

from simple_agent.llm import LLMClientError, OpenAICompatibleClient


class FakeCompletions:
    def __init__(self, *, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def make_response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def install_fake_openai(monkeypatch, completions: FakeCompletions):
    def fake_openai(*, api_key, base_url):
        fake_openai.api_key = api_key
        fake_openai.base_url = base_url
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    monkeypatch.setattr("simple_agent.llm.OpenAI", fake_openai)
    return fake_openai


@pytest.mark.parametrize(
    ("error_name", "expected"),
    [
        ("AuthenticationError", "认证失败"),
        ("PermissionDeniedError", "拒绝访问"),
        ("RateLimitError", "频率或额度"),
        ("APIConnectionError", "无法连接"),
        ("APITimeoutError", "超时"),
        ("BadRequestError", "参数无效"),
        ("UnexpectedProviderError", "模型调用失败"),
    ],
)
def test_friendly_errors(error_name, expected, monkeypatch):
    error_type = type(error_name, (Exception,), {})
    completions = FakeCompletions(error=error_type("provider details"))
    install_fake_openai(monkeypatch, completions)
    client = OpenAICompatibleClient(
        api_key="key",
        base_url="https://example.com/v1",
        provider="custom",
    )

    with pytest.raises(LLMClientError, match=expected) as exc_info:
        client.complete(model="model", messages=[], tools=[])

    assert "provider details" not in str(exc_info.value)


def test_debug_error_includes_details(monkeypatch):
    completions = FakeCompletions(error=ValueError("provider details"))
    install_fake_openai(monkeypatch, completions)
    client = OpenAICompatibleClient(
        api_key="key",
        base_url="https://example.com/v1",
        provider="custom",
        debug=True,
    )

    with pytest.raises(LLMClientError, match="provider details"):
        client.complete(model="model", messages=[], tools=[])


def test_complete_normalizes_response(monkeypatch):
    function = SimpleNamespace(name="calculator", arguments='{"expression":"2+2"}')
    message = SimpleNamespace(
        content=None,
        reasoning_content="I should calculate.",
        tool_calls=[SimpleNamespace(id="call-1", function=function)],
    )
    completions = FakeCompletions(response=make_response(message))
    fake_openai = install_fake_openai(monkeypatch, completions)

    client = OpenAICompatibleClient(
        api_key="key",
        base_url="https://example.com/v1",
        provider="custom",
        extra_body={"thinking": {"type": "disabled"}},
    )
    reply = client.complete(model="model", messages=[], tools=[{"type": "function"}])

    assert fake_openai.api_key == "key"
    assert fake_openai.base_url == "https://example.com/v1"
    assert completions.calls[0]["model"] == "model"
    assert completions.calls[0]["tool_choice"] == "auto"
    assert completions.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert reply.reasoning_content == "I should calculate."
    assert reply.tool_calls[0].name == "calculator"


def test_normalizes_list_content(monkeypatch):
    message = SimpleNamespace(
        content=[{"text": "hello"}, SimpleNamespace(text="world")],
        reasoning_content=None,
        tool_calls=None,
    )
    completions = FakeCompletions(response=make_response(message))
    install_fake_openai(monkeypatch, completions)

    client = OpenAICompatibleClient(
        api_key="key",
        base_url="https://example.com/v1",
        provider="custom",
    )
    reply = client.complete(model="model", messages=[], tools=[])

    assert reply.content == "hello\nworld"
