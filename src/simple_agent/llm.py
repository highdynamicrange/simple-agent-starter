from collections.abc import Sequence
from typing import Protocol

from simple_agent.models import Message, ModelReply, ToolCall


class LLMClientError(RuntimeError):
    """A provider error safe to present to a CLI user."""


class LLMClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        tools: list[dict],
    ) -> ModelReply: ...

    def supports_tools(self, model: str) -> bool | None: ...


class LiteLLMClient:
    def __init__(self, *, api_base: str | None = None, debug: bool = False) -> None:
        self.api_base = api_base or None
        self.debug = debug

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        tools: list[dict],
    ) -> ModelReply:
        from litellm import completion

        kwargs: dict = {
            "model": model,
            "messages": list(messages),
            "tools": tools,
            "tool_choice": "auto",
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base

        try:
            response = completion(**kwargs)
            message = response.choices[0].message
            calls = [
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
                for call in (message.tool_calls or [])
            ]
            return ModelReply(content=message.content, tool_calls=calls)
        except Exception as exc:
            raise LLMClientError(self._friendly_error(exc)) from exc

    def supports_tools(self, model: str) -> bool | None:
        from litellm import supports_function_calling

        try:
            return bool(supports_function_calling(model=model))
        except Exception:
            return None

    def _friendly_error(self, exc: Exception) -> str:
        error_name = type(exc).__name__
        messages = {
            "AuthenticationError": "模型认证失败，请检查对应厂商的 API Key。",
            "RateLimitError": "模型请求频率或额度受限，请稍后重试。",
            "APIConnectionError": "无法连接模型服务，请检查网络和 API 地址。",
            "Timeout": "模型请求超时，请稍后重试。",
            "APITimeoutError": "模型请求超时，请稍后重试。",
        }
        message = messages.get(error_name, "模型调用失败。")
        if self.debug:
            return f"{message} [{error_name}: {exc}]"
        return message
