from collections.abc import Sequence
from typing import Any, Protocol

from openai import OpenAI

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


class OpenAICompatibleClient:
    """Thin client for APIs that implement OpenAI chat completions."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        provider: str,
        debug: bool = False,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.debug = debug
        self.extra_body = extra_body or None
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        tools: list[dict],
    ) -> ModelReply:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "tools": tools,
            "tool_choice": "auto",
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        try:
            response = self._client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            calls = [
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
                for call in (message.tool_calls or [])
            ]
            return ModelReply(
                content=_normalize_content(message.content),
                reasoning_content=getattr(message, "reasoning_content", None),
                tool_calls=calls,
            )
        except Exception as exc:
            raise LLMClientError(self._friendly_error(exc)) from exc

    def _friendly_error(self, exc: Exception) -> str:
        error_name = type(exc).__name__
        messages = {
            "AuthenticationError": "模型认证失败，请检查对应厂商的 API Key。",
            "PermissionDeniedError": "模型服务拒绝访问，请检查账号权限或 API Key。",
            "RateLimitError": "模型请求频率或额度受限，请稍后重试。",
            "APIConnectionError": "无法连接模型服务，请检查网络和 API 地址。",
            "APITimeoutError": "模型请求超时，请稍后重试。",
            "BadRequestError": "模型请求参数无效，请检查模型名和工具调用支持。",
        }
        message = messages.get(error_name, "模型调用失败。")
        if self.debug:
            return f"{message} [{error_name}: {exc}]"
        return message


def _normalize_content(content: Any) -> str | None:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif hasattr(item, "text") and isinstance(item.text, str):
                parts.append(item.text)
        return "\n".join(parts) if parts else None
    return str(content)
