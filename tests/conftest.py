from collections.abc import Sequence

from simple_agent.models import Message, ModelReply


class FakeLLMClient:
    def __init__(
        self,
        replies: list[ModelReply] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.replies = list(replies or [])
        self.error = error
        self.calls: list[dict] = []

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        tools: list[dict],
    ) -> ModelReply:
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "tools": tools,
            }
        )
        if self.error:
            raise self.error
        return self.replies.pop(0)

    def supports_tools(self, model: str) -> bool | None:
        return True
