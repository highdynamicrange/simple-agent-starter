from collections.abc import Generator, Sequence

from simple_agent.models import Message, ModelReply, StreamChunk


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

    def complete_stream(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        tools: list[dict],
    ) -> Generator[StreamChunk, None, None]:
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "tools": tools,
            }
        )
        if self.error:
            raise self.error
        reply = self.replies.pop(0)
        if reply.content:
            yield StreamChunk(delta=reply.content)
        if reply.reasoning_content:
            yield StreamChunk(reasoning_delta=reply.reasoning_content)
        yield StreamChunk(tool_calls=reply.tool_calls, finished=True)
