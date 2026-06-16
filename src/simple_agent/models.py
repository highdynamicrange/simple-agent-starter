from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str


class ModelReply(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class StreamChunk(BaseModel):
    delta: str | None = None
    reasoning_delta: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finished: bool = False


Message = dict[str, Any]
