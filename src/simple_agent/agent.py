from simple_agent.llm import LLMClient
from simple_agent.models import Message, ModelReply
from simple_agent.tools import ToolRegistry

SYSTEM_PROMPT = """你是一个简洁、可靠的命令行助手。
需要计算、查询时间或读取 data 目录文件时，应调用相应工具。
不得声称执行了未实际执行的工具；工具返回错误时，应清楚说明并给出可行建议。
回答默认使用用户所用的语言。"""


class Agent:
    def __init__(
        self,
        *,
        client: LLMClient,
        tools: ToolRegistry,
        model: str,
        max_steps: int = 5,
        max_input_chars: int = 4000,
    ) -> None:
        self.client = client
        self.tools = tools
        self.model = model
        self.max_steps = max_steps
        self.max_input_chars = max_input_chars
        self._messages: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def reset(self) -> None:
        self._messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def set_model(self, model: str) -> None:
        model = model.strip()
        if not model:
            raise ValueError("模型名称不能为空。")
        self.model = model

    def set_client(self, client: LLMClient) -> None:
        self.client = client

    def remove_provider_specific_fields(self) -> None:
        for message in self._messages:
            message.pop("reasoning_content", None)

    def run(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            raise ValueError("输入不能为空。")
        if len(user_input) > self.max_input_chars:
            raise ValueError(f"输入不能超过 {self.max_input_chars} 个字符。")

        self._messages.append({"role": "user", "content": user_input})

        for _ in range(self.max_steps):
            reply = self.client.complete(
                model=self.model,
                messages=self._messages,
                tools=self.tools.schemas,
            )
            if not reply.tool_calls:
                content = (reply.content or "").strip() or "模型没有返回可显示的内容。"
                self._messages.append(_assistant_content_message(content, reply))
                return content

            self._messages.append(_assistant_tool_message(reply))
            for tool_call in reply.tool_calls:
                result = self.tools.execute(tool_call.name, tool_call.arguments)
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result,
                    }
                )

        message = f"Agent 已达到最大执行步数（{self.max_steps}），任务已安全停止。"
        self._messages.append({"role": "assistant", "content": message})
        return message


def _assistant_tool_message(reply: ModelReply) -> Message:
    message: Message = {
        "role": "assistant",
        "content": reply.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments,
                },
            }
            for call in reply.tool_calls
        ],
    }
    if reply.reasoning_content:
        message["reasoning_content"] = reply.reasoning_content
    return message


def _assistant_content_message(content: str, reply: ModelReply) -> Message:
    message: Message = {"role": "assistant", "content": content}
    if reply.reasoning_content:
        message["reasoning_content"] = reply.reasoning_content
    return message
