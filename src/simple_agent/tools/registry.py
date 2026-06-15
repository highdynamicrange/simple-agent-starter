import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from simple_agent.tools.calculator import CalculatorArguments, calculate
from simple_agent.tools.current_time import CurrentTimeArguments, current_time
from simple_agent.tools.file_reader import ReadTextFileArguments, read_text_file


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    arguments_model: type[BaseModel]
    handler: Callable[[BaseModel], Any]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arguments_model.model_json_schema(),
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[ToolDefinition]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    @property
    def schemas(self) -> list[dict]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, raw_arguments: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return _error(f"未知工具：{name}")

        try:
            arguments = tool.arguments_model.model_validate_json(raw_arguments)
            result = tool.handler(arguments)
            return json.dumps({"ok": True, "result": result}, ensure_ascii=False)
        except ValidationError as exc:
            return _error("工具参数校验失败。", details=exc.errors(include_url=False))
        except (ValueError, OSError) as exc:
            return _error(str(exc))
        except Exception:
            return _error("工具执行失败。")


def build_default_registry(data_dir: Path) -> ToolRegistry:
    return ToolRegistry(
        [
            ToolDefinition(
                name="calculator",
                description="安全计算一个只包含数字和基础运算符的数学表达式。",
                arguments_model=CalculatorArguments,
                handler=calculate,
            ),
            ToolDefinition(
                name="current_time",
                description="查询指定 IANA 时区的当前日期和时间。",
                arguments_model=CurrentTimeArguments,
                handler=current_time,
            ),
            ToolDefinition(
                name="read_text_file",
                description="读取项目 data 目录内允许类型的 UTF-8 文本文件。",
                arguments_model=ReadTextFileArguments,
                handler=lambda arguments: read_text_file(arguments, data_dir=data_dir),
            ),
        ]
    )


def _error(message: str, *, details: Any = None) -> str:
    payload: dict[str, Any] = {"ok": False, "error": message}
    if details is not None:
        payload["details"] = details
    return json.dumps(payload, ensure_ascii=False)
