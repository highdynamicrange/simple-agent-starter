import argparse

from pydantic import ValidationError

from simple_agent.agent import Agent
from simple_agent.config import Settings, default_data_dir
from simple_agent.llm import LiteLLMClient, LLMClientError
from simple_agent.tools import build_default_registry

HELP_TEXT = """可用命令：
  /help              显示帮助
  /model             查看当前模型
  /model <模型名称>   切换模型，例如 anthropic/claude-sonnet-4-5
  /reset             清空当前会话
  /exit              退出程序"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="简单的多模型命令行 Agent")
    parser.add_argument("--model", help="覆盖 .env 中的 LLM_MODEL")
    parser.add_argument("--debug", action="store_true", help="显示详细模型错误")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        settings = Settings()
    except ValidationError as exc:
        raise SystemExit(f"配置无效：{exc}") from exc

    model = args.model or settings.model
    client = LiteLLMClient(
        api_base=settings.api_base,
        debug=args.debug or settings.debug,
    )
    agent = Agent(
        client=client,
        tools=build_default_registry(default_data_dir()),
        model=model,
        max_steps=settings.max_steps,
        max_input_chars=settings.max_input_chars,
    )

    print("Simple Agent 已启动。输入 /help 查看命令。")
    print(f"当前模型：{agent.model}")
    support = client.supports_tools(agent.model)
    if support is False:
        print("警告：LiteLLM 标记该模型不支持工具调用，Agent 工具可能不可用。")
    elif support is None:
        print("提示：无法确认该模型是否支持工具调用。")

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return

        if not user_input:
            continue
        if user_input == "/exit":
            print("再见。")
            return
        if user_input == "/help":
            print(HELP_TEXT)
            continue
        if user_input == "/reset":
            agent.reset()
            print("会话已清空。")
            continue
        if user_input == "/model":
            print(f"当前模型：{agent.model}")
            continue
        if user_input.startswith("/model "):
            try:
                agent.set_model(user_input.removeprefix("/model "))
                print(f"已切换模型：{agent.model}")
            except ValueError as exc:
                print(f"错误：{exc}")
            continue
        if user_input.startswith("/"):
            print("未知命令。输入 /help 查看命令。")
            continue

        try:
            answer = agent.run(user_input)
            print(f"\nAgent：{answer}")
        except (ValueError, LLMClientError) as exc:
            print(f"\n错误：{exc}")
