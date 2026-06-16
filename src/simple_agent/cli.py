import argparse

from pydantic import ValidationError

from simple_agent.agent import Agent
from simple_agent.config import ProviderConfig, ProviderName, Settings, default_data_dir
from simple_agent.llm import LLMClientError, OpenAICompatibleClient
from simple_agent.tools import build_default_registry

PROVIDERS: tuple[ProviderName, ...] = ("mimo", "deepseek", "custom")

HELP_TEXT = """可用命令：
  /help                    显示帮助
  /provider                查看当前 Provider
  /provider <名称>          切换 Provider：mimo、deepseek、custom
  /model                   查看当前模型
  /model <模型名称>         切换当前 Provider 的模型
  /reset                   清空当前会话
  /exit                    退出程序"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="简单的多模型命令行 Agent")
    parser.add_argument("--provider", choices=PROVIDERS, help="覆盖 .env 中的 LLM_PROVIDER")
    parser.add_argument("--model", help="覆盖当前 Provider 的默认模型")
    parser.add_argument("--debug", action="store_true", help="显示详细模型错误")
    return parser


def build_client(config: ProviderConfig, *, debug: bool) -> OpenAICompatibleClient:
    config.validate_ready()
    return OpenAICompatibleClient(
        api_key=config.api_key,
        base_url=config.api_base,
        provider=config.name,
        debug=debug,
        extra_body=config.extra_body,
    )


def main() -> None:
    args = build_parser().parse_args()
    try:
        settings = Settings()
    except ValidationError as exc:
        raise SystemExit(f"配置无效：{exc}") from exc

    debug = args.debug or settings.debug
    current_provider: ProviderName = args.provider or settings.provider
    model_overrides: dict[ProviderName, str] = {}
    if args.model:
        model_overrides[current_provider] = args.model

    try:
        current_config = settings.provider_config(
            current_provider,
            model_override=model_overrides.get(current_provider),
        )
        client = build_client(current_config, debug=debug)
    except ValueError as exc:
        raise SystemExit(f"配置无效：{exc}") from exc

    agent = Agent(
        client=client,
        tools=build_default_registry(default_data_dir()),
        model=current_config.model,
        max_steps=settings.max_steps,
        max_input_chars=settings.max_input_chars,
    )

    print("Simple Agent 已启动。输入 /help 查看命令。")
    print(_provider_summary(current_config))

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
        if user_input == "/provider":
            print(_provider_summary(current_config))
            print(f"可用 Provider：{', '.join(PROVIDERS)}")
            continue
        if user_input.startswith("/provider "):
            provider_name = user_input.removeprefix("/provider ").strip()
            if provider_name not in PROVIDERS:
                print(f"错误：未知 Provider：{provider_name}")
                continue
            next_provider = provider_name  # type: ignore[assignment]
            try:
                current_config = settings.provider_config(
                    next_provider,
                    model_override=model_overrides.get(next_provider),
                )
                agent.set_client(build_client(current_config, debug=debug))
                agent.set_model(current_config.model)
                agent.remove_provider_specific_fields()
                current_provider = next_provider
                print(f"已切换 Provider：{current_provider}")
                print(_provider_summary(current_config))
            except ValueError as exc:
                print(f"错误：{exc}")
            continue
        if user_input == "/model":
            print(f"当前 Provider：{current_provider}")
            print(f"当前模型：{agent.model}")
            continue
        if user_input.startswith("/model "):
            try:
                agent.set_model(user_input.removeprefix("/model "))
                model_overrides[current_provider] = agent.model
                current_config = current_config.model_copy(update={"model": agent.model})
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


def _provider_summary(config: ProviderConfig) -> str:
    return (
        f"当前 Provider：{config.name}\n"
        f"当前模型：{config.model}\n"
        f"API 地址：{config.api_base}"
    )
