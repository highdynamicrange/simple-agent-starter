from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["mimo", "deepseek", "custom"]


class ProviderConfig(BaseModel):
    name: ProviderName
    model: str
    api_base: str
    api_key: str = ""
    deepseek_thinking: bool = False

    def validate_ready(self) -> None:
        missing = []
        if not self.model:
            missing.append("model")
        if not self.api_base:
            missing.append("api_base")
        if not self.api_key:
            missing.append("api_key")
        if missing:
            raise ValueError(f"{self.name} provider 缺少配置：{', '.join(missing)}。")

    @property
    def extra_body(self) -> dict | None:
        if self.name != "deepseek":
            return None
        thinking_type = "enabled" if self.deepseek_thinking else "disabled"
        return {"thinking": {"type": thinking_type}}


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: ProviderName = Field(default="mimo", alias="LLM_PROVIDER")

    mimo_model: str = Field(default="mimo-v2.5-pro", alias="MIMO_MODEL")
    mimo_api_base: str = Field(
        default="https://token-plan-cn.xiaomimimo.com/v1",
        alias="MIMO_API_BASE",
    )
    mimo_api_key: str = Field(default="", alias="MIMO_API_KEY")

    deepseek_model: str = Field(default="deepseek-v4-flash", alias="DEEPSEEK_MODEL")
    deepseek_api_base: str = Field(
        default="https://api.deepseek.com",
        alias="DEEPSEEK_API_BASE",
    )
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_thinking: bool = Field(default=False, alias="DEEPSEEK_THINKING")

    custom_model: str = Field(default="", alias="CUSTOM_MODEL")
    custom_api_base: str = Field(default="", alias="CUSTOM_API_BASE")
    custom_api_key: str = Field(default="", alias="CUSTOM_API_KEY")

    # Compatibility for the earlier LiteLLM/OpenAI-compatible MiMo config.
    legacy_model: str | None = Field(default=None, alias="LLM_MODEL")
    legacy_api_base: str | None = Field(default=None, alias="LLM_API_BASE")
    legacy_openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    max_steps: int = Field(default=5, ge=1, le=20, alias="AGENT_MAX_STEPS")
    max_input_chars: int = Field(
        default=4000,
        ge=1,
        le=100_000,
        alias="AGENT_MAX_INPUT_CHARS",
    )
    debug: bool = Field(default=False, alias="AGENT_DEBUG")

    def provider_config(
        self,
        provider: ProviderName | None = None,
        *,
        model_override: str | None = None,
    ) -> ProviderConfig:
        name = provider or self.provider
        if name == "mimo":
            return ProviderConfig(
                name="mimo",
                model=model_override or self._mimo_model(),
                api_base=self.mimo_api_base or self.legacy_api_base or "",
                api_key=self.mimo_api_key or self.legacy_openai_api_key,
            )
        if name == "deepseek":
            return ProviderConfig(
                name="deepseek",
                model=model_override or self.deepseek_model,
                api_base=self.deepseek_api_base,
                api_key=self.deepseek_api_key,
                deepseek_thinking=self.deepseek_thinking,
            )
        return ProviderConfig(
            name="custom",
            model=model_override or self.custom_model,
            api_base=self.custom_api_base,
            api_key=self.custom_api_key,
        )

    def _mimo_model(self) -> str:
        if self.mimo_model:
            return self.mimo_model
        if self.legacy_model:
            return self.legacy_model.removeprefix("openai/").removeprefix("xiaomi_mimo/")
        return "mimo-v2.5-pro"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    return project_root() / "data"
