from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: str = Field(default="openai/gpt-4.1-mini", alias="LLM_MODEL")
    api_base: str | None = Field(default=None, alias="LLM_API_BASE")
    max_steps: int = Field(default=5, ge=1, le=20, alias="AGENT_MAX_STEPS")
    max_input_chars: int = Field(
        default=4000,
        ge=1,
        le=100_000,
        alias="AGENT_MAX_INPUT_CHARS",
    )
    debug: bool = Field(default=False, alias="AGENT_DEBUG")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    return project_root() / "data"
