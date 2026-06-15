from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field


class CurrentTimeArguments(BaseModel):
    timezone: str = Field(default="Asia/Shanghai", min_length=1, max_length=100)


def current_time(arguments: CurrentTimeArguments) -> dict[str, str]:
    try:
        timezone = ZoneInfo(arguments.timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"未知时区：{arguments.timezone}") from exc

    now = datetime.now(timezone)
    return {
        "timezone": arguments.timezone,
        "datetime": now.isoformat(timespec="seconds"),
    }
