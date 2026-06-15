import pytest

from simple_agent.tools.current_time import CurrentTimeArguments, current_time


def test_current_time_uses_requested_timezone():
    result = current_time(CurrentTimeArguments(timezone="Asia/Shanghai"))

    assert result["timezone"] == "Asia/Shanghai"
    assert "+08:00" in result["datetime"]


def test_rejects_invalid_timezone():
    with pytest.raises(ValueError, match="未知时区"):
        current_time(CurrentTimeArguments(timezone="Not/A_Timezone"))
