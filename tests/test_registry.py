import json

from simple_agent.tools import build_default_registry


def test_registry_reports_unknown_tool(tmp_path):
    result = json.loads(build_default_registry(tmp_path).execute("missing", "{}"))

    assert result["ok"] is False
    assert "未知工具" in result["error"]


def test_registry_reports_invalid_json(tmp_path):
    result = json.loads(build_default_registry(tmp_path).execute("calculator", "{"))

    assert result["ok"] is False
    assert "参数校验" in result["error"]


def test_registry_contains_three_function_schemas(tmp_path):
    schemas = build_default_registry(tmp_path).schemas

    assert {schema["function"]["name"] for schema in schemas} == {
        "calculator",
        "current_time",
        "read_text_file",
    }
