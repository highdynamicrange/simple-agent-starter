import json

import pytest
from conftest import FakeLLMClient

from simple_agent.agent import Agent
from simple_agent.models import ModelReply, ToolCall
from simple_agent.tools import build_default_registry


def make_agent(tmp_path, replies, *, max_steps=5):
    client = FakeLLMClient(replies)
    return (
        Agent(
            client=client,
            tools=build_default_registry(tmp_path),
            model="test/model",
            max_steps=max_steps,
            max_input_chars=20,
        ),
        client,
    )


def test_returns_normal_answer(tmp_path):
    agent, client = make_agent(tmp_path, [ModelReply(content="你好")])

    assert agent.run("在吗") == "你好"
    assert client.calls[0]["model"] == "test/model"


def test_executes_tool_and_returns_final_answer(tmp_path):
    replies = [
        ModelReply(
            tool_calls=[
                ToolCall(id="call-1", name="calculator", arguments='{"expression":"2+3"}')
            ]
        ),
        ModelReply(content="结果是 5"),
    ]
    agent, client = make_agent(tmp_path, replies)

    assert agent.run("计算 2+3") == "结果是 5"
    tool_message = client.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert json.loads(tool_message["content"])["result"]["result"] == 5


def test_executes_multiple_tools_in_one_turn(tmp_path):
    replies = [
        ModelReply(
            tool_calls=[
                ToolCall(id="call-1", name="calculator", arguments='{"expression":"6*7"}'),
                ToolCall(
                    id="call-2",
                    name="current_time",
                    arguments='{"timezone":"Asia/Shanghai"}',
                ),
            ]
        ),
        ModelReply(content="完成"),
    ]
    agent, client = make_agent(tmp_path, replies)

    assert agent.run("计算并报时") == "完成"
    tool_messages = [
        message for message in client.calls[1]["messages"] if message["role"] == "tool"
    ]
    assert len(tool_messages) == 2


def test_stops_at_max_steps(tmp_path):
    call = ToolCall(id="call-1", name="calculator", arguments='{"expression":"1+1"}')
    agent, client = make_agent(
        tmp_path,
        [ModelReply(tool_calls=[call]), ModelReply(tool_calls=[call])],
        max_steps=2,
    )

    assert "最大执行步数" in agent.run("继续调用工具")
    assert len(client.calls) == 2


def test_input_validation_and_reset(tmp_path):
    agent, _ = make_agent(tmp_path, [ModelReply(content="ok")])

    with pytest.raises(ValueError, match="不能为空"):
        agent.run("  ")
    with pytest.raises(ValueError, match="20"):
        agent.run("x" * 21)

    agent.run("hello")
    agent.reset()
    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "system"


def test_model_switching(tmp_path):
    agent, client = make_agent(tmp_path, [ModelReply(content="ok")])

    agent.set_model("anthropic/example")
    agent.run("hello")

    assert client.calls[0]["model"] == "anthropic/example"
    with pytest.raises(ValueError, match="不能为空"):
        agent.set_model(" ")


def test_preserves_and_cleans_reasoning_content(tmp_path):
    call = ToolCall(id="call-1", name="calculator", arguments='{"expression":"2+2"}')
    agent, _client = make_agent(
        tmp_path,
        [
            ModelReply(reasoning_content="need math", tool_calls=[call]),
            ModelReply(content="4", reasoning_content="done"),
        ],
    )

    assert agent.run("算一下") == "4"
    assert any("reasoning_content" in message for message in agent.messages)

    new_client = FakeLLMClient([ModelReply(content="new")])
    agent.set_client(new_client)
    agent.remove_provider_specific_fields()

    assert all("reasoning_content" not in message for message in agent.messages)
