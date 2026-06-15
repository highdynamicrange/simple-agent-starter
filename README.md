# Simple Agent Starter

一个小而完整的多模型命令行 Agent。项目刻意不使用完整 Agent 框架，而是自己实现
“模型决定调用工具 → 程序校验并执行 → 工具结果返回模型”的循环，方便阅读和扩展。

## 架构

```text
CLI
 └── Agent 循环
      ├── LiteLLM 模型适配层
      │    └── OpenAI / Anthropic / Gemini / Ollama / 兼容接口
      └── Pydantic 工具注册表
           ├── calculator
           ├── current_time
           └── read_text_file
```

LiteLLM 在这里是多模型统一调用库，不是 Agent 框架。Agent 的循环、历史消息、工具权限和
停止条件都由本项目负责。

## 环境准备

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync --dev
cp .env.example .env
```

编辑 `.env`，填写模型和相应厂商的 API Key。不要提交 `.env`。

```env
LLM_MODEL=openai/gpt-4.1-mini
OPENAI_API_KEY=your-key
```

其他厂商示例：

```env
LLM_MODEL=anthropic/your-model-name
ANTHROPIC_API_KEY=your-key

LLM_MODEL=gemini/your-model-name
GEMINI_API_KEY=your-key

LLM_MODEL=ollama/your-model-name
LLM_API_BASE=http://localhost:11434
```

模型必须支持工具调用。准确的模型标识以对应厂商和 LiteLLM 文档为准。

## 使用

```bash
uv run simple-agent
uv run simple-agent --model anthropic/your-model-name
uv run simple-agent --debug
```

会话命令：

```text
/help              显示帮助
/model             查看当前模型
/model <模型名称>   切换模型
/reset             清空当前会话
/exit              退出
```

模型选择优先级为 `.env` 的 `LLM_MODEL`、命令行 `--model`、会话内 `/model`。后者覆盖前者。
对话记录只存在于当前进程，退出后不会保存。

## 内置工具与安全边界

- `calculator`：支持 `+ - * / // % **` 和括号；使用 AST 解析，不使用 `eval`。
- `current_time`：查询 IANA 时区，例如 `Asia/Shanghai`。
- `read_text_file`：读取 `data/` 内的 UTF-8 文本。

文件工具只接受相对路径，只允许 `.txt`、`.md`、`.json`、`.csv`、`.yaml`、`.yml`，
拒绝路径穿越和符号链接逃逸，单文件最大 100 KB。API Key 和 `.env` 不会进入工具可读范围。

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_MODEL` | `openai/gpt-4.1-mini` | LiteLLM 模型标识 |
| `LLM_API_BASE` | 空 | 可选兼容接口地址 |
| `AGENT_MAX_STEPS` | `5` | 单次任务最大模型调用轮数 |
| `AGENT_MAX_INPUT_CHARS` | `4000` | 单次用户输入长度限制 |
| `AGENT_DEBUG` | `false` | 是否显示模型异常详情 |

## 开发与测试

```bash
uv run ruff check .
uv run pytest
```

测试使用假的模型客户端，不请求真实模型，也不会消耗 API 额度。GitHub Actions 同样不需要
API Key。

手动冒烟测试建议：

1. 输入“计算 `(12 + 8) * 3`”。
2. 输入“现在上海几点？”
3. 输入“读取 `example.md` 并总结”。
4. 使用 `/model` 切换到另一个已配置 API Key 的厂商模型。

## 上传 GitHub

本地仓库已使用 `main` 分支。创建空的 GitHub 仓库后可执行：

```bash
git remote add origin git@github.com:<你的用户名>/simple-agent-starter.git
git push -u origin main
```

## License

[MIT](LICENSE)
