# Simple Agent Starter

一个小而完整的多模型命令行 Agent。项目刻意不使用完整 Agent 框架，而是自己实现
“模型决定调用工具 → 程序校验并执行 → 工具结果返回模型”的循环，方便阅读和扩展。

## 架构

```text
CLI
 └── Agent 循环
      ├── OpenAI 兼容模型客户端
      │    ├── MiMo
      │    ├── DeepSeek
      │    └── Custom OpenAI-compatible endpoint
      └── Pydantic 工具注册表
           ├── calculator
           ├── current_time
           └── read_text_file
```

项目不再使用 LiteLLM，也不依赖第三方模型能力表。MiMo、DeepSeek 和 custom Provider 都通过
官方 `openai` Python SDK 直接调用 OpenAI 兼容接口。

## 环境准备

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync --dev
cp .env.example .env
```

编辑 `.env`，填写你要使用的 Provider、模型和 API Key。不要提交 `.env`。

### MiMo

```env
LLM_PROVIDER=mimo
MIMO_MODEL=mimo-v2.5-pro
MIMO_API_BASE=https://token-plan-cn.xiaomimimo.com/v1
MIMO_API_KEY=tp-your-token-plan-key
```

中国、新加坡、欧洲集群的 Base URL 不同，应使用 MiMo 订阅管理页面显示的地址。Token Plan
Key 以 `tp-` 开头，不能与按量付费的 `sk-` Key 混用。

### DeepSeek

```env
LLM_PROVIDER=deepseek
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_API_KEY=your-key
DEEPSEEK_THINKING=false
```

DeepSeek 默认使用 `deepseek-v4-flash` 并关闭思考模式。如果需要更强推理，可改用
`deepseek-v4-pro`，并将 `DEEPSEEK_THINKING=true`。

### Custom

```env
LLM_PROVIDER=custom
CUSTOM_MODEL=your-model-name
CUSTOM_API_BASE=https://example.com/v1
CUSTOM_API_KEY=your-key
```

`custom` 用于其他 OpenAI 兼容接口。

## 使用

```bash
uv run simple-agent
uv run simple-agent --provider mimo
uv run simple-agent --provider deepseek
uv run simple-agent --provider custom --model your-model-name
uv run simple-agent --debug
```

会话命令：

```text
/help                    显示帮助
/provider                查看当前 Provider
/provider mimo           切换到 MiMo
/provider deepseek       切换到 DeepSeek
/provider custom         切换到自定义兼容接口
/model                   查看当前模型
/model <模型名称>         切换当前 Provider 的模型
/reset                   清空当前会话
/exit                    退出
```

跨 Provider 切换时会保留对话历史，同时移除厂商专属字段，例如 DeepSeek 的
`reasoning_content`，避免其他兼容端点拒绝请求。

## 内置工具与安全边界

- `calculator`：支持 `+ - * / // % **` 和括号；使用 AST 解析，不使用 `eval`。
- `current_time`：查询 IANA 时区，例如 `Asia/Shanghai`。
- `read_text_file`：读取 `data/` 内的 UTF-8 文本。

文件工具只接受相对路径，只允许 `.txt`、`.md`、`.json`、`.csv`、`.yaml`、`.yml`，
拒绝路径穿越和符号链接逃逸，单文件最大 100 KB。API Key 和 `.env` 不会进入工具可读范围。

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_PROVIDER` | `mimo` | `mimo`、`deepseek` 或 `custom` |
| `MIMO_MODEL` | `mimo-v2.5-pro` | MiMo 模型 |
| `MIMO_API_BASE` | `https://token-plan-cn.xiaomimimo.com/v1` | MiMo OpenAI 兼容地址 |
| `MIMO_API_KEY` | 空 | MiMo API Key |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | DeepSeek 模型 |
| `DEEPSEEK_API_BASE` | `https://api.deepseek.com` | DeepSeek OpenAI 兼容地址 |
| `DEEPSEEK_API_KEY` | 空 | DeepSeek API Key |
| `DEEPSEEK_THINKING` | `false` | 是否启用 DeepSeek 思考模式 |
| `CUSTOM_MODEL` | 空 | 自定义兼容模型 |
| `CUSTOM_API_BASE` | 空 | 自定义兼容地址 |
| `CUSTOM_API_KEY` | 空 | 自定义兼容 API Key |
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
4. 使用 `/provider deepseek` 或 `/provider mimo` 切换 Provider。

## 上传 GitHub

本地仓库已使用 `main` 分支。创建空的 GitHub 仓库后可执行：

```bash
git remote add origin git@github.com:<你的用户名>/simple-agent-starter.git
git push -u origin main
```

## License

[MIT](LICENSE)
