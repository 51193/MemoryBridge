# MemoryBridge

Agent 与大模型之间的记忆中间件。接收 Agent 的 OpenAI 兼容 API 请求，自动检索历史记忆并注入到上下文中，然后转发到 DeepSeek 官方 API，最后异步存储新的对话记忆。

## 架构

```
Agent ──(OpenAI API)──→ MemoryBridge ──(OpenAI API)──→ DeepSeek
                         │   ↑
                         │   ├── 检索记忆 (读了)
                         │   └── 存储记忆 (写)
                         │
                     ┌───┴────┐
                     │  Qdrant  │
                     └───┬────┘
                         │
                     Mem0 (记忆引擎)
```

详细架构文档见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 快速开始

### 环境要求

- Python 3.11+
- Linux x86_64 / macOS (arm64/x86_64)

### 1. 安装

```bash
git clone https://github.com/51193/MemoryBridge
cd MemoryBridge

# 安装 uv 包管理器 (如已安装则可跳过)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 安装项目依赖
uv sync
```

### 2. 初始化

`--setup` 会自动下载 Qdrant 二进制、创建数据目录、生成 .env 模板：

```bash
uv run python src/memory_bridge/host_manager.py --setup
```

### 3. 配置 API Key

编辑生成的 `.env` 文件，填入 API Key：

```bash
DEEPSEEK_API_KEY=sk-your-deepseek-key    # 从 platform.deepseek.com 获取
DASHSCOPE_API_KEY=sk-your-dashscope-key  # 从 bailian.console.aliyun.com 获取
```

### 4. 启动

```bash
uv run python src/memory_bridge/host_manager.py
```

### 5. 验证

```bash
curl http://localhost:8000/health
# → {"status": "ok", "qdrant": "connected"}
```

## 从 Release 安装

从 [GitHub Releases](https://github.com/51193/MemoryBridge/releases) 下载 `memorybridge.pyz`。

### 1. 下载

```bash
mkdir -p /opt/memorybridge && cd /opt/memorybridge
wget https://github.com/51193/MemoryBridge/releases/latest/download/memorybridge.pyz
```

### 2. 初始化 + 配置

```bash
python3 memorybridge.pyz --setup  # 自动下载 Qdrant + 创建 .env 模板
# 编辑 .env 填入 API Key
```

### 3. 启动

```bash
python3 memorybridge.pyz
```

服务器只需 Python 3.11+，所有依赖由 `.pyz` 自带，Qdrant 由 `--setup` 自动下载。

## Token 认证

MemoryBridge 默认启用 Token 认证，所有请求（除 `/health` 外）需要携带有效 Token。

### 初始化 Token 系统

```bash
# 源码开发
uv run python -m memory_bridge.host_manager --init-token

# Release
python memorybridge.pyz --init-token
```

### 管理 Token

```bash
# 创建 Token
bash scripts/token_admin.sh create --label "my-agent"

# 列出所有 Token
bash scripts/token_admin.sh list

# 删除 Token
bash scripts/token_admin.sh delete <token>
```

### 使用 Token

请求时通过 `Authorization: Bearer` 头携带 Token：

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer a1b2c3d4e5f6..." \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-pro","messages":[...],"agent_id":"agent-1","agent_session_id":"sess-1"}'
```

Token 管理系统独立于向量数据库，数据存储在 `data/tokens.db`。首次部署后必须先执行 `--init-token` 创建初始 Token，否则所有请求将被拒绝。

## 使用方式

### 基本请求

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "你好，我喜欢用 Python 写代码"}],
    "agent_id": "my-agent-1"
  }'
```

### 带会话的多轮对话

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "我刚才说我喜欢什么语言？"}],
    "agent_id": "my-agent-1",
    "agent_session_id": "sess-001"
  }'
```

### 启用思考模式

DeepSeek 模型支持在输出最终回答之前先生成思维链，提升推理准确性：

```json
{
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "9.11 和 9.8 谁更大？"}],
    "agent_id": "my-agent-1",
    "thinking_enabled": true,
    "reasoning_effort": "high"
}
```

思维链内容通过响应的 `reasoning_content` 字段返回。

### 流式输出

```json
{
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "讲个笑话"}],
    "agent_id": "my-agent-1",
    "stream": true
}
```

### 暂不启用记忆

```json
{
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "你好"}],
    "agent_id": "my-agent-1",
    "memory_enabled": false
}
```

## 请求参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | `string` | 是 | - | 模型名称，当前仅支持 `deepseek-chat` |
| `messages` | `array` | 是 | - | 对话消息数组，至少 1 条 |
| `messages[].role` | `string` | 是 | - | `system` / `user` / `assistant` / `tool` |
| `messages[].content` | `string` | 是 | - | 消息内容 |
| `temperature` | `float` | 否 | `0.7` | 采样温度，思考模式下不生效 |
| `max_tokens` | `int` | 否 | - | 最大输出 token 数 |
| `stream` | `bool` | 否 | `false` | 是否流式输出 (SSE) |
| `thinking_enabled` | `bool` | 否 | `false` | 启用思考模式 |
| `reasoning_effort` | `string` | 否 | - | 思考强度，`high` 或 `max` |
| `agent_id` | `string` | 是 | - | Agent 标识，用于记忆隔离 |
| `agent_session_id` | `string` | 否 | 自动生成 | 会话标识，用于多轮对话历史 |
| `memory_enabled` | `bool` | 否 | `true` | 是否检索历史记忆 |
| `memory_limit` | `int` | 否 | `5` | 检索记忆数量 (1-20) |

## 错误码

| HTTP 状态 | 场景 | 策略 |
|-----------|------|------|
| 422 | `agent_id` 缺失、`messages` 为空 | 拒绝请求 |
| 500 | 记忆检索失败 | 拒绝请求，不降级 |
| 502 | LLM Provider 调用失败或未注册 | 拒绝请求，不重试 |

## 开发

```bash
uv sync --extra dev      # 安装开发依赖
uv run mypy src/          # 类型检查
uv run ruff check src/    # Lint
uv run pytest -v          # 运行测试
```

## 项目结构

```
src/memory_bridge/
├── main.py              # FastAPI 应用
├── host_manager.py      # 进程管理器
├── config.py            # 配置
├── api/                 # 路由 + 依赖注入
├── core/                # 记忆 / 会话 / 上下文
├── providers/           # LLM Provider 适配器
└── models/              # 请求 / 响应模型
```

## 许可

Apache 2.0
