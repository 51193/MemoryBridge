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

`--init` 会自动下载 Qdrant 二进制、创建数据目录、生成 .env 模板、创建 API Token：

```bash
uv run python src/memory_bridge/host_manager.py --init
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
python3 memorybridge.pyz --init  # 自动下载 Qdrant + 创建 .env 模板 + 初始化 token
# 编辑 .env 填入 API Key
```

### 3. 启动

```bash
python3 memorybridge.pyz
```

服务器只需 Python 3.11+，所有依赖由 `.pyz` 自带，Qdrant 由 `--init` 自动下载。

## Token 认证

MemoryBridge 默认启用 Token 认证，所有请求（除 `/health` 外）需要携带有效 Token。

### 初始化 Token 系统

`--init` 自动创建初始 Token，如需单独生成新 Token：

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
  -d '{"messages":[...],"agent_id":"agent-1","agent_session_id":"sess-1"}'
```

Token 管理系统独立于向量数据库，数据存储在 `data/tokens.db`。首次部署后 `--init` 自动创建初始 Token，否则所有请求将被拒绝。

## 使用方式

### Session 管理

MemoryBridge 的会话历史存储在内存中，进程重启后丢失。通过 Session 导出/导入 API 可实现持久化恢复：

```bash
# 创建会话（可携带初始消息）
curl -X POST http://localhost:8000/v1/sessions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"agent-1","agent_session_id":"sess-001","initial_messages":[{"role":"user","content":"你好"}]}'

# 导出会话消息（user/assistant/tool，不含 system）
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/v1/sessions/agent-1/sess-001

# 进程重启后，用导出的消息恢复会话
curl -X POST http://localhost:8000/v1/sessions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"agent-1","agent_session_id":"sess-001","initial_messages":[...]}'
```

> **注意**：`system` 消息不存入会话历史。系统提示词每轮由 MemoryBridge 动态构建（注入长期记忆），外部可通过 `POST /v1/chat/completions` 的 `messages[0].role="system"` 传入自定义系统提示词。

### 基本请求

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好，我喜欢用 Python 写代码"}],
    "agent_id": "my-agent-1"
  }'
```

### 带会话的多轮对话

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "我刚才说我喜欢什么语言？"}],
    "agent_id": "my-agent-1",
    "agent_session_id": "sess-001"
  }'
```

### 启用思考模式

DeepSeek 模型支持在输出最终回答之前先生成思维链，提升推理准确性。通过环境变量配置思考模式：

```bash
# .env
DEEPSEEK_THINKING_ENABLED=true
DEEPSEEK_REASONING_EFFORT=high
```

思维链内容通过响应的 `reasoning_content` 字段返回。

### 流式输出

```json
{
    "messages": [{"role": "user", "content": "讲个笑话"}],
    "agent_id": "my-agent-1",
    "stream": true
}
```

### 暂不启用记忆

```json
{
    "messages": [{"role": "user", "content": "你好"}],
    "agent_id": "my-agent-1",
    "memory_enabled": false
}
```

## 请求参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `messages` | `array` | 是 | - | 对话消息数组，至少 1 条 |
| `messages[].role` | `string` | 是 | - | `system` / `user` / `assistant` / `tool` |
| `messages[].content` | `string` | 是 | - | 消息内容 |
| `temperature` | `float` | 否 | `0.7` | 采样温度，思考模式下不生效 |
| `max_tokens` | `int` | 否 | - | 最大输出 token 数 |
| `stream` | `bool` | 否 | `false` | 是否流式输出 (SSE) |
| `agent_id` | `string` | 是 | - | Agent 标识，用于记忆隔离 |
| `agent_session_id` | `string` | **是** | - | 会话标识，用于多轮对话历史 |
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
