# MemoryBridge 系统架构文档

## 一、系统定位

MemoryBridge 是 Agent 与 LLM Provider（大模型供应商）之间的中间件层。它不直接与用户对话，而是作为透明代理，拦截 Agent 发向 LLM 的请求，在转发前后注入记忆操作，从而为 Agent 提供**跨会话的长期记忆能力**。

```
Agent ──(OpenAI API)──→ MemoryBridge ──(OpenAI API)──→ DeepSeek
                         │   ↑
                         │   ├── 检索记忆 (读)
                         │   └── 存储记忆 (写)
                         │
                     ┌───┴────┐
                     │  Qdrant  │  向量数据库
                     └───┬────┘
                         │
                     Mem0 (记忆引擎)
                         提取事实 → 向量化 → 索引
```

---

## 二、进程模型

MemoryBridge 采用**双进程 + 本地网络通信**的架构：

```
Host Manager (Python)
  │
  ├─→ 子进程 1: Qdrant (Rust 二进制)
  │    地址: localhost:6333 (HTTP REST)
  │    数据: ./data/qdrant/
  │    作用: 向量存储与相似度检索
  │
  └─→ 子进程 2: MemoryBridge (Python FastAPI + Uvicorn)
       地址: localhost:8000
       作用: 请求代理 + 记忆管理
```

### 设计原理

**为什么一个服务一个进程？**

- **故障隔离**：Qdrant 崩溃不影响 MemoryBridge 接受请求（请求会因记忆不可用而返回 500，但进程不崩溃）
- **替换灵活**：将来可无缝更换向量数据库实现，只需改 Qdrant 进程的连接地址，MemoryBridge 代码无需修改
- **部署演进**：当前是 localhost 通信，未来可拆到不同机器上，只需改 IP 配置
- **启动顺序可控**：Host Manager 确保 Qdrant 先启动并通过健康检查后，才启动 MemoryBridge

**为什么用 Qdrant 二进制而不是 Docker？**

Docker 引入额外的守护进程、镜像管理和网络命名空间开销。Qdrant 作为单一 Rust 静态编译二进制（约 50MB），下载后直接 `./bin/qdrant` 即可运行，与 Python 进程生命周期完全一致，无需容器编排。

---

## 三、请求处理链路

以一次 `POST /v1/chat/completions` 请求为例，完整链路如下：

```
请求到达
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 1: 请求校验 (Pydantic Validation)                       │
│                                                             │
│ FastAPI 自动将 JSON body 解析为 ChatRequest Pydantic 模型     │
│                                                             │
│ 校验规则：                                                    │
│   - agent_id: 必填，不能为空字符串 → 否则 422                 │
│   - messages: 至少 1 条 → 否则 422                           │
│   - message.role: 只能是 system/user/assistant/tool → 否则 422│
│   - memory_limit: 必须在 1-20 之间 → 否则 422                │
│                                                             │
│ 校验失败 → 422 立即返回，不进入后续流程                        │
└──────────────────┬──────────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 2: Provider 路由 (ProviderRegistry)                     │
│                                                             │
│ 根据 request.model 查表获取对应的 LLM Provider 实例          │
│                                                             │
│ 注册表（应用启动时初始化）：                                   │
│   "deepseek-chat" → DeepSeekProvider                        │
│                                                             │
│ 找不到 → ProviderNotFoundError → 502 Bad Gateway             │
└──────────────────┬──────────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 3: 会话管理 (SessionStore)                               │
│                                                             │
│ agent_session_id 必填（ChatRequest 强制 min_length=1）        │
│                                                             │
│ 从 SessionStore 获取当前会话的消息历史：                       │
│   key = (agent_id, session_id)                              │
│   value = deque of messages (maxlen=50)                     │
│                                                             │
│ system 消息不进入 session：仅 user/assistant/tool 被存储        │
│ system prompt 由 ContextBuilder 每轮构建                       │
│ SessionStore 是纯内存结构，进程重启后丢失                      │
└──────────────────┬──────────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 4: 记忆检索 (Mem0.search)                               │
│                                                             │
│ 如果 memory_enabled = True：                                 │
│   - 提取 messages[-1].content 作为搜索 query                 │
│   - user_id = agent_id（实现 Agent 间记忆隔离）               │
│   - 检索 top_k = memory_limit 条最相关记忆                    │
│                                                             │
│ 检索失败 → MemorySearchError → 500 Internal Server Error     │
│ ⚠ 注意：不降级，不跳过。这是硬性约束                           │
│                                                             │
│ 如果 memory_enabled = False：                                │
│   跳过检索，不注入记忆                                        │
└──────────────────┬──────────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 5: 上下文注入 (ContextBuilder)                           │
│                                                             │
│ 将检索到的记忆拼接到 System Prompt 之前：                      │
│                                                             │
│   [相关历史记忆]                                              │
│   - 用户喜欢 Python                                          │
│   - 用户偏好使用 vim 编辑器                                   │
│                                                             │
│   [当前对话]                                                 │
│   你是一个有帮助的助手。                                       │
│                                                             │
│ 注入策略：                                                    │
│   - 已有 system message → 在原 content 前拼接记忆             │
│   - 无 system message → 在 messages 最前面插入新的 system 消息 │
│                                                             │
│ 原始 messages 列表不会被修改（不可变性）                        │
└──────────────────┬──────────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 6: LLM Provider 调用 (DeepSeekProvider)                 │
│                                                             │
│ 将增强后的消息转发给 DeepSeek API：                           │
│                                                             │
│ 非流式 (stream=false)：                                      │
│   POST https://api.deepseek.com/v1/chat/completions         │
│   → 等待完整响应 → 解析为 ChatResponse                        │
│                                                             │
│ 流式 (stream=true)：                                         │
│   POST ... (with stream=true, Accept: text/event-stream)    │
│   → async for line in response: parse SSE → yield chunk     │
│   → 包装为 StreamingResponse 返回给 Agent                    │
│                                                             │
│ Provider 调用失败 → 502 Bad Gateway                          │
│ ⚠ 注意：不重试，不切换 provider。这是硬性约束                  │
└──────────────────┬──────────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 7: 异步记忆存储 (BackgroundTasks)                        │
│                                                             │
│ 非流式：在返回响应前提交 BackgroundTask                        │
│ 流式：收集完整响应文本后直接调用 _store_memory()                 │
│                                                             │
│ 存储内容：                                                    │
│   - 原始请求消息 + 助手回复拼接为完整对话                       │
│   - 过滤 system 角色消息后存入 SessionStore                    │
│   - Mem0.add(messages, user_id=agent_id)                    │
│                                                             │
│ Mem0 内部流程：                                               │
│   对话消息 → Mem0 LLM 提取事实 → Embedding 向量化              │
│   → 写入 Qdrant 向量索引                                     │
│                                                             │
│ 存储失败 → 记录日志，不影响响应（fire-and-forget）              │
└──────────────────┬──────────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 8: 返回响应给 Agent                                     │
│                                                             │
│ 非流式：返回 ChatResponse JSON                               │
│ 流式：  返回 StreamingResponse (SSE)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、记忆子系统

### 4.1 架构概览

记忆子系统由三层组成：

```
┌─────────────────────────────────────┐
│  MemoryManager (自定义封装)          │  ← 提供给业务层的接口
│  search() / add()                   │
└──────────────┬──────────────────────┘
               │ 委托
┌──────────────▼──────────────────────┐
│  Mem0 (记忆引擎，npm/mem0ai)         │  ← 事实提取 + 向量化 + 索引
│  - ADD: 从对话中提取事实，向量化存储   │
│  - SEARCH: 语义搜索 + BM25 + 实体链接  │
└──────┬───────────────┬──────────────┘
       │               │
  ┌────▼─────┐   ┌─────▼──────┐
  │ LLM      │   │ Embedder   │
  │ DeepSeek │   │ DashScope  │
  │ 提取事实  │   │ text-emb   │
  │ (prompt) │   │ -v4 (1024d)│
  └──────────┘   └─────┬──────┘
                       │ 写入
               ┌───────▼──────┐
               │   Qdrant     │
               │   向量数据库   │
               └──────────────┘
```

### 4.2 Mem0 原理

Mem0 (v2.0.2) 是一个专为 AI Agent 设计的长期记忆库。其核心算法：

**ADD（存储）流程：**

1. 接收完整对话消息
2. 内部调用 LLM（DeepSeek）从对话中**提取结构化事实**（如"用户喜欢 Python"、"用户的工作是后端工程师"）
3. 将每条事实通过 Embedder（DashScope text-embedding-v4）**转换为 1024 维向量**
4. 向量 + 元数据写入 Qdrant

v2 版本采用 **Single-pass ADD-only** 策略：记忆只新增，不更新不删除。旧事实不会被覆盖，而是累积。这避免了因单一错误更新丢失全部历史的问题。

**SEARCH（检索）流程：**

1. 接收查询文本
2. Embedder 将查询转换为向量
3. 在 Qdrant 中执行**多信号融合检索**：
   - 语义相似度（向量余弦距离）
   - BM25 关键词匹配（全文倒排索引）
   - 实体链接增强（entity linking boost）
4. 返回 top_k 条最相关的记忆，附带相似度分数

### 4.3 Qdrant

Qdrant 是 Rust 实现的高性能向量搜索引擎。在本项目中以独立二进制进程运行。

关键配置：

| 参数 | 值 | 说明 |
|------|-----|------|
| Collection | `memory_bridge` | 由 Mem0 自动创建 |
| 向量维度 | 1024 | 与 text-embedding-v4 输出匹配 |
| 距离度量 | Cosine | 余弦相似度 |
| 存储 | `on_disk=True` | 数据持久化到 `./data/qdrant/` |

每条向量数据（Point）的 Payload 包含：

```json
{
    "user_id": "agent-1",
    "memory": "用户喜欢使用 Python 进行数据分析",
    "created_at": "2026-05-12T10:00:00Z"
}
```

### 4.4 阿里云 DashScope Embedding

目前 Embedding 模型采用阿里云百炼的 `text-embedding-v4`，通过 DashScope 提供的 OpenAI 兼容端点调用：

```
POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
Authorization: Bearer $DASHSCOPE_API_KEY
Content-Type: application/json

{
    "model": "text-embedding-v4",
    "input": "需要转向量的文本"
}
```

Mem0 内部使用 OpenAI embedder provider，通过配置 `openai_base_url` 指向 DashScope 兼容端点实现无缝切换。

**扩展性设计**：Embedder 配置中已预留 Ollama 本地模型的配置模板（注释形式），未来如果希望零外部依赖运行，只需切换 provider 为 `ollama` 并配置本地地址即可，无需修改任何业务代码。

### 4.5 记忆隔离

记忆中按 `user_id = agent_id` 进行多租户隔离。每个 Agent ID 拥有独立的记忆空间，互不干扰。

---

## 五、会话管理

### 5.1 SessionStore 设计

SessionStore 是纯内存数据结构，核心实现：

```python
_sessions: dict[tuple[str, str], deque[dict]] = {}
# key:   (agent_id, session_id)
# value: 固定容量双端队列（maxlen=50）
```

**特性：**

- **双重隔离**：`(agent_id, session_id)` 复合键确保不同 Agent、同一 Agent 的不同会话之间完全隔离
- **固定容量**：最多保留 50 条消息，超出时自动淘汰最早的消息
- **角色过滤**：`system` 角色消息在存储时自动过滤，仅 `user`、`assistant`、`tool` 进入 session deque。系统提示词由 ContextBuilder 每轮构建，不持久化到会话历史中
- **进程生命周期**：进程重启后所有会话数据丢失——长期记忆由 Mem0 承载，会话历史可通过 `GET /v1/sessions/` 导出并在重启后通过 `POST /v1/sessions` 恢复
- **零依赖**：不依赖 Redis、数据库等外部存储

### 5.2 会话生命周期

```
Agent 首次请求
  → 通过 POST /v1/sessions 创建会话（可携带 initial_messages）
  → auto-generated session_id (12位hex UUID) 或指定

后续请求 (携带相同 session_id)
  → 追加 user/assistant 消息到已有会话（system 被过滤）
  → 检索该会话的近期上下文

Session 导出 (GET /v1/sessions/{agent_id}/{session_id})
  → 返回完整会话消息列表（user/assistant/tool，不含 system）
  → Agent 可将消息持久化到外部存储

Session 恢复 (POST /v1/sessions)
  → 进程重启后，Agent 将持久化的消息作为 initial_messages 传入
  → 恢复完整的多轮对话上下文（session 不存在则创建，已存在则 409）

Agent 显式清除或进程重启
  → 会话数据被清除
  → Mem0 中长期记忆不受影响
```

---

## 六、上下文注入

### 6.1 完整消息组装流程

MemoryBridge 按以下顺序组装最终发给 LLM 的消息列表：

```
1. 记忆注入 (ContextBuilder)
   → 检索到的长期记忆拼接到 system prompt 前端
   → 如当前消息无 system → 创建新的 system 消息
   → 如当前消息有 system → 在原 content 前拼接记忆模板

2. 会话历史注入 (_inject_history)
   → system 消息（含记忆）放在最前面
   → 其后插入 session 存储的全部历史消息（无 system，纯对话）
   → 最后放当前 request 中的非 system 消息

最终结构：
  [system: 记忆模板 + 自定义提示词]
  [user/assistant: 历史消息 1...N]    ← 从 SessionStore 注入
  [user/assistant: 当前请求消息...]   ← 原始请求（system 已移除）
```

### 6.2 注入模板

```
[相关历史记忆]
- 记忆条目 1
- 记忆条目 2
...

[当前对话]
{原始 system prompt 或留空}
```

### 6.3 为什么注入 System Prompt？

- System Prompt 是 LLM 处理对话时最先看到的上下文，优先级最高
- 将记忆放在这里确保 LLM 在生成回复时能参考这些信息
- 不影响 user/assistant 角色的消息结构，保持对话格式一致

---

## 七、Provider 架构

### 7.1 抽象层设计

```
AbstractLLMProvider (ABC)
  │
  ├── chat(request) → ChatResponse         (非流式)
  └── chat_stream(request) → AsyncIterator (流式)
      │
      └── DeepSeekProvider (当前唯一实现)
```

### 7.2 ProviderRegistry

```python
class ProviderRegistry:
    _providers: dict[str, AbstractLLMProvider] = {}

    register(model_name, provider)   # 注册
    get(model_name) → provider       # 查找
```

按 `request.model` 字段路由到对应的 Provider 实例。当前只注册了 `deepseek-chat`。

**扩展性**：新增 OpenAI、Anthropic 等 Provider 只需：
1. 实现 `AbstractLLMProvider` 子类
2. 在应用启动时调用 `ProviderRegistry.register("gpt-5-mini", OpenAIProvider(...))`

### 7.3 DeepSeekProvider 内部实现

```
_client: httpx.AsyncClient (连接池复用)
  │
  ├── chat(request)
  │   POST /v1/chat/completions  (stream=false)
  │   → 解析 JSON → ChatResponse
  │
  └── chat_stream(request)
      POST /v1/chat/completions  (stream=true)
      → async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                yield StreamChunk(...)
```

**错误处理**：
- HTTP 非 200 → ProviderError → 502
- 连接错误 (ConnectError/Timeout) → ProviderError → 502
- DeepSeek SSE 流中断 → ProviderError → 502

---

## 八、错误处理哲学

MemoryBridge 采用**强硬约束**错误策略。与常见 Web 服务的"尽量容错"不同，本系统在设计上不允许任何环节降级。

| 错误场景 | HTTP 状态码 | 策略 |
|----------|-------------|------|
| `agent_id` 缺失/为空 | 422 | 拒绝请求，不设默认值 |
| `messages` 为空 | 422 | 拒绝请求 |
| 记忆检索失败 (Mem0.search 异常) | 500 | 拒绝请求，不跳过记忆步骤 |
| Provider 未注册 | 502 | 拒绝请求，不退回到默认 provider |
| LLM Provider 调用失败 | 502 | 拒绝请求，不重试不切换 |
| 记忆存储失败 (Mem0.add 异常) | *不阻断* | 仅记录日志，因为此时响应已生成 |

### 为什么记忆检索失败要阻断请求？

Agent 调用 MemoryBridge 的前提就是需要记忆增强的上下文。如果检索步骤失败，直接用无记忆的原始请求转发给 LLM，Agent 收到的回复质量会显著下降，且**Agent 不会知道这次回复缺少上下文**。与其给出一个"看似正常但缺失上下文"的回复，不如明确告诉 Agent 记忆服务不可用，由 Agent 决定如何处理。

### 为什么记忆存储失败不阻断请求？

存储发生在响应已生成之后。此时 LLM 的回复已经产生，阻断请求没有任何意义——Agent 已经收到了有效回复。将错误记录到日志，运维人员可以在事后排查。

---

## 九、配置体系

所有可变参数通过环境变量注入，代码内不允许硬编码。

```
.env
  ├── DEEPSEEK_API_KEY        ← DeepSeek 官方 API Key
  ├── DEEPSEEK_BASE_URL       ← 默认 https://api.deepseek.com
  ├── DEEPSEEK_MODEL          ← 默认 deepseek-chat
  │
  ├── DASHSCOPE_API_KEY       ← 阿里云百炼 API Key
  ├── EMBEDDING_MODEL         ← 默认 text-embedding-v4
  ├── EMBEDDING_DIMS          ← 默认 1024
  │
  ├── QDRANT_HOST             ← 默认 localhost
  ├── QDRANT_PORT             ← 默认 6333
  │
  ├── MEMORY_BRIDGE_HOST      ← 默认 0.0.0.0
  ├── MEMORY_BRIDGE_PORT      ← 默认 8000
  └── SESSION_MAX_HISTORY     ← 默认 50
```

pydantic-settings 自动从 `.env` 文件和系统环境变量加载，支持默认值。

应用启动时调用 `settings.validate_secrets()` 检查 API Key 是否已配置，缺失则退出并提示。

---

## 十、目录结构

```
MemoryBridge/
├── src/memory_bridge/
│   ├── main.py              # FastAPI 应用工厂 + lifespan
│   ├── host_manager.py      # Host 进程管理器
│   ├── config.py            # 配置
│   ├── exceptions.py        # 自定义异常
│   │
│   ├── api/
│   │   ├── router.py        # 路由: /health, /v1/chat/completions
│   │   └── dependencies.py  # FastAPI 依赖注入
│   │
│   ├── core/
│   │   ├── memory.py        # MemoryManager (Mem0 封装)
│   │   ├── session.py       # SessionStore (纯内存会话)
│   │   ├── context.py       # ContextBuilder (上下文注入)
│   │   ├── tokens.py        # TokenStore (SQLite Token 管理)
│   │   └── prompts.py       # 自定义记忆提取提示词加载
│   │
│   ├── providers/
│   │   ├── base.py          # AbstractLLMProvider
│   │   ├── deepseek.py      # DeepSeekProvider
│   │   └── registry.py      # ProviderRegistry
│   │
│   ├── api/
│   │   ├── router.py        # 路由: /health, /v1/sessions, /v1/chat/completions
│   │   ├── dependencies.py  # FastAPI 依赖注入
│   │   └── middleware.py    # TokenAuthMiddleware
│   │
│   └── models/
│       ├── request.py       # ChatRequest, Message
│       └── response.py      # ChatResponse, StreamChunk
│
├── tests/                   # 102 个测试用例
├── docs/                    # 本文档 + 开发计划
├── pyproject.toml           # 项目元数据 + 依赖 + 工具配置
├── .env.example             # 环境变量模板
├── AGENTS.md                # 开发准则
└── DEVELOPMENT_PLAN.md      # 开发计划
```

---

## 十一、部署流程

`host_manager.py` 是统一的入口，源码开发和 Release 部署走完全相同的代码路径。

### 源码开发

```bash
uv sync
uv run python src/memory_bridge/host_manager.py --setup  # 首次：下载 Qdrant + 创建 .env
uv run python src/memory_bridge/host_manager.py           # 启动
```

### Release 部署

```bash
python3 memorybridge.pyz --setup  # 首次：自动下载 Qdrant + 创建 .env
python3 memorybridge.pyz           # 启动
```

`--setup` 会自动检测平台 (Linux x86_64/aarch64, macOS x86_64/arm64) 并下载对应 Qdrant 二进制。
