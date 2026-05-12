# MemoryBridge 开发计划

## 一、项目概述

MemoryBridge 是一个 Agent 与大模型之间的中间件，负责：
1. 接收 Agent 的 LLM API 请求（OpenAI 兼容协议 + 扩展字段）
2. 检索相关历史记忆并注入到请求上下文
3. 将增强后的请求转发到 DeepSeek 官方 API
4. 异步存储对话记忆以供后续检索

### 技术栈

| 组件 | 选型 | 版本 |
|------|------|------|
| 语言 | Python | 3.11+ |
| Web 框架 | FastAPI + Uvicorn | latest |
| 记忆层 | Mem0 (library) | >=2.0.2 |
| 向量存储 | Qdrant (二进制，非 Docker) | latest stable |
| LLM Provider | DeepSeek 官方 API | - |
| Embedding | 阿里云 text-embedding-v4 (DashScope) | - |
| 异步 HTTP | httpx | >=0.28 |
| 配置 | pydantic-settings | >=2.7 |
| 进程管理 | 自研 Python Host Manager | - |

---

## 二、进程模型

```
┌──────────────────────────────────────────────────────────┐
│                    Host Manager (Python)                   │
│                    host_manager.py                         │
│                                                           │
│  职责：                                                    │
│  - 解析配置，初始化环境                                     │
│  - 启动/停止/监控所有子进程                                  │
│  - 信号处理（SIGINT/SIGTERM → 优雅关闭）                      │
│  - 健康检查                                                │
│                                                           │
│  ┌──────────────────────┐    ┌──────────────────────────┐ │
│  │   Qdrant 进程          │    │   MemoryBridge 进程       │ │
│  │   (Rust 二进制)        │    │   (Python FastAPI)        │ │
│  │                       │    │                           │ │
│  │   HTTP :6333          │◄──►│   HTTP :8000              │ │
│  │   gRPC :6334          │    │                           │ │
│  │   Data: ./data/qdrant │    │   Mem0 → Qdrant(localhost)│ │
│  │                       │    │   httpx → DeepSeek API    │ │
│  │                       │    │   httpx → DashScope Embed │ │
│  └──────────────────────┘    └──────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

进程间通信：全部通过 localhost TCP 通信，地址和端口可配置，天然具备替换任意组件的灵活度。

---

## 三、核心数据流

```
 Agent 请求 (POST /v1/chat/completions)
   │
   │  {
   │    "model": "deepseek-chat",
   │    "messages": [...],
   │    "stream": false,
   │    "agent_id": "my-agent",           ← 扩展字段
   │    "agent_session_id": "sess-001",   ← 扩展字段
   │    "memory_enabled": true,           ← 扩展字段
   │    "memory_limit": 5                 ← 扩展字段
   │  }
   ▼
┌────────────────────────────────────────────┐
│  1. 请求校验 (Pydantic)                     │
│     - agent_id 必填，否则 422               │
│     - messages 非空，否则 422               │
└────────────┬───────────────────────────────┘
             ▼
┌────────────────────────────────────────────┐
│  2. 检索记忆                                │
│     Mem0.search(                           │
│       query=messages[-1].content,          │
│       filters={"user_id": agent_id},       │
│       top_k=memory_limit                   │
│     )                                      │
│     失败 → 500, 不降级                      │
└────────────┬───────────────────────────────┘
             ▼ memories: list[str]
┌────────────────────────────────────────────┐
│  3. 上下文注入                              │
│     将检索到的记忆拼接到 system message 之前    │
│     构造增强后的 messages 数组                │
└────────────┬───────────────────────────────┘
             ▼ enriched_messages
┌────────────────────────────────────────────┐
│  4. 转发到 DeepSeek                        │
│     httpx → POST api.deepseek.com/v1/      │
│            chat/completions                │
│     支持 stream=True 时的 SSE 透传           │
│     失败 → 502, 不重试                      │
└────────────┬───────────────────────────────┘
             ▼ deepseek_response
┌────────────────────────────────────────────┐
│  5. 异步存储记忆 (fire-and-forget)           │
│     BackgroundTasks:                       │
│     Mem0.add(                             │
│       messages=[完整对话上下文],              │
│       user_id=agent_id,                    │
│     )                                      │
│     存储失败 → log error, 不影响响应           │
└────────────┬───────────────────────────────┘
             ▼
┌────────────────────────────────────────────┐
│  6. 更新会话历史                             │
│     SessionStore.append(                   │
│       agent_id, session_id, messages       │
│     )                                      │
└────────────┬───────────────────────────────┘
             ▼
┌────────────────────────────────────────────┐
│  7. 返回响应                                │
│     非流式: ChatResponse (JSON)              │
│     流式:   StreamingResponse (SSE)          │
└────────────────────────────────────────────┘
```

---

## 四、关键接口设计

### 4.1 API 层 — 对外接口

#### `POST /v1/chat/completions`

**请求体 — ChatRequest（OpenAI 兼容 + 扩展）**

```python
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

class ChatRequest(BaseModel):
    # ===== OpenAI 标准字段 =====
    model: str                                    # 必填
    messages: list[Message]                       # 必填，min_length=1
    temperature: float = 0.7                      # 可选
    max_tokens: int | None = None                 # 可选
    top_p: float = 1.0                            # 可选
    stream: bool = False                          # 可选
    stop: list[str] | None = None                 # 可选

    # ===== MemoryBridge 扩展字段 =====
    agent_id: str                                 # 必填, min_length=1
    agent_session_id: str | None = None            # 可选，不填自动生成 UUID
    memory_enabled: bool = True                   # 是否启用记忆检索
    memory_limit: int = Field(default=5, ge=1, le=20)
```

**响应体 — ChatResponse（OpenAI 兼容）**

```python
class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str | None = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None
```

**响应体 — 流式（SSE）**

```python
class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None

class StreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: str | None = None

class StreamChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]
```

#### `GET /health`

```json
{
    "status": "ok",
    "qdrant": "connected",
    "mem0": "ready"
}
```

### 4.2 Provider 抽象 — 对内接口

```python
class AbstractLLMProvider(ABC):
    """LLM Provider 抽象基类"""

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """非流式聊天完成"""
        ...

    @abstractmethod
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """流式聊天完成，返回异步迭代器"""
        ...
```

### 4.3 Memory 层接口

```python
class MemoryManager:
    """Mem0 封装，管理记忆的存储和检索"""

    def __init__(self, config: dict) -> None:
        self._memory: Memory = Memory.from_config(config)

    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        检索与 query 最相关的记忆。

        返回:
            [
                {"id": "uuid", "memory": "用户喜欢 Python", "score": 0.95},
                ...
            ]

        失败时抛出 MemorySearchError，调用方不降级。
        """
        ...

    def add(
        self,
        messages: list[dict],
        user_id: str,
        metadata: dict | None = None,
    ) -> None:
        """
        存储对话记忆。
        Mem0 v2 会自动从 messages 中提取事实并存储到 Qdrant。

        失败时记录日志，不阻塞调用方（fire-and-forget）。
        """
        ...
```

### 4.4 Session 层接口

```python
class SessionStore:
    """
    纯内存会话管理。

    会话 Key = (agent_id, session_id)
    会话 Value = deque[maxlen=max_history] of messages
    进程重启后全部丢失。
    """

    def __init__(self, max_history: int = 50) -> None: ...

    def get_or_create(self, agent_id: str, session_id: str) -> list[dict]:
        """获取会话消息历史，不存在则返回空列表（会话在此创建）"""
        ...

    def append(
        self,
        agent_id: str,
        session_id: str,
        messages: list[dict],
    ) -> None:
        """追加消息到会话历史"""
        ...

    def clear(self, agent_id: str, session_id: str) -> None:
        """清空指定会话"""
        ...
```

### 4.5 Context 构建接口

```python
class ContextBuilder:
    """
    将检索到的记忆注入到 System Prompt 中。
    """

    def build(
        self,
        messages: list[dict],
        memories: list[dict],
    ) -> list[dict]:
        """
        构造增强后的 messages 数组。

        策略:
        1. 如果 messages 中已有 system role，在其 content 之前拼接记忆
        2. 如果没有 system role，在 messages 最前面插入一条

        拼接模板:
            [相关历史记忆]
            - 记忆1
            - 记忆2
            ...

            [当前对话]
            {原始 system prompt 或留空}
        """
        ...
```

---

## 五、数据设计

### 5.1 Qdrant 存储

由 Mem0 全权管理，不直接操作。关键元数据：

| 字段 | 值 |
|------|-----|
| Collection | `memory_bridge` |
| Vector Dims | `1024` (text-embedding-v4) |
| Distance | Cosine |
| 每条 Point Payload | `{"user_id": str, "memory": str, "created_at": str, ...}` |

### 5.2 Mem0 历史库

Mem0 使用 SQLite 存储记忆提取历史，路径：`./data/mem0_history.db`。

### 5.3 会话数据（内存）

```python
# 数据结构
_sessions: dict[tuple[str, str], deque[dict]]
# Key: (agent_id, session_id)
# Value: 固定长度的双端队列，最多 max_history 条消息

# 示例
_sessions[("agent-1", "sess-001")] = deque([
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "Python 怎么用？"},
], maxlen=50)
```

### 5.4 目录结构

```
data/
├── qdrant/              # Qdrant 持久化数据（二进制进程管理）
└── mem0_history.db      # Mem0 SQLite 历史库
```

---

## 六、模块设计

### 项目文件树

```
MemoryBridge/
├── DEVELOPMENT_PLAN.md          # 本文件
├── pyproject.toml               # 项目元数据 + 依赖
├── .env.example                 # 环境变量模板
│
├── src/
│   └── memory_bridge/
│       ├── __init__.py
│       │
│       ├── main.py              # FastAPI app 工厂 + 生命周期
│       │
│       ├── host_manager.py      # Host 进程管理器
│       │   - 启动 Qdrant 子进程
│       │   - 启动 MemoryBridge 子进程（uvicorn）
│       │   - 信号处理，优雅关闭
│       │   - 健康检查轮询
│       │
│       ├── config.py            # pydantic-settings 配置
│       │   class Settings:
│       │       # DeepSeek
│       │       deepseek_api_key: str
│       │       deepseek_base_url: str
│       │       deepseek_model: str
│       │       # DashScope
│       │       dashscope_api_key: str
│       │       embedding_model: str
│       │       embedding_dims: int
│       │       # Qdrant
│       │       qdrant_host: str
│       │       qdrant_port: int
│       │       # MemoryBridge
│       │       host: str
│       │       port: int
│       │       session_max_history: int
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── router.py        # POST /v1/chat/completions, GET /health
│       │   └── dependencies.py  # FastAPI Depends 注入
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── memory.py        # MemoryManager
│       │   ├── session.py       # SessionStore
│       │   └── context.py       # ContextBuilder
│       │
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py          # AbstractLLMProvider
│       │   ├── deepseek.py      # DeepSeekProvider
│       │   └── registry.py      # 按 model 名路由到 Provider
│       │
│       └── models/
│           ├── __init__.py
│           ├── request.py       # ChatRequest, Message
│           └── response.py      # ChatResponse, StreamChunk
│
└── tests/
    ├── __init__.py
    ├── conftest.py              # fixtures (MemoryManager mock, test client)
    ├── test_core/
    │   ├── test_memory.py       # MemoryManager 单元测试
    │   ├── test_session.py      # SessionStore 单元测试
    │   └── test_context.py      # ContextBuilder 单元测试
    └── test_api/
        └── test_router.py       # API 集成测试
```

### 模块职责明细

| 模块 | 职责 | 依赖 |
|------|------|------|
| `host_manager.py` | 进程生命周期管理、信号处理 | `subprocess`, `signal`, `httpx` |
| `main.py` | FastAPI app 创建、lifespan（初始化/清理 Mem0） | `config`, `api/router`, `core/memory` |
| `config.py` | 读取 .env / 环境变量，提供类型化配置 | `pydantic-settings` |
| `api/router.py` | 请求路由、校验、编排 Memory → Context → Provider | `core/`, `providers/`, `models/` |
| `core/memory.py` | Mem0 初始化和操作封装 | `mem0ai`, `config` |
| `core/session.py` | 纯内存会话存储 | 无外部依赖 |
| `core/context.py` | 记忆注入到 System Prompt | 无外部依赖 |
| `providers/deepseek.py` | DeepSeek API 调用（http + sse） | `httpx`, `models/` |
| `models/request.py` | 请求 Pydantic 模型 | `pydantic` |
| `models/response.py` | 响应 Pydantic 模型 | `pydantic` |

---

## 七、Mem0 配置设计

```python
MEM0_CONFIG = {
    "llm": {
        "provider": "deepseek",
        "config": {
            "model": settings.deepseek_model,          # "deepseek-chat"
            "api_key": settings.deepseek_api_key,
            "deepseek_base_url": settings.deepseek_base_url,  # "https://api.deepseek.com"
            "temperature": 0.2,
            "max_tokens": 2000,
        }
    },
    "embedder": {
        "provider": "openai",                          # 利用 OpenAI 兼容端点
        "config": {
            "model": settings.embedding_model,          # "text-embedding-v4"
            "api_key": settings.dashscope_api_key,
            "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "embedding_dims": settings.embedding_dims,  # 1024
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": settings.qdrant_host,               # "localhost"
            "port": settings.qdrant_port,               # 6333
            "collection_name": "memory_bridge",
            "embedding_model_dims": settings.embedding_dims,  # 1024
            "on_disk": True,                            # 持久化到磁盘
        }
    },
    "history_db_path": "./data/mem0_history.db",
}
```

### Embedding 扩展性预留

Embedder 配置中保留 `ollama` provider 的配置模板（注释形式），切换时只需修改配置字典的 `provider` 和 `config` 字段，无需改动任何代码逻辑。

```python
# 预留：Ollama 本地 Embedding 配置（暂不实现）
# "embedder": {
#     "provider": "ollama",
#     "config": {
#         "model": "nomic-embed-text",
#         "embedding_dims": 768,
#         "ollama_base_url": "http://localhost:11434",
#     }
# },
```

---

## 八、Host Manager 设计

### 启动流程

```
host_manager.py start
    │
    ├─ 1. 检查 Qdrant 二进制是否存在 ./bin/qdrant
    │     不存在 → 提示运行 host_manager.py --setup
    │
    ├─ 2. 启动 Qdrant 子进程
    │     subprocess.Popen(["./bin/qdrant", "--storage-snapshot", "./data/qdrant"])
    │     等待 localhost:6333 返回 200 确认启动完成
    │     超时(30s) → 退出并报错
    │
    ├─ 3. 启动 MemoryBridge 子进程
    │     subprocess.Popen([
    │         "uvicorn",
    │         "memory_bridge.main:app",
    │         "--host", settings.host,
    │         "--port", str(settings.port),
    │     ])
    │     等待 localhost:8000/health 返回 200 确认启动完成
    │
    └─ 4. 注册信号处理
         SIGINT/SIGTERM → 依次终止 MemoryBridge → Qdrant
         输出 "MemoryBridge stopped."
```

### 关闭流程

```
收到 SIGINT/SIGTERM
    │
    ├─ 1. 向 MemoryBridge 子进程发送 SIGTERM
    │     等待 10s
    │     未退出 → SIGKILL
    │
    └─ 2. 向 Qdrant 子进程发送 SIGTERM
           等待 10s
           未退出 → SIGKILL
```

---

## 九、Provider 注册机制

```python
# providers/registry.py

class ProviderRegistry:
    """按 model 名称路由到对应的 LLM Provider"""

    _providers: dict[str, AbstractLLMProvider] = {}

    @classmethod
    def register(cls, model: str, provider: AbstractLLMProvider) -> None:
        cls._providers[model] = provider

    @classmethod
    def get(cls, model: str) -> AbstractLLMProvider:
        if model in cls._providers:
            return cls._providers[model]
        raise ProviderNotFoundError(f"No provider for model: {model}")


# 应用启动时注册
ProviderRegistry.register("deepseek-chat", DeepSeekProvider(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
))
```

当前只注册 `deepseek-chat` 一个 model，后续新增 Provider 只需实现 `AbstractLLMProvider` 并注册即可。

---

## 十、开发原则

1. **约束强硬，不降级不回退**
   - `agent_id` 缺失 → 422（不是 default）
   - 记忆检索失败 → 500（不跳过记忆步骤继续）
   - LLM Provider 失败 → 502（不重试，不换 provider）
   - 任何异常都不应该被静默吞掉

2. **最少依赖**
   - 除核心功能必需的库（fastapi、mem0ai、httpx、pydantic-settings、qdrant-client）外不引入额外依赖
   - 不使用 ORM、不用 Redis、不用消息队列
   - 进程间通信只用 HTTP，不用 RPC 框架

3. **单文件单职责**
   - 每个模块文件不超过 200 行
   - 类和函数保持单一职责

4. **类型覆盖**
   - 所有公共函数和方法的参数/返回值带完整类型标注
   - pyproject.toml 中启用 mypy strict 模式

5. **配置外置**
   - 所有可变参数（API Key、端口、模型名）通过 env / .env 注入
   - 代码内不含任何硬编码的 secret 或 URL
   - `.env.example` 列出所有必须配置的变量，值位为空

6. **错误显式化**
   - 自定义异常类，不使用裸 `Exception`
   - 每个异常带明确的错误消息

---

## 十一、开发阶段

### Phase 1：项目骨架 (预计 1 次提交)

- [ ] 创建 `pyproject.toml`，声明依赖和项目元数据
- [ ] 创建 `.env.example`
- [ ] 实现 `config.py`（pydantic-settings）
- [ ] 实现 `models/request.py` 和 `models/response.py`
- [ ] 实现 `main.py`（FastAPI app 工厂 + 空路由骨架）
- [ ] 启动应用确认路由注册成功

**验证**: `uvicorn memory_bridge.main:app` 启动，`/health` 返回 200，`/docs` 可见 OpenAPI 文档

### Phase 2：Qdrant 数据库进程

- [ ] 编写 `host_manager.py`：进程管理器（启停 Qdrant + MemoryBridge），内联 Qdrant 下载逻辑（`--setup`）
- [ ] `--setup` 自动检测平台并下载 Qdrant 二进制到 `./bin/`

**验证**: `python host_manager.py --setup` 后 `python host_manager.py` 启动，`curl localhost:6333` 和 `curl localhost:8000/health` 均返回成功

### Phase 3：Mem0 记忆层

- [ ] 实现 `core/memory.py`（Mem0 初始化、search、add）
- [ ] 在 `main.py` lifespan 中初始化 Mem0
- [ ] 在 `/health` 中加入 Qdrant 连接检查

**验证**: 调用 `search()` 和 `add()` 确认 Qdrant 中有数据落盘

### Phase 4：会话与上下文

- [ ] 实现 `core/session.py`（SessionStore）
- [ ] 实现 `core/context.py`（ContextBuilder）
- [ ] 编写单元测试

**验证**: pytest 通过

### Phase 5：DeepSeek Provider + API 路由

- [ ] 实现 `providers/deepseek.py`（非流式 chat + 流式 chat_stream）
- [ ] 实现 `providers/registry.py`
- [ ] 实现 `api/router.py`（完整请求管线：校验 → 记忆检索 → 上下文注入 → LLM 转发 → 记忆存储）
- [ ] 在 `main.py` lifespan 中注册 Provider

**验证**: `curl -X POST localhost:8000/v1/chat/completions ...` 返回 DeepSeek 响应，确认记忆被存储

### Phase 6：流式支持 + 集成测试

- [ ] 实现流式 SSE 响应透传
- [ ] 编写 API 集成测试
- [ ] 完整的端到端手动测试

**验证**: 流式和非流式请求均正确返回，记忆检索和存储正常

---

## 十二、验收标准

### 12.1 单模块验收（Phase 3-4）

| 测试项 | 验收方法 |
|--------|----------|
| Config 正确加载 | 修改 `.env` 后重启，新值生效 |
| Mem0.search 返回结果 | 先 add 数据后 search，确认返回相关记忆 |
| Mem0.add 落盘 | 调用 add 后检查 `./data/qdrant/` 有数据 |
| SessionStore 基本操作 | append → get 返回追加的消息；clear 后 get 返回空 |
| ContextBuilder 注入 | 输入空 system prompt + 3条记忆，输出带记忆的前置 system message |

### 12.2 端到端验收（Phase 5-6）

| 场景 | 请求 | 预期结果 |
|------|------|----------|
| 正常非流式请求 | agent_id=test, memory_enabled=true | 返回 DeepSeek 响应，`/data/qdrant/` 有新增记忆 |
| 正常流式请求 | stream=true | SSE 事件流完整返回，记忆异步存储 |
| 缺少 agent_id | agent_id 不传 | 422 Unprocessable Entity |
| 空 messages | messages=[] | 422 |
| DeepSeek 不可达 | 故意写错 base_url | 502 Bad Gateway，不降级 |
| 重复对话上下文 | 同一 session 两次请求 | 第二次检索到第一次的记忆 |

### 12.3 进程管理验收

| 场景 | 预期行为 |
|------|----------|
| 正常启动 | `python host_manager.py` 先后拉起 Qdrant 和 MemoryBridge |
| Ctrl+C 终止 | 两个子进程依次优雅退出，无僵尸进程 |
| Qdrant 未安装 | 启动时提示运行 host_manager.py --setup |
| 端口冲突 | 启动时明确报错端口被占用 |

---

## 十三、风险与对策

| 风险 | 可能性 | 影响 | 对策 |
|------|--------|------|------|
| DashScope `/v1/embeddings` 路径与 OpenAI 有差异 | 中 | 高 | 若 Mem0 OpenAI embedder 调用失败，改为自定义 DashScopeEmbedder 类，直接 HTTP 调用百炼 Embedding API |
| text-embedding-v4 维度非 1024 | 低 | 中 | 启动时用一条简单文本测试 embedding，从返回结果推断实际维度，若不符则调整配置 |
| Qdrant 二进制下载/兼容性 | 低 | 低 | 提供明确的版本锁定安装脚本 |
| Mem0 v2 算法变化 | 低 | 中 | 锁定 mem0ai 版本号在 pyproject.toml |

---

## 十四、环境变量清单

```bash
# .env.example

# ===== DeepSeek =====
DEEPSEEK_API_KEY=           # 从 platform.deepseek.com 获取
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# ===== DashScope (阿里云百炼) =====
DASHSCOPE_API_KEY=          # 从 bailian.console.aliyun.com 获取
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMS=1024

# ===== Qdrant =====
QDRANT_HOST=localhost
QDRANT_PORT=6333

# ===== MemoryBridge =====
MEMORY_BRIDGE_HOST=0.0.0.0
MEMORY_BRIDGE_PORT=8000
SESSION_MAX_HISTORY=50
```
