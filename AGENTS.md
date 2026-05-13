# AGENTS.md — MemoryBridge 开发准则

## 项目定位

MemoryBridge 是 Agent 与大模型之间的中间件，负责记忆检索、上下文注入和请求代理。

## 技术栈

| 组件 | 选型 |
|------|------|
| 语言 | Python 3.11+ |
| Web 框架 | FastAPI + Uvicorn |
| 记忆层 | Mem0 >=2.0.2 (library) |
| 向量存储 | Qdrant (Rust 二进制，localhost:6333) |
| LLM Provider | DeepSeek 官方 API |
| Embedding | 阿里云 text-embedding-v4 (DashScope) |
| 异步 HTTP | httpx |
| 配置 | pydantic-settings |
| 类型检查 | mypy (strict) |
| 包管理 | uv |
| 打包 | pyproject.toml (hatchling) + shiv |

## 文档导航

| 文档 | 用途 |
|------|------|
| `README.md` | 项目介绍、快速开始、API 使用方式 |
| `docs/ARCHITECTURE.md` | 系统架构、请求链路、各子系统原理 |
| `docs/INTEGRATION_TEST.md` | 集成冒烟测试步骤、预期输出、清理方法 |
| `AGENTS.md` | 本文档 — 开发准则与工程约束 |

### 测试命令速查

```bash
uv run mypy src/          # 类型检查
uv run ruff check src/    # Lint
uv run pytest -v          # 单元/集成测试 (102 cases)
LOG_LEVEL=DEBUG uv run python src/memory_bridge/host_manager.py  # 启动（DEBUG 日志）
python tests/integration/smoke_test.py   # 集成冒烟测试（需先启动服务）
python tests/integration/cleanup.py      # 清理冒烟测试数据
bash scripts/token_admin.sh create       # 创建 API Token
```

## 核心开发原则

### 1. 约束强硬，不降级不回退

- `agent_id` 缺失 → 422（不设默认值，不回退）
- 记忆检索失败 → 500（不跳过记忆步骤继续）
- LLM Provider 失败 → 502（不重试，不切换 provider）
- 任何异常不应被静默吞掉，必须显式抛出或日志记录

### 2. 最少依赖

- 只引入核心功能必需的库：fastapi、uvicorn、mem0ai、httpx、pydantic-settings、qdrant-client
- 不用 ORM、不用 Redis、不用消息队列、不用 Docker
- 进程间通信只用 HTTP over localhost

### 3. 进程模型

```
Host Manager (Python)
  ├─ 拉起 Qdrant 二进制子进程 (localhost:6333)
  └─ 拉起 MemoryBridge FastAPI 子进程 (localhost:8000)
```
- 一个服务一个进程，本地 TCP 通信解耦
- 进程管理器通过 subprocess 启动/停止/监控子进程
- SIGINT/SIGTERM 时顺序优雅关闭

### 4. 单文件单职责

- 每个模块文件不超过 200 行
- 类和函数保持单一职责
- 模块按职责分层：`api/` → `core/` → `providers/` → `models/`

### 5. 完整类型覆盖

- 所有变量、属性、函数参数、返回值必须显式声明类型标注
- 禁止依赖类型推断省略声明（如 `x = 1` 应写为 `x: int = 1`）
- 所有公共函数和方法的参数、返回值带完整类型标注
- `pyproject.toml` 启用 mypy strict 模式
- 不写 `Any`，不跳过类型检查

### 6. 配置外置

- 所有可变参数通过 env / `.env` 注入
- 代码内不含硬编码的 secret、URL、端口
- `.env.example` 列出全部必需变量，值为空

### 7. 错误显式化

- 自定义异常类，命名清晰（如 `MemorySearchError`、`ProviderNotFoundError`）
- 禁止使用裸 `Exception` 或 `except: pass`
- 每个异常携带明确的错误消息

### 8. 测试驱动

- 核心模块（MemoryManager、SessionStore、ContextBuilder）必须有单元测试
- API 层有集成测试
- 不追求覆盖率数字，但核心链路必须覆盖
- 测试使用 pytest

### 9. 扩展性预留但不提前实现

- Provider 架构支持注册多个 LLM 后端，但当前只实现 DeepSeek
- Embedder 配置中 Ollama 选项以注释形式预留，不写实现代码
- Provider 按 model 名路由到对应适配器
- 预留空间但不为未来过度设计

### 10. 可部署性

- 项目必须能通过单次命令构建可分发的产物（wheel 或单文件）
- 目标服务器上只需 Python 3.11+ 和 Qdrant 二进制即可运行
- 部署流程最多 3 步：传文件 → 装依赖 → 启动
- 详见下方"打包方案"章节

## 代码风格

- 遵循 PEP 8
- 使用 `black` 代码格式化（line-length=100）
- import 顺序：标准库 → 第三方 → 项目内部，每组间空一行
- 不使用 `from module import *`
- 类名 PascalCase，函数/变量名 snake_case，常量 UPPER_SNAKE_CASE

## 环境准备

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv sync

# 运行类型检查
uv run mypy src/

# 运行 lint
uv run ruff check src/

# 运行测试
uv run pytest
```

## Git 工作流

- 提交信息使用英文，简明描述变更原因
- 遵循 Conventional Commits（`feat:`, `fix:`, `chore:`, `test:`, `docs:`）
- 每次提交前通过 mypy 类型检查
- 不提交 `.env`、`data/`、`bin/qdrant` 到仓库

---

# 打包方案

## 选型结论

| 层级 | 工具 | 用途 |
|------|------|------|
| 构建 | `hatchling` (via pyproject.toml) | 生成 wheel |
| 单文件打包 | `shiv` | 将所有 Python 依赖打成 .pyz 文件 |
| 二进制依赖 | `host_manager.py --setup` | 自动下载 Qdrant 二进制 |
| 部署脚本 | 无额外脚本 | 启动 + 初始化均通过 `host_manager.py` 入口 |

## 选型理由

**shiv** 适合本项目的原因：

- 生成**单个 .pyz 文件**（zipapp），`scp` 一份到服务器即可
- 启动时自动解压，运行时零额外 I/O
- 不改变任何代码，不需要 Docker
- 目标服务器只需有 Python 3.11+，无需安装依赖
- 文件大小可控（本项目依赖少，预计 ~40MB）

**不用 Docker**：用户明确要求避免引入 Docker 复杂度。

**不用 PyInstaller**：PyInstaller 依赖平台编译，跨平台部署需要对应平台构建，且二进制体积大（~60MB+）。

**不用 PEX**：功能类似 shiv 但配置更复杂。

## 打包流程

```bash
# 本地构建（含 lint + typecheck + test）
bash scripts/build.sh

# 或仅构建（跳过检查）
bash scripts/build.sh --build

# 开发机上手动构建
uv run shiv \
    --compile-pyc \
    --console-script host-manager \
    --output-file dist/memorybridge.pyz \
    .

# 复制到服务器
scp dist/memorybridge.pyz server:/opt/memorybridge/

# 服务器上（首次）
cd /opt/memorybridge
python memorybridge.pyz --setup    # 下载 qdrant 二进制、创建 data 目录

# 每次启动
python memorybridge.pyz
```

## .gitignore 补充

```
# 构建产物
dist/
*.pyz
*.whl

# Qdrant 二进制
bin/qdrant

# 运行时数据
data/
*.db

# Qdrant 运行时
.qdrant-initialized
snapshots/

# 环境
.env
.venv/
__pycache__/
*.pyc

# 工具缓存
.mypy_cache/
.pytest_cache/
.ruff_cache/
```

## 目录结构（部署视角）

```
/opt/memorybridge/
├── memorybridge.pyz      # shiv 单文件（Python 代码 + 所有依赖）
├── bin/
│   └── qdrant            # Qdrant Rust 二进制
├── .env                  # 环境变量配置
└── data/                 # 自动创建
    ├── qdrant/           # Qdrant 持久化数据
    └── mem0_history.db   # Mem0 历史
```
