# 集成冒烟测试

验证 MemoryBridge 完整请求管线的端到端正确性。

## 前置条件

- `./bin/qdrant` 已下载并启动
- `.env` 已填入 `DEEPSEEK_API_KEY` 和 `DASHSCOPE_API_KEY`
- MemoryBridge 服务已启动：`python src/memory_bridge/host_manager.py`

## 执行

```bash
# 默认连接 localhost:8000
python tests/integration/smoke_test.py

# DEBUG 日志模式（观察记忆检索/注入细节）
LOG_LEVEL=DEBUG python src/memory_bridge/host_manager.py

# 指定地址
python tests/integration/smoke_test.py --base-url http://10.0.0.1:8000
```

## 测试步骤

### Phase A — 短期记忆（Session 上下文）

验证 SessionStore 的会话历史注入功能（持久化到 SQLite）。

| 步骤 | 操作 | 预期 |
|------|------|------|
| 1 | 创建空 Session `sess-st` | 201, agent_session_id=sess-st |
| 2 | 同一 Session 发送"你好，我叫小明"（`memory_enabled=false`） | 200 |
| 3 | **sleep 0.5s**（等待 Session DB 写入完成） | — |
| 4 | 同一 Session 发送"我叫什么名字？"（`memory_enabled=false`） | 响应含"小明" |

**原理**：第 2 步的 user + assistant 对写入 SQLite，第 4 步从数据库读取历史上下文注入。`memory_enabled=false` 关闭长期记忆检索，回复只能来自 SessionStore 注入的会话历史。

### Phase B — 长期记忆（Mem0 + Qdrant）

验证 Mem0 长期记忆的跨 Session 共享。

| 步骤 | 操作 | 预期 |
|------|------|------|
| 5 | 创建 `sess-lt` | 201 |
| 6 | 发送"我最喜欢的颜色是蓝色，最喜欢的动物是猫"（memory_enabled=true） | 200 |
| 7 | **sleep 3s**（等待 BackgroundTasks 写 Mem0 完成） | — |
| 8 | 创建**新 Session** `sess-lt-verify`（同 agent，不同 session） | 201 |
| 9 | 新 Session 发送"我喜欢什么颜色和动物？" | 响应同时含"蓝"和"猫" |

**原理**：新 Session 没有会话历史，回答正确只能来自 Mem0 长期记忆检索。如果响应不含这些信息，说明长期记忆管线断裂。

### Phase C — Agent 隔离

验证不同 Agent 之间记忆完全隔离。

| 步骤 | 操作 | 预期 |
|------|------|------|
| 10 | 创建 `agent-other/sess-iso` | 201 |
| 11 | `agent-other` 问"我喜欢什么颜色和动物？"（memory_enabled=true） | 响应**不含**"蓝"和"猫" |

### Phase D — 边界条件

| 步骤 | 操作 | 预期 |
|------|------|------|
| 12 | 不带 `agent_session_id` | 422 |
| 13 | 不存在的 session_id | 404, `SESSION_NOT_FOUND` |
| 14 | 重复创建相同 session | 409, `SESSION_EXISTS` |
| 15 | 思考模式 + 流式（`thinking_enabled=true, stream=true`） | SSE 事件流，含 `reasoning_content` + `[DONE]` |

## 成功输出

```
[1/19] CREATE session st (empty)... 201 ✓ (session_id=sess-st)
[2/19] CHAT first turn (establishes session history)... 200 ✓
[3/19] SLEEP 0.5s (wait for session DB write)... slept 0.5s ✓
[4/19] CHAT short-term memory (memory_enabled=false)... 200 ✓ (contains '小明')
[5/19] CREATE session lt... 201 ✓
[6/19] CHAT write long-term facts... 200 ✓
[7/19] SLEEP 3s... slept 3s ✓
[8/19] CREATE session lt-verify... 201 ✓
[9/19] CHAT verify long-term memory... 200 ✓ (contains '蓝'+'猫')
[10/19] CREATE session other agent... 201 ✓
[11/19] CHAT verify isolation... 200 ✓ (no '蓝'/'猫')
[12/19] VALIDATE missing session_id... 422 ✓
[13/19] VALIDATE unknown session... 404 ✓
[14/19] VALIDATE duplicate session... 409 ✓
[15/19] CHAT thinking+stream... 200 ✓ (SSE+reasoning)
[16/19] CHAT custom prompt... 200 ✓
[17/19] SLEEP 3s... slept 3s ✓
[18/19] CHAT verify custom prompt... 200 ✓ (contains '旺财'+'狗')
[19/19] Health check... 200 ✓

==================================================
ALL PASSED: XX/XX
```

## DEBUG 日志对照

设置 `LOG_LEVEL=DEBUG` 后，可观察每个阶段的完整日志：

```
[Phase B - 步骤 6/9：长期记忆写入与检索]
memory search → results count=0 scores=[]                    ← 首次无记忆
├─ context build memories_count=0
├─ memory store messages_count=3                             ← 写入 Mem0
├─ memory stored
...sleep 3s...
memory search → results count=2 scores=[0.92,0.78]           ← 记忆命中！
│  top_memory="用户最喜欢的水果是苹果"
├─ context build memories_count=2
│  memory block preview="[相关历史记忆]\n- 用户最喜欢的..."
└─ ← 200 latency_ms=2340

[Phase C - 步骤 11：Agent 隔离]
memory search → results count=0 scores=[]                    ← agent-other 无记忆
├─ context build memories_count=0
└─ LLM response: "我还不知道你喜欢什么颜色和动物..."
```

## 清理测试数据

测试在 Qdrant 中留下了 `agent-smoke` 和 `agent-other` 的记忆向量。检查完数据后手动清理：

```bash
# 先检查数据（通过 MemoryBridge 健康检查确认 Qdrant 连接正常）
curl http://localhost:8000/health

# 确认无误后清理
python tests/integration/cleanup.py

# 若要彻底删除 Qdrant 数据
rm -rf ./data/qdrant/
```

**注意**：cleanup 是**手动步骤**，不会自动执行。这是有意设计——确保测试后有机会检查数据库状态。
