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

验证 SessionStore 的会话历史注入功能。

| 步骤 | 操作 | 预期 |
|------|------|------|
| 1 | 创建 `sess-st`，附初始消息 `{"role":"user","content":"你好，我叫小明"}` | 201, message_count=2 |
| 2 | 同一 Session 发送"我叫什么名字？"（`memory_enabled=false`） | 响应含"小明" |

**原理**：`memory_enabled=false` 关闭长期记忆检索，回复只能来自 SessionStore 注入的会话历史。

### Phase B — 长期记忆（Mem0 + Qdrant）

验证 Mem0 长期记忆的跨 Session 共享。

| 步骤 | 操作 | 预期 |
|------|------|------|
| 3 | 创建 `sess-lt` | 201 |
| 4 | 发送"我最喜欢的颜色是蓝色，最喜欢的动物是猫"（memory_enabled=true） | 200 |
| 5 | **sleep 3s**（等待 BackgroundTasks 写 Mem0 完成） | — |
| 6 | 创建**新 Session** `sess-lt-verify`（同 agent，不同 session） | 201 |
| 7 | 新 Session 发送"我喜欢什么颜色和动物？" | 响应同时含"蓝"和"猫" |

**原理**：新 Session 没有会话历史，回答正确只能来自 Mem0 长期记忆检索。如果响应不含这些信息，说明长期记忆管线断裂。

### Phase C — Agent 隔离

验证不同 Agent 之间记忆完全隔离。

| 步骤 | 操作 | 预期 |
|------|------|------|
| 8 | 创建 `agent-other/sess-iso` | 201 |
| 9 | `agent-other` 问"我喜欢什么颜色和动物？"（memory_enabled=true） | 响应**不含**"蓝"和"猫" |

### Phase D — Session 导出与恢复（Export/Import Round-trip）

验证会话历史的导出和重新导入能力，支持进程重启后恢复对话上下文。

| 步骤 | 操作 | 预期 |
|------|------|------|
| 10 | 创建 `sess-export-src`，附初始消息 `{"role":"user","content":"你好，我叫小刚"}` | 201, message_count=2 |
| 11 | 同一 Session 发送"回复一个字：好"（`memory_enabled=false`） | 200，响应含"好" |
| 12 | `GET /v1/sessions/agent-smoke/sess-export-src` 导出会话 | 200，messages >= 4 条 |
| 13 | `POST /v1/sessions` 以导出的 messages 创建新 Session `sess-export-dst` | 201, message_count=导出条数 |
| 14 | 新 Session 发送"我叫什么名字？请只回答名字"（`memory_enabled=false`） | 响应含"小刚" |

**原理**：导出再重建的 Session 应有完整的历史上下文，`memory_enabled=false` 确保回答来自 SessionStore 而非长期记忆。

### Phase E — 边界条件

| 步骤 | 操作 | 预期 |
|------|------|------|
| 15 | 不带 `agent_session_id` | 422 |
| 16 | 不存在的 session_id | 404, `SESSION_NOT_FOUND` |
| 17 | 重复创建相同 session | 409, `SESSION_EXISTS` |
| 18 | 思考模式 + 流式（`thinking_enabled=true, stream=true`） | SSE 事件流，含 `reasoning_content` + `[DONE]` |
| 19 | 导出不存在的 session | 404, `SESSION_NOT_FOUND` |

## 成功输出

```
[1/19] CREATE session st... 201 ✓ (message_count=2)
[2/19] CHAT short-term memory... 200 ✓ (contains '小明')
[3/19] CREATE session lt... 201 ✓
[4/19] CHAT write long-term facts... 200 ✓
[5/19] SLEEP 3s...
       slept 3s ✓
[6/19] CREATE session lt-verify... 201 ✓
[7/19] CHAT verify long-term memory... 200 ✓ (contains '蓝'+'猫')
[8/19] CREATE session other agent... 201 ✓
[9/19] CHAT verify isolation... 200 ✓ (no '蓝'/'猫')
[10/19] CREATE session export-src... 201 ✓
[11/19] CHAT one turn in export-src... 200 ✓
[12/19] EXPORT session... 200 ✓ (system not in exported)
[13/19] IMPORT session... 201 ✓
[14/19] CHAT verify restored session... 200 ✓ (contains '小刚')
[15/19] VALIDATE missing session_id... 422 ✓
[16/19] VALIDATE unknown session... 404 ✓
[17/19] VALIDATE duplicate session... 409 ✓
[18/19] CHAT thinking+stream... 200 ✓ (SSE+reasoning)
[19/19] Health check... 200 ✓

==================================================
ALL PASSED: 27/27
```

## DEBUG 日志对照

设置 `LOG_LEVEL=DEBUG` 后，可观察每个阶段的完整日志：

```
[Phase B - 步骤 4/7：长期记忆写入与检索]
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

[Phase C - 步骤 9：Agent 隔离]
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
