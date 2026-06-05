# 记忆系统升级设计文档

> 项目：医疗智能助手 (agentic-rag-for-dummies)  
> 日期：2026-06-05  
> 状态：设计评审

---

## 1. 现状分析

### 1.1 当前记忆架构

系统目前采用三层记忆，但实现不完整：

| 层级 | 实现 | 存储 | 生命周期 | 问题 |
|------|------|------|----------|------|
| 工作记忆 | LangGraph `State` 字段 | 内存 | 单次图执行 | ✅ 够用 |
| 短期记忆 | `RedisSessionMemory` | Redis / 内存回退 | 会话内，TTL 24h | ✅ 够用，滑动窗口 + 回退 |
| 长期记忆 | `SummaryStore` | PostgreSQL | 跨会话持久 | ❌ 仅线程级摘要，无用户级知识 |

### 1.2 核心短板

**短板 1：无跨会话用户级记忆**

用户在会话 A 中说了"我有高血压"，新开会话 B 后系统完全不知道。`SummaryStore` 只按 `thread_id` 存摘要，没有 `user_id` 维度的知识积累。

```
现状：user_id=alice, thread_1 → 摘要1（独立）
      user_id=alice, thread_2 → 摘要2（独立）
      thread_1 和 thread_2 之间没有任何记忆共享

期望：user_id=alice → 用户知识库（跨线程共享）
      thread_1 学到的偏好/病史 → thread_2 也能用
```

**短板 2：记忆无重要性区分**

所有记忆平等对待——用户随口一句"好的"和"我对青霉素过敏"获得相同权重。无重要性评分，无衰减策略，无法优先召回关键信息。

**短板 3：记忆提取过于粗糙**

当前 `summarize_history` 节点只做整体摘要，不提取结构化知识（偏好、病史、决策）。摘要丢失细节，且无法按类型检索。

```
现状：整段对话 → 一段摘要文本
期望：整段对话 → {偏好: [...], 事实: [...], 病史: [...], 决策: [...]}
```

### 1.3 已有基础设施（可直接复用）

| 组件 | 状态 | 说明 |
|------|------|------|
| `user_id` 身份体系 | ✅ 已有 | `AuthenticatedUser.user_id`，`chat_sessions.owner_user_id` |
| pgvector 向量检索 | ✅ 已有 | `child_chunks.embedding VECTOR(1024)` + 余弦相似度 |
| 混合检索 (dense + sparse) | ✅ 已有 | RAG 管道已实现，可复用到记忆检索 |
| 摘要管道 | ✅ 已有 | `summarize_history` 节点 + `SummaryStore` |
| Benchmark 框架 | ✅ 已有 | `QAEvalSample`、`AblationStudy`、memory token benchmark |
| Redis + 内存回退 | ✅ 已有 | `RedisSessionMemory` 优雅降级模式 |

---

## 2. 技术选项

### 方案 A：增强型三层记忆（推荐落地）

**核心理念**：在现有架构上渐进增强，补齐跨会话记忆 + 重要性评分 + 结构化提取。

**新增组件**：

```
                    ┌──────────────────────────┐
                    │     用户级记忆 Store      │
                    │  (user_memories 表)       │
                    │  ┌──────────────────────┐ │
                    │  │ preference  偏好      │ │
                    │  │ fact        事实      │ │
                    │  │ medical     病史      │ │
                    │  │ decision    决策      │ │
                    │  │ reflection  反思      │ │
                    │  └──────────────────────┘ │
                    │  + importance 评分        │
                    │  + embedding 向量         │
                    │  + created_at 时间戳      │
                    └──────────┬───────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
  记忆提取管道          记忆检索管道            记忆合并管道
  (对话 → 结构化)      (加权三因子排序)        (去重 + 合并)
```

**新增表 `user_memories`**：

```sql
CREATE TABLE IF NOT EXISTS user_memories (
    id              BIGSERIAL PRIMARY KEY,
    user_id         VARCHAR(128) NOT NULL,
    memory_type     VARCHAR(32) NOT NULL,  -- preference/fact/medical/decision/reflection
    content         TEXT NOT NULL,
    source_thread   VARCHAR(128),          -- 来源会话
    importance      SMALLINT NOT NULL DEFAULT 5,  -- 1-10，LLM 评分
    embedding       VECTOR(1024),          -- 语义检索
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    merged_from     JSONB DEFAULT '[]',    -- 合并来源记录
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_user_memories_user_id ON user_memories(user_id);
CREATE INDEX idx_user_memories_type ON user_memories(user_id, memory_type);
CREATE UNIQUE INDEX uq_user_memories_content ON user_memories(user_id, memory_type, content);
```

**记忆提取管道**（对话结束时异步执行）：

```
对话历史 → LLM 提取 → 结构化记忆条目列表
                        ↓
                   重要性评分 (1-10)
                        ↓
                   去重检查 (embedding > 0.9 相似 → 合并)
                        ↓
                   写入 user_memories
```

LLM 提取 Prompt 示例：

```
从以下对话中提取结构化记忆，返回 JSON 数组：
[
  {"type": "medical", "content": "用户对青霉素过敏", "importance": 9},
  {"type": "preference", "content": "用户偏好简洁回答", "importance": 4},
  {"type": "fact", "content": "用户父亲有糖尿病", "importance": 7}
]

对话内容：
{conversation}
```

**记忆检索管道**（新会话启动时）：

```
用户提问 → embedding
              ↓
     user_memories 向量检索 (top_k=10)
              ↓
     三因子加权排序：
     score = 0.3 * recency + 0.4 * importance + 0.3 * relevance
              ↓
     取 top 5 注入 conversation_summary / topic_focus
```

**各因子计算**：

- **recency**: `exp(-0.01 * hours_since_creation)` — 24h 内接近 1.0，一周后约 0.19
- **importance**: `score / 10` — 直接归一化
- **relevance**: `cosine_similarity(query_embedding, memory_embedding)` — pgvector 原生支持

**改动清单**：

| 新增 | 修改 |
|------|------|
| `project/memory/user_memory_store.py` — 用户级记忆 CRUD + 向量检索 | `project/core/chat_interface.py` — 对话结束时触发提取，新会话注入记忆 |
| `project/memory/memory_extractor.py` — LLM 提取 + 重要性评分 | `project/rag_agent/routing_nodes.py` — 注入用户记忆到路由上下文 |
| `project/db/sql/V2__user_memories.sql` — 建表迁移 | `project/rag_agent/graph_state.py` — State 新增 `user_memories` 字段 |
| `tests/test_user_memory_store.py` | `project/memory/summary_store.py` — 提取后更新线程摘要 |

---

### 方案 B：MemGPT 风格自主记忆管理

**核心理念**：将记忆操作作为工具暴露给 LLM Agent，让 Agent 自己决定何时存、何时取、何时改。

**在方案 A 基础上新增**：

| 组件 | 说明 |
|------|------|
| `memory_save` 工具 | Agent 主动保存记忆到用户 store |
| `memory_search` 工具 | Agent 主动检索相关记忆 |
| `memory_update` 工具 | Agent 修改已有记忆 |
| `memory_delete` 工具 | Agent 删除过时记忆 |
| 核心记忆区 (Core Memory) | 始终在 context 中的结构化文本，Agent 可通过 `core_memory_update()` 修改 |

**优势**：Agent 理解力更强，能主动判断什么该记什么该忘  
**劣势**：每轮交互增加工具调用 → token 成本 +20~40%；医疗场景下自主操作可能出错（误删过敏史）；需要更多 prompt engineering

---

### 方案 C：完整认知记忆架构

**核心理念**：融合 Generative Agents + MemGPT + LangGraph Store，构建六层记忆。

**在方案 B 基础上新增**：

| 层 | 名称 | 说明 |
|---|---|---|
| L4 | 情景记忆 | 完整对话时间线，支持"你上周二问我什么来着" |
| L6 | 反思记忆 | 定期从情景/语义记忆中合成更高层抽象 |

反思示例：

```
原始记忆：
  - "用户第1次问头痛怎么办"
  - "用户第3次问头痛缓解方法"
  - "用户第5次问长期头痛需要做什么检查"

反思合成：
  "用户有慢性头痛困扰，已持续关注，可能需要建议做进一步检查"
```

**优势**：最完整，最接近人类记忆模型  
**劣势**：实现复杂度最高；反思层 token 消耗大；六层交互需要精细调优；医疗场景下反思可能引入错误推理

---

### 2.4 方案对比与选型

| 维度 | 方案 A (增强三层) | 方案 B (自主管理) | 方案 C (六层认知) |
|------|---|---|---|
| **新增代码量** | ~400 行 | ~700 行 | ~1200 行 |
| **新增表** | 1 张 | 1 张 | 3 张 |
| **Token 增量** | 对话结束时一次性 +500 | 每轮 +200~400 | 每轮 +300~600 |
| **跨会话记忆** | ✅ | ✅ | ✅ |
| **重要性评分** | ✅ | ✅ | ✅ |
| **结构化提取** | ✅ | ✅ | ✅ |
| **Agent 自主性** | ❌ | ✅ | ✅ |
| **反思/抽象** | ❌ | ❌ | ✅ |
| **医疗安全性** | 高（被动提取，可审计） | 中（Agent 可能误操作） | 中（反思可能引入推理错误） |
| **可测试性** | 高 | 中 | 低 |
| **面试讲述清晰度** | 高（before/after 明确） | 中 | 低（层次多难讲清） |
| **演进路径** | → B → C | → C | — |

**选型结论：方案 A**

理由：
1. 医疗场景记忆准确性 > 自主性，被动提取比 Agent 主动管理更可靠
2. Before/after 对比最清晰，面试时好讲
3. Token 增量可控，不影响用户体验
4. 方案 A 是 B/C 的基础，后续可渐进演进

---

## 3. 实现设计（方案 A 详细设计）

### 3.1 数据模型

```sql
-- 用户级记忆表
CREATE TABLE IF NOT EXISTS user_memories (
    id              BIGSERIAL PRIMARY KEY,
    user_id         VARCHAR(128) NOT NULL,
    memory_type     VARCHAR(32) NOT NULL,  -- preference/fact/medical/decision
    content         TEXT NOT NULL,
    source_thread   VARCHAR(128),
    importance      SMALLINT NOT NULL DEFAULT 5,
    embedding       VECTOR(1024),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    merged_from     JSONB DEFAULT '[]'::jsonb,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_user_memories_user_id ON user_memories(user_id);
CREATE INDEX idx_user_memories_user_type ON user_memories(user_id, memory_type);
```

### 3.2 记忆类型定义

| 类型 | 含义 | 重要性典型范围 | 示例 |
|------|------|---|---|
| `preference` | 用户偏好 | 3-6 | "偏好简洁回答"、"不想看太长的解释" |
| `fact` | 事实信息 | 5-8 | "用户是退休教师"、"用户家住北京" |
| `medical` | 医疗相关 | 7-10 | "对青霉素过敏"、"有高血压病史" |
| `decision` | 用户决策 | 5-8 | "决定下周一去心内科就诊" |

### 3.3 记忆提取管道

**触发时机**：对话轮次达到 `SUMMARY_REFRESH_THRESHOLD` 时，在 `summarize_history` 节点之后异步执行。

**流程**：

```
Step 1: 对话历史 → LLM → 结构化记忆条目列表
        Prompt: "从对话中提取 preference/fact/medical/decision 四类记忆"

Step 2: 每条记忆 → LLM → importance 评分 (1-10)
        Prompt: "评估这条记忆的重要性，1=琐事，10=生命攸关"
        （可与 Step 1 合并为一次 LLM 调用节省 token）

Step 3: 新记忆 embedding → 与已有记忆做余弦相似度
        相似度 > 0.9 → 合并（保留 importance 更高的，记录 merged_from）
        相似度 ≤ 0.9 → 新增

Step 4: 写入 user_memories 表
```

**合并策略**：

```python
def merge_memories(existing, new):
    """合并两条相似记忆"""
    if new.importance >= existing.importance:
        # 新记忆更重要或同等重要 → 更新
        merged = {
            "content": new.content,
            "importance": max(existing.importance, new.importance),
            "merged_from": existing.merged_from + [existing.content],
        }
    else:
        # 旧记忆更重要 → 保留但追加新信息
        merged = {
            "content": existing.content,
            "importance": existing.importance,
            "merged_from": existing.merged_from + [new.content],
        }
    return merged
```

### 3.4 记忆检索管道

**触发时机**：每次新对话开始，在 `analyze_turn` 节点执行时。

**流程**：

```
Step 1: 用户当前问题 → embedding

Step 2: pgvector 检索该用户 top_k=10 条记忆
        SELECT *, 1 - (embedding <=> $query_vector) AS relevance
        FROM user_memories
        WHERE user_id = $user_id AND is_active = TRUE
        ORDER BY embedding <=> $query_vector
        LIMIT 10

Step 3: 三因子加权排序
        for memory in candidates:
            recency = exp(-0.01 * hours_since(memory.created_at))
            importance = memory.importance / 10.0
            relevance = cosine_similarity(query_embedding, memory.embedding)
            memory.score = 0.3 * recency + 0.4 * importance + 0.3 * relevance

Step 4: 取 top 5 注入 State
        State.user_memories = sorted_candidates[:5]
```

**三因子权重说明**：

| 因子 | 权重 | 理由 |
|------|------|------|
| importance | 0.4 | 医疗场景最重要的信息（过敏、病史）应始终优先 |
| relevance | 0.3 | 语义相关性确保召回当前话题相关的记忆 |
| recency | 0.3 | 近期记忆通常更相关，但不应该压过重要性 |

### 3.5 记忆注入方式

检索到的用户记忆通过两种方式注入对话上下文：

1. **State 字段注入**：`State.user_memories: List[dict]` — 在 `analyze_turn` 中填充
2. **Prompt 模板增强**：在 `routing_nodes.py` 和 `rag_nodes.py` 的 prompt 中加入用户记忆段

```
[用户记忆]
- [medical|9] 对青霉素过敏
- [fact|7] 父亲有糖尿病史
- [preference|4] 偏好简洁回答

[当前问题]
{user_query}
```

### 3.6 错误处理

| 场景 | 处理 |
|------|------|
| 记忆提取 LLM 失败 | 静默跳过，不阻断对话流程，记录 warning 日志 |
| pgvector 不可用 | 退化为 keyword-only 检索（tsvector） |
| embedding 服务不可用 | 跳过记忆注入，对话正常进行 |
| 合并去重冲突 | 保留 importance 更高者，记录 merged_from 审计链 |

---

## 4. 效果评估

### 4.1 评估维度

| 维度 | 指标 | 测试方法 |
|------|------|----------|
| **跨会话记忆召回率** | 新会话中正确引用历史信息的比例 | 构造多轮会话测试集，自动验证 |
| **记忆检索精度** | 返回的记忆与当前问题的相关性 | 人工标注 + embedding 相似度 |
| **Token 效率** | 注入记忆后的 token 增量 | memory token benchmark 对比 |
| **端到端回答质量** | 有/无记忆注入时回答的关键词覆盖率和安全性 | QA benchmark 对比 |
| **提取质量** | 提取的记忆条目的准确性和完整性 | 人工评估 + LLM-as-judge |
| **去重准确率** | 合并操作是否正确识别重复 | 人工构造重复场景测试 |

### 4.2 测试方法

#### 测试 1：跨会话记忆召回测试

**构造**：5 组多会话场景，每组包含 2-3 个连续会话

```json
{
  "user_id": "test-alice",
  "sessions": [
    {
      "thread_id": "t1",
      "turns": [
        {"role": "user", "content": "我有高血压，最近头晕"},
        {"role": "assistant", "content": "头晕可能与血压波动有关..."}
      ],
      "expected_memories": [
        {"type": "medical", "content_contains": "高血压", "importance_min": 7},
        {"type": "medical", "content_contains": "头晕", "importance_min": 6}
      ]
    },
    {
      "thread_id": "t2",
      "query": "我该吃什么药？",
      "expected_recall": ["高血压"],
      "eval": "回答应提及高血压相关的用药建议"
    }
  ]
}
```

**执行**：
1. 运行 session 1 → 验证提取的记忆条目与 `expected_memories` 匹配
2. 开新会话 session 2 → 验证检索到的记忆包含 `expected_recall`
3. 对比有/无记忆注入时的回答质量

**指标**：
- 记忆提取命中率 = 匹配的 expected_memories / 总 expected_memories
- 跨会话召回率 = 正确召回的 expected_recall / 总 expected_recall
- 回答质量提升 = 有记忆时 answer_score - 无记忆时 answer_score

#### 测试 2：三因子检索排序测试

**构造**：预置 20 条不同类型、重要性、时间的记忆，给定 5 个查询

```python
test_cases = [
    {
        "query": "我头痛吃什么药",
        "expected_top_types": ["medical"],    # 病史应排前面
        "expected_top_min_importance": 7,     # 高重要性记忆优先
        "memories": [
            {"content": "有偏头痛病史", "type": "medical", "importance": 9, "age_hours": 48},
            {"content": "偏好简洁回答", "type": "preference", "importance": 4, "age_hours": 1},
            {"content": "对阿司匹林过敏", "type": "medical", "importance": 10, "age_hours": 720},
        ]
    }
]
```

**指标**：
- 排序准确率 = 排在 top 5 中的正确记忆数 / 期望的正确记忆数
- 医疗记忆优先率 = medical 类型在 top 5 中的比例

#### 测试 3：Ablation 对比实验

利用现有 `AblationStudy` 框架，对比四个配置：

| 配置 | 说明 |
|------|------|
| `baseline` | 当前系统（无用户级记忆） |
| `+memory_extract` | 仅开启记忆提取，不注入检索 |
| `+memory_retrieval` | 仅开启检索注入，用预置记忆 |
| `+memory_full` | 完整管道（提取 + 检索 + 注入） |

**指标对比**：

| 指标 | baseline | +extract | +retrieval | +full |
|------|----------|----------|------------|-------|
| 跨会话召回率 | 0% | 0%（未注入） | 预置数据 | 预期 70%+ |
| 回答关键词覆盖率 | 当前值 | 当前值 | 提升 | 预期 +15% |
| Token 增量 | 0 | +0 | +50~100/轮 | +50~100/轮 |
| 安全关键词覆盖率 | 当前值 | 当前值 | 提升 | 预期 +10% |

#### 测试 4：去重合并测试

**构造**：模拟用户多次表述相同信息

```
会话1: "我有高血压"  → 提取: {type: medical, content: "有高血压", importance: 8}
会话2: "我是高血压患者" → 应合并而非新增
会话3: "高血压好几年了" → 应合并，更新 importance
```

**指标**：
- 去重准确率 = 正确合并次数 / 应合并次数
- 误合并率 = 错误合并次数 / 总合并次数（应接近 0，医疗场景宁可不合并也不误合并）

### 4.3 预期效果

| 指标 | 当前 | 预期 | 提升幅度 |
|------|------|------|----------|
| 跨会话记忆召回率 | 0% | 70%+ | 从无到有 |
| 回答关键词覆盖率 | 基线值 | +15% | 记忆注入提升回答针对性 |
| 安全关键词覆盖率 | 基线值 | +10% | 过敏/病史自动纳入 |
| Token 增量/轮 | 0 | +50~100 | 可接受 |
| 记忆提取耗时 | N/A | +2~3s | 异步执行，不阻塞用户 |

---

## 5. 实现步骤

| 阶段 | 任务 | 产出 |
|------|------|------|
| **P1: 基础设施** | 建表、UserMemoryStore CRUD | `V2__user_memories.sql`、`user_memory_store.py` |
| **P2: 记忆提取** | LLM 提取 + 重要性评分 + 去重合并 | `memory_extractor.py` |
| **P3: 记忆检索** | 三因子排序 + pgvector 检索 | `user_memory_store.py` 扩展 |
| **P4: 管道集成** | 对话结束触发提取、新会话注入记忆 | 改 `chat_interface.py`、`routing_nodes.py` |
| **P5: 评估** | 多会话测试集 + ablation 对比 | benchmark 报告 |
| **P6: 演进准备** | 预留 LLM 自主记忆工具接口 | `tools.py` 扩展点 |

---

## 6. 演进路线

```
方案 A（当前）           方案 B（+Agent 自主）      方案 C（+六层认知）
┌──────────────┐      ┌──────────────┐       ┌──────────────┐
│ 用户级 Store  │      │ + 记忆工具    │       │ + 情景记忆    │
│ 重要性评分    │  →   │ + Core Memory │  →    │ + 反思记忆    │
│ 结构化提取    │      │ + Agent 自主  │       │ + 六层架构    │
│ 三因子检索    │      │   读写删改    │       │   递归理解    │
└──────────────┘      └──────────────┘       └──────────────┘
  400 行, 1 表          +300 行, 0 表          +500 行, 2 表
  Token: 异步一次性      Token: 每轮 +200       Token: 每轮 +400
```

方案 A 是 B/C 的基础。P6 阶段预留的工具接口使得后续升级到方案 B 时只需在 `tools.py` 中注册新工具，无需改动底层 Store。
