# P3: 任务分解 / 规划 — 设计文档

**日期**: 2026-06-29
**范围**: 五阶段"从 RAG 到 Agent"升级的第 3 阶段
**依赖**: P1（agentic 检索循环已落地，作为可复用单元被 fan-out 调用）
**状态**: 待 review

## 1. 背景与动机

P1 让**检索**成环（`evaluate_evidence` → `route_after_evidence`，在 AgentState 子图内）。P2 让**回答**成环（`answer_grounding_check` → `route_after_grounding` → `revise_answer`，在主图 State 内）。P3 补上"agent"还缺的一块：**自主规划**——复合医学问题先被分解成多个独立 facet 子问题，并行检索，再聚合。

### 当前检索路径的真实情况（代码事实）

medical_rag 路径（`edges.py:8-126`）：
```
analyze_turn → intent_router → rewrite_query → route_after_rewrite → plan_retrieval_queries → route_after_query_plan → [单 Send] agent → grounded_answer_generation → answer_grounding_check → [P2 回路] → END
```

- `rewrite_query`（`rag_nodes.py:129`）已能产出 1-3 个**同义变体**（prompt 让 LLM 重写成 retrieval-friendly 查询），但都是**同一个问题的变体**，不是分解成不同子问题。
- `plan_retrieval_queries`（`rag_nodes.py:280`）是**规则式**节点（LLM 调用已移除，注释说规则扩展质量相当且零成本）：取第一个重写问题，追加"症状/治疗/注意事项"等后缀，产出 `planned_queries`（2-3 条，仍是同问题变体）。
- `route_after_query_plan`（`edges.py:94-126`）用 `Send` 但**只返回单元素列表** `[Send("agent", {question_index: 0, ...})]`——没有真正的 fan-out。
- `AgentState.question_index` 恒为 0；`State.agent_answers` 的 `accumulate_or_reset` reducer（`graph_state.py:6`）按 index 累加，`grounded_answer_generation`（`rag_nodes.py:606`）按 index 排序合成——**这套 fan-in 基建已存在但闲置**。
- `RetrievalQueryPlan` schema（`schemas.py:165`）存在但未被使用。

**结论**：系统知道"查询列表"，但列表里全是同一问题的变体，没有"独立子问题"概念、没有真正的并行 fan-out。fan-in 基建却已为它准备就绪。P3 的接入点是 `plan_retrieval_queries` 的位置——用真正的"分解"取代"同义变体扩展"。

### 本阶段目标

新增 `decompose_tasks` 节点（light LLM 判断复合 → 产出 1-N 个独立 facet 子问题）+ 把 `route_after_query_plan` 从单 Send 扩为多 Send 并行 fan-out。每个子问题进同一个 `agent` 子图，各自独立跑 **P1 的检索循环**——把 P1 的循环当可复用单元。fan-in 复用现有 `accumulate_or_reset` + `grounded_answer_generation`。

## 2. 非目标（YAGNI）

- **不**做 supervisor / 子问题间协调 / 动态重规划（P4）。
- **不**做子问题间证据共享（并行 fan-out，各子图独立）。
- **不**改 P1 的 `evaluate_evidence` / `route_after_evidence` / AgentState。
- **不**改 P2 的回答反思回路（P3 产出多条 `agent_answers` 后，`grounded_answer_generation` 合成单条最终回答，再进 P2——P2 看到的是合成后的单回答，行为不变）。
- **不**改 `rewrite_query`（它仍负责语义重写 + 意图再判；P3 接在它后面）。
- **不**改 `grounded_answer_generation` / `collect_answer`（fan-in 基建已就绪，P3 直接复用）。
- **不**碰非 medical_rag 意图路径（greeting/appointment/triage/clarification 在 P3 之前已分流，不进分解）。

## 3. 架构：主图拓扑变更

### 现有 medical_rag 段

```
rewrite_query → route_after_rewrite → plan_retrieval_queries → route_after_query_plan → [单 Send] agent → grounded_answer_generation → ...
```

### P3 后的 medical_rag 段

```
rewrite_query → route_after_rewrite → decompose_tasks → route_after_query_plan → [Send×N 并行] agent → grounded_answer_generation → ...
                                                                                              ↑
                                          每个 Send 独立进 agent 子图，各自跑 P1 检索循环
                                          collect_answer 写 agent_answers(index=i)
                                          accumulate_or_reset reducer 自动合并
                                          grounded_answer_generation 按 index 排序合成
```

**关键变化**：
1. `route_after_rewrite` 的 medical_rag 目标从 `plan_retrieval_queries` 改为 `decompose_tasks`。
2. `decompose_tasks`（新增节点）取代 `plan_retrieval_queries` 的位置：判断复合 → 产出 `sub_questions`（1-N 个独立子问题）。
3. `route_after_query_plan` 从返回 `[Send]`（单元素）改为返回 `[Send, Send, ...]`（每子问题一个），LangGraph 自动并行调度。
4. 每个子图独立跑 P1 检索循环（P3 不碰 AgentState）。
5. fan-in 复用现有 `accumulate_or_reset` + `grounded_answer_generation`，**零 fan-in 新代码**。

**为什么 fan-out 用 Send 而非 Command**：`Send` 是 LangGraph 并行 fan-out 的标准机制（P1 的 `route_after_query_plan` 已用 `Send`，只是单元素）。多 `Send` 即多并行子图实例。`Command` 用于动态改写状态+跳转（如 `should_compress_context`），不适合 fan-out。

**为什么取代 plan_retrieval_queries 而非叠加**：`plan_queries`（`tools.py:137`）给同问题加后缀产出变体；P1 的 `evaluate_evidence` 循环已在证据不足时自动 refine query 重检索——预计算变体不再必需。且对复合问题，每个 facet 各自聚焦检索比一组混合变体更准。净效果非回退，多数更优（§6 取舍详述）。

## 4. 组件设计

### 4.1 `decompose_tasks` 节点（`rag_nodes.py` 新增）

**职责**：判断医学问题是否复合；复合则拆成 1-N 个独立 facet 子问题；不复合则输出单元素列表（= 原问题，退化为今天的单 Send 路径）。

**输入**（从 `State` 读）：
- `originalQuery`：原始问题。
- `rewrittenQuestions`：rewrite_query 产出的重写列表（取首条作为分解对象；rewrite 已做语义规整）。
- `conversation_summary` / `recent_context` / `topic_focus`：上下文，用于理解"复合"。

**逻辑**：
1. 取 `primary_query`（`rewrittenQuestions` 首条，回退到 `originalQuery`）。
2. 调 light LLM（`_light_llm`）用 `TaskDecomposition` schema 判断：
   - `needs_decomposition: bool`（是否真复合——含多个独立 facet，如"高血压合并痛风吃什么药 + 怎么监测血压"）。
   - `sub_questions: List[str]`（不复合时为 `[primary_query]`；复合时为 1-N 个独立子问题，N ≤ `MAX_SUB_QUESTIONS`）。
   - `reason: str`（简短判断理由）。
3. 失败兜底：`_structured_output_llm` 永不抛（`node_helpers.py:114`，LLM/解析失败返回 schema 默认 `sub_questions=[]`）。若 `sub_questions` 为空或 `needs_decomposition=False` → 退回 `[primary_query]` 单路径，绝不挂图。
4. 截断：`sub_questions = sub_questions[:MAX_SUB_QUESTIONS]`。
5. 写回 State：`{"sub_questions": subs}`。

**输出 schema**（`schemas.py` 新增 `TaskDecomposition`）：
```python
class TaskDecomposition(BaseModel):
    needs_decomposition: bool = Field(description="用户问题是否包含多个可独立检索的子问题/facet。")
    sub_questions: List[str] = Field(description="分解后的独立子问题；不复合时为仅含原问题的单元素列表。")
    reason: str = Field(description="简短说明是否复合的判断依据。")
```

> 取舍：是否复用 `RetrievalQueryPlan`（已存在但闲置）？不复用——它的语义是"有序的检索变体查询"，P3 需要的是"是否复合 + 独立子问题 + 理由"，字段语义不同。新建 `TaskDecomposition` 更清晰，且让闲置的 `RetrievalQueryPlan` 仍可后续清理（本阶段不动它，避免 scope 蔓延）。

### 4.2 `route_after_query_plan` 改为 fan-out（`edges.py`）

```python
def route_after_query_plan(state) -> list[Send]:
    """P3: fan out one Send per sub-question; LangGraph runs them in parallel."""
    primary = (state.get("rewrittenQuestions") or [state.get("originalQuery", "")])[0]
    subs = state.get("sub_questions") or [primary]
    subs = [s for s in subs if str(s or "").strip()][:config.MAX_SUB_QUESTIONS]
    if not subs:
        subs = [primary]
    payload_base = {
        "messages": [],
        "context_summary": state.get("context_summary", ""),
        "recent_context": state.get("recent_context", ""),
        "topic_focus": state.get("topic_focus", ""),
        "user_memories": state.get("user_memories", ""),
    }
    return [
        Send("agent", {
            **payload_base,
            "question": q,
            "question_index": i,
            "query_plan": [q],
        })
        for i, q in enumerate(subs)
    ]
```

- 每子问题 `question_index=i`（终于用上非 0 index）；`query_plan=[q]`（子问题自身作检索起点，P1 循环在不足时 refine）。
- 其余 payload（context 等）各子图共享同一份（并行，无依赖）。
- 返回 `Send` 列表 → LangGraph 并行调度各子图实例。

### 4.3 `route_after_rewrite` 改指向（`edges.py`）

`route_after_rewrite` 当前 medical_rag 目标 `"plan_retrieval_queries"` 改为 `"decompose_tasks"`。其余映射（request_clarification / handle_appointment_skill / recommend_department / __end__ / skill targets）不变。

### 4.4 配置新增（`config.py`）

- `MAX_SUB_QUESTIONS = int(os.environ.get("MAX_SUB_QUESTIONS", "3"))`：fan-out 上限。
- `ENABLE_TASK_DECOMPOSITION = os.environ.get("ENABLE_TASK_DECOMPOSITION", "true").lower() == "true"`：开关。`False` 时 `decompose_tasks` 直接输出 `[primary_query]`（单路径，等价今天），可回滚。

### 4.5 State 新增字段（`graph_state.py` 的 `State`）

```python
sub_questions: List[str] = []
```

无 reducer（`decompose_tasks` 单点写，`route_after_query_plan` 单点读）。

### 4.6 图接线（`graph.py`）

- 删除 `plan_retrieval_queries` 节点注册 + `route_after_query_plan` 旧连边（`graph.py:91`、`graph.py:169`）。
- 注册 `decompose_tasks`（`partial(decompose_tasks, llm=_light_llm)`）。
- 连边：`decompose_tasks → route_after_query_plan`（条件边，返回 Send 列表）。
- `route_after_rewrite` 的条件映射 medical_rag 目标改为 `decompose_tasks`。
- `ENABLE_TASK_DECOMPOSITION=False` 时：`decompose_tasks` 节点仍注册但内部直接退回 `[primary_query]`（保持单路径），无需改拓扑——开关只控节点行为，不控拓扑，更简单可回滚。

> 注意：`plan_retrieval_queries` 节点函数 + `plan_queries` 工具 + `RetrievalQueryPlan` schema 本阶段**保留不删**（避免连带改动 `tools.py`、`schemas.py` 引入回归风险），仅从图中摘除接线。后续清理单列任务。

## 5. 数据流（一次"复合问题"的分解+并行）

```
用户："高血压合并痛风吃什么药安全，另外怎么在家监测血压？"
rewrite_query → rewrittenQuestions=["高血压合并痛风用药安全"]
  → route_after_rewrite (medical_rag) → decompose_tasks
decompose_tasks (light LLM):
  needs_decomposition=True
  sub_questions=["高血压合并痛风吃什么药安全", "高血压患者如何在家监测血压"]
  → 写 sub_questions
  → route_after_query_plan: 返回 [Send(q0, idx0), Send(q1, idx1)]
[并行]
  agent 子图 #0 (q0): orchestrator→tools→evaluate_evidence→...→collect_answer
    → agent_answers += {index:0, question:q0, answer:A0, ...}
  agent 子图 #1 (q1): orchestrator→tools→evaluate_evidence→...→collect_answer
    → agent_answers += {index:1, question:q1, answer:A1, ...}
[fan-in]
  grounded_answer_generation: 按 index 排序 [A0,A1], LLM 合成单条最终回答
  → answer_grounding_check (P2 回路)
  → END
```

简单问题路径（不复合）：`decompose_tasks` 输出 `[primary]` → `route_after_query_plan` 返回单 `Send` → 等价今天单路径。

## 6. 错误处理（复用现有，无新逻辑）

按已确认范围"聚合全部，部分降级"——全部是现有基建的默认行为，P3 不写新错误处理：

| 失败模式 | 现有处理（P3 不改） |
|---|---|
| 某子问题 P1 检索循环耗尽 | 该子图走 `fallback_response` → `collect_answer` 产出 `no_evidence`/`low` 条目 |
| 部分子答案低置信 | `grounded_answer_generation`（`rag_nodes.py:638`）：任一 `no_evidence` 则整体 `confidence_bucket=no_evidence`，贴免责声明 |
| 整体合成回答未 grounded | P2 反思回路接管（重写/复检） |
| `decompose_tasks` LLM 失败 | `_structured_output_llm` 返回默认 `sub_questions=[]` → 退回 `[primary_query]` 单路径 |
| `ENABLE_TASK_DECOMPOSITION=False` | `decompose_tasks` 退回 `[primary_query]`，等价今天单路径 |

## 7. 一个特意标注的取舍

`decompose_tasks` 取代 `plan_retrieval_queries` 后，**每子问题不再有预计算的检索变体**（原 `plan_queries` 给同问题加"症状/治疗"后缀）。但 P1 的 `evaluate_evidence` 循环已在证据不足时自动 refine query 重检索——预计算变体不再必需，P1 兜底。且对复合问题，每个 facet 各自聚焦检索比一组混合变体更准。**净效果：非回退，多数情况更优；单子问题路径行为等价今天**。我会用测试覆盖"单子问题路径等价"以证明非回退。

## 8. 测试策略

### 8.1 单元（`tests/test_task_decomposition.py` 新增）

- `decompose_tasks`：
  - 复合问题 → `needs_decomposition=True`，`sub_questions` 长度 2-3，写回 State。
  - 简单问题 → `needs_decomposition=False`，`sub_questions == [primary]`。
  - LLM 失败（返回默认 `sub_questions=[]`）→ 退回 `[primary]`，不抛异常。
  - `MAX_SUB_QUESTIONS` 截断：LLM 返回 5 个 → 截到 3。
  - `ENABLE_TASK_DECOMPOSITION=False` → 直接 `[primary]`，不调 LLM。
- `route_after_query_plan`：
  - N 子问题 → 返回 N 个 `Send`，`question_index` 0..N-1，`question`/`query_plan` 对应正确。
  - 空 sub_questions → 退回单 `Send`（primary）。

### 8.2 集成（编译图，仿 P1/P2 的 `TestCompiledGraphStateHandoff`）

- 编译最小主图段（decompose_tasks → route_after_query_plan → [agent sink]），fake LLM 返回 2 子问题 → 断言 2 个并行 agent 调用、2 条 `agent_answers`（index 0/1）。证明 fan-out 真走通 LangGraph 的 Send 机制。
- 终止/简单用例：1 子问题 → 单 Send。

### 8.3 回归

- 更新 `tests/test_routing_edges.py`：`route_after_rewrite` medical_rag 目标断言从 `plan_retrieval_queries` 改为 `decompose_tasks`；`route_after_query_plan` 断言从单 Send 改为可多 Send。
- `tests/test_retrieval_quality_loop.py`：若触达 `plan_queries` 的用例需调整（P3 从图中摘除 `plan_retrieval_queries`，但 `plan_queries` 工具函数保留，单测它应仍绿）。
- P1 的 20 个测试（`test_agentic_retrieval`）不受影响（P3 不碰 AgentState）。
- P2 的 16 个测试（`test_answer_reflection`）不受影响（P3 产出多 `agent_answers` 后由 `grounded_answer_generation` 合成单回答进 P2，P2 看到的是单回答）。
- 全量 `unittest discover`：新用例全 PASS；既有失败与 main 基线一致（mojibake / 缺 token），**P3 不引入新回归**。

## 9. 对面试叙事的直接支撑

完成后，"agent"补上自主规划这一块：

> 复合医学问题不是一把梭检索。我加了 `decompose_tasks`：light LLM 先判断问题是否含多个独立 facet，复合就拆成 1-3 个子问题，用 LangGraph 的 `Send` 并行 fan-out，每个子问题独立跑 P1 的检索循环（把检索循环当可复用单元），最后复用已有的 `agent_answers` 聚合基建合成。简单问题自动退化为单路径，零额外延迟。

连同 P1（检索成环）、P2（回答成环），系统有了"检索-回答-规划"三类 agent 行为。P4 多 agent supervisor 会把这套重新组织。

## 10. 风险与取舍

- **并行无证据共享**：各子图独立检索，可能重复检索相似内容。缓解：复合问题的 facet 本就应不同；重复检索的开销可接受（pgvector 查询快）；P4 可加共享证据池。
- **延迟**：并行 fan-out 延迟 ≈ 最慢子图（非 N 倍）。分解多一次 light LLM 调用（~几百 ms）。简单问题退单路径无额外延迟。
- **分解质量**：LLM 可能误判复合/过度拆分。缓解：`MAX_SUB_QUESTIONS=3` 上限；不复合明确退单路径；P2 反思兜底合成质量。
- **取代 plan_retrieval_queries 的非回退性**：§7 已述，用"单子问题路径等价"测试覆盖。
- **fan-in 依赖现有基建**：`grounded_answer_generation` 原为单 answer 设计但已遍历 `agent_answers` 列表——需测试确认多 answer 合成正确（无拼接错乱）。

## 11. 验收标准

1. `decompose_tasks` 节点 + `TaskDecomposition` schema + `sub_questions` State 字段落地。
2. `route_after_query_plan` 扩为多 `Send` fan-out（每子问题一个，`question_index` 0..N-1）。
3. `route_after_rewrite` medical_rag 目标改为 `decompose_tasks`。
4. `plan_retrieval_queries` 从图中摘除（节点函数/s工具/schema 保留）。
5. `MAX_SUB_QUESTIONS` 截断生效；`ENABLE_TASK_DECOMPOSITION` 开关可回滚单路径。
6. 复合问题并行 fan-out → 多 `agent_answers` → `grounded_answer_generation` 合成（集成测试证明）。
7. 简单问题退化为单路径，行为等价今天（测试覆盖非回退）。
8. 全量 unittest 不引入新回归；P1 的 20 个、P2 的 16 个不受影响。
9. 至少一个编译图集成用例证明 fan-out+fan-in 真走通 LangGraph。
