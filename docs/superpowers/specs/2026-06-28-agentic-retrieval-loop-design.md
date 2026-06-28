# P1: Agentic 检索循环 — 设计文档

**日期**: 2026-06-28
**范围**: 五阶段"从 RAG 到 Agent"升级的第 1 阶段（地基阶段）
**状态**: 待 review

## 1. 背景与动机

面试官评价本项目"也就一个 RAG"。代码盘点后确认根因：**检索、规划、回答三个阶段都是一次性管道，没有 agent 循环。** 本阶段针对其中最关键的一环——**检索**——把它从"单发 + 写死的一次重试"升级为"自我评估的多轮循环"。

### 当前检索的真实情况（代码事实）

- 检索证据是否"够用"由 `check_sufficiency()`（`tools.py:190`）判定，这是一段**确定性规则**（按 high_grade 数量和分数阈值），不是 LLM 反思。
- 不够用时，`_search_child_chunks`（`tools.py:552-672`）在工具**内部**用 `retry_query` 再查一次，上限 `RAG_RETRY_LIMIT = 1`。**LLM 从不知道、也不参与**这次重试决策。
- orchestrator 子图（`graph.py:56-71`）本身是 ReAct 式工具调用循环，被 `MAX_ITERATIONS=10` / `MAX_TOOL_CALLS=8` 和两个重复检测守卫（`edges.py:207-229`）约束。但"证据够不够"这个判断没有暴露给 LLM，循环只是被动地"再调一次工具"。
- orchestrator prompt 里只有一句软提示"first retrieval weak 时可试一个 alternate query"（`rag_nodes.py:330-332`），不是结构化反思。

**结论**：检索虽有多轮壳子，但没有"反思 → 主动改写 query → 重检索"的 agent 内核。这就是"还是 RAG"的硬证据之一。

### 本阶段目标

把证据充分性判断从**工具内的死规则**提升为**graph 层的 LLM 反思节点**，并据此驱动"改写 query → 重检索"的闭环，带显式终止守卫。完成后：

- 检索不再是单发管道，而是自我评估的多轮循环。
- 留下 `evaluate_evidence` 这个可独立 demo、可独立讲原理的节点，为 P2（反思回路）和 P3（任务分解，把检索循环当可复用单元）打地基。

## 2. 非目标（YAGNI）

- **不**改 `plan_retrieval_queries` 的规则式查询生成（那是 P3 的任务分解）。
- **不**改 `answer_grounding_check` 的"闸门 → END"行为（那是 P2 的反思回路）。
- **不**做多 agent / supervisor（P4）。
- **不**接在线自评（P5）。
- **不**移除 `check_sufficiency`——它降级为 LLM 反思的廉价前置快路径和失败兜底，保留以降低成本和保证可用性。

## 3. 架构：子图拓扑变更

### 现有 agent 子图

```
START -> orchestrator
  orchestrator --[route_after_orchestrator_call]--> tools | fallback_response | collect_answer
  tools -> should_compress_context
  should_compress_context --[token_budget]--> compress_context | orchestrator
  compress_context -> orchestrator
  fallback_response -> collect_answer
  collect_answer -> END
```

### P1 后的 agent 子图

```
START -> orchestrator
  orchestrator --[route_after_orchestrator_call]--> tools | fallback_response | collect_answer
  tools -> evaluate_evidence                ← 新增节点
  evaluate_evidence --[route_after_evidence]--> should_compress_context | fallback_response
  should_compress_context --[token_budget]--> compress_context | orchestrator   ← 不变
  compress_context -> orchestrator
  fallback_response -> collect_answer
  collect_answer -> END
```

**关键变化**：
1. `tools` 不再直接连 `should_compress_context`，而是先经过新节点 `evaluate_evidence`。
2. `evaluate_evidence` 产出结构化判断后由新边 `route_after_evidence` 路由：
   - **证据充分** 或 **证据不足但仍有进展空间** → `should_compress_context`（压缩逻辑不变），随后回到 `orchestrator`：充分时 orchestrator 不再发工具调用 → `collect_answer`；不足时由 4.3 的 refined_query 注入驱动下一次 `search_child_chunks`。
   - **证据不足且预算耗尽 / 无进展** → `fallback_response`（走已有的降级回答路径，不经压缩）。
3. `should_compress_context` 及其 token 预算逻辑**完全不变**，只是入边来源从 `tools` 改为 `evaluate_evidence`。压缩与证据评估是正交关注点，不混在一个节点里；且 `evaluate_evidence` 必须在压缩之前读到**完整的最新工具结果**，故顺序是 tools → evaluate_evidence → should_compress_context → orchestrator。

**为什么 evaluate_evidence 是独立节点而不是塞进 orchestrator**：独立节点让"反思"成为一等公民——可单测、可观测、可 demo，且符合"agent 循环"叙事。塞进 orchestrator 会让它退回成一句软提示。

## 4. 组件设计

### 4.1 `evaluate_evidence` 节点（`rag_nodes.py` 新增）

**职责**：拿到最近一次检索的工具结果 + 原始/改写后的问题，判断证据是否足以回答；不足时给出 critique 和 refined_query。

**输入**（从 `AgentState` 读）：
- `messages`：取最后一条 `ToolMessage` 作为本次检索结果；取 `question` / `query_plan` 作为问题上下文。
- `evidence_rounds`：已反思轮数（新字段）。
- `refined_queries`：历史 refined_query（新字段，用于无进展检测）。

**逻辑**：
1. **快路径**：若 `check_sufficiency()`（规则）判定 `is_sufficient=True`，直接返回充分结论，**跳过 LLM 调用**——省成本，保留规则的价值。
2. **LLM 反思路径**：规则判为不足时，调用轻量 LLM（`_light_llm`，与意图分类同档），用结构化输出 schema `EvidenceSufficiencyCheck`：
   - `sufficient: bool`
   - `critique: str`（为什么不够：证据偏离问题 / 只覆盖部分 / 分数过低 / 噪声多）
   - `refined_query: str | None`（不足时给出一个更优检索式；充分时为 None）
3. **LLM 失败兜底**：反思调用抛错或解析失败 → 退回 `check_sufficiency` 的结论（含其 `retry_query`）。保证 LLM 不可用时系统不挂。
4. 写回 state：`evidence_critique`、`last_refined_query`、`evidence_rounds += 1`，并把 `refined_query` 追加进 `refined_queries`。

**输出 schema**（`schemas.py` 新增 `EvidenceSufficiencyCheck`）：
```python
class EvidenceSufficiencyCheck(BaseModel):
    sufficient: bool
    critique: str = ""
    refined_query: str | None = None
```

### 4.2 `route_after_evidence` 边（`edges.py` 新增）

```
route_after_evidence(state) -> "should_compress_context" | "fallback_response"
```

**判定顺序**：
1. 若 `evaluate_evidence` 写入的本次判断为 `sufficient=True` → `"should_compress_context"`（随后 orchestrator 走向 collect_answer）。
2. 终止守卫触发 → `"fallback_response"`：
   - `evidence_rounds >= MAX_EVIDENCE_ROUNDS`（新配置，默认 2）。
   - `last_refined_query` 与 `refined_queries` 中已有 query 重复（无进展，复用 `_has_repeated_search_query` 模式）。
   - `_has_repeated_no_evidence(state)` 仍命中（连续 NO_EVIDENCE）。
3. 否则（不足且有进展空间）→ `"should_compress_context"`，随后回到 `orchestrator`，由 4.3 的 refined_query 注入驱动重检索。

> 两个分支都先经 `should_compress_context` 管理预算再回 orchestrator；只有终止分支直接走 `fallback_response`，不经压缩。

注意：`MAX_ITERATIONS` / `MAX_TOOL_CALLS` 仍在 `route_after_orchestrator_call` 里兜底，两层守卫叠加，不会因新增循环而失控。

### 4.3 refined_query 注入（`orchestrator` 节点改动）

`orchestrator`（`rag_nodes.py:291`）在构造 `base_messages` 时，若 state 中存在 `last_refined_query` 且本轮是"反思后的重检索"，追加一条 HumanMessage 提示：

> 上一次检索证据不足，原因：{evidence_critique}。请用以下检索式重新调用 search_child_chunks，不要重复之前的查询：{last_refined_query}

注入后清掉 `last_refined_query` 标记，避免反复注入同一条。这样 LLM 的下一次工具调用自然带上 refined_query，而不是靠软提示自己猜。

### 4.4 工具内部重试的处置（`tools.py` 改动）

`_search_child_chunks` 内部的 `check_sufficiency` + 一次 retry **保留但降级**：
- 保留：作为工具内最末端的兜底，且 `check_sufficiency` 现在还服务 4.1 的快路径。
- 降级：把"重试决策权"上移到 graph 层的 `evaluate_evidence`。工具内 retry 仍可触发一次（处理工具自身视角的明显空召回），但 graph 层的反思循环成为主路径。
- 不删除 `check_sufficiency` 和 `RAG_RETRY_LIMIT`，避免破坏既有基准测试和 fallback 行为。

> 取舍说明：彻底移除工具内 retry 风险高、收益小（要回归大量检索用例）。把它留作"工具内最后一搏"，让 graph 层反思做主导，是更稳的渐进路线。P2/P3 视情况再决定是否完全上移。

### 4.5 配置新增（`config.py`）

- `MAX_EVIDENCE_ROUNDS = 2`：反思循环上限。结合 `MAX_ITERATIONS=10`，最坏情况检索相关迭代受 `min` 约束，不会爆。
- `ENABLE_AGENTIC_RETRIEVAL = True`：开关。`False` 时 graph 回退到 P1 前拓扑（`tools -> should_compress_context`），用于 A/B 和回滚。

### 4.6 State 新增字段（`graph_state.py` 的 `AgentState`）

```python
class AgentState(MessagesState):
    ...
    evidence_rounds: int = 0
    evidence_critique: str = ""
    last_refined_query: str = ""
    refined_queries: Annotated[List[str], operator.add] = []  # 累积，供无进展检测
```

`evidence_rounds` 用 `operator.add` 还是覆盖？反思节点每轮显式 `+1` 返回，故用默认覆盖语义即可（节点返回 `{"evidence_rounds": current + 1}`）。

## 5. 数据流（一次"需要两轮"的检索）

```
user: "高血压合并痛风吃什么药安全"   (知识库初次召回偏向单一病种)
  rewrite_query -> plan_retrieval_queries -> [Send] -> agent subgraph
  orchestrator --search_child_chunks("高血压 痛风 用药")--> tools
  tools -> evaluate_evidence
    check_sufficiency 判 False -> LLM 反思:
      critique="证据只覆盖高血压用药，未涉及与痛风的交互"
      refined_query="高血压 合并痛风 药物 相互作用 禁忌"
    -> evidence_rounds=1
  route_after_evidence: 不足且有进展 -> should_compress_context -> orchestrator (注入 refined_query)
  orchestrator --search_child_chunks(refined_query)--> tools
  tools -> evaluate_evidence
    LLM 反思 sufficient=True (命中指南中联合用药禁忌段落)
  route_after_evidence: 充分 -> should_compress_context -> orchestrator (无 tool_calls) -> collect_answer -> END
```

无进展场景：refined_query 与上一轮重复 → `route_after_evidence` → `fallback_response`（已有降级回答 + 免责提示）。

## 6. 错误处理

| 失败模式 | 处理 |
|---|---|
| LLM 反思调用抛错 / 超时 | 退回 `check_sufficiency` 结论，记日志，不中断 |
| 结构化输出解析失败 | 同上，退回规则结论 |
| refined_query 为空但判不足 | 视为无进展，走 `fallback_response` |
| 反思循环达 `MAX_EVIDENCE_ROUNDS` | `route_after_evidence` → `fallback_response` |
| `MAX_ITERATIONS` / `MAX_TOOL_CALLS` 仍被触发 | `route_after_orchestrator_call` 原有兜底，不变 |
| `ENABLE_AGENTIC_RETRIEVAL=False` | graph 编译时用旧拓扑，零行为变化，可随时回滚 |

所有降级路径都落到已有的 `fallback_response`（含医疗免责与高风险就医提醒），不会产出"伪装成有证据"的回答——这是现有安全兜底，本阶段不削弱。

## 7. 测试策略

### 7.1 单元（`tests/test_agentic_retrieval.py` 新增）

- `evaluate_evidence` 快路径：规则判充分时不调 LLM（mock 计数验证 0 次 LLM 调用）。
- `evaluate_evidence` 反思路径：规则判不足、LLM 返回 `sufficient=True` → state 写入正确，边路由到 `orchestrator`。
- `evaluate_evidence` LLM 失败兜底：mock LLM 抛错 → 退回 `check_sufficiency.retry_query`，不抛异常。
- `route_after_evidence` 守卫：`evidence_rounds` 达上限 → `fallback_response`；refined_query 重复 → `fallback_response`。

### 7.2 集成（复用 `tests/test_api_app.py` 模式）

- 一个需要 refined_query 才能命中的 fixture：第一轮召回弱证据 → 断言发生了第二次 `search_child_chunks` 且 query 是 refined_query → 最终答案 grounded。
- 一个单轮即充分的 fixture：断言只检索一次，未触发反思重检索（不回归现有行为，不加延迟）。

### 7.3 回归

- `python -m unittest discover -s tests -v` 全绿。
- `scripts/smoke_split_app.ps1 -SkipChat` 通过。
- 用 `ENABLE_AGENTIC_RETRIEVAL=False` 跑一遍，确认拓扑回退后行为与改动前一致。

### 7.4 评测（可选，复用 `project/benchmarks/`）

- 在现有 NHC/WHO 指南集上对比 P1 前/后 Precision@5 与 Top-1，记录趋势（预期：原本需要多轮的复杂/多病种问题命中提升；简单问题持平）。**不**预先承诺具体数字，只在跑出后据实记录。

## 8. 对面试叙事的直接支撑

完成后，"还是 RAG"的批评可正面回答：

> 检索不是单发管道。我加了 `evaluate_evidence` 反思节点，让轻量 LLM 判断证据是否足以回答；不足时主动生成 refined_query 重检索，带轮次上限和无进展检测守卫。规则 `check_sufficiency` 保留为廉价快路径和 LLM 失败兜底。这样检索变成了自我评估的多轮循环，而不是一次性召回。

并且这是 P2（回答反思回路）和 P3（任务分解把检索循环当可复用单元）的地基——三个阶段都成环后，才构成"从 RAG 到 Agent"的完整叙事。

## 9. 风险与取舍

- **延迟增加**：反思路径多一次轻量 LLM 调用。缓解：快路径跳过 LLM；用 light tier；`MAX_EVIDENCE_ROUNDS` 收紧到 2。
- **工具内 retry 与 graph 反思重复**：短期内两者并存，可能多查一次。取舍为换稳定渐进，P3 时再统一。
- **复杂度**：子图多一个节点和一条边。可接受，因为可被 `ENABLE_AGENTIC_RETRIEVAL` 开关隔离。

## 10. 验收标准

1. `evaluate_evidence` 节点 + `route_after_evidence` 边 + 4 个新 state 字段落地。
2. 快路径（规则充分）不调 LLM；反思路径驱动重检索；失败降级到规则。
3. 三个终止守卫（轮次上限、refined_query 重复、连续 NO_EVIDENCE）均生效。
4. `ENABLE_AGENTIC_RETRIEVAL` 开关可回滚到旧拓扑。
5. 全量 unittest + smoke 通过；现有检索用例不回归。
6. 至少一个"需要两轮"的集成用例证明闭环真的会重检索并改进结果。
