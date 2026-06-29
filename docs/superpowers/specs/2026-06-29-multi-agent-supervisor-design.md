# P4: 多 Agent Supervisor — 设计文档

**日期**: 2026-06-29
**范围**: 五阶段"从 RAG 到 Agent"升级的第 4 阶段
**依赖**: P1（agentic 检索循环）、P2（回答反思回路）、P3（任务分解并行）均已落地于 main
**状态**: 待 review

## 1. 背景与动机

P1-P3 让医学问答路径有了真正的 agent 行为：检索成环（P1）、回答成环（P2）、自主规划（P3）。但顶层仍是**静态分类链**——`analyze_turn`（规则）→ `intent_router`/`rewrite_query`（LLM 分类）→ `route_after_rewrite` 固定边映射。路径一旦进入 medical_rag，就一路跑到 `answer_grounding_check → END`，没有任何节点在医疗答案产出后**观察答案并动态决定是否派发下一个 agent**。

面试官说"也就一个 rag，没有真正的 agent 特征"——缺的正是**多 agent 协作**：一个 supervisor 协调多个专家 agent，根据上一个 agent 的产出决定下一步。

### 当前 medical_rag 终止路径（代码事实）

```
grounded_answer_generation → answer_grounding_check → [P2: revise_answer → answer_grounding_check] → END
```

- `route_after_grounding`（`edges.py:281`）：grounded/预算耗尽 → `__end__`；未 grounded 有预算 → `revise_answer`。**P2 回路结束即 END，无下游 agent 派发。**
- `route_after_action`（`edges.py:174`）：被 4 个动作型 specialist（`handle_appointment_skill`/`handle_appointment`/`handle_cancel_appointment`/`recommend_department`）共用，检查 `pending_clarification` / `secondary_intent`+`deferred_user_question` → `request_clarification`/`prepare_secondary_turn`/`__end__`。**只处理动作型 specialist 的收尾，不处理 medical_rag 收尾。**
- `prepare_secondary_turn`（`routing_nodes.py:921`）：仅当 `secondary_intent`+`deferred_user_question` 同时存在时把 deferred 问题重注入，跑第二意图——**只在动作型 specialist 优先（appointment-first + medical-deferred）时触发**；medical-primary 轮若规则没拆出 secondary，医疗答案产出后直接 END，无法在同轮内 handoff 去挂号/分诊。

**结论**：缺一个位于 medical_rag 出口的 **supervisor**——观察医疗答案 + 原始查询，用 LLM 判断用户是否暗示了一个后续动作意图（挂号/分诊），若有则派发对应 specialist，循环至上限或 FINISH。这是补齐"多 agent 协作"这一块。

### 本阶段目标

新增 `supervise` 节点（light LLM）+ `route_after_supervisor` 边，包住 medical_rag 出口：医疗答案（经 P2 反思）产出后，supervisor 决定同轮内是否派发 {appointment, triage} specialist；specialist 跑完回 supervisor，可再链一轮，上限 `MAX_SUPERVISOR_ROUNDS`。非 medical_rag 意图（greeting/appointment/triage/cancel 的静态快路径）**不动**。

## 2. 非目标（YAGNI）

- **不**取代顶层路由（`route_after_rewrite`/`route_after_intent`）。非医疗意图仍走静态分类快路径。
- **不**把 clarification 纳入 supervisor roster。澄清由现有 intent/clarification 机器 + `interrupt_before=["request_clarification"]` 处理，引入 supervisor 会和跨轮 interrupt 状态纠缠，YAGNI。
- **不**让 supervisor 重派 medical_rag agent（RAG→RAG 重试）。RAG 的自我纠正已由 P1（检索成环）+ P2（回答成环）覆盖；supervisor 不重复它们。
- **不**改 AgentState 子图（P1 检索循环）。supervisor 在主图 State 层。
- **不**改 `prepare_secondary_turn` / `route_after_action` 的既有 secondary-turn 逻辑（动作优先型复合请求仍由它处理）；supervisor 是**附加**的 medical-primary 型 handoff 路径。
- **不**做子问题间证据共享 / 动态重规划（P3 已并行，P4 不碰）。
- **不**碰 appointment pending 状态机 / `route_after_action` 的既有三条分支语义，只**新增**一条 `supervisor_active` 分支。

## 3. 架构：主图拓扑变更

### 现有 medical_rag 终止段

```
grounded_answer_generation → answer_grounding_check → [P2: revise_answer → answer_grounding_check] → END
```

### P4 后

```
grounded_answer_generation → answer_grounding_check → [P2 回路] → supervise (NEW)
                                                                    ↓ route_after_supervisor
                                              ┌─────────────────────┼────────────────────┐
                                              ▼                     ▼                    ▼
                                  handle_appointment_skill   recommend_department        END (FINISH)
                                  (appointment agent)        (triage agent)
                                              │                     │
                                              └─────────┬───────────┘
                                                        ▼ route_after_action
                                          [无 pending + supervisor_active] → supervise (循环)
                                          [有 pending_clarification]     → request_clarification (既有)
                                          [有 secondary+deferred]        → prepare_secondary_turn (既有)
                                          [否则]                          → END (既有)
```

- **新增入口边**：`START → reset_supervisor_state → analyze_turn`（`reset_supervisor_state` 在 analyze_turn 前清掉上一轮残留的 `supervisor_active`/`supervisor_rounds`，防跨轮泄漏，§6）。
- **supervisor 位置**：`answer_grounding_check` 的 `__end__` 出口改为 `supervise`（supervisor 开启时）；`revise_answer` 回路不变。
- **循环回路**：specialist 跑完经 `route_after_action`，当无 pending 且 `supervisor_active=True` 时回到 `supervise` 而非 END。
- **非医疗路径不变**：greeting/appointment/triage/cancel 从 `intent_router` 静态分流，不经 supervisor。

**为什么用条件边循环而非 `Command`**：P1（`route_after_evidence`）和 P2（`route_after_grounding`）都是"条件边 + 计数器"模式。supervisor 循环沿用同一模式，测试/审阅心智一致；`Command` 在本代码库仅用于状态改写（`should_compress_context`），不用于控制流。

## 4. 组件设计

### 4.1 `supervise` 节点（`rag_nodes.py` 新增，light tier）

**职责**：观察医疗答案 + 原始查询，用 light LLM 判断是否派发一个 peer action-agent（appointment/triage）；否则 FINISH。

**输入**（从 `State` 读）：
- `originalQuery` / `primary_user_query`：用户原始问题。
- `agent_answers`：刚产出的医疗答案（取最后一条的 answer/confidence）。
- `secondary_intent` / `deferred_user_question`：analyze_turn 可能已拆出的第二意图（规则复合检测的产物）——supervisor 把它当作 handoff 的强信号输入，但**不直接信任**它做派发（LLM 再判一次）。
- `conversation_summary` / `recent_context` / `topic_focus`：上下文。
- `supervisor_rounds`：已循环轮数（预算 guard）。

**逻辑**：
1. **预算/开关 guard**：`not ENABLE_MULTI_AGENT_SUPERVISOR` 或 `supervisor_rounds >= MAX_SUPERVISOR_ROUNDS` → 返回 `{"supervisor_active": False, "supervisor_rounds": 0, "supervisor_next": "FINISH"}`（**不调 LLM**，重置计数）。
2. 否则调 light LLM + `SupervisorDecision` schema：
   - `next_agent: Literal["appointment", "triage", "FINISH"]`。
   - `reason: str`（简短判断依据）。
3. **FINISH**：返回 `{"supervisor_active": False, "supervisor_rounds": 0, "supervisor_next": "FINISH"}`（结束并重置，下一轮 medical_rag 从 0 开始）。
4. **派发**：返回 `{"supervisor_active": True, "supervisor_rounds": rounds+1, "supervisor_next": <agent>, "secondary_intent": "", "deferred_user_question": ""}`。清空 `secondary_intent`/`deferred_user_question` 是**关键**——supervisor 已消费这个 handoff 信号，必须清掉，否则 `route_after_action` 的 `prepare_secondary_turn` 分支会把它当未处理复合请求再触发，重复派发。
5. **失败兜底**：`_structured_output_llm` 永不抛（`node_helpers.py`，LLM/解析失败返回 schema 默认）。`SupervisorDecision` 的 `next_agent` 是 `Literal` 字段——P3 修复的 `_default()` 对非 str 标量字段默认为 `""`，对 `Literal["a","b","c"]` 会落到 `""`。supervise 节点在拿到 verdict 后**显式校验** `next_agent ∈ {"appointment","triage","FINISH"}`，不合法一律当 FINISH。绝不挂图。

**输出 schema**（`schemas.py` 新增 `SupervisorDecision`）：
```python
class SupervisorDecision(BaseModel):
    next_agent: Literal["appointment", "triage", "FINISH"] = Field(
        description="下一步派发的专家 agent；无需后续动作时为 FINISH。"
    )
    reason: str = Field(description="简短说明为何派发该 agent 或 FINISH。")
```

**prompt**（`prompts.py` 新增 `get_supervisor_prompt()`）：给 LLM 角色"医疗助手 supervisor"，列出 roster 与各自职责，要求：仅当用户原始查询明确暗示了挂号或分诊需求、且医疗答案尚未满足该需求时派发；纯医学知识问答、闲聊、已完成动作 → FINISH。严格 JSON 输出。

### 4.2 `reset_supervisor_state` 节点（`rag_nodes.py` 新增，无 LLM）

**职责**：每轮入口清掉上一轮残留的 supervisor 状态。单职责、零 LLM、不改 `analyze_turn` 的 6 条返回路径（低回归）。

```python
def reset_supervisor_state(state: State):
    """P4: clear supervisor loop flags at turn start to prevent cross-turn leak."""
    return {"supervisor_active": False, "supervisor_rounds": 0}
```

第一轮是 no-op（State 默认即 False/0）。**为什么必须**：LangGraph checkpointer 跨轮持久化 State。若 supervisor 派发的 specialist 触发 `request_clarification`（如挂号需澄清）→ `interrupt_before` → 下一轮用户回复恢复 specialist，此时残留的 `supervisor_active=True` 会让 `route_after_action` 错误地把 specialist 路由回 `supervise` 而非正常收尾。

### 4.3 `route_after_supervisor` 边（`edges.py` 新增）

```python
def route_after_supervisor(state: State) -> str:
    """P4: dispatch the supervisor's chosen agent, or finish."""
    nxt = str(state.get("supervisor_next", "FINISH") or "FINISH").strip()
    if nxt == "appointment":
        return "handle_appointment_skill"
    if nxt == "triage":
        return "recommend_department"
    return "__end__"
```

- `appointment` → `handle_appointment_skill`（与 `route_after_rewrite` 一致的 appointment 入口）。
- `triage` → `recommend_department`。
- `FINISH`/未知 → `__end__`。

### 4.4 `route_after_grounding` 改指向（`edges.py`）

`answer_grounding_check` 的 `__end__` 出口，在 supervisor 开启时改为 `supervise`：

```python
def route_after_grounding(state: State) -> str:
    if bool(state.get("grounding_passed", False)):
        return "supervise" if config.ENABLE_MULTI_AGENT_SUPERVISOR else "__end__"
    rounds = int(state.get("grounding_rounds", 0) or 0)
    if rounds < MAX_GROUNDING_ROUNDS:
        return "revise_answer"
    return "supervise" if config.ENABLE_MULTI_AGENT_SUPERVISOR else "__end__"
```

（注：原返回类型 `Literal["__end__", "revise_answer"]` 改为 `str`，因新增 `supervise`。）`revise_answer → answer_grounding_check` 回路不变——P2 反思先跑完，supervisor 在 P2 收尾后才介入。

> edges.py 使用直接名导入（`from config import ...`），需把 `ENABLE_MULTI_AGENT_SUPERVISOR` 加进 line-5 的 import。

### 4.5 `route_after_action` 新增 supervisor 分支（`edges.py`）

现有三条分支（`request_clarification` / `prepare_secondary_turn` / `__end__`）不变，**新增**一条优先级最低的 `supervisor_active` 分支：

```python
def route_after_action(state: State) -> str:
    if state.get("pending_clarification") and state.get("clarification_target"):
        return "request_clarification"
    if (
        state.get("secondary_intent")
        and state.get("deferred_user_question")
        and not state.get("pending_clarification")
        and not state.get("pending_action_type")
        and not state.get("pending_candidates")
        and not state.get("deferred_confirmation_action")
    ):
        return "prepare_secondary_turn"
    if bool(state.get("supervisor_active", False)):
        return "supervise"
    return "__end__"
```

返回类型从 `Literal[...]` 改 `str`（新增 `supervise`）。`supervisor_active` 分支放最后——pending clarification / secondary turn 优先级更高（这些是更明确的收尾信号）。

4 个 specialist 的 conditional map（`graph.py` 中 `handle_appointment_skill`/`handle_appointment`/`handle_cancel_appointment`/`recommend_department`）各加 `"supervise": "supervise"`。

### 4.6 State 新增字段（`graph_state.py` 的 `State`）

```python
supervisor_active: bool = False
supervisor_rounds: int = 0
supervisor_next: str = "FINISH"
```

无 reducer（`supervise` 单点写，`route_after_*` 单点读；`reset_supervisor_state` 单点写覆盖）。

### 4.7 配置新增（`config.py`，接在 P3 块后 line 94 之后）

```python
# P4: multi-agent supervisor — LLM-coordinated agent handoff after medical_rag
MAX_SUPERVISOR_ROUNDS = int(os.environ.get("MAX_SUPERVISOR_ROUNDS", "3"))
ENABLE_MULTI_AGENT_SUPERVISOR = os.environ.get("ENABLE_MULTI_AGENT_SUPERVISOR", "true").lower() == "true"
```

`False` 时：`route_after_grounding` 直接 `__end__`，`supervise` 不注册/不调用——等价今天 P1-P3 行为，可回滚。

### 4.8 图接线（`graph.py`）

- 注册 `supervise`（`partial(supervise, llm=_light_llm)`）、`reset_supervisor_state`（无 partial）。
- `START → analyze_turn` 改 `START → reset_supervisor_state → analyze_turn`。
- `answer_grounding_check` 的条件边（P2 块内）mapping：`{"__end__": END, ...}` → `{"__end__": END, "supervise": "supervise", ...}`（supervisor 开启时；关闭时保持原样）。`route_after_grounding` 在开启时返回 `supervise`。
- 新增 `supervise` 的条件边：`add_conditional_edges("supervise", route_after_supervisor, {"handle_appointment_skill": "handle_appointment_skill", "recommend_department": "recommend_department", "__end__": END})`。
- 4 个 specialist 的 `route_after_action` mapping 各加 `"supervise": "supervise"`。
- `ENABLE_MULTI_AGENT_SUPERVISOR=False` 时：不注册 `supervise`，不改 `answer_grounding_check` 出口（保持 P2 的 `__end__`），不连 `route_after_supervisor`——拓扑回退到 P3 终态。`reset_supervisor_state` 仍注册（无害 no-op）。

### 4.9 `SILENT_NODES`（`chat_interface.py`）

`SILENT_NODES`（line 25）加 `"supervise"`、`"reset_supervisor_state"`——都是规划/协调节点，不应在前端流式 UI 冒 token（与 `decompose_tasks`/`answer_grounding_check` 同类）。

## 5. 数据流（一次"医疗问答 + 顺带挂号"的同轮 handoff）

```
用户："我高血压用药期间能打疫苗吗，顺便帮我挂个心内科"
START → reset_supervisor_state (清零 supervisor 标志)
  → analyze_turn (规则未拆出 secondary，primary_intent=medical_rag)
  → rewrite_query → decompose_tasks (P3: 单子问题，不复合)
  → [Send×1] agent 子图 (P1 检索循环 + collect_answer)
  → grounded_answer_generation (合成单条医疗答案 A0)
  → answer_grounding_check (P2: grounded)
  → supervise (light LLM 读 A0 + 原始查询):
      next_agent="appointment", reason="用户明示挂号心内科，医疗答案未覆盖挂号"
      → 写 supervisor_active=True, supervisor_rounds=1, 清空 secondary_intent/deferred
  → route_after_supervisor → handle_appointment_skill
      (appointment agent: 发现/规划/动作，可能 interrupt 澄清)
  → route_after_action:
      若 appointment 跑完无 pending → supervisor_active=True → supervise
  → supervise (rounds=1<3): light LLM 再判 → next_agent="FINISH" (挂号已处理)
      → supervisor_active=False, rounds=0
  → route_after_supervisor → END
```

简单医疗问答（无 handoff）：
```
... → answer_grounding_check → supervise → FINISH (一次 LLM 调用，无派发) → END
```

## 6. 跨轮状态泄漏与 `reset_supervisor_state`（特意标注）

LangGraph checkpointer（`PersistentInMemorySaver`）跨轮持久化整个 State。supervisor 引入两个跨轮敏感标志：`supervisor_active`、`supervisor_rounds`。若不清零：

- **场景**：supervisor 派发 appointment → appointment 触发 `request_clarification`（需用户确认医生/时段）→ `interrupt_before=["request_clarification"]` → 当前轮中断。下一轮用户回复"上午的"，恢复 specialist 跑完 → `route_after_action` 看到 `supervisor_active=True`（上一轮残留）→ 错误回 `supervise` 而非 END。
- **修复**：`reset_supervisor_state` 在每轮 `START` 后、`analyze_turn` 前无条件清零。这样上一轮的 supervisor 循环状态不会污染本轮的 specialist 收尾路由。`supervisor_rounds` 也随之归零——每轮 supervisor 从 0 开始计预算（语义正确：用户每轮新输入是一个新的 supervisor 决策起点）。

**为什么是独立节点而非塞进 `analyze_turn`**：`analyze_turn` 有 6 条复杂返回路径（pending resume、clarification resume、compound split 等），每条都返回大 dict。往里塞 supervisor 清零会污染所有路径、放大回归面。独立节点单职责、可独立测试、对 `analyze_turn` 零侵入。

## 7. 错误处理

| 失败模式 | 处理 |
|---|---|
| `supervise` LLM 失败 | `_structured_output_llm` 返回默认（`next_agent` 默认 `""`）→ 节点内显式校验不合法 → 当 FINISH → END。永不挂图。 |
| supervisor 循环失控 | `supervisor_rounds >= MAX_SUPERVISOR_ROUNDS` → 强制 FINISH（无 LLM 调用） |
| specialist 需澄清 | 既有 `interrupt_before=["request_clarification"]`；下一轮 `reset_supervisor_state` 清掉残留 `supervisor_active` |
| specialist 自身失败 | 各 specialist 既有 fallback（如 `recommend_department` 解析失败回全科）→ `route_after_action` → supervisor 继续 or FINISH |
| `ENABLE_MULTI_AGENT_SUPERVISOR=False` | `route_after_grounding` 直接 `__end__`，supervise 不注册；等价 P3 行为 |
| `supervisor_next` 残留非法值 | `route_after_supervisor` 未知值一律 `__end__` |

## 8. 测试策略

### 8.1 单元（`tests/test_multi_agent_supervisor.py` 新增）

- **config/state/schema**：`MAX_SUPERVISOR_ROUNDS`/`ENABLE_MULTI_AGENT_SUPERVISOR` 存在且类型对；`State` 有三新字段默认值；`SupervisorDecision` schema 字段。
- **`supervise` 节点**：
  - `ENABLE_MULTI_AGENT_SUPERVISOR=False` → FINISH，不调 LLM。
  - `supervisor_rounds >= MAX` → FINISH，不调 LLM。
  - LLM 返回 `appointment` → `supervisor_active=True`、`supervisor_rounds+1`、`supervisor_next="appointment"`、清空 `secondary_intent`/`deferred_user_question`。
  - LLM 返回 `triage` → 同上指向 triage。
  - LLM 返回 `FINISH` → `supervisor_active=False`、`rounds=0`。
  - **LLM 失败**（返回非法 `next_agent=""`）→ 当 FINISH，不抛异常（验证 §4.1 step 5 显式校验）。
  - **真路径 LLM 失败**（传 bare MagicMock LLM，不 patch `_structured_output_llm`）→ `_default()` 被触发 → FINISH，不抛（回归 P3 的 `_default()` 修复对 `Literal` 字段的行为）。
- **`reset_supervisor_state`**：返回 `{"supervisor_active": False, "supervisor_rounds": 0}`，不读其他字段。
- **`route_after_supervisor`**：appointment→`handle_appointment_skill`；triage→`recommend_department`；FINISH/未知→`__end__`。
- **`route_after_grounding`**：supervisor 开启时 grounded/预算耗尽 → `supervise`；关闭时 → `__end__`；未 grounded 有预算 → `revise_answer`（不变）。
- **`route_after_action`**：`supervisor_active=True` 且无 pending → `supervise`；有 pending_clarification 优先 → `request_clarification`；有 secondary+deferred 优先 → `prepare_secondary_turn`；否则 `__end__`。

### 8.2 集成（编译图，仿 P1/P2/P3 的 `TestCompiled*`）

- **多轮 handoff**：编译最小主图段（`answer_grounding_check → supervise → route_after_supervisor → [specialist sink] → route_after_action → supervise`），fake LLM：第一次 supervise 返回 `appointment`，specialist 跑完回 supervise，第二次返回 `FINISH`。断言：specialist 被调用 1 次、supervise 被调用 2 次、最终 `supervisor_active=False`、`supervisor_rounds` 归零。证明 supervisor 循环真走通 LangGraph 条件边。
- **简单 FINISH**：fake LLM supervise 直接返回 FINISH → 断言 specialist 未被调用、supervise 调用 1 次、END。

### 8.3 回归

- **`test_answer_reflection.py`**：`route_after_grounding` 的断言从 `__end__` 改为 `supervise`（supervisor 默认开启）。P2 回路（`revise_answer`）行为不变。
- **`test_routing_edges.py`**：若有 `route_after_grounding`/`route_after_action` 的期望值断言，按新返回值更新。
- **`test_task_decomposition.py`**（P3，19 个）：supervisor 在 `agent` 子图之外，P3 不触达 → 不受影响。
- **`test_agentic_retrieval.py`**（P1，20 个）：supervisor 在 AgentState 子图之外 → 不受影响。
- **`test_chat_interface.py`**：`SILENT_NODES` 断言加 `supervise`/`reset_supervisor_state`。
- **全量 `unittest discover`**：新用例全 PASS；既有失败与 main 基线一致（mojibake/缺 token），**P4 不引入新回归**。

## 9. 对面试叙事的直接支撑

> 不是单一 RAG 了。医疗 agent（检索成环 P1 + 回答反思 P2 + 任务分解 P3）产出答案后，一个 LLM **supervisor** 观察答案与原始查询，决定是否在同轮派发 peer agent——挂号或分诊——循环至上限或 FINISH。所以"我高血压用药期间能打疫苗吗，顺便挂个心内科"会先拿到 grounded 医疗答案，再被 supervisor 协调 handoff 去挂号，LLM 决策而非硬编码路由。P1-P3 成了医疗 agent 的内部；supervisor 是缺的那层多 agent 协调器。

连同 P1（检索成环）、P2（回答成环）、P3（自主规划），系统有了"检索-回答-规划-多 agent 协作"四类 agent 行为。P5 在线自评做收尾。

## 10. 风险与取舍

- **supervisor 多一次 light LLM 调用**：每轮 medical_rag 出口 +~几百 ms。简单问答也调一次（判 FINISH）。缓解：light tier（14B 级，~1s 内）；预算/开关 guard 在已 exhausted 时不调。可接受。
- **`route_after_grounding` 返回类型从 `Literal` 改 `str`**：弱化类型约束。缓解：`route_after_supervisor` 内显式校验合法值；测试覆盖所有返回值。
- **specialist interrupt 跨轮泄漏**：§6 已述，`reset_supervisor_state` 修复。集成测试覆盖"specialist 触发澄清后下一轮不误回 supervise"场景（若可低成本构造）。
- **supervisor 误派发**（纯医学问答被误判要挂号）：LLM 可能误判。缓解：prompt 强调"仅当用户明示动作意图"；specialist 自身有澄清门槛（挂号需确认）；最坏多一轮 specialist 后 FINISH，不破坏正确性。
- **与 `prepare_secondary_turn` 的功能重叠**：两者都能处理"医疗+挂号"复合。`prepare_secondary_turn` 走规则复合检测 + 动作优先重排；supervisor 走 LLM 观察 medical-primary 答案后派发。**两者并存**：规则拆出 secondary 时 `route_after_action` 优先走 `prepare_secondary_turn`（更明确），supervisor 兜底规则没拆出的情况。supervisor 派发时清空 `secondary_intent`/`deferred_user_question` 防双触发。测试覆盖"secondary 已存在时 supervisor 不重复派发"。
- **supervisor 默认开启改 P2 测试断言**：`route_after_grounding` 期望值变化。属预期回归，测试同步更新。

## 11. 验收标准

1. `supervise` 节点 + `SupervisorDecision` schema + 三新 State 字段（`supervisor_active`/`supervisor_rounds`/`supervisor_next`）落地。
2. `reset_supervisor_state` 节点落地，接在 `START → analyze_turn` 之间。
3. `route_after_supervisor` 边落地（appointment/triage/FINISH 三分支）。
4. `route_after_grounding` 在 supervisor 开启时出口改为 `supervise`；关闭时回退 `__end__`。
5. `route_after_action` 新增 `supervisor_active` 分支（最低优先级），4 specialist map 加 `supervise`。
6. `MAX_SUPERVISOR_ROUNDS` 预算截断生效；`ENABLE_MULTI_AGENT_SUPERVISOR` 开关可回滚 P3 行为。
7. 集成测试证明多步 handoff 循环真走通（supervise→specialist→supervise→FINISH）。
8. 简单医疗问答退化为单次 supervise→FINISH（无派发），测试覆盖。
9. `supervise` LLM 失败永不挂图（显式校验 + `_default()` 兜底），测试覆盖真路径。
10. 全量 unittest 不引入新回归；P1（20）/P3（19）不受影响；P2 测试断言按新返回值同步更新。
11. `SILENT_NODES` 加 `supervise`/`reset_supervisor_state`，前端流式不冒 token。
