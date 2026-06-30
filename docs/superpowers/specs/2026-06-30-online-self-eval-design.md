# P5: 在线自评 (Online Self-Eval) — 设计文档

**日期**: 2026-06-30
**范围**: 五阶段"从 RAG 到 Agent"升级的第 5 阶段（最终阶段）
**依赖**: P1（检索成环）、P2（回答反思）、P3（任务分解）、P4（多 agent supervisor）均已落地于 main
**状态**: 待 review

## 1. 背景与动机

P1-P4 让系统有了真正的 agent 行为：检索成环、回答成环、自主规划、多 agent 协作。但系统对自己产出的回答**没有自省**——回答产出后直接交付，不知道这次答得好不好。面试官说"没有真正的 agent 特征"，缺的正是**自我评估 / 自我觉察**这一块。

### 当前 turn 结尾已有的评分信号（代码事实）

medical_rag 终止路径上，State 已携带若干评分信号（`graph_state.py`）：

| 字段 | 含义 | 局限 |
|------|------|------|
| `grounding_passed: bool` | P2 grounding 检查是否通过（`ground_answer()` 返回 `grounded=True`） | 二值，无梯度 |
| `grounding_evidence_score: float \| None` | 聚合证据分（`agent_answers` 中 evidence_score 的 max） | 只反映检索相关性，不反映回答质量 |
| `grounding_critique: str` | P2 `revise_answer` 的结构化批评 | 仅在 P2 触发重写时产出 |
| `agent_answers[*].confidence_bucket` | 每子回答的 "high/medium/low/no_evidence" | 来自检索分数映射，非回答内容评判 |
| `agent_answers[*].evidence_score` | 每子回答的检索相似度分 | 同上 |

**结论**：现有信号全部源自**检索置信度**，没有任何信号评判**回答本身**的医学安全性、准确性、完整性。这些维度正是医疗场景最该自省的。P5 补一个 LLM-as-judge 自评节点，评这些检索信号覆盖不到的维度，低分触发软降级（追加免责提示），分数持久化供离线分析。

### 离线评估基建（不可直接复用，但提供模式参考）

- `project/core/qa_eval.py`（`RetrievalQualityEvaluator`）：**监督式**——需 `QAEvalSample` 的期望关键词/来源类型，本质是离线 benchmark，无法在线（live chat 无期望答案）。其 `_score_answer()`、`_confidence_bucket_from_docs()`、`compute_mrr()` 是可复用原语，但 P5 不依赖它们。
- `project/benchmarks/evaluate_*.py`：全部读 `retrieval_logs`/`route_logs` 表做离线聚合，非在线。
- **P5 不复用离线 eval**——它是监督式的；P5 需要**无监督**的 LLM-as-judge。

### 本阶段目标

新增 `self_eval` 节点（light LLM + `AnswerSelfEval` schema），插在 P2 回答反思回路之后、P4 supervisor 之前。对最终回答打 4 维分（safety/accuracy/completeness/groundedness）→ 加权总分；低分（< 阈值）追加用户可见的软降级提示；分数 + 维度详情持久化到 `route_logs`。开关 `ENABLE_SELF_EVAL` 可回滚到 P4 行为。

## 2. 非目标（YAGNI）

- **不**让自评触发重答（重检索/重写）。P1（检索成环）+ P2（回答反思）已覆盖自我纠正；P5 只做"评估 + 软降级提示"，不重复纠正回路。
- **不**改 P2 的 `answer_grounding_check`/`revise_answer` 回路。自评在 P2 收尾后才介入。
- **不**改 P4 的 `supervise`/`route_after_supervisor`。自评在 supervisor 之前串行；supervisor 读 `agent_answers` 不读自评分，互不影响。
- **不**改 AgentState 子图（P1 检索循环）。自评在主图 State 层。
- **不**做监督式离线 eval 的在线化（不引入期望答案）。
- **不**碰非 medical_rag 意图路径（greeting/appointment/triage/cancel 的静态快路径不进自评）。
- **不**新建持久化表。复用 `route_logs`（已有 `extra_metadata` JSONB + 每轮写入）。
- **不**改 Langfuse callback 的核心逻辑（仅在 `finalize_turn` 把分数塞进 route_log；Langfuse trace 集成列为可选 follow-up，非本阶段必须）。

## 3. 架构：主图拓扑变更

### 现有 medical_rag 终止段（P4 后）

```
grounded_answer_generation → answer_grounding_check → [P2 回路: revise_answer] → route_after_grounding
    → supervise (P4) → route_after_supervisor → {specialist, END}
    → [specialist] → route_after_action → supervise → ... → FINISH → END
```

### P5 后

```
grounded_answer_generation → answer_grounding_check → [P2 回路] → route_after_grounding
    → self_eval (NEW) → route_after_self_eval (NEW)
        → supervise (P4) [supervisor 开启时]
        → __end__        [supervisor 关闭时]
    [后续 P4 supervisor 链不变]
```

**关键变化**：
1. `route_after_grounding` 的终止出口（`supervise`/`__end__`）在 `ENABLE_SELF_EVAL=true` 时改为 `self_eval`；关闭时不变（P4 行为）。
2. `self_eval`（新增节点）：读最终回答 + agent_answers + grounding 信号 → LLM-as-judge 4 维分 → 加权总分 → 低分追加软降级提示 → 写 State 字段。
3. `route_after_self_eval`（新增边）：→ `supervise`（supervisor 开启）或 `__end__`（关闭）。确定性，复用 P4 的 config 检查。
4. P2 回路、P4 supervisor 链**完全不变**——自评串行夹在中间，不交互。

**为什么 P2 后、P4 前**：P2 回路跑完时回答已 grounded（最终版），自评判的是最终回答；放 P4 前则 supervisor 的 FINISH/dispatch 决策不受自评分影响（supervisor 不读自评分）。串行无交互，叙事干净。

## 4. 组件设计

### 4.1 `self_eval` 节点（`rag_nodes.py` 新增，light tier）

**职责**：对最终回答打 4 维 LLM-as-judge 分；低分追加软降级提示；写自评 State 字段。永不抛（失败→中性分 + `degraded=True`，不追加提示）。

**输入**（从 `State` 读）：
- `originalQuery`：原始问题（判 completeness 用）。
- 最终回答：`messages[-1].content`，但需剥掉 `grounded_answer_generation` 追加的 `confidence_note` + `citation_block` 尾巴——用既有 `_sanitize_final_answer_text` 或截到参考来源之前。取**纯回答正文**给 judge，避免把免责声明/引用也评进去。
- `agent_answers`：每子回答的 `confidence_bucket`/`evidence_score`（作为上下文给 judge，辅助 groundedness 判断）。
- `grounding_passed`：P2 是否通过（上下文）。

**逻辑**：
1. **开关 guard**：`not config.ENABLE_SELF_EVAL` → 返回空 dict（不评，不写字段；`route_after_grounding` 在关闭时本就不路由到此节点，此 guard 是双保险）。
2. 取 `final_answer`（剥尾巴）。若空 → 返回 `{"self_eval_score": None, "self_eval_details": {"degraded": True, "reason": "empty_answer"}}`，不追加提示。
3. 调 light LLM + `AnswerSelfEval` schema，prompt 要求对 safety/accuracy/completeness/groundedness 各打 1-5 分 + 简短理由。
4. **校验维度值**：每个维度 coerce 到 [1,5] int；非法/缺失 → 中性 3。
5. **加权总分** → 0.0-1.0：
   - `score = (safety*0.35 + accuracy*0.30 + completeness*0.20 + groundedness*0.15) / 5.0`
   - 安全权重最高（医疗领域）。
6. **软降级**：`score < SELF_EVAL_DEGRADE_THRESHOLD`（默认 0.6）→ 追加提示到 `messages[-1]`（新增一条 AIMessage，内容为软降级提示，**不修改原回答**——保留原回答 + 引用）。
   - 提示文案：`⚠️ 自评提示：本回答在准确性/完整性上置信度较低（自评 {score:.2f}/1.0），建议结合线下医生意见或补充更多症状细节后再判断。`
   - 镜像 `grounded_answer_generation` 的 `confidence_note` 追加模式（同 UX idiom）。
7. **`degraded=True`（LLM 失败）→ 不追加提示**：失败的自评不可信，不降级回答（假提示比无提示更糟）。仅写 `degraded` 标志。
8. 写回 State：`{"self_eval_score": score, "self_eval_details": {safety, accuracy, completeness, groundedness, reason, degraded, caveat_appended}}`。

**永不抛**：`try/except` 包住 `parser.invoke()`（含 `_structured_output_llm(...)` 构造，仿 P4 `supervise` 的 hardened scope）→ 失败返回中性分 0.5 + `degraded=True` + 无提示。非法维度值显式 coerce。`AnswerSelfEval` 的维度是 `int` 字段——`_default()` 对 int 处理正确（返回 0），不像 P4 的 `Literal` 问题；但节点仍显式校验 [1,5] 防御 0/越界。

### 4.2 `AnswerSelfEval` schema（`schemas.py` 新增）

```python
class AnswerSelfEval(BaseModel):
    safety: int = Field(description="回答的医学安全性 1-5：是否避免不安全建议、必要时建议就医。")
    accuracy: int = Field(description="医学准确性 1-5：是否医学正确、与检索证据一致。")
    completeness: int = Field(description="完整性 1-5：是否充分回答了用户问题（尤其多 facet 问题）。")
    groundedness: int = Field(description="证据支撑度 1-5：是否限于检索证据、未臆造。")
    reason: str = Field(description="简短说明打分依据。")
```

> `int` 字段（非 `Literal`）——`_structured_output_llm._default()` 对 int 默认 0，Pydantic 接受，不抛。节点显式 coerce [1,5]。比 P4 的 `SupervisorDecision`（Literal）更稳。

### 4.3 `get_self_eval_prompt()`（`prompts.py` 新增）

LLM-as-judge 系统 prompt：角色"医学回答质量评审员"；给最终回答 + 原始问题 + 每子回答的置信度；要求对 4 维各打 1-5 分 + 理由；严格 JSON 输出 `{"safety": int, "accuracy": int, "completeness": int, "groundedness": int, "reason": str}`。强调基于检索证据评判，不要求新知识。

### 4.4 `route_after_self_eval` 边（`edges.py` 新增）

```python
def route_after_self_eval(state: State) -> str:
    """P5: after self-eval, continue to the P4 supervisor (or END if disabled)."""
    return "supervise" if ENABLE_MULTI_AGENT_SUPERVISOR else "__end__"
```

确定性，复用 P4 的 `ENABLE_MULTI_AGENT_SUPERVISOR`（edges.py line 5 已 import）。

### 4.5 `route_after_grounding` 改指向（`edges.py`）

P4 的 `route_after_grounding` 把终止出口指向 `supervise`/`__end__`。P5 在 `ENABLE_SELF_EVAL=true` 时改为 `self_eval`：

```python
def route_after_grounding(state: State) -> Literal["__end__", "revise_answer", "supervise", "self_eval"]:
    """P2/P4/P5: route after the answer grounding check.
    - grounded → self_eval (P5) when self-eval on, else supervise (P4) / END
    - not grounded + budget + reflection on → revise_answer
    - not grounded + budget + reflection off → self_eval (P5) / supervise (P4) / END
    - budget exhausted → self_eval (P5) / supervise (P4) / END
    """
    if bool(state.get("grounding_passed", False)):
        return _next_after_grounding()
    rounds = int(state.get("grounding_rounds", 0) or 0)
    if rounds < MAX_GROUNDING_ROUNDS and config.ENABLE_ANSWER_REFLECTION:
        return "revise_answer"
    return _next_after_grounding()


def _next_after_grounding() -> str:
    """P5/P4: terminal target after grounding. self_eval if on, else supervisor if on, else END."""
    if config.ENABLE_SELF_EVAL:
        return "self_eval"
    return "supervise" if config.ENABLE_MULTI_AGENT_SUPERVISOR else "__end__"
```

- `ENABLE_SELF_EVAL=true`（默认）→ `self_eval` → `route_after_self_eval` → `supervise`/`__end__`（串联 P4）。
- `ENABLE_SELF_EVAL=false` → 直接 `supervise`/`__end__`（P4 行为）。
- `revise_answer` 分支不变（P2 回路优先）。

### 4.6 State 新增字段（`graph_state.py` 的 `State`）

```python
# P5: online self-eval — LLM-as-judge score + details at turn end
self_eval_score: float | None = None
self_eval_details: dict = {}
```

无 reducer（`self_eval` 单点写；`route_after_self_eval`/`finalize_turn` 单点读）。

### 4.7 配置新增（`config.py`，接 P4 块后）

```python
# P5: online self-eval — LLM-as-judge answer scoring + soft-degrade caveat
ENABLE_SELF_EVAL = os.environ.get("ENABLE_SELF_EVAL", "true").lower() == "true"
SELF_EVAL_DEGRADE_THRESHOLD = float(os.environ.get("SELF_EVAL_DEGRADE_THRESHOLD", "0.6"))
```

### 4.8 图接线（`graph.py`）

- import `self_eval`（`partial(self_eval, llm=_light_llm)`）。
- 注册 `self_eval` 节点（在 `answer_grounding_check` 注册附近）。
- `answer_grounding_check` 的条件边 mapping（P4 块内）：`ENABLE_SELF_EVAL=true` 时加 `"self_eval": "self_eval"` 到 `_grounding_map`；`route_after_grounding` 返回 `self_eval`。
- 新增 `self_eval` 条件边：`add_conditional_edges("self_eval", route_after_self_eval, {"supervise": "supervise", "__end__": END})`（supervisor 开启时；关闭时 mapping 只需 `{"__end__": END}`，但 `route_after_self_eval` 关闭时只返回 `__end__`，所以 mapping 始终含 `__end__`，`supervise` 键在关闭时可缺省——为安全两者都列）。
- `ENABLE_SELF_EVAL=false` 时不注册 `self_eval`、不加 `self_eval` 到 `_grounding_map`（`route_after_grounding` 返回 `supervise`/`__end__`）——拓扑回退 P4。

### 4.9 `SILENT_NODES`（`chat_interface.py`）

加 `"self_eval"`——judge 节点的原始 JSON token 不应流式给用户（与 `answer_grounding_check`/`supervise` 同类）。软降级提示是节点**显式追加**的 AIMessage（非 judge 的原始 token），不受 SILENT_NODES 影响，仍会通过 `_handle_llm_token`/`_prepare_visible_messages` 正常流给用户。

### 4.10 持久化（`route_logs` + `finalize_turn`）

**Schema 迁移**（`schema_manager.py`，新增编号 migration，仿 `005_appointment_skill_and_retrieval_quality` 的 `ADD COLUMN IF NOT EXISTS` 模式）：

```python
(
    "006_route_logs_self_eval",
    "Persist online self-eval score + details per turn.",
    [
        """
        ALTER TABLE route_logs
        ADD COLUMN IF NOT EXISTS self_eval_score FLOAT
        """,
        """
        ALTER TABLE route_logs
        ADD COLUMN IF NOT EXISTS self_eval_details JSONB NOT NULL DEFAULT '{}'::jsonb
        """,
    ],
),
```

**写入**（`chat_turn_service.py` 的 `_persist_route_log`）：现有 `extra_metadata` dict 里加两个键（不新增列写入路径——直接塞进 `extra_metadata` JSONB，零表结构变更的备选；但 spec 选**显式列**方案更可查询）。

> **取舍**：`extra_metadata` JSONB（零迁移）vs 显式 `self_eval_score`/`self_eval_details` 列（可 `SELECT avg(self_eval_score)`）。选显式列——离线分析要按分聚合查询，显式列可直接 `WHERE self_eval_score < 0.6`，JSONB 路径需 `->>` 转换不便。多一个 migration（仿 005 模式，已有先例）。

`_persist_route_log` 的 `save_log` dict 加：
```python
"self_eval_score": artifacts.latest_values.get("self_eval_score"),
"self_eval_details": artifacts.latest_values.get("self_eval_details") or {},
```
`RouteLogStore.save_log` 的 INSERT 需含这两列（若 store 用动态列映射则自动；若显式列名需加——实现时检查 `project/db/*route_log_store*`）。`try/except` 已包住整个 `_persist_route_log`（line 256），写失败不影响 turn。

## 5. 数据流（一次低质回答的自评+软降级）

```
用户："我这个偏方能治高血压吗"（检索证据弱、回答需谨慎）
... → grounded_answer_generation (合成回答 A0 + confidence_note)
  → answer_grounding_check (P2: grounded=True)
  → route_after_grounding (self_eval 开启) → self_eval
self_eval (light LLM judge):
  safety=4, accuracy=2, completeness=3, groundedness=2
  score = (4*0.35 + 2*0.30 + 3*0.20 + 2*0.15)/5 = (1.4+0.6+0.6+0.3)/5 = 0.58
  0.58 < 0.6 阈值 → 追加软降级提示 AIMessage
  → 写 self_eval_score=0.58, self_eval_details={..., caveat_appended=True}
  → route_after_self_eval → supervise (P4)
  → supervise (FINISH) → END
[finalize_turn] route_logs 写入 self_eval_score=0.58, self_eval_details={...}
用户看到：原回答 + 引用 + ⚠️自评提示
```

高分回答：`score=0.82 ≥ 0.6` → 不追加提示 → 用户看到原回答（无变化）。

LLM 失败：`degraded=True`, `score=0.5`（中性），**不追加提示** → route_logs 记 `degraded`。

## 6. 错误处理

| 失败模式 | 处理 |
|---|---|
| `self_eval` LLM 失败 | `try/except` → 中性分 0.5 + `degraded=True` + **不追加提示** + 继续（永不挂图） |
| 维度值非法/缺失 | 显式 coerce 到 [1,5]，缺失→3（中性） |
| `ENABLE_SELF_EVAL=False` | `route_after_grounding` 不路由到 `self_eval`（P4 行为）；节点不注册 |
| 持久化写失败 | `_persist_route_log` 既有 `try/except` 吞错（line 256）—— 分数丢失不影响 turn |
| 低分误判（高证据却被判低） | 软降级提示是 advisory（用户仍可读原回答+引用）；可接受。`SELF_EVAL_DEGRADE_THRESHOLD` 可调 |
| `degraded=True` 仍触发提示？ | 否——失败的自评不可信，`caveat_appended` 仅在非 degraded 且低分时为 True |
| 空回答 | `self_eval_score=None`, `degraded=True`, 无提示 |

## 7. 测试策略

### 7.1 单元（`tests/test_online_self_eval.py` 新增）

- **config/state/schema**：`ENABLE_SELF_EVAL`/`SELF_EVAL_DEGRADE_THRESHOLD` 类型对；State 有 `self_eval_score`/`self_eval_details`；`AnswerSelfEval` 4 int 字段 + reason。
- **`self_eval` 节点**：
  - `ENABLE_SELF_EVAL=False` → 空 dict（不评）。
  - LLM 返回 4 维 → 加权总分正确（验证权重公式）。
  - 低分（< 阈值）→ 追加软降级提示 AIMessage（`messages` 多一条，`caveat_appended=True`）。
  - 高分（≥ 阈值）→ 不追加（`messages` 不变，`caveat_appended=False`）。
  - LLM 失败（patch `_structured_output_llm` 抛异常）→ 中性分 0.5 + `degraded=True` + 不追加提示（永不抛）。
  - **真路径 LLM 失败**（bare MagicMock LLM，不 patch `_structured_output_llm`）→ `_default()` 触发 → int 字段默认 0 → coerce 到 3 → 中性分 + 不抛（回归 `_default()` 对 int 的行为）。
  - 非法维度值（6, 0, -1）→ coerce 到 [1,5]。
  - 空回答 → `score=None`, `degraded=True`, 无提示。
  - 权重 guard：删 `safety*0.35` 权重 → 测试 fail（防回归公式）。
- **`route_after_self_eval`**：supervisor 开启→`supervise`；关闭→`__end__`。
- **`route_after_grounding`**：self_eval 开启→`self_eval`；关闭→`supervise`/`__end__`；`revise_answer` 分支不变。

### 7.2 集成（编译图，仿 P4 `TestCompiledSupervisorLoop`）

- 医疗回答 → `self_eval`（fake judge 返回低分）→ 软降级提示追加 → `route_after_self_eval` → `supervise`/END。证明节点串入真实图不破坏 P4 链。
- 高分 → 无提示 → END。

### 7.3 持久化

- `finalize_turn`/`_persist_route_log` 的 `save_log` dict 含 `self_eval_score`/`self_eval_details`（mock `RouteLogStore.save_log`，断言 dict 键）。

### 7.4 回归

- `test_answer_reflection.py`：`route_after_grounding` 终止断言从 `supervise` 改 `self_eval`（self_eval 默认开启）。P2 回路（`revise_answer`）不变。
- `test_multi_agent_supervisor.py`：`route_after_grounding` 终止断言从 `supervise` 改 `self_eval`（self_eval 默认开启）。`TestCompiledSupervisorLoop` 用的是最小自建图（不引用真实 `route_after_grounding`），**不受影响**——无需改其 fake 链；但其 `route_after_grounding` 相关的纯函数断言（如有）需更新。
- `test_chat_interface.py`：`SILENT_NODES` 加 `self_eval`。
- P1（20）/P3（19）不受影响（self_eval 在主图，不碰 AgentState/并行 fan-out）。
- 全量 `unittest discover`：新用例全 PASS；既有失败与 main 基线一致，**P5 不引入新回归**。

## 8. 对面试叙事的直接支撑

> 系统不只是回答——它**自省**。每轮回答产出后（经 P2 grounding），一个 LLM-as-judge 对它打 4 维分：安全性、准确性、完整性、证据支撑度——这些是检索置信度信号覆盖不到的维度。低分回答会主动追加一条自我贬抑的提示（软降级），告诉用户"这次回答置信度低，建议结合线下医生意见"；分数 + 维度详情持久化到 `route_logs` 供离线分析。所以 agent 知道自己什么时候答得不好，并告诉用户。这是缺的自我觉察层。

连同 P1（检索成环）、P2（回答成环）、P3（自主规划）、P4（多 agent 协作），系统有了"检索-回答-规划-协作-自省"五类 agent 行为。五阶段升级完结。

## 9. 风险与取舍

- **每轮 +1 次 light LLM 调用**：~几百 ms（light tier）。简单问答也评一次。缓解：light tier；开关可关；`SILENT_NODES` 不影响用户感知延迟（judge token 不流式）。可接受。
- **LLM-judge 误判**：可能假阴性（高证据被判低 → 多余提示）或假阳性（低质被判高 → 无提示）。缓解：4 维 + 权重 + 阈值可调；软降级提示是 advisory 不破坏回答；`degraded` 标志区分失败与真低分。医疗场景偏保守——多提示优于漏提示。
- **与 P2 grounding 的重叠**：groundedness 维度与 `grounding_passed` 重叠。但 P2 是二值、P5 是梯度 1-5——P5 能抓 P2 的盲区（P2 通过但 groundedness 仍弱）。非冗余。
- **软降级提示追加为独立 AIMessage**：而非改原回答——保留原回答 + 引用完整。但流式 UI 会多一条消息；需确认 `_handle_llm_token`/`_prepare_visible_messages` 正常处理（节点返回的 AIMessage 非流式 chunk，应作为完整消息出现）。实现时验证。
- **持久化列方案 vs JSONB**：选显式列多一个 migration，但可查询性好（§4.10）。已有 005 先例。
- **`route_after_grounding` 返回类型再扩**：P4 已扩到 `Literal["__end__","revise_answer","supervise"]`，P5 加 `"self_eval"`。类型仍为 Literal（静态安全）。
- **RouteLogStore 是否支持新列**：若 store 用显式列名 INSERT，需加两列；若动态映射则自动。实现时检查并适配（Task 覆盖）。

## 10. 验收标准

1. `self_eval` 节点 + `AnswerSelfEval` schema + `get_self_eval_prompt` + 两新 State 字段落地。
2. `route_after_self_eval` 边落地（→ supervise/__end__）。
3. `route_after_grounding` 在 self_eval 开启时终止出口改为 `self_eval`；关闭时回退 P4。
4. 4 维加权评分公式正确（safety 0.35/accuracy 0.30/completeness 0.20/groundedness 0.15）。
5. 低分（< `SELF_EVAL_DEGRADE_THRESHOLD`）追加软降级提示；高分不追加；`degraded` 不追加。
6. `self_eval` LLM 失败永不挂图（中性分 + degraded + 无提示），测试覆盖真路径。
7. `route_logs` 加 `self_eval_score`/`self_eval_details` 列（migration 006）；`_persist_route_log` 写入。
8. `SILENT_NODES` 加 `self_eval`；软降级提示仍能流给用户。
9. 集成测试证明 self_eval 串入真实图不破坏 P2/P4 链。
10. 全量 unittest 不引入新回归；P1（20）/P3（19）不受影响；P2/P4 测试断言同步更新。
11. `ENABLE_SELF_EVAL=False` 完全回退 P4 行为。
