# P2: 回答反思回路 — 设计文档

**日期**: 2026-06-29
**范围**: 五阶段"从 RAG 到 Agent"升级的第 2 阶段
**依赖**: P1（agentic 检索循环已落地）— P2 复用 P1 的状态字段命名约定与开关风格，但不跨子图回流
**状态**: 待 review

## 1. 背景与动机

P1 让**检索**变成了自我评估的多轮循环。本阶段把同样的"agent 循环"思想带到**回答**阶段：让回答在交付前先被批判，未 grounded 时自我重写，而不是一次性生成完就 END。

### 当前回答阶段的真实情况（代码事实）

- `grounded_answer_generation`（`rag_nodes.py:604`）合成最终答案 → 写入 `messages`。
- `answer_grounding_check`（`rag_nodes.py:687`）调用 `ground_answer(...)`（`tools.py:205`），返回 `{grounded: bool, revised_answer: str, note: str}`。
- **`grounded` 字段被算出来，但从未用于分支**：节点只取 `revised_answer`，若与原文相同返回 `{}`，否则用被动加了免责声明的版本覆盖 `messages`。
- 图边硬编码 `answer_grounding_check -> END`（`graph.py:190`）。
- 两条快路径跳过检查：证据强时（`rag_nodes.py:698`）直接返回 `{}`；revised 与原文相同时（`rag_nodes.py:736`）返回 `{}`。

**结论**：回答阶段是一次性管道。未 grounded 的回答只是被"贴免责声明"后照常输出，没有"批判 → 重写"的纠错循环。`grounded` 这个已经算出来的布尔值是整个代码库最干净的接入点。

### 本阶段目标

把 `answer_grounding_check` 从"被动贴声明 → END"提升为"**主动批判 → 必要时重写 → 再校验**"的反思回路，启用 `grounded` 字段做分支。带轮次上限，防止反复重写。完成后：

- 回答会自我纠错：判定未 grounded 时，先生成 critique（指出哪里超证据），再让 LLM 基于现有证据重写（收窄到证据范围内），然后回校验节点复检。
- 不跨子图、不重检索（按已确认范围）。重写只基于现有 `agent_answers` 证据。
- 与 P1 的 agentic 检索正交：P1 管"证据够不够"，P2 管"回答有没有超出证据"。

## 2. 非目标（YAGNI）

- **不**回 P1 重检索（已确认范围；若重写后仍不 grounded，终止并走现有降级，不跨子图回流）。
- **不**改 `ground_answer` 的免责声明逻辑（保留为终止分支的兜底，与现有安全行为一致）。
- **不**改 `grounded_answer_generation` 的合成逻辑（它只负责首轮生成）。
- **不**做多 agent / supervisor（P4）。
- **不**改 P1 的 `evaluate_evidence` / `route_after_evidence`。

## 3. 架构：主图拓扑变更

### 现有主图回答段

```
agent -> grounded_answer_generation -> answer_grounding_check -> END
```

### P2 后的主图回答段

```
agent -> grounded_answer_generation -> answer_grounding_check
  answer_grounding_check --[route_after_grounding]--> END | revise_answer
  revise_answer -> answer_grounding_check          ← 回到校验节点复检（反思回路）
```

**关键变化**：
1. `answer_grounding_check -> END` 改为条件边 `route_after_grounding`。
2. 新增节点 `revise_answer`：当 `grounded=False` 且重写预算未耗尽时，生成 critique 并让 LLM 基于证据重写，然后回到 `answer_grounding_check` 复检。
3. `route_after_grounding` 三条出口：
   - **已 grounded** → `END`（包括原有快路径：证据强 / revised 与原文相同）。
   - **未 grounded 且预算允许** → `revise_answer`。
   - **未 grounded 但预算耗尽** → `END`（保留现有 `ground_answer` 的免责声明降级行为，不削弱安全兜底）。

**为什么 revise_answer 回到 answer_grounding_check 而不是 grounded_answer_generation**：复检应走轻量的 grounding 校验（已有），不应重跑整个合成。`answer_grounding_check` 本就是校验入口，回它最自然，且 `grounded` 字段在那里被计算。

**为什么是主图节点而非子图节点**：`grounded_answer_generation` 和 `answer_grounding_check` 都在主图（`State`，非 `AgentState`）。P2 的反思回路在主图回答段，与 P1 的子图检索循环是两个独立的 agent 循环。

## 4. 组件设计

### 4.1 `revise_answer` 节点（`rag_nodes.py` 新增）

**职责**：拿到未 grounded 的回答 + critique + 现有证据，让 LLM 基于证据重写回答（收窄到证据范围内，不编造）。

**输入**（从 `State` 读）：
- `messages`：最后一条 `AIMessage` 是当前未 grounded 的回答。
- `agent_answers`：现有证据（answer + sources + evidence_score）。
- `originalQuery`：原始问题。
- `grounding_critique`：本回路已生成的 critique（新字段，首次进入时为空）。
- `grounding_rounds`：已重写轮数（新字段）。

**逻辑**：
1. 取当前回答文本 + 证据（从 `agent_answers` 拼 evidence block）。
2. 调用 `ground_answer` 先拿到 `note`（它内部已判 grounded=False，`note` 说明原因）——但这里不直接用它的 `revised_answer`（那是被动贴声明版），而是自己生成结构化 critique：调用轻量 LLM（`_light_llm`）用 `GroundingCritique` schema：
   - `critique: str`（哪些论断超出证据 / 缺证据 / 与证据矛盾）
   - `revised_answer: str`（基于证据重写后的回答，收窄范围，不加免责声明——声明留给终止分支）
3. LLM 失败兜底：若 critique 调用抛错或 `revised_answer` 为空，退回 `ground_answer` 的 `revised_answer`（被动贴声明版）作为本轮重写结果，并记 `grounding_rounds += 1`。保证不挂。
4. 写回 state：用 `revised_answer` 覆盖 `messages` 的最后一条 AIMessage，`grounding_critique` 存 critique，`grounding_rounds += 1`。
5. 返回后由边路由回 `answer_grounding_check` 复检。

**输出 schema**（`schemas.py` 新增 `GroundingCritique`）：
```python
class GroundingCritique(BaseModel):
    critique: str = Field(description="哪些回答内容超出检索证据、缺证据或与证据矛盾。")
    revised_answer: str = Field(description="基于现有证据重写后的回答，收窄到证据范围内，不加免责声明。")
```

> 取舍：是否复用已有的 `GroundedAnswerCheck`（`grounded/revised_answer/note`）？不复用——它的语义是"判定 + 保守改写"，而 P2 需要的是"主动 critique + 基于证据重写"，字段语义不同（critique vs note）。新建 `GroundingCritique` 更清晰。

### 4.2 `route_after_grounding` 边（`edges.py` 新增）

```
route_after_grounding(state) -> "END" | "revise_answer"
```

**判定顺序**：
1. **已 grounded** → `END`。判定依据：`answer_grounding_check` 写入的 `grounding_passed: bool`（新字段，见 4.3）为 True。
2. **未 grounded 且预算允许** → `revise_answer`：`grounding_rounds < MAX_GROUNDING_ROUNDS`（新配置，默认 1）。
3. **未 grounded 且预算耗尽** → `END`（终止，依赖 `ground_answer` 已贴的免责声明）。

> 默认 `MAX_GROUNDING_ROUNDS=1`：回答反思通常一轮即可收窄；多轮易反复横跳且增延迟。可配置。

### 4.3 `answer_grounding_check` 改动（`rag_nodes.py`）

当前节点返回 `{}` 或 `{"messages": [AIMessage(revised)]}`。P2 改为始终明确写出 `grounding_passed`，并保留原有的 messages 覆盖与快路径：

1. **快路径不变**：证据强时（`rag_nodes.py:698`）返回 `{"grounding_passed": True}`（原返回 `{}`，现多带一个字段，行为等价但让边能判断）。
2. **正常路径**：调 `ground_answer` 后：
   - 若 `grounded["grounded"]` 为 True → `{"grounding_passed": True}`（revised 与原文相同时也归此）。
   - 若 False → 用 `revised_answer`（被动贴声明版）覆盖 messages（保留现有降级），并 `{"grounding_passed": False}`。**注意**：这里先贴声明是为了"预算耗尽终止时"输出的是带声明的安全版本；若进了 `revise_answer` 重写，会再用基于证据的重写版覆盖掉这个声明版。
3. `grounding_rounds` 不在此节点累加（由 `revise_answer` 累加），此节点只读。

> 关键：`answer_grounding_check` 是回路里"判定"角色，`revise_answer` 是"纠错"角色。判定节点每次都重算 `grounded`，复检时若重写成功 grounded 变 True 则终止。

### 4.4 配置新增（`config.py`）

- `MAX_GROUNDING_ROUNDS = 1`：回答反思上限。
- `ENABLE_ANSWER_REFLECTION = True`：开关。`False` 时图回退到 P2 前拓扑（`answer_grounding_check -> END` 硬边），可回滚。

### 4.5 State 新增字段（`graph_state.py` 的 `State`）

```python
class State(MessagesState):
    ...
    grounding_passed: bool = False
    grounding_critique: str = ""
    grounding_rounds: int = 0
```

均无 reducer（每轮覆盖），因为每轮只由一个节点写。

## 5. 数据流（一次"需要重写"的回答）

```
grounded_answer_generation 合成回答 A（含一句超出证据的论断）
  -> answer_grounding_check
    ground_answer 判 grounded=False（"超证据"）
    grounding_passed=False; messages 覆盖为贴声明版
  -> route_after_grounding: 未 grounded + rounds(0) < 1 -> revise_answer
  -> revise_answer
    LLM critique: "第三句关于剂量推荐超出证据"
    revised_answer: 基于证据重写（去掉剂量推荐，收窄）
    messages 覆盖为重写版; grounding_rounds=1
  -> answer_grounding_check (复检)
    ground_answer 判 grounded=True
    grounding_passed=True
  -> route_after_grounding: grounded -> END
```

预算耗尽场景：第二轮重写后仍不 grounded → `route_after_grounding`：`rounds(1) >= 1` → `END`，输出 `ground_answer` 贴的免责声明版（安全降级）。

## 6. 错误处理

| 失败模式 | 处理 |
|---|---|
| critique LLM 调用抛错 / 超时 | 退回 `ground_answer` 的 `revised_answer`（贴声明版），`grounding_rounds += 1`，回校验 |
| `GroundingCritique` 解析失败 / revised_answer 空 | 同上，退回 `ground_answer.revised_answer` |
| 重写后仍不 grounded 且预算耗尽 | `route_after_grounding` → `END`，输出贴声明版 |
| `ENABLE_ANSWER_REFLECTION=False` | 图回退硬边 `answer_grounding_check -> END`，零行为变化 |
| 证据强（原有快路径） | `grounding_passed=True` → END，不进回路 |

所有终止路径都落到 `ground_answer` 的免责声明降级（含医疗免责与高风险就医提醒），不产出"伪装 grounded"的回答——与现有安全兜底一致，本阶段不削弱。

## 7. 测试策略

### 7.1 单元（`tests/test_answer_reflection.py` 新增）

- `revise_answer`：未 grounded + critique LLM 返回有效 → messages 被重写版覆盖，`grounding_critique` 写入，`grounding_rounds += 1`。
- `revise_answer` LLM 失败兜底：mock 抛错 → 退回 `ground_answer.revised_answer`，`grounding_rounds += 1`，不抛异常。
- `answer_grounding_check` grounded=True → 返回 `grounding_passed=True`，不覆盖 messages。
- `answer_grounding_check` grounded=False → 返回 `grounding_passed=False`，messages 覆盖为贴声明版。
- `answer_grounding_check` 快路径（证据强）→ `grounding_passed=True`。
- `route_after_grounding`：grounded→END；未grounded+预算→revise_answer；未grounded+耗尽→END。

### 7.2 集成（编译图，仿 P1 的 `TestCompiledGraphStateHandoff`）

- 编译最小主图段（grounded_answer_generation→answer_grounding_check→route_after_grounding→{END, revise_answer}→answer_grounding_check），fake LLM：首轮生成超证据回答 → 进 revise_answer → 复检 grounded → END。证明回路真的走通 LangGraph 机器。
- 终止用例：重写后仍不 grounded + 预算耗尽 → END（输出贴声明版）。

### 7.3 回归

- `python -m unittest discover -s tests -v`：既有用例不回归（特别是 `test_api_app` 里触达 grounding 的）。
- `ENABLE_ANSWER_REFLECTION=False`：回退硬边，行为与改动前一致。
- P1 的 20 个测试不受影响（P2 不碰子图）。

## 8. 对面试叙事的直接支撑

完成后，"回答"阶段也成环：

> 回答不是一次性生成。我让 `answer_grounding_check` 启用之前算出却没用过的 `grounded` 字段做分支：未 grounded 时进 `revise_answer`，让 LLM 先生成 critique（指出哪里超证据），再基于现有证据重写收窄，然后回校验节点复检，带轮次上限。证据不足的终止路径仍走 `ground_answer` 的免责声明降级。这样回答会自我纠错，而不是生成完就交付。

连同 P1，检索和回答两个阶段都从管道变成了 agent 循环。P3 任务分解会把检索循环当可复用单元，进一步成环。

## 9. 风险与取舍

- **延迟**：未 grounded 时多一轮 critique + 重写 LLM 调用。缓解：默认 `MAX_GROUNDING_ROUNDS=1`；快路径跳过；用 light tier。
- **反复横跳**：重写后可能仍不 grounded。缓解：默认 1 轮上限，耗尽即终止走降级。
- **声明版与重写版覆盖顺序**：判定节点先贴声明版（为终止兜底），重写节点再用基于证据版覆盖。若进了重写，最终输出是重写版（无声明，因已收窄到证据内）；若没进重写或耗尽，输出声明版。逻辑自洽，但需测试覆盖两种终态。
- **不重检索的局限**：若未 grounded 的根因是证据本身不够，重写只能收窄不能补证据。这是已确认的范围取舍；P3 再评估是否需要回 P1。

## 10. 验收标准

1. `revise_answer` 节点 + `route_after_grounding` 边 + 3 个新 state 字段落地。
2. `grounded` 字段被用于分支（不再仅算不用）。
3. `answer_grounding_check` 写出 `grounding_passed`，快路径与正常路径都覆盖。
4. critique LLM 失败降级到 `ground_answer.revised_answer`。
5. `MAX_GROUNDING_ROUNDS` 上限生效；耗尽走 END + 免责声明。
6. `ENABLE_ANSWER_REFLECTION` 开关可回滚硬边。
7. 全量 unittest 不回归；P1 的 20 个测试不受影响。
8. 至少一个编译图集成用例证明回路真的走通（重写→复检→END）。
