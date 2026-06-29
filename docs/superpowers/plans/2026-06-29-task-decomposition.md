# P3 任务分解 / 规划 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把复合医学问题分解成多个独立 facet 子问题，用 LangGraph `Send` 并行 fan-out，每个子问题独立跑 P1 检索循环，最后复用已有 `agent_answers` 聚合基建合成——补上 agent 的"自主规划"能力。

**Architecture:** 新增 `decompose_tasks` 节点（light LLM 判复合 → 1-N 子问题）取代 `plan_retrieval_queries` 在图中的位置；`route_after_query_plan` 从单 `Send` 扩为多 `Send` fan-out；fan-in 复用现有 `accumulate_or_reset` reducer + `grounded_answer_generation`。简单问题退单路径，`ENABLE_TASK_DECOMPOSITION` 可回滚。

**Tech Stack:** LangGraph StateGraph + `Send` 并行 fan-out, Pydantic schema, `_structured_output_llm`（regex JSON 解析，SiliconFlow/Qwen 兼容），Python unittest。

**Spec:** `docs/superpowers/specs/2026-06-29-task-decomposition-design.md`

**Scope guard:** 本计划只改动 `project/rag_agent/`、`project/config.py`、`tests/test_task_decomposition.py`、`tests/test_routing_edges.py`、`docs/superpowers/plans/2026-06-29-task-decomposition.md`。工作区有一批已 stash 的 frontend/api/db 改动（`stash@{0}`），**不得** pop、不得触碰——P3 commit 只 stage 本计划列出的精确路径，**绝不**用 `git add -A`/`.`/`-am`。

---

## 关键代码事实（实现者必读）

1. **`route_after_rewrite`（`edges.py:55-91`）** 当前 medical_rag 默认返回 `"plan_retrieval_queries"`（line 91）。P3 改为返回 `"decompose_tasks"`。

2. **`route_after_query_plan`（`edges.py:94-126`）** 当前返回**单元素** `[Send("agent", {...question_index:0, query_plan: deduped_plan, ...})]`。`deduped_plan` 来自 `planned_queries or rewrittenQuestions`。P3 改为返回 `[Send(...) for each sub_question]`，每子问题 `question_index=i`、`query_plan=[q]`。

3. **`Send` 已导入** `edges.py:3`（`from langgraph.types import Send`）。`route_after_query_plan` 返回 `Send` 列表是 LangGraph 并行 fan-out 的标准机制——多元素列表 = 多并行子图实例。

4. **fan-in 基建已就绪**（P3 不改）：
   - `State.agent_answers` 用 `accumulate_or_reset` reducer（`graph_state.py:6`）——多子图各写一条（`collect_answer` 写 `index=i`），reducer 自动合并。
   - `grounded_answer_generation`（`rag_nodes.py:606`）按 `index` 排序遍历 `agent_answers` 合成。
   - `AgentState.question_index`（`graph_state.py:73`）恒为 0 → P3 终于让它取 0..N-1。

5. **`plan_retrieval_queries`（`rag_nodes.py:280`）** 是规则式节点。P3 从图中**摘除接线**（删 graph.py 注册 + 连边），但**保留**节点函数、`plan_queries` 工具（`tools.py`）、`RetrievalQueryPlan` schema——避免连带回归。`plan_retrieval_queries` 仍在 `rag_nodes.py` 的 `__all__`（line 832）和 graph.py import（line 21）里，保留不动（摘除的是图接线，不是符号）。

6. **`_structured_output_llm(llm, schema, max_tokens=N)`（`node_helpers.py:114`）**：返回带 `.invoke()` 的 parser；LLM/解析失败返回 schema 默认值（`sub_questions: []`、`needs_decomposition: False`、`reason: ""`），**永不抛异常**。所以 `decompose_tasks` 里 `parser.invoke(...)` 不会抛——失败表现为 `sub_questions == []`，走兜底。

7. **配置风格**：`config.py` P2 块在 line 89-90。P3 块追加其后。布尔用 `os.environ.get("X", "true").lower() == "true"`，int 用 `int(os.environ.get("X", "3"))`。

8. **`route_after_query_plan` 返回类型**：当前无类型注解（返回 `list`）。P3 保持无注解或加 `-> list[Send]`——`Send` 已在 edges.py 顶部导入，可用。保持与现有风格一致即可（现有函数无注解，P3 也保持无注解，避免引入不一致）。

9. **`route_after_query_plan` 在图中是条件边**：`graph.py:170` `graph_builder.add_conditional_edges("plan_retrieval_queries", route_after_query_plan)`。P3 把源头节点改为 `decompose_tasks`：`graph_builder.add_conditional_edges("decompose_tasks", route_after_query_plan)`。`route_after_query_plan` 返回 `Send` 列表时，LangGraph 用 `Send.node` 字段定位目标节点（这里是 `"agent"`），**不需要**显式 mapping dict（与现有 line 170 一致——现有也无 mapping dict）。

10. **运行测试**：`PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_task_decomposition -v`（Windows bash）。测试文件首行 `sys.path.insert(0, .../project)`。

11. **现有 `test_routing_edges.py:89` 测试** `test_route_after_rewrite_passes_recent_context_to_agent_subgraph`：断言 `route_after_rewrite(...) == "plan_retrieval_queries"`（line 99）+ `route_after_query_plan` 返回 1 个 Send 且 `query_plan == ["高血压应该注意什么"]`（line 110-117）。P3 后：route_after_rewrite 改返回 `"decompose_tasks"`（line 99 需改）；route_after_query_plan 部分**无需改**——无 `sub_questions` 时退回 `[rewrittenQuestions[0]]` 单 Send，`query_plan=[q]` 仍等于 `["高血压应该注意什么"]`，断言保持绿。Task 7 会精确处理。

---

## File Structure

| 文件 | 责任 | 改动 |
|---|---|---|
| `project/config.py` | 新增 `MAX_SUB_QUESTIONS`、`ENABLE_TASK_DECOMPOSITION` | 追加 2 行 |
| `project/rag_agent/graph_state.py` | `State` 新增 `sub_questions` 字段 | 追加 1 行 |
| `project/rag_agent/schemas.py` | 新增 `TaskDecomposition` schema | 追加 1 class |
| `project/rag_agent/prompts.py` | 新增 `get_task_decomposition_prompt()` | 追加 1 func |
| `project/rag_agent/rag_nodes.py` | 新增 `decompose_tasks` 节点；imports + `__all__` | 加 1 函数 + 改 imports |
| `project/rag_agent/edges.py` | `route_after_query_plan` 改 fan-out；`route_after_rewrite` 改指向 | 改 2 函数 |
| `project/rag_agent/graph.py` | 摘除 `plan_retrieval_queries` 接线、注册 `decompose_tasks`、改连边 | 改接线 + 改 import |
| `tests/test_task_decomposition.py` | 全部单元 + 集成测试 | 新建 |
| `tests/test_routing_edges.py` | 更新 route_after_rewrite 断言 | 改 2 行 |

---

## Task 1: config 新增开关与上限

**Files:**
- Modify: `project/config.py:90`（P2 块之后）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_task_decomposition.py`：

```python
"""Tests for P3 task decomposition: decompose_tasks + route_after_query_plan fan-out."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from unittest.mock import MagicMock, patch  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.types import Send  # noqa: E402


class TestConfigFields(unittest.TestCase):
    def test_decomposition_config_fields_exist(self):
        import config
        self.assertTrue(hasattr(config, "MAX_SUB_QUESTIONS"))
        self.assertEqual(config.MAX_SUB_QUESTIONS, 3)
        self.assertTrue(hasattr(config, "ENABLE_TASK_DECOMPOSITION"))
        self.assertTrue(config.ENABLE_TASK_DECOMPOSITION)


def _make_main_state(**extra):
    base = {
        "messages": [],
        "originalQuery": "高血压合并痛风吃什么药安全，另外怎么在家监测血压？",
        "rewrittenQuestions": ["高血压合并痛风吃什么药安全"],
        "conversation_summary": "",
        "recent_context": "",
        "topic_focus": "",
        "user_memories": "",
        "sub_questions": [],
    }
    base.update(extra)
    return base


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_task_decomposition.TestConfigFields -v`
Expected: FAIL — `AttributeError`/断言失败（config 尚无 `MAX_SUB_QUESTIONS`）

- [ ] **Step 3: 实现**

在 `project/config.py` 的 P2 块（line 89-90）之后追加：

```python
# P3: task decomposition — parallel sub-question fan-out
MAX_SUB_QUESTIONS = int(os.environ.get("MAX_SUB_QUESTIONS", "3"))
ENABLE_TASK_DECOMPOSITION = os.environ.get("ENABLE_TASK_DECOMPOSITION", "true").lower() == "true"
```

- [ ] **Step 4: 运行测试验证通过**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_task_decomposition.TestConfigFields -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add project/config.py tests/test_task_decomposition.py
git commit -m "feat(p3): add MAX_SUB_QUESTIONS and ENABLE_TASK_DECOMPOSITION config"
```

---

## Task 2: State 新增 sub_questions 字段

**Files:**
- Modify: `project/rag_agent/graph_state.py`（`State` 类内，`planned_queries` 之后）
- Test: `tests/test_task_decomposition.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_task_decomposition.py` 的 `_make_main_state` 之前插入：

```python
class TestStateFields(unittest.TestCase):
    def test_sub_questions_field_exists(self):
        from project.rag_agent.graph_state import State
        self.assertIn("sub_questions", State.__annotations__)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_task_decomposition.TestStateFields -v`
Expected: FAIL — `'sub_questions' not found in ...`

- [ ] **Step 3: 实现**

在 `project/rag_agent/graph_state.py` 的 `State` 类中，`planned_queries: List[str] = []` 行（line 37）之后追加：

```python
    sub_questions: List[str] = []
```

无 reducer（`decompose_tasks` 单点写，`route_after_query_plan` 单点读）。

- [ ] **Step 4: 运行测试验证通过**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_task_decomposition.TestStateFields -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add project/rag_agent/graph_state.py tests/test_task_decomposition.py
git commit -m "feat(p3): add sub_questions field to State"
```

---

## Task 3: TaskDecomposition schema + decompose prompt

**Files:**
- Modify: `project/rag_agent/schemas.py`（末尾追加）
- Modify: `project/rag_agent/prompts.py`（末尾追加）
- Test: `tests/test_task_decomposition.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_task_decomposition.py` 的 `TestStateFields` 之后插入：

```python
class TestTaskDecompositionSchema(unittest.TestCase):
    def test_schema_fields(self):
        from project.rag_agent.schemas import TaskDecomposition
        inst = TaskDecomposition(needs_decomposition=True, sub_questions=["a", "b"], reason="复合")
        self.assertTrue(inst.needs_decomposition)
        self.assertEqual(inst.sub_questions, ["a", "b"])
        self.assertEqual(inst.reason, "复合")

    def test_prompt_function_exists_and_mentions_json(self):
        from project.rag_agent.prompts import get_task_decomposition_prompt
        text = get_task_decomposition_prompt()
        self.assertIn("needs_decomposition", text)
        self.assertIn("sub_questions", text)
        self.assertIn("JSON", text)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_task_decomposition.TestTaskDecompositionSchema -v`
Expected: FAIL — `ImportError: cannot import name 'TaskDecomposition'`

- [ ] **Step 3: 实现 schema**

在 `project/rag_agent/schemas.py` 末尾（`GroundingCritique` 之后）追加：

```python
class TaskDecomposition(BaseModel):
    needs_decomposition: bool = Field(description="用户问题是否包含多个可独立检索的子问题/facet。")
    sub_questions: List[str] = Field(description="分解后的独立子问题；不复合时为仅含原问题的单元素列表。")
    reason: str = Field(description="简短说明是否复合的判断依据。")
```

- [ ] **Step 4: 实现 prompt**

在 `project/rag_agent/prompts.py` 末尾追加：

```python
def get_task_decomposition_prompt() -> str:
    """System prompt for the decompose_tasks node (P3).

    The LLM judges whether a medical question contains multiple independent
    facets and, if so, splits it into 1-3 sub-questions for parallel retrieval.
    Output must be strict JSON matching the TaskDecomposition schema:
    {"needs_decomposition": bool, "sub_questions": [str], "reason": str}.
    """
    return (
        "你是一名医学问题分析员。判断用户问题是否包含多个可独立检索的子问题/facet"
        "（例如同时问用药和监测、或同时问两种不同病症）。\n\n"
        "判定标准：\n"
        "- needs_decomposition=true：问题含 2 个及以上独立 facet，应分别检索。\n"
        "- needs_decomposition=false：单一 facet，sub_questions 只放原问题本身。\n"
        "要求：\n"
        "- sub_questions 每条都是自足的、可直接用于检索的子问题，不超过 3 条。\n"
        "- 不复合时 sub_questions 必须为仅含原问题的单元素列表。\n\n"
        "严格输出 JSON，不要输出多余文字：\n"
        '{"needs_decomposition": true/false, "sub_questions": ["子问题1", "子问题2"], "reason": "简短依据"}'
    )
```

- [ ] **Step 5: 运行测试验证通过**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_task_decomposition.TestTaskDecompositionSchema -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add project/rag_agent/schemas.py project/rag_agent/prompts.py tests/test_task_decomposition.py
git commit -m "feat(p3): add TaskDecomposition schema and decompose prompt"
```

---

## Task 4: decompose_tasks 节点

**Files:**
- Modify: `project/rag_agent/rag_nodes.py`（imports 块 + 新增 `decompose_tasks` 函数 + `__all__`）
- Test: `tests/test_task_decomposition.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_task_decomposition.py` 的 `TestTaskDecompositionSchema` 之后插入：

```python
class TestDecomposeTasks(unittest.TestCase):
    def test_compound_question_yields_multiple_sub_questions(self):
        """LLM says compound → write multiple sub_questions."""
        from project.rag_agent.rag_nodes import decompose_tasks
        from project.rag_agent.schemas import TaskDecomposition
        state = _make_main_state()
        parser = MagicMock()
        parser.invoke.return_value = TaskDecomposition(
            needs_decomposition=True,
            sub_questions=["高血压合并痛风吃什么药安全", "高血压患者如何在家监测血压"],
            reason="含用药与监测两个 facet",
        )
        with patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser):
            result = decompose_tasks(state, MagicMock())
        self.assertEqual(len(result["sub_questions"]), 2)
        self.assertIn("高血压合并痛风吃什么药安全", result["sub_questions"])

    def test_simple_question_yields_single_sub_question(self):
        """LLM says not compound → sub_questions == [primary]."""
        from project.rag_agent.rag_nodes import decompose_tasks
        from project.rag_agent.schemas import TaskDecomposition
        state = _make_main_state(rewrittenQuestions=["高血压应该注意什么"])
        parser = MagicMock()
        parser.invoke.return_value = TaskDecomposition(
            needs_decomposition=False,
            sub_questions=["高血压应该注意什么"],
            reason="单一 facet",
        )
        with patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser):
            result = decompose_tasks(state, MagicMock())
        self.assertEqual(result["sub_questions"], ["高血压应该注意什么"])

    def test_empty_llm_result_falls_back_to_primary(self):
        """LLM failure (empty sub_questions) → fall back to [primary], never crash."""
        from project.rag_agent.rag_nodes import decompose_tasks
        from project.rag_agent.schemas import TaskDecomposition
        state = _make_main_state()
        parser = MagicMock()
        parser.invoke.return_value = TaskDecomposition(needs_decomposition=False, sub_questions=[], reason="")
        with patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser):
            result = decompose_tasks(state, MagicMock())
        self.assertEqual(result["sub_questions"], ["高血压合并痛风吃什么药安全"])

    def test_max_sub_questions_truncation(self):
        """LLM returns 5 sub-questions → truncated to MAX_SUB_QUESTIONS (3)."""
        import config
        from project.rag_agent.rag_nodes import decompose_tasks
        from project.rag_agent.schemas import TaskDecomposition
        state = _make_main_state()
        parser = MagicMock()
        parser.invoke.return_value = TaskDecomposition(
            needs_decomposition=True,
            sub_questions=["q1", "q2", "q3", "q4", "q5"],
            reason="复合",
        )
        with patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser):
            result = decompose_tasks(state, MagicMock())
        self.assertEqual(len(result["sub_questions"]), config.MAX_SUB_QUESTIONS)

    def test_disabled_flag_skips_llm(self):
        """ENABLE_TASK_DECOMPOSITION=False → return [primary] without calling LLM."""
        import config
        from project.rag_agent.rag_nodes import decompose_tasks
        state = _make_main_state()
        with patch.object(config, "ENABLE_TASK_DECOMPOSITION", False), \
             patch("project.rag_agent.rag_nodes._structured_output_llm") as mock_so:
            result = decompose_tasks(state, MagicMock())
            mock_so.assert_not_called()
        self.assertEqual(result["sub_questions"], ["高血压合并痛风吃什么药安全"])
```

- [ ] **Step 2: 运行测试验证失败**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_task_decomposition.TestDecomposeTasks -v`
Expected: FAIL — `ImportError: cannot import name 'decompose_tasks'`

- [ ] **Step 3a: 改 imports**

在 `project/rag_agent/rag_nodes.py` 顶部的 schemas import 块（line 18-24）追加 `TaskDecomposition`：

```python
from .schemas import (
    QueryAnalysis,
    RetrievalQueryPlan,
    GroundedAnswerCheck,
    EvidenceSufficiency,
    GroundingCritique,
    TaskDecomposition,
)
```

在 prompts import 块（line 25-33）追加 `get_task_decomposition_prompt`：

```python
from .prompts import (
    get_rewrite_query_prompt,
    get_retrieval_query_plan_prompt,
    get_orchestrator_prompt,
    get_fallback_response_prompt,
    get_context_compression_prompt,
    get_aggregation_prompt,
    get_evidence_sufficiency_prompt,
    get_grounding_critique_prompt,
    get_task_decomposition_prompt,
)
```

- [ ] **Step 3b: 新增 decompose_tasks 函数**

在 `project/rag_agent/rag_nodes.py` 的 `plan_retrieval_queries` 函数之后（line 291 之后）插入：

```python
def decompose_tasks(state: State, llm):
    """P3: judge whether a medical question is compound and split into sub-questions.

    A light LLM (via TaskDecomposition schema) decides whether the question
    contains multiple independent facets. If compound, it produces 1-N
    self-contained sub-questions (N <= MAX_SUB_QUESTIONS); if not, the result
    is a single-element list holding the primary query — which makes the
    downstream route_after_query_plan fall back to today's single-Send path.
    LLM failure (empty sub_questions) falls back to [primary] so the node never
    breaks. When ENABLE_TASK_DECOMPOSITION is False the LLM is skipped and
    [primary] is returned directly (rollback).
    """
    rewritten = [str(q).strip() for q in (state.get("rewrittenQuestions") or []) if str(q).strip()]
    original_query = str(state.get("originalQuery") or state.get("primary_user_query") or "").strip()
    primary = rewritten[0] if rewritten else original_query

    if not config.ENABLE_TASK_DECOMPOSITION or not primary:
        return {"sub_questions": [primary] if primary else []}

    sys_msg = SystemMessage(content=get_task_decomposition_prompt())
    user_payload = (
        f"用户原始问题：{original_query}\n"
        f"重写后问题：{rewritten}\n"
        f"上下文摘要：{state.get('conversation_summary', '')}\n"
        f"近期对话：{state.get('recent_context', '')}\n"
        f"话题焦点：{state.get('topic_focus', '')}"
    )
    parser = _structured_output_llm(llm, TaskDecomposition, max_tokens=config.LLM_STRUCTURED_MAX_TOKENS)
    verdict = parser.invoke([sys_msg, HumanMessage(content=user_payload)])

    subs = [str(s).strip() for s in (getattr(verdict, "sub_questions", []) or []) if str(s).strip()]

    # Fallback: LLM gave no usable sub-questions → single primary path.
    if not subs:
        return {"sub_questions": [primary] if primary else []}

    return {"sub_questions": subs[: config.MAX_SUB_QUESTIONS]}
```

- [ ] **Step 3c: 加入 __all__**

在 `project/rag_agent/rag_nodes.py` 末尾的 `__all__` 中，按字母序在 `collect_answer` 之后插入 `"decompose_tasks",`：

```python
__all__ = [
    "aggregate_answers",
    "answer_grounding_check",
    "collect_answer",
    "compress_context",
    "decompose_tasks",
    "evaluate_evidence",
    "fallback_response",
    "grounded_answer_generation",
    "orchestrator",
    "plan_retrieval_queries",
    "revise_answer",
    "rewrite_query",
    "should_compress_context",
]
```

- [ ] **Step 4: 运行测试验证通过**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_task_decomposition.TestDecomposeTasks -v`
Expected: PASS（5 个）

- [ ] **Step 5: 提交**

```bash
git add project/rag_agent/rag_nodes.py tests/test_task_decomposition.py
git commit -m "feat(p3): add decompose_tasks node with compound-question splitting"
```

---

## Task 5: route_after_query_plan 改 fan-out

**Files:**
- Modify: `project/rag_agent/edges.py:94-126`（`route_after_query_plan` 函数体）
- Test: `tests/test_task_decomposition.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_task_decomposition.py` 的 `TestDecomposeTasks` 之后插入：

```python
class TestRouteAfterQueryPlanFanOut(unittest.TestCase):
    def test_fan_out_one_send_per_sub_question(self):
        """N sub-questions → N Sends, question_index 0..N-1, query_plan=[q] each."""
        from project.rag_agent.edges import route_after_query_plan
        state = _make_main_state(
            rewrittenQuestions=["高血压合并痛风吃什么药安全"],
            sub_questions=["高血压合并痛风吃什么药安全", "高血压患者如何在家监测血压"],
        )
        sends = route_after_query_plan(state)
        self.assertEqual(len(sends), 2)
        self.assertTrue(all(isinstance(s, Send) for s in sends))
        self.assertEqual(sends[0].arg["question"], "高血压合并痛风吃什么药安全")
        self.assertEqual(sends[0].arg["question_index"], 0)
        self.assertEqual(sends[0].arg["query_plan"], ["高血压合并痛风吃什么药安全"])
        self.assertEqual(sends[1].arg["question"], "高血压患者如何在家监测血压")
        self.assertEqual(sends[1].arg["question_index"], 1)
        self.assertEqual(sends[1].arg["query_plan"], ["高血压患者如何在家监测血压"])

    def test_single_sub_question_returns_one_send(self):
        """One sub-question → one Send (today's single-path behavior)."""
        from project.rag_agent.edges import route_after_query_plan
        state = _make_main_state(
            rewrittenQuestions=["高血压应该注意什么"],
            sub_questions=["高血压应该注意什么"],
        )
        sends = route_after_query_plan(state)
        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0].arg["question_index"], 0)

    def test_no_sub_questions_falls_back_to_primary(self):
        """Empty sub_questions → single Send with primary from rewrittenQuestions."""
        from project.rag_agent.edges import route_after_query_plan
        state = _make_main_state(rewrittenQuestions=["高血压应该注意什么"], sub_questions=[])
        sends = route_after_query_plan(state)
        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0].arg["question"], "高血压应该注意什么")
        self.assertEqual(sends[0].arg["query_plan"], ["高血压应该注意什么"])

    def test_context_fields_propagated_to_each_send(self):
        """Each Send carries the shared context fields."""
        from project.rag_agent.edges import route_after_query_plan
        state = _make_main_state(
            conversation_summary="摘要",
            recent_context="近期",
            topic_focus="焦点",
            user_memories="记忆",
            sub_questions=["q1", "q2"],
        )
        sends = route_after_query_plan(state)
        for s in sends:
            self.assertEqual(s.arg["context_summary"], "摘要")
            self.assertEqual(s.arg["recent_context"], "近期")
            self.assertEqual(s.arg["topic_focus"], "焦点")
            self.assertEqual(s.arg["user_memories"], "记忆")
            self.assertEqual(s.arg["messages"], [])
```

- [ ] **Step 2: 运行测试验证失败**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_task_decomposition.TestRouteAfterQueryPlanFanOut -v`
Expected: FAIL — 当前 `route_after_query_plan` 返回单 Send，`test_fan_out_one_send_per_sub_question` 期望 2 个；且当前 `query_plan` 是 deduped 列表而非 `[q]`。

- [ ] **Step 3: 实现**

把 `project/rag_agent/edges.py` 中 `route_after_query_plan`（line 94-126）整段替换为：

```python
def route_after_query_plan(state: State):
    """P3: fan out one Send per sub-question; LangGraph runs them in parallel.

    Each sub-question enters the agent subgraph independently and runs the P1
    retrieval loop. collect_answer writes agent_answers with question_index=i;
    the accumulate_or_reset reducer merges them; grounded_answer_generation
    sorts by index and synthesizes. With a single sub-question this is exactly
    today's single-Send path.
    """
    summary = state.get("conversation_summary", "")
    recent_context = state.get("recent_context", "")
    topic_focus = state.get("topic_focus", "")
    user_memories = state.get("user_memories", "")

    rewritten = [str(q).strip() for q in (state.get("rewrittenQuestions") or []) if str(q).strip()]
    primary = (rewritten[0] if rewritten else "") or state.get("originalQuery", "") or state.get("primary_user_query", "")

    subs = [str(s).strip() for s in (state.get("sub_questions") or []) if str(s).strip()]
    if not subs:
        subs = [primary] if primary else []
    subs = subs[:MAX_SUB_QUESTIONS]

    payload_base = {
        "messages": [],
        "context_summary": summary,
        "recent_context": recent_context,
        "topic_focus": topic_focus,
        "user_memories": user_memories,
    }
    return [
        Send(
            "agent",
            {
                **payload_base,
                "question": q,
                "question_index": i,
                "query_plan": [q],
            },
        )
        for i, q in enumerate(subs)
    ]
```

`MAX_SUB_QUESTIONS` 已在 edges.py 顶部直接导入（见下）——edges.py 约定用直接名字（`MAX_EVIDENCE_ROUNDS`，非 `config.MAX_EVIDENCE_ROUNDS`），故函数体内写 `MAX_SUB_QUESTIONS`（非 `config.MAX_SUB_QUESTIONS`）。确认：edges.py line 5 `from config import MAX_ITERATIONS, MAX_TOOL_CALLS, MAX_EVIDENCE_ROUNDS, MAX_GROUNDING_ROUNDS`，需**追加** `MAX_SUB_QUESTIONS`。把 line 5 改为：

```python
from config import MAX_ITERATIONS, MAX_TOOL_CALLS, MAX_EVIDENCE_ROUNDS, MAX_GROUNDING_ROUNDS, MAX_SUB_QUESTIONS
```

- [ ] **Step 4: 运行测试验证通过**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_task_decomposition.TestRouteAfterQueryPlanFanOut -v`
Expected: PASS（4 个）

- [ ] **Step 5: 提交**

```bash
git add project/rag_agent/edges.py tests/test_task_decomposition.py
git commit -m "feat(p3): route_after_query_plan fans out one Send per sub-question"
```

---

## Task 6: route_after_rewrite 改指向 + graph 接线

**Files:**
- Modify: `project/rag_agent/edges.py:91`（`route_after_rewrite` 默认返回值）
- Modify: `project/rag_agent/graph.py`（import line 12-23、节点注册 line 92、连边 line 159-170）
- Test: `tests/test_task_decomposition.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_task_decomposition.py` 的 `TestRouteAfterQueryPlanFanOut` 之后插入：

```python
class TestGraphWiring(unittest.TestCase):
    def test_route_after_rewrite_targets_decompose_tasks(self):
        """medical_rag default route target is now decompose_tasks."""
        from project.rag_agent.edges import route_after_rewrite
        self.assertEqual(
            route_after_rewrite({"questionIsClear": True, "intent": "medical_rag",
                                 "rewrittenQuestions": ["高血压日常注意事项"]}),
            "decompose_tasks",
        )

    def test_graph_source_references_decomposition_wiring(self):
        """graph.py must register decompose_tasks and wire it into the rewrite edge."""
        import inspect
        import project.rag_agent.graph as graph_mod
        src = inspect.getsource(graph_mod)
        self.assertIn("decompose_tasks", src)
        self.assertIn("route_after_query_plan", src)
        # plan_retrieval_queries should no longer be wired as a node (kept as symbol only).
        self.assertNotIn('add_node("plan_retrieval_queries"', src)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_task_decomposition.TestGraphWiring -v`
Expected: FAIL — `route_after_rewrite` 仍返回 `"plan_retrieval_queries"`；graph.py 仍 `add_node("plan_retrieval_queries", ...)`

- [ ] **Step 3a: 改 route_after_rewrite 默认返回**

把 `project/rag_agent/edges.py` line 91 的 `return "plan_retrieval_queries"` 改为：

```python
    # Default: medical RAG pipeline (P3: decompose into sub-questions first)
    return "decompose_tasks"
```

- [ ] **Step 3b: 改 graph.py import**

在 `project/rag_agent/graph.py` 的 rag_nodes import 块（line 12-23）按字母序加入 `decompose_tasks`（在 `compress_context` 之后、`evaluate_evidence` 之前）：

```python
from .rag_nodes import (
    answer_grounding_check,
    collect_answer,
    compress_context,
    decompose_tasks,
    evaluate_evidence,
    fallback_response,
    grounded_answer_generation,
    orchestrator,
    plan_retrieval_queries,
    revise_answer,
    rewrite_query,
    should_compress_context,
)
```

> 注：`plan_retrieval_queries` 保留在 import 中（符号仍被 `__all__` 导出，且保留可回滚），但下文不再 `add_node` 它。

- [ ] **Step 3c: 改节点注册 + 连边**

把 `project/rag_agent/graph.py` line 92：
```python
    graph_builder.add_node("plan_retrieval_queries", partial(plan_retrieval_queries, llm=_light_llm))
```
替换为：
```python
    graph_builder.add_node("decompose_tasks", partial(decompose_tasks, llm=_light_llm))
```

把 line 159-169 的 `route_after_rewrite` 条件边映射里，`"plan_retrieval_queries": "plan_retrieval_queries"` 改为 `"decompose_tasks": "decompose_tasks"`，并把 skill-targets 排除列表里的 `"plan_retrieval_queries"` 改为 `"decompose_tasks"`：

```python
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite, {
        "request_clarification": "request_clarification",
        "decompose_tasks": "decompose_tasks",
        "handle_appointment_skill": "handle_appointment_skill",
        "recommend_department": "recommend_department",
        "__end__": END,
        **{node_name: node_name for node_name in _skill_route_targets.values()
           if node_name not in ("request_clarification", "decompose_tasks",
                                "handle_appointment_skill", "recommend_department",
                                "rewrite_query")},
    })
```

把 line 170 的条件边源头节点从 `plan_retrieval_queries` 改为 `decompose_tasks`：
```python
    graph_builder.add_conditional_edges("decompose_tasks", route_after_query_plan)
```

- [ ] **Step 4: 编译检查 + 运行测试**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m compileall project/rag_agent/graph.py project/rag_agent/edges.py`
Expected: 无输出

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_task_decomposition.TestGraphWiring -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add project/rag_agent/edges.py project/rag_agent/graph.py tests/test_task_decomposition.py
git commit -m "feat(p3): retarget rewrite→decompose_tasks and rewire graph (replace plan_retrieval_queries)"
```

---

## Task 7: 编译图集成测试 + routing_edges 回归 + 全量回归

**Files:**
- Test: `tests/test_task_decomposition.py`（追加集成测试）
- Modify: `tests/test_routing_edges.py:99,128`（route_after_rewrite 断言）

- [ ] **Step 1: 写集成测试**

在 `tests/test_task_decomposition.py` 的 `TestGraphWiring` 之后、`if __name__` 之前插入：

```python
class TestCompiledDecompositionFanOut(unittest.TestCase):
    """Verify decompose_tasks -> route_after_query_plan fan-out survives LangGraph's real
    state machinery: N sub-questions produce N parallel agent invocations whose
    agent_answers entries (index 0..N-1) are merged by the accumulate_or_reset reducer."""

    def _build_graph(self, llm):
        from langgraph.graph import StateGraph, START, END
        from project.rag_agent.graph_state import State
        from project.rag_agent.rag_nodes import decompose_tasks
        from project.rag_agent.edges import route_after_query_plan
        from functools import partial

        builder = StateGraph(State)
        builder.add_node("decompose_tasks", partial(decompose_tasks, llm=llm))

        # Sink records every Send the edge dispatched (one per sub-question).
        dispatched = {"sends": []}

        def _agent_sink(state):
            # Mimic collect_answer: append an agent_answers entry tagged by question_index.
            idx = state.get("question_index", 0)
            q = state.get("question", "")
            dispatched["sends"].append({"index": idx, "question": q})
            return {"agent_answers": [{
                "index": idx,
                "question": q,
                "answer": f"answer-{idx}",
                "query_plan": [q],
                "confidence_bucket": "high",
                "evidence_score": 0.9,
                "sources": [],
            }]}

        builder.add_node("agent", _agent_sink)
        builder.add_edge(START, "decompose_tasks")
        builder.add_conditional_edges("decompose_tasks", route_after_query_plan)
        builder.add_edge("agent", END)
        return builder.compile(), dispatched

    def test_two_sub_questions_dispatch_two_parallel_agents(self):
        """Compound question → 2 Sends → 2 agent_sink invocations → 2 agent_answers (index 0,1)."""
        from project.rag_agent.schemas import TaskDecomposition
        parser = MagicMock()
        parser.invoke.return_value = TaskDecomposition(
            needs_decomposition=True,
            sub_questions=["高血压合并痛风吃什么药安全", "高血压患者如何在家监测血压"],
            reason="复合",
        )
        with patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser):
            graph, dispatched = self._build_graph(MagicMock())
            state = _make_main_state()
            final = graph.invoke(state, {"recursion_limit": 20})
        # Two parallel agent invocations happened.
        self.assertEqual(len(dispatched["sends"]), 2)
        indices = sorted(s["index"] for s in dispatched["sends"])
        self.assertEqual(indices, [0, 1])
        # fan-in: agent_answers merged with both indices.
        answer_indices = sorted(a["index"] for a in final["agent_answers"])
        self.assertEqual(answer_indices, [0, 1])

    def test_single_sub_question_dispatches_one_agent(self):
        """Simple question → 1 Send → 1 agent_sink → 1 agent_answer (today's path)."""
        from project.rag_agent.schemas import TaskDecomposition
        parser = MagicMock()
        parser.invoke.return_value = TaskDecomposition(
            needs_decomposition=False,
            sub_questions=["高血压应该注意什么"],
            reason="单一 facet",
        )
        with patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser):
            graph, dispatched = self._build_graph(MagicMock())
            state = _make_main_state(rewrittenQuestions=["高血压应该注意什么"])
            final = graph.invoke(state, {"recursion_limit": 20})
        self.assertEqual(len(dispatched["sends"]), 1)
        self.assertEqual(len(final["agent_answers"]), 1)
        self.assertEqual(final["agent_answers"][0]["index"], 0)
```

- [ ] **Step 2: 运行集成测试**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_task_decomposition.TestCompiledDecompositionFanOut -v`
Expected: PASS（2 个）。若失败，检查 `Send` fan-out 是否真的触发并行 agent_sink（LangGraph 对返回 `Send` 列表的条件边自动 fan-out），以及 `accumulate_or_reset` 是否合并了多条 `agent_answers`。

- [ ] **Step 3: 跑整个 test_task_decomposition 文件**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_task_decomposition -v`
Expected: 全部 PASS（Config + State + Schema + DecomposeTasks + RouteFanOut + GraphWiring + CompiledFanOut）

- [ ] **Step 4: 更新 test_routing_edges 回归断言**

`tests/test_routing_edges.py` 有两处断言 `route_after_rewrite(...) == "plan_retrieval_queries"`，需改为 `"decompose_tasks"`：

line 99（`test_route_after_rewrite_passes_recent_context_to_agent_subgraph` 内）：
```python
            "plan_retrieval_queries",
```
改为：
```python
            "decompose_tasks",
```

line 128（`test_route_after_rewrite_treats_medical_rag_skill_as_retrieval_pipeline` 内）：
```python
            "plan_retrieval_queries",
```
改为：
```python
            "decompose_tasks",
```

> 注：line 102-117 的 `route_after_query_plan` 断言**无需改**——无 `sub_questions` 时退回 `[rewrittenQuestions[0]]` 单 Send，`query_plan=[q]` 仍等于 `["高血压应该注意什么"]`，断言保持绿。改前先 Read 该文件确认确切文本（whitespace 需匹配）。

- [ ] **Step 5: 跑 test_routing_edges**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_routing_edges -v`
Expected: 全部 PASS（更新后的断言）

- [ ] **Step 6: P1 + P2 回归（确保不碰子图/回答回路）**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_agentic_retrieval tests.test_answer_reflection -v`
Expected: P1 的 20 个 + P2 的 16 个全 PASS（P3 不改 AgentState / evaluate_evidence / answer_grounding_check / revise_answer）

- [ ] **Step 7: 编译检查**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m compileall project tests`
Expected: 无输出

- [ ] **Step 8: 全量 unittest 回归**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest discover -s tests -v`
Expected: P3 新增用例全 PASS；既有失败与 main 基线一致（已知：中文断言 mojibake、缺 LLM token 的环境性失败），**P3 不引入新回归**。若出现新的失败且 traceback 提及 `decompose_tasks`/`route_after_query_plan`/`sub_questions`/`Send`，需排查 P3 改动。

- [ ] **Step 9: 提交**

```bash
git add tests/test_task_decomposition.py tests/test_routing_edges.py
git commit -m "test(p3): compiled-graph fan-out integration + routing_edges regression"
```

---

## 验收对照（spec §11）

| 验收项 | 任务 |
|---|---|
| 1. decompose_tasks + TaskDecomposition schema + sub_questions 字段 | Task 2, 3, 4 |
| 2. route_after_query_plan 扩为多 Send fan-out（question_index 0..N-1） | Task 5 |
| 3. route_after_rewrite medical_rag 目标改 decompose_tasks | Task 6 |
| 4. plan_retrieval_queries 从图中摘除（符号保留） | Task 6 |
| 5. MAX_SUB_QUESTIONS 截断；ENABLE_TASK_DECOMPOSITION 可回滚 | Task 1, 4 (test_disabled_flag_skips_llm) |
| 6. 复合并行 fan-out → 多 agent_answers → 合成（集成测试） | Task 7 |
| 7. 简单问题退单路径等价今天 | Task 4 (test_simple) + Task 7 (test_single) |
| 8. 全量不引入新回归；P1/P2 不受影响 | Task 7 |
| 9. 编译图集成用例证明 fan-out+fan-in 走通 | Task 7 (2 个) |
