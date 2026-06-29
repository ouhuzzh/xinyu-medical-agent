# P2 回答反思回路 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把回答阶段从"一次性生成 → END"提升为"判定 → 必要时批判重写 → 复检"的反思回路，启用此前算出却从未用于分支的 `grounded` 字段。

**Architecture:** 在主图回答段把硬边 `answer_grounding_check -> END` 改为条件边 `route_after_grounding -> END | revise_answer`，新增 `revise_answer` 节点（light LLM + `GroundingCritique` schema）做"批判 + 基于证据重写"，复检回 `answer_grounding_check`。`answer_grounding_check` 写出新的 `grounding_passed` 字段供边判断。带 `MAX_GROUNDING_ROUNDS` 上限，`ENABLE_ANSWER_REFLECTION` 开关可回滚硬边。不重检索（不跨 P1 子图）。

**Tech Stack:** LangGraph StateGraph + conditional edges, Pydantic schema, `_structured_output_llm`（regex JSON 解析，SiliconFlow/Qwen 兼容），Python unittest。

**Spec:** `docs/superpowers/specs/2026-06-29-answer-reflection-loop-design.md`

**Scope guard:** 本计划只改动 `project/rag_agent/`、`project/config.py`、`tests/test_answer_reflection.py`、`docs/superpowers/plans/2026-06-29-answer-reflection-loop.md`。**不得**触碰工作区里那批未提交的 frontend/api/db 改动（`frontend/src/*`、`project/api/routes/chat.py`、`project/api/schemas.py`、`project/db/*`、`tests/test_api_app.py`）——它们单独处理，不混入 P2 commit。

---

## 关键代码事实（实现者必读）

1. **`answer_grounding_check`（`rag_nodes.py:687`）当前返回 `{}` 或 `{"messages": [AIMessage(revised)]}`**。`grounded` 字段由 `ground_answer(...)` 算出但**从未用于分支**——这是 P2 的接入点。

2. **`messages` 用 `MessagesState` 的 `add_messages` reducer（追加，非替换）**。返回 `{"messages": [AIMessage(x)]}` 是**追加**一条；后续节点读 `state["messages"][-1]` 取最新。现有 `answer_grounding_check` 已是这套"追加修正版，最后一条为准"的模式。P2 沿用，**不要**用 `RemoveMessage`。

3. **`ground_answer`（`tools.py:205`，rule-based 无 LLM）** 返回 `{grounded: bool, revised_answer: str, note: str}`：
   - `grounded=True` 时 `revised_answer == 原文`（不改）。
   - `grounded=False` 时 `revised_answer` 是被动贴了免责声明的版本。

4. **`_structured_output_llm(llm, schema, max_tokens=N)`（`node_helpers.py:114`）**：返回带 `.invoke()` 的 parser；LLM 失败/解析失败时返回 schema 默认值（str 字段为 `""`），**永不抛异常**。所以 `revise_answer` 里 `parser.invoke(...)` 不会抛——失败表现为 `revised_answer == ""`，走 fallback。

5. **边返回 END 的约定**：返回字符串 `"__end__"`，graph.py 在 conditional_edges 映射里 `{"__end__": END}`（见 `edges.py:163` 的 `route_after_action`）。

6. **`edges.py` 顶部已有** `from config import MAX_ITERATIONS, MAX_TOOL_CALLS, MAX_EVIDENCE_ROUNDS`（line 5），需追加 `MAX_GROUNDING_ROUNDS`。

7. **`graph.py:189-190`** 现有：
   ```python
   graph_builder.add_edge("grounded_answer_generation", "answer_grounding_check")
   graph_builder.add_edge("answer_grounding_check", END)
   ```
   第二行是 P2 改条件边的位置。`answer_grounding_check` 当前绑定 `partial(answer_grounding_check, llm=_strong_llm)`（line 101），`llm` 参数在节点内未被使用（`ground_answer` 是 rule-based），保持不动。

8. **运行测试**：`PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection -v`（Windows bash）。测试文件首行已 `sys.path.insert(0, .../project)`。

---

## File Structure

| 文件 | 责任 | 改动 |
|---|---|---|
| `project/config.py` | 新增 `MAX_GROUNDING_ROUNDS`、`ENABLE_ANSWER_REFLECTION` | 追加 2 行 |
| `project/rag_agent/graph_state.py` | `State` 新增 3 个 grounding 反思字段 | 追加 3 行 |
| `project/rag_agent/schemas.py` | 新增 `GroundingCritique` schema | 追加 1 class |
| `project/rag_agent/prompts.py` | 新增 `get_grounding_critique_prompt()` | 追加 1 func |
| `project/rag_agent/rag_nodes.py` | `answer_grounding_check` 写出 `grounding_passed`；新增 `revise_answer` 节点；imports + `__all__` | 改 1 函数 + 加 1 函数 |
| `project/rag_agent/edges.py` | 新增 `route_after_grounding` 边；import 追加 | 加 1 函数 + 改 import |
| `project/rag_agent/graph.py` | 条件接线 + 注册 `revise_answer` 节点 + import | 改接线 + 改 import |
| `tests/test_answer_reflection.py` | 全部单元 + 集成测试 | 新建 |

---

## Task 1: config 新增开关与上限

**Files:**
- Modify: `project/config.py:86`（P1 块之后）

- [ ] **Step 1: 写失败测试**

创建 `tests/test_answer_reflection.py`：

```python
"""Tests for P2 answer reflection loop: revise_answer + route_after_grounding."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from unittest.mock import MagicMock, patch  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage  # noqa: E402
from langchain_core.documents import Document  # noqa: E402


class TestConfigFields(unittest.TestCase):
    def test_grounding_config_fields_exist(self):
        import config
        self.assertTrue(hasattr(config, "MAX_GROUNDING_ROUNDS"))
        self.assertEqual(config.MAX_GROUNDING_ROUNDS, 1)
        self.assertTrue(hasattr(config, "ENABLE_ANSWER_REFLECTION"))
        self.assertTrue(config.ENABLE_ANSWER_REFLECTION)


def _make_main_state(messages, **extra):
    base = {
        "messages": messages,
        "originalQuery": "高血压合并痛风吃什么药安全",
        "agent_answers": [],
        "grounding_evidence_score": None,
        "grounding_rounds": 0,
        "grounding_critique": "",
        "grounding_passed": False,
    }
    base.update(extra)
    return base


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection.TestConfigFields -v`
Expected: FAIL — `AttributeError: module 'config' has no attribute 'MAX_GROUNDING_ROUNDS'`

- [ ] **Step 3: 实现**

在 `project/config.py` 的 P1 块（line 84-86）之后追加：

```python
# P2: answer reflection — LLM grounding-critique rewrite loop
MAX_GROUNDING_ROUNDS = int(os.environ.get("MAX_GROUNDING_ROUNDS", "1"))
ENABLE_ANSWER_REFLECTION = os.environ.get("ENABLE_ANSWER_REFLECTION", "true").lower() == "true"
```

- [ ] **Step 4: 运行测试验证通过**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection.TestConfigFields -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add project/config.py tests/test_answer_reflection.py
git commit -m "feat(p2): add MAX_GROUNDING_ROUNDS and ENABLE_ANSWER_REFLECTION config"
```

---

## Task 2: State 新增 grounding 反思字段

**Files:**
- Modify: `project/rag_agent/graph_state.py`（`State` 类内，`grounding_evidence_score` 之后）
- Test: `tests/test_answer_reflection.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_answer_reflection.py` 的 `_make_main_state` 之前插入：

```python
class TestStateFields(unittest.TestCase):
    def test_grounding_reflection_fields_exist(self):
        from project.rag_agent.graph_state import State
        defaults = State.__annotations__
        for field in ("grounding_passed", "grounding_critique", "grounding_rounds"):
            self.assertIn(field, defaults, f"State missing field: {field}")
```

- [ ] **Step 2: 运行测试验证失败**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection.TestStateFields -v`
Expected: FAIL — `AssertionError: 'grounding_passed' not found in ...`（State 尚无该字段）

- [ ] **Step 3: 实现**

在 `project/rag_agent/graph_state.py` 的 `State` 类中，`grounding_evidence_score` 行（line 61）之后追加：

```python
    grounding_passed: bool = False
    grounding_critique: str = ""
    grounding_rounds: int = 0
```

三个字段均无 reducer（每轮单点写入，last-write-wins）。

- [ ] **Step 4: 运行测试验证通过**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection.TestStateFields -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add project/rag_agent/graph_state.py tests/test_answer_reflection.py
git commit -m "feat(p2): add grounding reflection fields to State"
```

---

## Task 3: GroundingCritique schema + critique prompt

**Files:**
- Modify: `project/rag_agent/schemas.py`（末尾追加）
- Modify: `project/rag_agent/prompts.py`（末尾追加）
- Test: `tests/test_answer_reflection.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_answer_reflection.py` 的 `TestStateFields` 之后插入：

```python
class TestGroundingCritiqueSchema(unittest.TestCase):
    def test_schema_fields(self):
        from project.rag_agent.schemas import GroundingCritique
        inst = GroundingCritique(critique="c", revised_answer="r")
        self.assertEqual(inst.critique, "c")
        self.assertEqual(inst.revised_answer, "r")

    def test_prompt_function_exists_and_mentions_json(self):
        from project.rag_agent.prompts import get_grounding_critique_prompt
        text = get_grounding_critique_prompt()
        self.assertIn("critique", text)
        self.assertIn("revised_answer", text)
        self.assertIn("JSON", text)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection.TestGroundingCritiqueSchema -v`
Expected: FAIL — `ImportError: cannot import name 'GroundingCritique'`

- [ ] **Step 3: 实现 schema**

在 `project/rag_agent/schemas.py` 末尾（`GroundedAnswerCheck` 之后）追加：

```python
class GroundingCritique(BaseModel):
    critique: str = Field(description="哪些回答内容超出检索证据、缺证据或与证据矛盾。")
    revised_answer: str = Field(description="基于现有证据重写后的回答，收窄到证据范围内，不加免责声明。")
```

- [ ] **Step 4: 实现 prompt**

在 `project/rag_agent/prompts.py` 末尾追加：

```python
def get_grounding_critique_prompt() -> str:
    """System prompt for the revise_answer node (P2).

    The LLM critiques an already-generated answer against the retrieved
    evidence (which claims exceed / lack / contradict evidence) and produces
    an evidence-bounded rewrite. Output must be strict JSON matching the
    GroundingCritique schema:
    {"critique": str, "revised_answer": str}.
    """
    return (
        "你是一名严谨的回答 grounding 评审员。给定用户问题、检索证据和一份已生成的回答，"
        "判断回答中哪些内容超出了证据范围、缺少证据支撑或与证据矛盾，并基于现有证据重写回答"
        "（收窄到证据范围内，不得编造新事实）。\n\n"
        "要求：\n"
        "- critique：逐条指出超证据 / 缺证据 / 与证据矛盾的论断。\n"
        "- revised_answer：基于现有证据重写后的回答，只保留有证据支撑的内容，收窄表述，"
        "不加免责声明（声明由系统统一处理）。\n\n"
        "严格输出 JSON，不要输出多余文字：\n"
        '{"critique": "逐条问题", "revised_answer": "重写后的回答"}'
    )
```

- [ ] **Step 5: 运行测试验证通过**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection.TestGroundingCritiqueSchema -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add project/rag_agent/schemas.py project/rag_agent/prompts.py tests/test_answer_reflection.py
git commit -m "feat(p2): add GroundingCritique schema and critique prompt"
```

---

## Task 4: answer_grounding_check 写出 grounding_passed

**Files:**
- Modify: `project/rag_agent/rag_nodes.py:687-738`（`answer_grounding_check` 函数体）
- Test: `tests/test_answer_reflection.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_answer_reflection.py` 的 `TestGroundingCritiqueSchema` 之后插入：

```python
class TestAnswerGroundingCheck(unittest.TestCase):
    def test_fast_path_writes_grounding_passed_true(self):
        """Strong evidence fast-path → skip ground_answer, write grounding_passed=True."""
        from project.rag_agent.rag_nodes import answer_grounding_check
        state = _make_main_state(
            [AIMessage(content="某回答")],
            agent_answers=[{"confidence_bucket": "high", "evidence_score": 0.9, "answer": "证据", "source": "src"}],
            grounding_evidence_score=0.9,
        )
        with patch("project.rag_agent.rag_nodes.ground_answer") as mock_g:
            result = answer_grounding_check(state, MagicMock())
            mock_g.assert_not_called()
        self.assertEqual(result, {"grounding_passed": True})

    def test_grounded_true_returns_passed_true_no_overwrite(self):
        """ground_answer says grounded=True (revised==current) → grounding_passed=True, no message append."""
        from project.rag_agent.rag_nodes import answer_grounding_check
        state = _make_main_state(
            [AIMessage(content="有证据的回答")],
            agent_answers=[{"confidence_bucket": "low", "evidence_score": 0.5, "answer": "证据文本", "source": "src"}],
            grounding_evidence_score=0.5,
        )
        with patch("project.rag_agent.rag_nodes.ground_answer",
                   return_value={"grounded": True, "revised_answer": "有证据的回答", "note": "grounded"}):
            result = answer_grounding_check(state, MagicMock())
        self.assertTrue(result["grounding_passed"])
        self.assertNotIn("messages", result)

    def test_not_grounded_appends_disclaimer_and_marks_false(self):
        """ground_answer says grounded=False → append disclaimer version, grounding_passed=False."""
        from project.rag_agent.rag_nodes import answer_grounding_check
        state = _make_main_state(
            [AIMessage(content="超证据回答")],
            agent_answers=[{"confidence_bucket": "low", "evidence_score": 0.5, "answer": "证据", "source": "src"}],
            grounding_evidence_score=0.5,
        )
        with patch("project.rag_agent.rag_nodes.ground_answer",
                   return_value={"grounded": False, "revised_answer": "超证据回答【声明】", "note": "low_confidence_guardrail"}):
            result = answer_grounding_check(state, MagicMock())
        self.assertFalse(result["grounding_passed"])
        self.assertEqual(len(result["messages"]), 1)
        self.assertIn("【声明】", result["messages"][0].content)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection.TestAnswerGroundingCheck -v`
Expected: FAIL — fast-path 测试期望 `{"grounding_passed": True}`，当前返回 `{}`；其余两个期望 `grounding_passed` 键，当前无此键。

- [ ] **Step 3: 实现**

把 `project/rag_agent/rag_nodes.py` 中 `answer_grounding_check`（line 687-738）整段替换为：

```python
def answer_grounding_check(state: State, llm):
    latest_message = state["messages"][-1] if state.get("messages") else None
    current_answer = str(getattr(latest_message, "content", "") or "").strip()
    confidence_levels = [
        str(item.get("confidence_bucket") or "").strip().lower()
        for item in state.get("agent_answers") or []
        if str(item.get("confidence_bucket") or "").strip()
    ]
    # Fast-path: skip grounding check when evidence is clearly strong
    has_low = any(c in ("low", "no_evidence") for c in confidence_levels)
    evidence_score = state.get("grounding_evidence_score")
    if not has_low and evidence_score is not None and evidence_score >= config.RAG_HIGH_CONFIDENCE_SCORE:
        return {"grounding_passed": True}
    # Build evidence docs from agent_answers (each answer carries its retrieval
    # evidence metadata — score, source citation).  If agent_answers is empty,
    # fall back to the legacy numerical score.
    evidence_docs = []
    for item in (state.get("agent_answers") or []):
        if isinstance(item, dict):
            content = item.get("answer", "") or item.get("content", "") or ""
            score = item.get("score", item.get("evidence_score", None))
            source = item.get("source", "") or item.get("citation", "") or ""
            if not source and score is not None:
                source = f"evidence_score={score}"
            if source:
                evidence_docs.append(Document(page_content=str(content), metadata={"score": float(score) if score else 0.0, "source": str(source)}))
    if not evidence_docs and evidence_score is not None:
        evidence_docs = [Document(page_content="", metadata={"score": float(evidence_score)})]
    if not evidence_docs:
        if "no_evidence" in confidence_levels:
            evidence_docs = []
        else:
            evidence_docs = [Document(page_content="", metadata={"score": 0.88 if "high" in confidence_levels else 0.68})]
    original_query = state.get("originalQuery", "")
    risk_level = _infer_risk_level(original_query, state.get("risk_level", "normal"))
    grounded = ground_answer(
        current_answer,
        evidence_docs,
        question=original_query,
        medical_mode=_looks_like_medical_request(
            original_query,
            conversation_summary=state.get("conversation_summary", ""),
            recent_context=state.get("recent_context", ""),
            topic_focus=state.get("topic_focus", ""),
        ),
        high_risk=_needs_strict_medical_safety(original_query, risk_level),
    )
    final_answer = _strip_leading_query_plan_blob(grounded.get("revised_answer", current_answer))
    is_grounded = bool(grounded.get("grounded"))
    delta: dict = {"grounding_passed": is_grounded}
    # Append the (passive disclaimer) revised answer only when it differs —
    # this is the termination-branch safe degrade; if revise_answer runs next
    # it appends an evidence-bounded rewrite that becomes the latest message.
    if final_answer != current_answer:
        delta["messages"] = [AIMessage(content=final_answer)]
    return delta
```

变更要点：快路径 `return {}` → `return {"grounding_passed": True}`；末尾用 `is_grounded` 显式写 `grounding_passed`，仅在 `final_answer != current_answer` 时追加 messages（与原行为等价，但把 `grounded` 用于分支）。

- [ ] **Step 4: 运行测试验证通过**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection.TestAnswerGroundingCheck -v`
Expected: PASS（3 个）

- [ ] **Step 5: 提交**

```bash
git add project/rag_agent/rag_nodes.py tests/test_answer_reflection.py
git commit -m "feat(p2): answer_grounding_check writes grounding_passed for branching"
```

---

## Task 5: revise_answer 节点

**Files:**
- Modify: `project/rag_agent/rag_nodes.py`（imports 块 + 新增 `revise_answer` 函数 + `__all__`）
- Test: `tests/test_answer_reflection.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_answer_reflection.py` 的 `TestAnswerGroundingCheck` 之后插入：

```python
class TestReviseAnswer(unittest.TestCase):
    def test_llm_rewrite_appends_and_increments_round(self):
        """LLM returns a valid critique+rewrite → rewrite appended, critique recorded, rounds+1."""
        from project.rag_agent.rag_nodes import revise_answer
        from project.rag_agent.schemas import GroundingCritique
        state = _make_main_state(
            [AIMessage(content="超证据回答")],
            agent_answers=[{"answer": "证据文本", "evidence_score": 0.5, "source": "src"}],
            grounding_rounds=0,
        )
        parser = MagicMock()
        parser.invoke.return_value = GroundingCritique(critique="第三句剂量推荐超证据", revised_answer="收窄版回答")
        with patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser), \
             patch("project.rag_agent.rag_nodes.ground_answer",
                   return_value={"grounded": False, "revised_answer": "fallback声明版", "note": "low_confidence_guardrail"}):
            result = revise_answer(state, MagicMock())
        self.assertEqual(result["messages"][0].content, "收窄版回答")
        self.assertEqual(result["grounding_critique"], "第三句剂量推荐超证据")
        self.assertEqual(result["grounding_rounds"], 1)

    def test_empty_llm_result_falls_back_to_ground_answer(self):
        """LLM returns empty (default-on-failure shape) → use ground_answer.revised_answer + note."""
        from project.rag_agent.rag_nodes import revise_answer
        from project.rag_agent.schemas import GroundingCritique
        state = _make_main_state(
            [AIMessage(content="超证据回答")],
            agent_answers=[{"answer": "证据", "evidence_score": 0.5, "source": "src"}],
            grounding_rounds=0,
        )
        parser = MagicMock()
        parser.invoke.return_value = GroundingCritique(critique="", revised_answer="")
        with patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser), \
             patch("project.rag_agent.rag_nodes.ground_answer",
                   return_value={"grounded": False, "revised_answer": "fallback声明版", "note": "low_confidence_guardrail"}):
            result = revise_answer(state, MagicMock())
        self.assertEqual(result["messages"][0].content, "fallback声明版")
        self.assertEqual(result["grounding_critique"], "low_confidence_guardrail")
        self.assertEqual(result["grounding_rounds"], 1)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection.TestReviseAnswer -v`
Expected: FAIL — `ImportError: cannot import name 'revise_answer'`

- [ ] **Step 3a: 改 imports**

在 `project/rag_agent/rag_nodes.py` 顶部的 schemas import 块（line 18-23）追加 `GroundingCritique`：

```python
from .schemas import (
    QueryAnalysis,
    RetrievalQueryPlan,
    GroundedAnswerCheck,
    EvidenceSufficiency,
    GroundingCritique,
)
```

在 prompts import 块（line 24-32）追加 `get_grounding_critique_prompt`：

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
)
```

- [ ] **Step 3b: 新增 revise_answer 函数**

在 `project/rag_agent/rag_nodes.py` 的 `answer_grounding_check` 函数之后、`aggregate_answers` 之前插入：

```python
def revise_answer(state: State, llm):
    """P2: critique an un-grounded answer and rewrite it within evidence bounds.

    Reads the current (un-grounded) answer from the last message, the evidence
    from agent_answers, and asks the light LLM (via GroundingCritique schema)
    for a structured critique + an evidence-bounded rewrite. The rewrite is
    appended as the latest message; control returns to answer_grounding_check
    for a re-check. LLM failure (empty revised_answer) falls back to
    ground_answer's passive-disclaimer revised_answer so the node never breaks.
    """
    latest_message = state["messages"][-1] if state.get("messages") else None
    current_answer = str(getattr(latest_message, "content", "") or "").strip()

    # Evidence block from agent_answers (same shape answer_grounding_check builds).
    evidence_docs = []
    for item in (state.get("agent_answers") or []):
        if isinstance(item, dict):
            content = item.get("answer", "") or item.get("content", "") or ""
            score = item.get("score", item.get("evidence_score", None))
            source = item.get("source", "") or item.get("citation", "") or ""
            if not source and score is not None:
                source = f"evidence_score={score}"
            if source:
                evidence_docs.append(Document(page_content=str(content), metadata={"score": float(score) if score else 0.0, "source": str(source)}))
    evidence_text = "\n\n".join(
        f"[证据{i}] {getattr(d, 'page_content', '')}".strip()
        for i, d in enumerate(evidence_docs, start=1)
    ) or "（无结构化证据）"

    original_query = str(state.get("originalQuery", "") or "").strip()
    rounds = int(state.get("grounding_rounds", 0) or 0)
    risk_level = _infer_risk_level(original_query, state.get("risk_level", "normal"))

    # Passive-disclaimer fallback (used if the LLM critique yields no rewrite).
    fallback = ground_answer(
        current_answer,
        evidence_docs,
        question=original_query,
        medical_mode=_looks_like_medical_request(
            original_query,
            conversation_summary=state.get("conversation_summary", ""),
            recent_context=state.get("recent_context", ""),
            topic_focus=state.get("topic_focus", ""),
        ),
        high_risk=_needs_strict_medical_safety(original_query, risk_level),
    )
    fallback_revised = _strip_leading_query_plan_blob(fallback.get("revised_answer", current_answer))

    sys_msg = SystemMessage(content=get_grounding_critique_prompt())
    user_payload = (
        f"用户问题：{original_query}\n"
        f"检索证据：\n{evidence_text[:2000]}\n"
        f"待评审回答：\n{current_answer}"
    )
    parser = _structured_output_llm(llm, GroundingCritique, max_tokens=config.LLM_STRUCTURED_MAX_TOKENS)
    verdict = parser.invoke([sys_msg, HumanMessage(content=user_payload)])

    critique = (getattr(verdict, "critique", "") or "").strip()
    revised = (getattr(verdict, "revised_answer", "") or "").strip()

    if not revised:
        revised = fallback_revised
        if not critique:
            critique = (fallback.get("note", "") or "").strip()

    return {
        "messages": [AIMessage(content=revised)],
        "grounding_critique": critique,
        "grounding_rounds": rounds + 1,
    }
```

- [ ] **Step 3c: 加入 __all__**

在 `project/rag_agent/rag_nodes.py` 末尾的 `__all__`（line 745-757）中，按字母序在 `plan_retrieval_queries` 之后插入 `"revise_answer",`：

```python
__all__ = [
    "aggregate_answers",
    "answer_grounding_check",
    "collect_answer",
    "compress_context",
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

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection.TestReviseAnswer -v`
Expected: PASS（2 个）

- [ ] **Step 5: 提交**

```bash
git add project/rag_agent/rag_nodes.py tests/test_answer_reflection.py
git commit -m "feat(p2): add revise_answer node with critique + evidence-bounded rewrite"
```

---

## Task 6: route_after_grounding 边

**Files:**
- Modify: `project/rag_agent/edges.py`（import 行 line 5 + 末尾追加函数）
- Test: `tests/test_answer_reflection.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_answer_reflection.py` 的 `TestReviseAnswer` 之后插入：

```python
class TestRouteAfterGrounding(unittest.TestCase):
    def test_grounded_routes_to_end(self):
        from project.rag_agent.edges import route_after_grounding
        state = _make_main_state([], grounding_passed=True)
        self.assertEqual(route_after_grounding(state), "__end__")

    def test_not_grounded_with_budget_routes_to_revise(self):
        from project.rag_agent.edges import route_after_grounding
        state = _make_main_state([], grounding_passed=False, grounding_rounds=0)
        self.assertEqual(route_after_grounding(state), "revise_answer")

    def test_not_grounded_budget_exhausted_routes_to_end(self):
        import config
        from project.rag_agent.edges import route_after_grounding
        state = _make_main_state([], grounding_passed=False, grounding_rounds=config.MAX_GROUNDING_ROUNDS)
        self.assertEqual(route_after_grounding(state), "__end__")
```

- [ ] **Step 2: 运行测试验证失败**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection.TestRouteAfterGrounding -v`
Expected: FAIL — `ImportError: cannot import name 'route_after_grounding'`

- [ ] **Step 3a: 改 import**

把 `project/rag_agent/edges.py` line 5 改为：

```python
from config import MAX_ITERATIONS, MAX_TOOL_CALLS, MAX_EVIDENCE_ROUNDS, MAX_GROUNDING_ROUNDS
```

- [ ] **Step 3b: 新增边函数**

在 `project/rag_agent/edges.py` 末尾追加：

```python
def route_after_grounding(state: State) -> Literal["__end__", "revise_answer"]:
    """P2: route after the answer grounding check.

    - grounded (grounding_passed=True) → END
    - not grounded + budget remaining (grounding_rounds < MAX_GROUNDING_ROUNDS) → revise_answer
    - not grounded + budget exhausted → END (passive-disclaimer degrade already appended)
    """
    if bool(state.get("grounding_passed", False)):
        return "__end__"
    rounds = int(state.get("grounding_rounds", 0) or 0)
    if rounds < MAX_GROUNDING_ROUNDS:
        return "revise_answer"
    return "__end__"
```

- [ ] **Step 4: 运行测试验证通过**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection.TestRouteAfterGrounding -v`
Expected: PASS（3 个）

- [ ] **Step 5: 提交**

```bash
git add project/rag_agent/edges.py tests/test_answer_reflection.py
git commit -m "feat(p2): add route_after_grounding edge"
```

---

## Task 7: graph 条件接线

**Files:**
- Modify: `project/rag_agent/graph.py`（import 块 line 12-23 + 接线 line 189-190）
- Test: `tests/test_answer_reflection.py`（仅编译性回归，下个 task 覆盖运行时）

- [ ] **Step 1: 写失败测试**

在 `tests/test_answer_reflection.py` 的 `TestRouteAfterGrounding` 之后插入：

```python
class TestGraphWiring(unittest.TestCase):
    def test_graph_source_references_reflection_wiring(self):
        """graph.py must wire the conditional edge + revise_answer node under ENABLE_ANSWER_REFLECTION."""
        import inspect
        import project.rag_agent.graph as graph_mod
        src = inspect.getsource(graph_mod)
        self.assertIn("revise_answer", src)
        self.assertIn("route_after_grounding", src)
        self.assertIn("ENABLE_ANSWER_REFLECTION", src)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection.TestGraphWiring -v`
Expected: FAIL — `AssertionError: 'revise_answer' not found in ...`（graph.py 尚未引用）

- [ ] **Step 3a: 改 import**

在 `project/rag_agent/graph.py` 的 rag_nodes import 块（line 12-23）按字母序加入 `revise_answer`：

```python
from .rag_nodes import (
    answer_grounding_check,
    collect_answer,
    compress_context,
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

- [ ] **Step 3b: 改接线**

把 `project/rag_agent/graph.py` line 189-190：

```python
    graph_builder.add_edge("grounded_answer_generation", "answer_grounding_check")
    graph_builder.add_edge("answer_grounding_check", END)
```

替换为：

```python
    graph_builder.add_edge("grounded_answer_generation", "answer_grounding_check")
    if config.ENABLE_ANSWER_REFLECTION:
        # P2: answer reflection loop — critique + evidence-bounded rewrite, re-checked.
        # revise_answer is a LIGHT-tier task (critique/rewrite-class, like evaluate_evidence).
        graph_builder.add_node("revise_answer", partial(revise_answer, llm=_light_llm))
        graph_builder.add_conditional_edges(
            "answer_grounding_check",
            route_after_grounding,
            {"__end__": END, "revise_answer": "revise_answer"},
        )
        graph_builder.add_edge("revise_answer", "answer_grounding_check")
    else:
        graph_builder.add_edge("answer_grounding_check", END)
```

`route_after_grounding` 通过 `from .edges import *`（line 9）已可用（edges.py 无 `__all__`，P1 的 `route_after_evidence` 同样靠 `*` 引入）。

- [ ] **Step 4: 编译检查 + 运行测试**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m compileall project/rag_agent/graph.py`
Expected: 无输出（编译通过）

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection.TestGraphWiring -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add project/rag_agent/graph.py tests/test_answer_reflection.py
git commit -m "feat(p2): wire answer_grounding_check -> route_after_grounding -> END|revise_answer"
```

---

## Task 8: 编译图集成测试（回路真的走通）

**Files:**
- Test: `tests/test_answer_reflection.py`

仿 P1 的 `TestCompiledGraphStateHandoff`，编译最小主图段并用真实 `StateGraph` + `graph.invoke()` 跑通回路。

- [ ] **Step 1: 写测试**

在 `tests/test_answer_reflection.py` 的 `TestGraphWiringImport` 之后、`if __name__` 之前插入：

```python
class TestCompiledGroundingLoop(unittest.TestCase):
    """Verify the answer_grounding_check -> route_after_grounding -> revise_answer -> answer_grounding_check
    handoff survives LangGraph's real state machinery (MessagesState append reducer + state fields)."""

    def _build_graph(self, llm):
        from langgraph.graph import StateGraph, START, END
        from project.rag_agent.graph_state import State
        from project.rag_agent.rag_nodes import answer_grounding_check, revise_answer
        from project.rag_agent.edges import route_after_grounding
        from functools import partial

        builder = StateGraph(State)
        builder.add_node("answer_grounding_check", partial(answer_grounding_check, llm=llm))
        builder.add_node("revise_answer", partial(revise_answer, llm=llm))
        sink = {"hit": None}

        def _sink_end(state):
            sink["hit"] = "__end__"
            return {}

        builder.add_node("__end_sink", _sink_end)
        builder.add_edge(START, "answer_grounding_check")
        builder.add_conditional_edges(
            "answer_grounding_check",
            route_after_grounding,
            {"__end__": "__end_sink", "revise_answer": "revise_answer"},
        )
        builder.add_edge("revise_answer", "answer_grounding_check")
        builder.add_edge("__end_sink", END)
        return builder.compile(), sink

    def test_rewrite_succeeds_then_terminates(self):
        """Un-grounded answer -> revise_answer -> re-check grounded -> END.

        ground_answer side effects (in call order):
          1. answer_grounding_check iter1  -> grounded=False (append disclaimer)
          2. revise_answer fallback        -> unused (LLM succeeds)
          3. answer_grounding_check iter2  -> grounded=True (revised==current, no append)
        """
        from project.rag_agent.schemas import GroundingCritique
        ground_results = [
            {"grounded": False, "revised_answer": "超证据【声明】", "note": "low_confidence_guardrail"},
            {"grounded": False, "revised_answer": "fallback", "note": "low"},
            {"grounded": True, "revised_answer": "收窄版回答", "note": "grounded"},
        ]
        parser = MagicMock()
        parser.invoke.return_value = GroundingCritique(critique="超证据", revised_answer="收窄版回答")
        with patch("project.rag_agent.rag_nodes.ground_answer", side_effect=ground_results), \
             patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser):
            graph, sink = self._build_graph(MagicMock())
            state = _make_main_state(
                [AIMessage(content="超证据")],
                agent_answers=[{"answer": "证据", "evidence_score": 0.5, "source": "src", "confidence_bucket": "low"}],
                grounding_evidence_score=0.5,
            )
            graph.invoke(state, {"recursion_limit": 20})
        self.assertEqual(sink["hit"], "__end__")

    def test_budget_exhausted_terminates_at_end(self):
        """Rewrite still not grounded + rounds exhausted -> END (disclaimer degrade)."""
        from project.rag_agent.schemas import GroundingCritique
        ground_result = {"grounded": False, "revised_answer": "答【声明】", "note": "low_confidence_guardrail"}
        parser = MagicMock()
        parser.invoke.return_value = GroundingCritique(critique="仍超证据", revised_answer="重写版")
        with patch("project.rag_agent.rag_nodes.ground_answer", return_value=ground_result), \
             patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser):
            graph, sink = self._build_graph(MagicMock())
            state = _make_main_state(
                [AIMessage(content="超证据")],
                agent_answers=[{"answer": "证据", "evidence_score": 0.5, "source": "src", "confidence_bucket": "low"}],
                grounding_evidence_score=0.5,
            )
            graph.invoke(state, {"recursion_limit": 20})
        self.assertEqual(sink["hit"], "__end__")
```

- [ ] **Step 2: 运行测试验证通过**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection.TestCompiledGroundingLoop -v`
Expected: PASS（2 个）。若失败，检查 `ground_answer` 调用顺序与 MessagesState 追加语义（见"关键代码事实"第 2 条）。

- [ ] **Step 3: 跑整个 test_answer_reflection 文件**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection -v`
Expected: 全部 PASS（Config + State + Schema + GroundingCheck + Revise + Route + Wiring + CompiledLoop）

- [ ] **Step 4: 提交**

```bash
git add tests/test_answer_reflection.py
git commit -m "test(p2): compiled-graph integration for answer reflection loop"
```

---

## Task 9: 回滚用例 + 全量回归

**Files:**
- Test: `tests/test_answer_reflection.py`（追加回滚用例）

- [ ] **Step 1: 写回滚（硬边）测试**

在 `tests/test_answer_reflection.py` 的 `TestCompiledGroundingLoop` 之后、`if __name__` 之前插入：

```python
class TestReflectionDisabledRollback(unittest.TestCase):
    """When ENABLE_ANSWER_REFLECTION=False the graph uses the hard edge
    answer_grounding_check -> END (no revise_answer). This proves that topology
    terminates for both grounded and un-grounded answers (no infinite loop)."""

    def _build_hard_edge_graph(self, llm):
        from langgraph.graph import StateGraph, START, END
        from project.rag_agent.graph_state import State
        from project.rag_agent.rag_nodes import answer_grounding_check
        from functools import partial

        builder = StateGraph(State)
        builder.add_node("answer_grounding_check", partial(answer_grounding_check, llm=llm))
        sink = {"hit": None}

        def _sink(state):
            sink["hit"] = "end"
            return {}

        builder.add_node("_sink", _sink)
        builder.add_edge(START, "answer_grounding_check")
        builder.add_edge("answer_grounding_check", "_sink")  # rollback hard edge
        builder.add_edge("_sink", END)
        return builder.compile(), sink

    def test_hard_edge_terminates_un_grounded(self):
        with patch("project.rag_agent.rag_nodes.ground_answer",
                   return_value={"grounded": False, "revised_answer": "答【声明】", "note": "low"}):
            graph, sink = self._build_hard_edge_graph(MagicMock())
            state = _make_main_state(
                [AIMessage(content="答")],
                agent_answers=[{"answer": "证据", "evidence_score": 0.5, "source": "src", "confidence_bucket": "low"}],
                grounding_evidence_score=0.5,
            )
            graph.invoke(state, {"recursion_limit": 10})
        self.assertEqual(sink["hit"], "end")
```

- [ ] **Step 2: 运行该测试**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection.TestReflectionDisabledRollback -v`
Expected: PASS

- [ ] **Step 3: 跑 P2 全部测试**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection -v`
Expected: 全部 PASS

- [ ] **Step 4: P1 回归（确保 P2 不碰子图）**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_agentic_retrieval -v`
Expected: 20 个全 PASS（P2 不改动 AgentState / evaluate_evidence / route_after_evidence）

- [ ] **Step 5: 编译检查**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m compileall project tests`
Expected: 无输出

- [ ] **Step 6: 全量 unittest 回归**

Run: `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest discover -s tests -v`
Expected: P2 新增用例全 PASS；既有失败用例与 main 基线（commit eda9c90）一致（已知：中文断言 mojibake、缺 LLM token 的环境性失败），**P2 不引入新回归**。若出现新的 grounding 相关失败（尤其 `test_api_app` 里触达 grounding 的），需排查 P2 改动。

- [ ] **Step 7: 提交**

```bash
git add tests/test_answer_reflection.py
git commit -m "test(p2): rollback hard-edge topology + full regression"
```

---

## 验收对照（spec §10）

| 验收项 | 任务 |
|---|---|
| 1. revise_answer + route_after_grounding + 3 个 state 字段 | Task 2, 5, 6 |
| 2. `grounded` 字段被用于分支 | Task 4 (写) + Task 6 (读) |
| 3. answer_grounding_check 写出 grounding_passed，快路径+正常路径 | Task 4 |
| 4. critique LLM 失败降级到 ground_answer.revised_answer | Task 5 (test_empty_llm_result_falls_back) |
| 5. MAX_GROUNDING_ROUNDS 上限生效；耗尽走 END + 免责声明 | Task 6 + Task 8 (test_budget_exhausted) |
| 6. ENABLE_ANSWER_REFLECTION 开关可回滚硬边 | Task 7 (graph 分支) + Task 9 (硬边测试) |
| 7. 全量 unittest 不回归；P1 的 20 个测试不受影响 | Task 9 |
| 8. 至少一个编译图集成用例证明回路走通 | Task 8 (2 个) |
