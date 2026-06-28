# P1: Agentic 检索循环 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把检索证据充分性判断从工具内死规则提升为 graph 层 LLM 反思节点 `evaluate_evidence`，驱动"改写 query → 重检索"闭环并带终止守卫。

**Architecture:** 在现有 agent 子图的 `tools` 与 `should_compress_context` 之间插入 `evaluate_evidence` 节点：规则 `check_sufficiency` 判充分则跳过 LLM（快路径），判不足则用轻量 LLM + 复用 `EvidenceSufficiency` schema 反思并产出 `refined_query`，由新边 `route_after_evidence` 决定回 orchestrator 重检索或走 `fallback_response`。新增 `MAX_EVIDENCE_ROUNDS` 上限与无进展检测守卫。`ENABLE_AGENTIC_RETRIEVAL` 开关可回滚到旧拓扑。

**Tech Stack:** Python, LangGraph (StateGraph/ToolNode/Command), LangChain messages, Pydantic, unittest。

**Reference spec:** `docs/superpowers/specs/2026-06-28-agentic-retrieval-loop-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `project/config.py` | Modify (~line 82, ~250) | 新增 `MAX_EVIDENCE_ROUNDS`、`ENABLE_AGENTIC_RETRIEVAL` 配置 |
| `project/rag_agent/graph_state.py` | Modify (`AgentState`) | 新增 4 个证据反思状态字段 |
| `project/rag_agent/rag_nodes.py` | Modify | 新增 `evaluate_evidence` 节点；改 `orchestrator` 注入 refined_query |
| `project/rag_agent/edges.py` | Modify | 新增 `route_after_evidence` 边 + `_has_repeated_refined_query` 守卫 |
| `project/rag_agent/graph.py` | Modify (`create_agent_graph`) | 条件接线：插入 evaluate_evidence，重连 tools 边 |
| `project/rag_agent/prompts.py` | Modify | 新增 `get_evidence_sufficiency_prompt` |
| `tests/test_agentic_retrieval.py` | Create | evaluate_evidence + route_after_evidence 单元测试 |

既有约定参考：
- 节点签名：`def node(state, llm)` 返回 state-delta dict。
- 结构化输出：`from .structured_output import _structured_output_llm`，调用 `parser = _structured_output_llm(llm, Schema); result = parser.invoke(messages)`。失败返回 schema 默认实例，不抛异常。
- `EvidenceSufficiency` schema 已存在于 `project/rag_agent/schemas.py:175`，字段 `is_sufficient: bool / reason: str / retry_query: str`。
- `check_sufficiency(query, docs)` 已存在于 `project/rag_agent/tools.py:190`，返回 `{"is_sufficient", "reason", "retry_query"}`。

---

## Task 1: 配置开关与上限

**Files:**
- Modify: `project/config.py:82` (RAG_RETRY_LIMIT 附近) 和 `project/config.py:250` (MAX_ITERATIONS 附近)

- [ ] **Step 1: 新增配置项**

在 `project/config.py` 中 `RAG_RETRY_LIMIT` 行下方添加：

```python
RAG_RETRY_LIMIT = int(os.environ.get("RAG_RETRY_LIMIT", "1"))

# P1: agentic retrieval — LLM evidence-sufficiency reflection loop
MAX_EVIDENCE_ROUNDS = int(os.environ.get("MAX_EVIDENCE_ROUNDS", "2"))
ENABLE_AGENTIC_RETRIEVAL = os.environ.get("ENABLE_AGENTIC_RETRIEVAL", "true").lower() == "true"
```

（放在 RAG_RETRY_LIMIT 附近，语义聚合。`MAX_ITERATIONS`/`MAX_TOOL_CALLS` 在第 250 行区，无需改。）

- [ ] **Step 2: 语法校验**

Run: `.\venv\Scripts\python.exe -m compileall project/config.py`
Expected: 无输出（编译通过）

- [ ] **Step 3: 验证配置可读**

Run: `.\venv\Scripts\python.exe -c "from project import config; print(config.MAX_EVIDENCE_ROUNDS, config.ENABLE_AGENTIC_RETRIEVAL)"`
Expected: `2 True`

- [ ] **Step 4: Commit**

```bash
git add project/config.py
git commit -m "feat: add MAX_EVIDENCE_ROUNDS and ENABLE_AGENTIC_RETRIEVAL config"
```

---

## Task 2: AgentState 新增反思状态字段

**Files:**
- Modify: `project/rag_agent/graph_state.py:67-80` (`AgentState`)
- Test: `tests/test_agentic_retrieval.py` (新建)

- [ ] **Step 1: 先写字段可用的失败测试**

创建 `tests/test_agentic_retrieval.py`：

```python
"""Tests for P1 agentic retrieval loop: evaluate_evidence + route_after_evidence."""

import unittest
from project.rag_agent.graph_state import AgentState


class TestAgentStateFields(unittest.TestCase):
    def test_evidence_reflection_fields_exist(self):
        """AgentState must carry the four evidence-reflection fields."""
        defaults = AgentState.__annotations__
        for field in ("evidence_rounds", "evidence_critique", "last_refined_query", "refined_queries"):
            self.assertIn(field, defaults, f"AgentState missing field: {field}")

    def test_refined_queries_is_accumulating(self):
        """refined_queries accumulates across rounds (operator.add reducer)."""
        from project.rag_agent.graph_state import AgentState
        # reducer is annotated via typing.Annotated; check the metadata carries operator.add
        import typing
        hints = typing.get_type_hints(AgentState, include_extras=True)
        meta = typing.get_args(hints["refined_queries"])
        import operator
        self.assertIn(operator.add, meta, "refined_queries must use operator.add reducer")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.\venv\Scripts\python.exe -m unittest tests.test_agentic_retrieval -v`
Expected: FAIL（字段不存在 / reducer 不匹配）

- [ ] **Step 3: 实现字段**

修改 `project/rag_agent/graph_state.py` 的 `AgentState`，在 `iteration_count` 行下方添加：

```python
class AgentState(MessagesState):
    """State for individual agent subgraph"""
    question: str = ""
    question_index: int = 0
    query_plan: List[str] = []
    context_summary: str = ""
    recent_context: str = ""
    topic_focus: str = ""
    retrieval_keys: Annotated[Set[str], set_union] = set()
    final_answer: str = ""
    agent_answers: List[dict] = []
    tool_call_count: Annotated[int, operator.add] = 0
    iteration_count: Annotated[int, operator.add] = 0
    # P1: agentic retrieval — evidence-sufficiency reflection loop
    evidence_rounds: int = 0
    evidence_critique: str = ""
    last_refined_query: str = ""
    refined_queries: Annotated[List[str], operator.add] = []
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.\venv\Scripts\python.exe -m unittest tests.test_agentic_retrieval -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add project/rag_agent/graph_state.py tests/test_agentic_retrieval.py
git commit -m "feat: add evidence-reflection fields to AgentState"
```

---

## Task 3: 证据充分性反思 prompt

**Files:**
- Modify: `project/rag_agent/prompts.py` (末尾追加)

- [ ] **Step 1: 查看现有 prompt 函数风格**

Run: `.\venv\Scripts\python.exe -c "from project.rag_agent.prompts import get_orchestrator_prompt; print(type(get_orchestrator_prompt()))"`
Expected: `<class 'str'>`（确认是返回 str 的无参/轻参函数）

- [ ] **Step 2: 追加 prompt 函数**

在 `project/rag_agent/prompts.py` 末尾添加：

```python
def get_evidence_sufficiency_prompt() -> str:
    """System prompt for the evaluate_evidence reflection node (P1).

    The LLM judges whether the retrieved evidence can answer the user's
    question and, if not, produces one improved retry query. Output must be
    strict JSON matching the EvidenceSufficiency schema:
    {"is_sufficient": bool, "reason": str, "retry_query": str}.
    """
    return (
        "你是一名严谨的医疗知识库证据评审员。判断当前检索到的证据是否足以回答用户问题。\n\n"
        "判定标准：\n"
        "- is_sufficient=true：证据直接覆盖问题核心，可支撑回答。\n"
        "- is_sufficient=false：证据偏离问题、只覆盖部分、分数过低或噪声过多。\n"
        "当判为 false 时，retry_query 必须给出一个**与原检索式不同**的更优检索式（同义词、补充关键病种/药物、换表述），用于下一轮检索；判为 true 时 retry_query 留空。\n\n"
        "严格输出 JSON，不要输出多余文字：\n"
        '{"is_sufficient": true/false, "reason": "简短原因", "retry_query": "改进检索式或空"}'
    )
```

- [ ] **Step 3: 语法校验与导入检查**

Run: `.\venv\Scripts\python.exe -c "from project.rag_agent.prompts import get_evidence_sufficiency_prompt; print(len(get_evidence_sufficiency_prompt()) > 50)"`
Expected: `True`

- [ ] **Step 4: Commit**

```bash
git add project/rag_agent/prompts.py
git commit -m "feat: add evidence sufficiency reflection prompt"
```

---

## Task 4: evaluate_evidence 节点

**Files:**
- Modify: `project/rag_agent/rag_nodes.py` (新增节点函数 + 导入)
- Test: `tests/test_agentic_retrieval.py` (追加)

- [ ] **Step 1: 先写失败测试**

在 `tests/test_agentic_retrieval.py` 顶部 import 区追加，并在文件末尾追加测试类：

```python
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.documents import Document
from project.rag_agent.rag_nodes import evaluate_evidence
```

```python
def _make_state(messages, **extra):
    base = {
        "messages": messages,
        "question": "高血压合并痛风吃什么药安全",
        "query_plan": [],
        "evidence_rounds": 0,
        "evidence_critique": "",
        "last_refined_query": "",
        "refined_queries": [],
    }
    base.update(extra)
    return base


class TestEvaluateEvidenceFastPath(unittest.TestCase):
    def test_sufficient_evidence_skips_llm(self):
        """Rule path returns sufficient=True WITHOUT calling the LLM."""
        tool_msg = ToolMessage(
            content="[DOC1] 高血压合并痛风患者应避免使用噻嗪类利尿剂，因可能加重高尿酸血症。",
            tool_call_id="1",
        )
        state = _make_state([tool_msg])

        llm = MagicMock()
        with patch("project.rag_agent.rag_nodes.check_sufficiency") as mock_check:
            mock_check.return_value = {"is_sufficient": True, "reason": "direct_evidence", "retry_query": ""}
            with patch("project.rag_agent.rag_nodes._structured_output_llm") as mock_so:
                result = evaluate_evidence(state, llm)
                mock_so.assert_not_called()  # fast path skips LLM

        self.assertEqual(result["evidence_critique"], "direct_evidence")
        self.assertEqual(result["evidence_rounds"], 0)  # no round counted on fast-path success


class TestEvaluateEvidenceReflection(unittest.TestCase):
    def test_insufficient_triggers_llm_and_records_refined_query(self):
        """Rule says insufficient → LLM reflects → refined_query recorded, round counted."""
        tool_msg = ToolMessage(content="[DOC1] 高血压常规用药包括ACEI。", tool_call_id="1")
        state = _make_state([tool_msg])

        llm = MagicMock()
        parser = MagicMock()
        from project.rag_agent.schemas import EvidenceSufficiency
        parser.invoke.return_value = EvidenceSufficiency(
            is_sufficient=False,
            reason="证据只覆盖高血压用药，未涉及与痛风的交互",
            retry_query="高血压 合并痛风 药物 相互作用 禁忌",
        )

        with patch("project.rag_agent.rag_nodes.check_sufficiency") as mock_check, \
             patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser) as mock_so:
            mock_check.return_value = {"is_sufficient": False, "reason": "weak", "retry_query": "x"}
            result = evaluate_evidence(state, llm)

            mock_so.assert_called_once()
        self.assertFalse(result["_evidence_sufficient"])
        self.assertEqual(result["evidence_rounds"], 1)
        self.assertEqual(result["last_refined_query"], "高血压 合并痛风 药物 相互作用 禁忌")
        self.assertEqual(result["refined_queries"], ["高血压 合并痛风 药物 相互作用 禁忌"])
        self.assertIn("痛风", result["evidence_critique"])


class TestEvaluateEvidenceLLMFailureFallback(unittest.TestCase):
    def test_llm_failure_falls_back_to_rule_retry_query(self):
        """If the LLM reflection returns insufficient with empty retry_query (parse failure default), fall back to rule's retry_query."""
        tool_msg = ToolMessage(content="[DOC1] 略", tool_call_id="1")
        state = _make_state([tool_msg])

        llm = MagicMock()
        parser = MagicMock()
        from project.rag_agent.schemas import EvidenceSufficiency
        # Simulate _structured_output_llm default: is_sufficient=False, retry_query=""
        parser.invoke.return_value = EvidenceSufficiency(is_sufficient=False, reason="", retry_query="")

        with patch("project.rag_agent.rag_nodes.check_sufficiency") as mock_check, \
             patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser):
            mock_check.return_value = {"is_sufficient": False, "reason": "weak", "retry_query": "高血压 痛风 医学资料"}
            result = evaluate_evidence(state, llm)

        # fallback to rule retry_query when LLM gave empty refined_query
        self.assertEqual(result["last_refined_query"], "高血压 痛风 医学资料")
        self.assertFalse(result["_evidence_sufficient"])
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.\venv\Scripts\python.exe -m unittest tests.test_agentic_retrieval.TestEvaluateEvidenceFastPath tests.test_agentic_retrieval.TestEvaluateEvidenceReflection tests.test_agentic_retrieval.TestEvaluateEvidenceLLMFailureFallback -v`
Expected: FAIL（`evaluate_evidence` 未定义 / `check_sufficiency` 未导入到 rag_nodes）

- [ ] **Step 3: 添加导入**

在 `project/rag_agent/rag_nodes.py` 的 `.schemas` 导入块中追加 `EvidenceSufficiency`，并在 `.tools` 相关导入旁补 `check_sufficiency`，并导入结构化输出与 prompt：

修改第 18-22 行的 schemas 导入块为：

```python
from .schemas import (
    QueryAnalysis,
    RetrievalQueryPlan,
    GroundedAnswerCheck,
    EvidenceSufficiency,
)
```

在文件已有 `from .prompts import (...)` 块中追加 `get_evidence_sufficiency_prompt`（在该块的圆括号内加一行）。

在文件已有 tools 导入附近（搜 `from .tools import` 或 `from .tools` 确认 `check_sufficiency` 是否已被导入；若未导入则添加）：

```python
from .tools import check_sufficiency
from .structured_output import _structured_output_llm
```

> 实现者注意：先用 `grep -n "check_sufficiency\|_structured_output_llm\|from .tools" project/rag_agent/rag_nodes.py` 确认现有导入行，避免重复 import。若 `check_sufficiency` 已在某行导入则不重复添加；`_structured_output_llm` 同理。

- [ ] **Step 4: 实现 evaluate_evidence 节点**

在 `project/rag_agent/rag_nodes.py` 的 `orchestrator` 函数之前（或 `plan_retrieval_queries` 之后）添加：

```python
def _latest_tool_message(state: AgentState):
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, ToolMessage):
            return msg
    return None


def evaluate_evidence(state: AgentState, llm):
    """P1: reflect on retrieved evidence; decide sufficiency and refine query.

    Fast path: rule-based check_sufficiency says sufficient → skip LLM.
    Reflection path: rule says insufficient → light LLM judges via
    EvidenceSufficiency schema and produces a refined retry_query.
    Failure path: LLM parse/empty → fall back to rule's retry_query.
    """
    tool_msg = _latest_tool_message(state)
    evidence_text = str(getattr(tool_msg, "content", "") or "")
    question = str(state.get("question") or "").strip()
    query_plan = [str(q).strip() for q in (state.get("query_plan") or []) if str(q).strip()]

    # Build a pseudo doc list for the rule check from the latest tool result text.
    # check_sufficiency keys off relevance_grade/score metadata; with raw text we
    # pass a single doc and rely on the empty/weak fallback path, which is the
    # case we want the LLM to overrule.
    pseudo_docs = [Document(page_content=evidence_text, metadata={"score": 0.0, "relevance_grade": "low"})] if evidence_text else []

    rule = check_sufficiency(question or (query_plan[0] if query_plan else ""), pseudo_docs)

    rounds = int(state.get("evidence_rounds", 0) or 0)

    # Fast path: rule says sufficient — skip LLM entirely.
    if rule.get("is_sufficient"):
        return {
            "evidence_critique": rule.get("reason", ""),
            "last_refined_query": "",
            "evidence_rounds": rounds,
            "_evidence_sufficient": True,
        }

    # Reflection path.
    import config
    sys_msg = SystemMessage(content=get_evidence_sufficiency_prompt())
    user_payload = (
        f"用户问题：{question}\n"
        f"已有检索式：{query_plan}\n"
        f"最新检索证据：\n{evidence_text[:2000]}"
    )
    parser = _structured_output_llm(llm, EvidenceSufficiency, max_tokens=config.LLM_STRUCTURED_MAX_TOKENS)
    verdict = parser.invoke([sys_msg, HumanMessage(content=user_payload)])

    refined = (verdict.retry_query or "").strip()
    critique = (verdict.reason or rule.get("reason", "")).strip()

    # Failure fallback: LLM gave insufficient but no usable refined query.
    if not verdict.is_sufficient and not refined:
        refined = (rule.get("retry_query") or "").strip()

    is_sufficient = bool(verdict.is_sufficient)

    delta = {
        "evidence_critique": critique,
        "evidence_rounds": rounds + 1 if not is_sufficient else rounds,
        "_evidence_sufficient": is_sufficient,
    }
    if refined and not is_sufficient:
        delta["last_refined_query"] = refined
        delta["refined_queries"] = [refined]
    else:
        delta["last_refined_query"] = ""
    return delta
```

> 说明：`_evidence_sufficient` 是临时内部字段，仅供 `route_after_evidence` 在同一轮读取；不写入 AgentState 声明（LangGraph 允许 state-delta 携带未声明键，会在该次 invoke 内可见但不会持久化为 schema 字段——若该版本 LangGraph 严格校验，Task 5 会改用 state 已声明的派生方式，见 Task 5 Step 3 备注）。

- [ ] **Step 5: 运行测试确认通过**

Run: `.\venv\Scripts\python.exe -m unittest tests.test_agentic_retrieval.TestEvaluateEvidenceFastPath tests.test_agentic_retrieval.TestEvaluateEvidenceReflection tests.test_agentic_retrieval.TestEvaluateEvidenceLLMFailureFallback -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add project/rag_agent/rag_nodes.py tests/test_agentic_retrieval.py
git commit -m "feat: add evaluate_evidence reflection node"
```

---

## Task 5: route_after_evidence 边与终止守卫

**Files:**
- Modify: `project/rag_agent/edges.py` (新增边函数 + 守卫)
- Test: `tests/test_agentic_retrieval.py` (追加)

- [ ] **Step 1: 先写失败测试**

在 `tests/test_agentic_retrieval.py` 顶部 import 区追加：

```python
from project.rag_agent.edges import route_after_evidence
```

末尾追加测试类：

```python
class TestRouteAfterEvidence(unittest.TestCase):
    def test_sufficient_routes_to_compress(self):
        state = _make_state([], _evidence_sufficient=True, evidence_rounds=0)
        self.assertEqual(route_after_evidence(state), "should_compress_context")

    def test_insufficient_with_budget_routes_to_compress(self):
        """Insufficient but under round limit and novel query → loop back via compress."""
        state = _make_state(
            [],
            _evidence_sufficient=False,
            evidence_rounds=1,
            last_refined_query="新检索式A",
            refined_queries=["新检索式A"],
        )
        self.assertEqual(route_after_evidence(state), "should_compress_context")

    def test_round_limit_reached_routes_to_fallback(self):
        import config
        state = _make_state(
            [],
            _evidence_sufficient=False,
            evidence_rounds=config.MAX_EVIDENCE_ROUNDS,
            last_refined_query="q",
            refined_queries=["q"],
        )
        self.assertEqual(route_after_evidence(state), "fallback_response")

    def test_repeated_refined_query_routes_to_fallback(self):
        """Refined query repeats a prior one (no progress) → fallback."""
        state = _make_state(
            [],
            _evidence_sufficient=False,
            evidence_rounds=1,
            last_refined_query="重复检索式",
            refined_queries=["别的", "重复检索式"],
        )
        self.assertEqual(route_after_evidence(state), "fallback_response")

    def test_no_progress_when_refined_query_empty_routes_to_fallback(self):
        state = _make_state(
            [],
            _evidence_sufficient=False,
            evidence_rounds=1,
            last_refined_query="",
            refined_queries=[],
        )
        self.assertEqual(route_after_evidence(state), "fallback_response")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.\venv\Scripts\python.exe -m unittest tests.test_agentic_retrieval.TestRouteAfterEvidence -v`
Expected: FAIL（`route_after_evidence` 未定义）

- [ ] **Step 3: 实现 route_after_evidence 与守卫**

在 `project/rag_agent/edges.py` 的 `_has_repeated_search_query` 函数之后添加：

```python
def _has_repeated_refined_query(state: AgentState) -> bool:
    """No-progress guard: the latest refined query already appeared earlier."""
    refined = [str(q or "").strip().lower() for q in (state.get("refined_queries") or []) if str(q or "").strip()]
    if len(refined) < 2:
        return False
    return refined[-1] in refined[:-1]


def route_after_evidence(state: AgentState) -> Literal["should_compress_context", "fallback_response"]:
    """P1: route after evidence reflection.

    - sufficient → should_compress_context (then orchestrator → collect_answer)
    - insufficient + budget + progress → should_compress_context (re-search)
    - insufficient + exhausted/no-progress → fallback_response
    """
    import config
    is_sufficient = bool(state.get("_evidence_sufficient", False))
    if is_sufficient:
        return "should_compress_context"

    rounds = int(state.get("evidence_rounds", 0) or 0)
    last_refined = str(state.get("last_refined_query", "") or "").strip()

    # Termination guards.
    if rounds >= config.MAX_EVIDENCE_ROUNDS:
        return "fallback_response"
    if not last_refined:
        return "fallback_response"
    if _has_repeated_refined_query(state):
        return "fallback_response"
    if _has_repeated_no_evidence(state):
        return "fallback_response"

    return "should_compress_context"
```

> 备注：`_evidence_sufficient` 由 `evaluate_evidence` 在同一轮写入 state-delta，`route_after_evidence` 在紧随其后的边读取。LangGraph 在单次 invoke 内把节点返回的 delta 合并入 state 后再调用边函数，因此该临时键在边内可见。若实测发现严格 schema 校验丢弃了未声明键，则改方案：让 `evaluate_evidence` 不返回 `_evidence_sufficient`，而由 `route_after_evidence` 根据 `last_refined_query` 是否为空 + `evidence_critique` 是否为成功标记来推断——但优先用临时键方案，更清晰。

- [ ] **Step 4: 运行测试确认通过**

Run: `.\venv\Scripts\python.exe -m unittest tests.test_agentic_retrieval.TestRouteAfterEvidence -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add project/rag_agent/edges.py tests/test_agentic_retrieval.py
git commit -m "feat: add route_after_evidence edge with termination guards"
```

---

## Task 6: orchestrator 注入 refined_query

**Files:**
- Modify: `project/rag_agent/rag_nodes.py:291` (`orchestrator`)

- [ ] **Step 1: 先写失败测试**

在 `tests/test_agentic_retrieval.py` 末尾追加：

```python
class TestOrchestratorRefinedQueryInjection(unittest.TestCase):
    def test_refined_query_is_injected_as_hint(self):
        """When last_refined_query is set, orchestrator appends a re-search hint."""
        from langchain_core.messages import HumanMessage
        from project.rag_agent.rag_nodes import orchestrator

        state = {
            "question": "高血压合并痛风吃什么药安全",
            "query_plan": ["高血压 痛风"],
            "last_refined_query": "高血压 合并痛风 药物 相互作用 禁忌",
            "evidence_critique": "证据未涉及与痛风的交互",
            "messages": [],
            "context_summary": "",
            "recent_context": "",
            "topic_focus": "",
            "user_memories": "",
        }

        llm_with_tools = MagicMock()
        response = MagicMock()
        response.tool_calls = [{"name": "search_child_chunks", "args": {"query": "高血压 合并痛风 药物 相互作用 禁忌"}, "id": "1"}]
        response.content = ""
        llm_with_tools.invoke.return_value = response

        result = orchestrator(state, llm_with_tools)

        # The injected hint should mention the critique and refined query.
        invoked_messages = llm_with_tools.invoke.call_args[0][0]
        joined = "\n".join(str(getattr(m, "content", "")) for m in invoked_messages)
        self.assertIn("高血压 合并痛风 药物 相互作用 禁忌", joined)
        self.assertIn("证据未涉及与痛风的交互", joined)

    def test_no_injection_when_refined_query_absent(self):
        from project.rag_agent.rag_nodes import orchestrator

        state = {
            "question": "普通问题",
            "query_plan": [],
            "last_refined_query": "",
            "messages": [],
            "context_summary": "",
            "recent_context": "",
            "topic_focus": "",
            "user_memories": "",
        }
        llm_with_tools = MagicMock()
        response = MagicMock()
        response.tool_calls = []
        response.content = "answer"
        llm_with_tools.invoke.return_value = response

        orchestrator(state, llm_with_tools)
        joined = "\n".join(str(getattr(m, "content", "")) for m in llm_with_tools.invoke.call_args[0][0])
        self.assertNotIn("上一次检索证据不足", joined)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.\venv\Scripts\python.exe -m unittest tests.test_agentic_retrieval.TestOrchestratorRefinedQueryInjection -v`
Expected: FAIL（注入逻辑尚未实现）

- [ ] **Step 3: 实现注入**

在 `project/rag_agent/rag_nodes.py` 的 `orchestrator` 函数中，定位到构造 `base_messages` 的两处分支（`if not state.get("messages"):` 块和其后的复用块）。在两处构造消息列表之后、`llm_with_tools.invoke(...)` 之前，注入 refined_query 提示。

在 `if not state.get("messages"):` 分支内，把 `retrieval_hint` 注入逻辑之后、`response = llm_with_tools.invoke(base_messages)` 之前，插入：

```python
        # P1: inject refined-query hint after evidence reflection found evidence insufficient.
        refined_query = str(state.get("last_refined_query", "") or "").strip()
        if refined_query:
            critique = str(state.get("evidence_critique", "") or "").strip()
            refined_hint = (
                f"上一次检索证据不足，原因：{critique}。"
                f"请用以下检索式重新调用 search_child_chunks，不要重复之前的查询：{refined_query}"
            )
            base_messages.append(HumanMessage(content=refined_hint))
```

在复用分支（`response = llm_with_tools.invoke([sys_msg] + ... + state["messages"])` 那一行）之前，构造消息列表时同样追加 refined hint。将该行改为先组装再注入：

```python
    reuse_messages = [sys_msg] + summary_injection + recent_context_injection + topic_focus_injection + user_memories_injection + query_plan_injection + state["messages"]
    refined_query = str(state.get("last_refined_query", "") or "").strip()
    if refined_query:
        critique = str(state.get("evidence_critique", "") or "").strip()
        reuse_messages.append(HumanMessage(
            content=f"上一次检索证据不足，原因：{critique}。请用以下检索式重新调用 search_child_chunks，不要重复之前的查询：{refined_query}"
        ))
    response = llm_with_tools.invoke(reuse_messages)
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {"messages": [response], "tool_call_count": len(tool_calls) if tool_calls else 0, "iteration_count": 1, "last_refined_query": ""}
```

> 注意第二处返回时把 `last_refined_query` 清空，避免同一条 refined_query 被反复注入。第一处分支也应在 return 中清空：

把第一处分支的 `return {"messages": [human_msg, response], "tool_call_count": len(response.tool_calls or []), "iteration_count": 1}` 改为追加 `"last_refined_query": ""`。

- [ ] **Step 4: 运行测试确认通过**

Run: `.\venv\Scripts\python.exe -m unittest tests.test_agentic_retrieval.TestOrchestratorRefinedQueryInjection -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add project/rag_agent/rag_nodes.py tests/test_agentic_retrieval.py
git commit -m "feat: inject refined-query hint into orchestrator after evidence reflection"
```

---

## Task 7: graph 条件接线（含开关回滚）

**Files:**
- Modify: `project/rag_agent/graph.py:56-71` (`create_agent_graph` 子图构建段)

- [ ] **Step 1: 阅读当前接线**

Run: `.\venv\Scripts\python.exe -c "import inspect, project.rag_agent.graph as g; print(inspect.getsource(g.create_agent_graph)[:1500])"`
确认当前 `tools -> should_compress_context` 边与节点注册行。

- [ ] **Step 2: 改写子图接线**

在 `project/rag_agent/graph.py` 的 `create_agent_graph` 中，找到子图构建段（`agent_builder.add_node(...)` 与 `agent_builder.add_edge(...)` 区）。

在节点注册区追加 `evaluate_evidence` 节点（紧邻 `agent_builder.add_node("should_compress_context", should_compress_context)` 之后或之前）：

```python
    if config.ENABLE_AGENTIC_RETRIEVAL:
        agent_builder.add_node("evaluate_evidence", partial(evaluate_evidence, llm=_light_llm))
```

> 用 `_light_llm`（反思是轻任务，与意图分类同档）。确认 `_light_llm` 已在该函数作用域内（前文 `_light_llm = llm_router.get_llm("light") if has_tiers else llm`，见 graph.py:43-46）。

然后改接线。原始：

```python
    agent_builder.add_edge("tools", "should_compress_context")
```

改为条件接线：

```python
    if config.ENABLE_AGENTIC_RETRIEVAL:
        agent_builder.add_edge("tools", "evaluate_evidence")
        agent_builder.add_conditional_edges(
            "evaluate_evidence",
            route_after_evidence,
            {"should_compress_context": "should_compress_context", "fallback_response": "fallback_response"},
        )
    else:
        agent_builder.add_edge("tools", "should_compress_context")
```

在文件顶部 import 区确认引入（若缺失则添加）：

```python
from .rag_nodes import orchestrator, compress_context, fallback_response, should_compress_context, collect_answer, evaluate_evidence, plan_retrieval_queries
from .edges import route_after_orchestrator_call, route_after_evidence
import config
```

> 实现者注意：先 `grep -n "from .rag_nodes import\|from .edges import\|^import config\|from .* import config" project/rag_agent/graph.py` 看现有 import 行，按最小改动合并，不要重复导入。

- [ ] **Step 3: 语法校验**

Run: `.\venv\Scripts\python.exe -m compileall project/rag_agent/graph.py`
Expected: 无输出

- [ ] **Step 4: 图可编译校验（开关开）**

Run:
```bash
.\venv\Scripts\python.exe -c "import os; os.environ['ENABLE_AGENTIC_RETRIEVAL']='true'; from project.rag_agent.graph import create_agent_graph; from unittest.mock import MagicMock; g=create_agent_graph(MagicMock(), []); print('compiled OK', 'evaluate_evidence' in [n for n,_ in getattr(g,'nodes',{}).items()] if hasattr(g,'nodes') else 'n/a')"
```
Expected: `compiled OK ...`（不抛异常）

- [ ] **Step 5: 图可编译校验（开关关，回滚拓扑）**

Run:
```bash
.\venv\Scripts\python.exe -c "import os; os.environ['ENABLE_AGENTIC_RETRIEVAL']='false'; import importlib, project.config as c; importlib.reload(c); from project.rag_agent.graph import create_agent_graph; from unittest.mock import MagicMock; g=create_agent_graph(MagicMock(), []); print('compiled OK (rollback)')"
```
Expected: `compiled OK (rollback)`（不抛异常，回退旧拓扑）

- [ ] **Step 6: Commit**

```bash
git add project/rag_agent/graph.py
git commit -m "feat: wire evaluate_evidence into agent subgraph with rollback switch"
```

---

## Task 8: 集成测试 — 两轮检索闭环

**Files:**
- Modify: `tests/test_agentic_retrieval.py` (追加集成测试)

- [ ] **Step 1: 先写集成测试**

末尾追加（在 `if __name__` 之前）：

```python
class TestAgenticRetrievalLoopIntegration(unittest.TestCase):
    """End-to-end-ish: weak first retrieval → reflection → refined re-search → grounded answer."""

    def test_weak_first_then_refined_second_completes(self):
        from unittest.mock import patch, MagicMock
        from langchain_core.messages import AIMessage, ToolMessage
        from project.rag_agent.rag_nodes import evaluate_evidence, orchestrator
        from project.rag_agent.edges import route_after_evidence

        # Round 1: weak evidence in tool result.
        state1 = _make_state(
            [ToolMessage(content="[DOC1] 高血压常规用药ACEI。", tool_call_id="1")],
            evidence_rounds=0,
        )
        rule1 = {"is_sufficient": False, "reason": "weak", "retry_query": "高血压 痛风 医学资料"}
        from project.rag_agent.schemas import EvidenceSufficiency
        parser = MagicMock()
        parser.invoke.return_value = EvidenceSufficiency(
            is_sufficient=False,
            reason="未覆盖痛风交互",
            retry_query="高血压 合并痛风 药物 相互作用 禁忌",
        )
        with patch("project.rag_agent.rag_nodes.check_sufficiency", return_value=rule1), \
             patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser):
            delta1 = evaluate_evidence(state1, MagicMock())

        state1.update(delta1)
        # Insufficient + novel query + under budget → loop back via compress.
        self.assertEqual(route_after_evidence(state1), "should_compress_context")

        # Round 2: refined retrieval yields strong evidence.
        state2 = dict(state1)
        state2["messages"] = [ToolMessage(
            content="[DOC] 高血压合并痛风者禁用噻嗪类利尿剂，首选CCB；避免非甾体抗炎药。",
            tool_call_id="2",
        )]
        rule2 = {"is_sufficient": True, "reason": "direct_evidence", "retry_query": ""}
        with patch("project.rag_agent.rag_nodes.check_sufficiency", return_value=rule2):
            delta2 = evaluate_evidence(state2, MagicMock())
        state2.update(delta2)
        self.assertTrue(state2["_evidence_sufficient"])
        self.assertEqual(route_after_evidence(state2), "should_compress_context")  # sufficient → compress → collect

    def test_loop_terminates_on_round_limit(self):
        """Two insufficient rounds with distinct queries hit MAX_EVIDENCE_ROUNDS → fallback."""
        import config
        from unittest.mock import patch, MagicMock
        from langchain_core.messages import ToolMessage
        from project.rag_agent.rag_nodes import evaluate_evidence
        from project.rag_agent.edges import route_after_evidence
        from project.rag_agent.schemas import EvidenceSufficiency

        state = _make_state(
            [ToolMessage(content="[DOC] 略", tool_call_id="1")],
            evidence_rounds=config.MAX_EVIDENCE_ROUNDS,  # already at limit
            last_refined_query="某检索式",
            refined_queries=["别的", "某检索式"],
            _evidence_sufficient=False,
        )
        self.assertEqual(route_after_evidence(state), "fallback_response")
```

- [ ] **Step 2: 运行测试确认通过**

Run: `.\venv\Scripts\python.exe -m unittest tests.test_agentic_retrieval.TestAgenticRetrievalLoopIntegration -v`
Expected: PASS (2 tests)

> 若失败，常见原因：`_evidence_sufficient` 临时键未被 LangGraph 透传到边——本测试直接在 state dict 上 `.update(delta)`，故不受 LangGraph schema 校验影响，应通过。若真实 graph 运行时该键被丢弃，需在 Task 5 备注方案落地（改用声明字段 `evidence_sufficient: bool` 替代临时键），此时同步更新 evaluate_evidence 与 route_after_evidence 与所有测试。

- [ ] **Step 3: Commit**

```bash
git add tests/test_agentic_retrieval.py
git commit -m "test: agentic retrieval two-round loop and termination integration"
```

---

## Task 9: 全量回归与 smoke

**Files:** 无修改，仅运行

- [ ] **Step 1: 编译全量**

Run: `.\venv\Scripts\python.exe -m compileall project tests`
Expected: 无错误

- [ ] **Step 2: 全量单测**

Run: `.\venv\Scripts\python.exe -m unittest discover -s tests -v 2>&1 | tail -30`
Expected: 全绿（含既有 test_api_app 等）。若有既有失败，需确认与本改动无关（用 `git stash` 对比）。

- [ ] **Step 3: P1 专项全绿**

Run: `.\venv\Scripts\python.exe -m unittest tests.test_agentic_retrieval -v`
Expected: 全部 PASS

- [ ] **Step 4: smoke（无模型）**

Run: `.\scripts\smoke_split_app.ps1 -SkipChat`
Expected: 通过（脚本退出码 0）

- [ ] **Step 5: 回滚拓扑回归**

Run: `$env:ENABLE_AGENTIC_RETRIEVAL='false'; .\venv\Scripts\python.exe -m unittest tests.test_api_app -v`
Expected: 通过（确认开关关闭时既有行为不回归）

> 注意：`test_api_app` 若依赖真实 graph 编译，关闭开关应回退旧拓扑。若该测试不触达子图接线，则此步主要验证不报错。

- [ ] **Step 6: 收尾 commit（若有未提交改动）**

```bash
git status
# 若干净则跳过；若有 docs/notes 改动：
git add -A && git commit -m "test: P1 agentic retrieval regression green"
```

---

## Self-Review Checklist（实现完成后自查，非步骤）

- [ ] spec 第 3 节拓扑：tools → evaluate_evidence → route_after_evidence → {should_compress_context | fallback_response} 已落地（Task 7）。
- [ ] spec 第 4.1 快路径不调 LLM（Task 4 测试覆盖）。
- [ ] spec 第 4.1 LLM 失败兜底回退规则 retry_query（Task 4 测试覆盖）。
- [ ] spec 第 4.2 三守卫（轮次上限 / refined_query 重复 / 连续 NO_EVIDENCE）落地（Task 5 测试覆盖）。
- [ ] spec 第 4.3 refined_query 注入并清空（Task 6 测试覆盖）。
- [ ] spec 第 4.5 开关回滚（Task 7 Step 5）。
- [ ] spec 第 7 节"需要两轮"集成用例（Task 8）。
- [ ] 无占位符；类型/字段名跨任务一致（`evidence_rounds`/`evidence_critique`/`last_refined_query`/`refined_queries`/`_evidence_sufficient`）。
