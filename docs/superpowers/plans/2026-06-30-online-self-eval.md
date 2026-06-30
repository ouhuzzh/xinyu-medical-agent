# P5 Online Self-Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LLM-as-judge `self_eval` node at the medical_rag exit (after P2 grounding, before P4 supervisor) that scores the final answer on safety/accuracy/completeness/groundedness, appends a soft-degrade caveat when the score is low, and persists the score to `route_logs`.

**Architecture:** New `self_eval` node (light tier) slots between `answer_grounding_check` and `supervise`. `route_after_grounding`'s terminal target becomes `self_eval` (when enabled); a new `route_after_self_eval` edge continues to `supervise`/`__end__`. Score + details persist as new `route_logs` columns. `ENABLE_SELF_EVAL` toggles the whole feature off (rollback to P4). Mirrors the conditional-edge + `_structured_output_llm` + SILENT_NODES patterns of P1-P4.

**Tech Stack:** LangGraph StateGraph, conditional edges, `_structured_output_llm` helper, Pydantic schemas, PostgreSQL `ADD COLUMN IF NOT EXISTS` migration, Python `unittest`.

**Spec:** `docs/superpowers/specs/2026-06-30-online-self-eval-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `project/config.py` | `ENABLE_SELF_EVAL`, `SELF_EVAL_DEGRADE_THRESHOLD` | Modify (append after P4 block) |
| `project/rag_agent/graph_state.py` | `State` adds `self_eval_score`/`self_eval_details` | Modify |
| `project/rag_agent/schemas.py` | `AnswerSelfEval` schema | Modify (append) |
| `project/rag_agent/prompts.py` | `get_self_eval_prompt()` | Modify (append) |
| `project/rag_agent/rag_nodes.py` | `self_eval` node; imports; `__all__` | Modify |
| `project/rag_agent/edges.py` | `route_after_self_eval` (new); modify `route_after_grounding`; import | Modify |
| `project/rag_agent/graph.py` | wire `self_eval`, rewire grounding edge | Modify |
| `project/core/chat_interface.py` | `SILENT_NODES` adds `self_eval` | Modify |
| `project/db/schema_manager.py` | migration `006_route_logs_self_eval` | Modify |
| `project/db/route_log_store.py` | `save_log` INSERT adds 2 columns | Modify |
| `project/core/chat_turn_service.py` | `_persist_route_log` payload adds score/details | Modify |
| `tests/test_online_self_eval.py` | new test module | Create |
| `tests/test_answer_reflection.py` | update `route_after_grounding` assertions | Modify |
| `tests/test_multi_agent_supervisor.py` | update `route_after_grounding` assertions | Modify |
| `tests/test_chat_interface.py` | `SILENT_NODES` assertion | Modify |

**Scope guard (CRITICAL):** The working tree has a stashed batch of unrelated changes (`stash@{0}`). Stage ONLY the P5 files listed above per task. NEVER use `git add -A`/`git add .`/`git commit -am`. Tests run via `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest <module> -v` (Windows bash). Test files use `sys.path.insert(0, .../project)`.

---

## Task 1: Config + State fields

**Files:**
- Modify: `project/config.py` (after the P4 block — the `MAX_SUPERVISOR_ROUNDS`/`ENABLE_MULTI_AGENT_SUPERVISOR` lines)
- Modify: `project/rag_agent/graph_state.py` (after the P4 supervisor fields — `supervisor_next: str = "FINISH"`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_online_self_eval.py`:

```python
"""Tests for P5 online self-eval: self_eval node + route_after_self_eval +
route_after_grounding rewire + route_logs persistence."""
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))


def _make_main_state(**extra):
    base = {
        "messages": [],
        "originalQuery": "高血压用药期间能打疫苗吗",
        "primary_user_query": "高血压用药期间能打疫苗吗",
        "rewrittenQuestions": ["高血压用药期间能打疫苗吗"],
        "conversation_summary": "",
        "recent_context": "",
        "topic_focus": "",
        "user_memories": "",
        "sub_questions": ["高血压用药期间能打疫苗吗"],
        "agent_answers": [{"index": 0, "question": "高血压用药期间能打疫苗吗",
                           "answer": "一般可以接种，但需先咨询医生。", "confidence_bucket": "medium",
                           "evidence_score": 0.78}],
        "grounding_passed": True,
        "grounding_rounds": 0,
        "grounding_evidence_score": 0.78,
        "supervisor_active": False,
        "supervisor_rounds": 0,
        "supervisor_next": "FINISH",
        "self_eval_score": None,
        "self_eval_details": {},
    }
    base.update(extra)
    return base


class TestConfigFields(unittest.TestCase):
    def test_self_eval_config_fields_exist(self):
        import config
        self.assertIsInstance(config.ENABLE_SELF_EVAL, bool)
        self.assertIsInstance(config.SELF_EVAL_DEGRADE_THRESHOLD, float)
        self.assertGreater(config.SELF_EVAL_DEGRADE_THRESHOLD, 0.0)
        self.assertLess(config.SELF_EVAL_DEGRADE_THRESHOLD, 1.0)


class TestStateFields(unittest.TestCase):
    def test_state_has_self_eval_fields(self):
        from project.rag_agent.graph_state import State
        defaults = State.__annotations__
        self.assertIn("self_eval_score", defaults)
        self.assertIn("self_eval_details", defaults)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_online_self_eval.TestConfigFields tests.test_online_self_eval.TestStateFields -v
```
Expected: FAIL with `AttributeError: module 'config' has no attribute 'ENABLE_SELF_EVAL'`.

- [ ] **Step 3: Implement config + state**

In `project/config.py`, after the P4 block (`ENABLE_MULTI_AGENT_SUPERVISOR = ...`), insert:

```python

# P5: online self-eval — LLM-as-judge answer scoring + soft-degrade caveat
ENABLE_SELF_EVAL = os.environ.get("ENABLE_SELF_EVAL", "true").lower() == "true"
SELF_EVAL_DEGRADE_THRESHOLD = float(os.environ.get("SELF_EVAL_DEGRADE_THRESHOLD", "0.6"))
```

In `project/rag_agent/graph_state.py`, after the line `supervisor_next: str = "FINISH"` (the P4 block), insert (4-space indent matching surrounding State fields):

```python
    # P5: online self-eval — LLM-as-judge score + details at turn end
    self_eval_score: float | None = None
    self_eval_details: dict = {}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_online_self_eval.TestConfigFields tests.test_online_self_eval.TestStateFields -v
```
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add project/config.py project/rag_agent/graph_state.py tests/test_online_self_eval.py
git commit -m "feat(p5): add self-eval config fields and State fields"
```

---

## Task 2: AnswerSelfEval schema + prompt

**Files:**
- Modify: `project/rag_agent/schemas.py` (append after `SupervisorDecision`)
- Modify: `project/rag_agent/prompts.py` (append after `get_supervisor_prompt`)
- Modify: `tests/test_online_self_eval.py` (append 2 test classes)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_online_self_eval.py` (before the `if __name__` block):

```python
class TestAnswerSelfEvalSchema(unittest.TestCase):
    def test_schema_fields(self):
        from project.rag_agent.schemas import AnswerSelfEval
        fields = AnswerSelfEval.model_fields
        for name in ("safety", "accuracy", "completeness", "groundedness", "reason"):
            self.assertIn(name, fields)
        # The 4 scoring dims are Literal[1,2,3,4,5] (enforces range + makes _default() raise → degraded path)
        from typing import get_args
        for name in ("safety", "accuracy", "completeness", "groundedness"):
            self.assertEqual(set(get_args(fields[name].annotation)), {1, 2, 3, 4, 5})

    def test_schema_accepts_valid_dims(self):
        from project.rag_agent.schemas import AnswerSelfEval
        v = AnswerSelfEval(safety=5, accuracy=4, completeness=4, groundedness=3, reason="ok")
        self.assertEqual(v.safety, 5)


class TestSelfEvalPrompt(unittest.TestCase):
    def test_prompt_exists(self):
        from project.rag_agent.prompts import get_self_eval_prompt
        p = get_self_eval_prompt()
        for token in ("safety", "accuracy", "completeness", "groundedness", "JSON"):
            self.assertIn(token, p)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_online_self_eval.TestAnswerSelfEvalSchema tests.test_online_self_eval.TestSelfEvalPrompt -v
```
Expected: FAIL with `ImportError: cannot import name 'AnswerSelfEval'`.

- [ ] **Step 3: Implement schema + prompt**

Append to `project/rag_agent/schemas.py`:

```python


class AnswerSelfEval(BaseModel):
    safety: Literal[1, 2, 3, 4, 5] = Field(description="回答的医学安全性 1-5：是否避免不安全建议、必要时建议就医。")
    accuracy: Literal[1, 2, 3, 4, 5] = Field(description="医学准确性 1-5：是否医学正确、与检索证据一致。")
    completeness: Literal[1, 2, 3, 4, 5] = Field(description="完整性 1-5：是否充分回答了用户问题（尤其多 facet 问题）。")
    groundedness: Literal[1, 2, 3, 4, 5] = Field(description="证据支撑度 1-5：是否限于检索证据、未臆造。")
    reason: str = Field(description="简短说明打分依据。")
```

Append to `project/rag_agent/prompts.py`:

```python


def get_self_eval_prompt() -> str:
    """System prompt for the self_eval node (P5).

    LLM-as-judge: rates the final medical answer on 4 dimensions (1-5 each).
    Output must be strict JSON matching AnswerSelfEval:
    {"safety": 1-5, "accuracy": 1-5, "completeness": 1-5, "groundedness": 1-5, "reason": str}.
    """
    return (
        "你是一名医学回答质量评审员。基于检索到的证据评判以下医学回答，不要求你引入新知识。\n\n"
        "对 4 个维度各打 1-5 分：\n"
        "- safety：是否避免不安全医学建议、必要时建议就医。\n"
        "- accuracy：是否医学正确、与检索证据一致。\n"
        "- completeness：是否充分回答了用户问题（尤其多 facet 问题）。\n"
        "- groundedness：是否限于检索证据、未臆造。\n\n"
        "判定原则：\n"
        "- 5 分优秀，3 分及格，1 分很差。\n"
        "- 仅基于给定证据与回答内容评判，不臆测未给出的信息。\n\n"
        "严格输出 JSON，不要输出多余文字：\n"
        '{"safety": 1-5, "accuracy": 1-5, "completeness": 1-5, "groundedness": 1-5, "reason": "简短依据"}'
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_online_self_eval.TestAnswerSelfEvalSchema tests.test_online_self_eval.TestSelfEvalPrompt -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add project/rag_agent/schemas.py project/rag_agent/prompts.py tests/test_online_self_eval.py
git commit -m "feat(p5): add AnswerSelfEval schema and self-eval prompt"
```

---

## Task 3: `self_eval` node

**Files:**
- Modify: `project/rag_agent/rag_nodes.py` (imports; add `self_eval` function; `__all__`)
- Modify: `tests/test_online_self_eval.py` (append test class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_online_self_eval.py`:

```python
class _FakeStructuredLLM:
    """Mimics _structured_output_llm.invoke returning a schema instance."""
    def __init__(self, verdict):
        self._verdict = verdict
    def invoke(self, messages):
        return self._verdict


class TestSelfEvalNode(unittest.TestCase):
    def _state_with_answer(self, answer="一般可以接种，但需先咨询医生。", **extra):
        from langchain_core.messages import AIMessage
        return _make_main_state(messages=[AIMessage(content=answer)], **extra)

    def test_disabled_returns_empty(self):
        import project.rag_agent.rag_nodes as mod
        with unittest.mock.patch.object(mod.config, "ENABLE_SELF_EVAL", False):
            result = mod.self_eval(self._state_with_answer(), MagicMock())
        self.assertEqual(result, {})

    def test_four_dims_produce_weighted_score(self):
        """safety*0.35 + accuracy*0.30 + completeness*0.20 + groundedness*0.15, /5."""
        import project.rag_agent.rag_nodes as mod
        from project.rag_agent.schemas import AnswerSelfEval
        verdict = AnswerSelfEval(safety=5, accuracy=5, completeness=5, groundedness=5, reason="perfect")
        with unittest.mock.patch.object(mod, "_structured_output_llm",
                                        return_value=_FakeStructuredLLM(verdict)):
            result = mod.self_eval(self._state_with_answer(), MagicMock())
        self.assertAlmostEqual(result["self_eval_score"], 1.0)
        self.assertFalse(result["self_eval_details"].get("caveat_appended", False))

    def test_low_score_appends_caveat(self):
        """score < threshold → caveat AIMessage appended, caveat_appended=True."""
        import project.rag_agent.rag_nodes as mod
        from project.rag_agent.schemas import AnswerSelfEval
        # safety=4, accuracy=2, completeness=3, groundedness=2 → 0.58 < 0.6
        verdict = AnswerSelfEval(safety=4, accuracy=2, completeness=3, groundedness=2, reason="weak")
        with unittest.mock.patch.object(mod, "_structured_output_llm",
                                        return_value=_FakeStructuredLLM(verdict)):
            result = mod.self_eval(self._state_with_answer(), MagicMock())
        self.assertLess(result["self_eval_score"], 0.6)
        self.assertTrue(result["self_eval_details"].get("caveat_appended"))
        # A caveat AIMessage was appended
        self.assertTrue(any("自评提示" in str(getattr(m, "content", "")) for m in result.get("messages", [])))

    def test_high_score_no_caveat(self):
        import project.rag_agent.rag_nodes as mod
        from project.rag_agent.schemas import AnswerSelfEval
        verdict = AnswerSelfEval(safety=4, accuracy=4, completeness=4, groundedness=4, reason="good")
        with unittest.mock.patch.object(mod, "_structured_output_llm",
                                        return_value=_FakeStructuredLLM(verdict)):
            result = mod.self_eval(self._state_with_answer(), MagicMock())
        self.assertGreaterEqual(result["self_eval_score"], 0.6)
        self.assertFalse(result["self_eval_details"].get("caveat_appended"))
        self.assertNotIn("messages", result)

    def test_llm_failure_degrades_neutral_no_caveat(self):
        """patch _structured_output_llm to raise → neutral 0.5, degraded=True, no caveat, no raise."""
        import project.rag_agent.rag_nodes as mod
        with unittest.mock.patch.object(mod, "_structured_output_llm",
                                        side_effect=Exception("boom")):
            result = mod.self_eval(self._state_with_answer(), MagicMock())
        self.assertEqual(result["self_eval_score"], 0.5)
        self.assertTrue(result["self_eval_details"].get("degraded"))
        self.assertFalse(result["self_eval_details"].get("caveat_appended"))

    def test_real_llm_failure_exercises_default_fallback(self):
        """Bare MagicMock LLM (no patch of _structured_output_llm) → _default() path.
        AnswerSelfEval dims are Literal[1-5], so _default() sets "" → Pydantic rejects
        → _default() raises → self_eval's try/except → degraded path: score 0.5,
        degraded=True, NO caveat, never raises. (Mirrors P4 supervise's never-raise test.)"""
        import project.rag_agent.rag_nodes as mod
        result = mod.self_eval(self._state_with_answer(), MagicMock())
        self.assertEqual(result["self_eval_score"], 0.5)
        self.assertTrue(result["self_eval_details"].get("degraded"))
        self.assertFalse(result["self_eval_details"].get("caveat_appended"))

    def test_illegal_dims_coerced(self):
        """dims out of [1,5] coerced into range."""
        import project.rag_agent.rag_nodes as mod
        class _Bogus:
            safety = 9
            accuracy = 0
            completeness = -1
            groundedness = 6
            reason = "bogus"
        with unittest.mock.patch.object(mod, "_structured_output_llm",
                                        return_value=_FakeStructuredLLM(_Bogus())):
            result = mod.self_eval(self._state_with_answer(), MagicMock())
        d = result["self_eval_details"]
        self.assertTrue(1 <= d["safety"] <= 5)
        self.assertTrue(1 <= d["accuracy"] <= 5)
        self.assertTrue(1 <= d["completeness"] <= 5)
        self.assertTrue(1 <= d["groundedness"] <= 5)

    def test_empty_answer_degrades(self):
        import project.rag_agent.rag_nodes as mod
        result = mod.self_eval(_make_main_state(messages=[]), MagicMock())
        self.assertIsNone(result["self_eval_score"])
        self.assertTrue(result["self_eval_details"].get("degraded"))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_online_self_eval.TestSelfEvalNode -v
```
Expected: FAIL with `AttributeError: module 'project.rag_agent.rag_nodes' has no attribute 'self_eval'`.

- [ ] **Step 3: Implement the self_eval node**

In `project/rag_agent/rag_nodes.py`:

(a) The `.schemas` import block (already has `SupervisorDecision` from P4). Add `AnswerSelfEval` (alphabetical, before `EvidenceSufficiency`). The block looks like:
```python
from .schemas import (
    QueryAnalysis,
    RetrievalQueryPlan,
    GroundedAnswerCheck,
    EvidenceSufficiency,
    GroundingCritique,
    SupervisorDecision,
    TaskDecomposition,
)
```
Insert `AnswerSelfEval,` after `GroundedAnswerCheck,` (alphabetical):
```python
from .schemas import (
    AnswerSelfEval,
    QueryAnalysis,
    RetrievalQueryPlan,
    GroundedAnswerCheck,
    EvidenceSufficiency,
    GroundingCritique,
    SupervisorDecision,
    TaskDecomposition,
)
```

(b) The `.prompts` import block (already has `get_supervisor_prompt`). Add `get_self_eval_prompt,` (alphabetical, before `get_supervisor_prompt`). The block:
```python
from .prompts import (
    get_rewrite_query_prompt,
    ...
    get_grounding_critique_prompt,
    get_supervisor_prompt,
    get_task_decomposition_prompt,
)
```
Insert `get_self_eval_prompt,` before `get_supervisor_prompt,`:
```python
    get_grounding_critique_prompt,
    get_self_eval_prompt,
    get_supervisor_prompt,
    get_task_decomposition_prompt,
```

(c) Add the `self_eval` function. Place it immediately AFTER the `supervise` function (which is after `reset_supervisor_state`, before `decompose_tasks`):

```python
# P5 self-eval weights (safety weighted highest — medical domain)
_SELF_EVAL_WEIGHTS = {"safety": 0.35, "accuracy": 0.30, "completeness": 0.20, "groundedness": 0.15}


def _coerce_dim(value, default=3):
    """Coerce a self-eval dimension to an int in [1, 5]; default on illegal/missing."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    if v < 1:
        return 1
    if v > 5:
        return 5
    return v


def _extract_answer_body(content: str) -> str:
    """Strip the trailing confidence_note + citation_block appended by
    grounded_answer_generation, returning the pure answer body for judging."""
    text = str(content or "")
    for marker in ("\n\n参考来源：", "\n\n证据强度：", "\n\n版本提醒："):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def self_eval(state: State, llm):
    """P5: online self-eval — LLM-as-judge on the final medical answer.

    Scores safety/accuracy/completeness/groundedness (1-5 each) → weighted
    0.0-1.0. Low score appends a soft-degrade caveat AIMessage. Never raises:
    on any LLM/parse failure it degrades to a neutral score (0.5) with
    degraded=True and NO caveat (a failed eval must not falsify a warning).
    """
    if not config.ENABLE_SELF_EVAL:
        return {}

    latest = state["messages"][-1] if state.get("messages") else None
    raw_answer = str(getattr(latest, "content", "") or "")
    answer_body = _extract_answer_body(raw_answer)
    if not answer_body:
        return {"self_eval_score": None,
                "self_eval_details": {"degraded": True, "reason": "empty_answer"}}

    original_query = str(state.get("originalQuery") or state.get("primary_user_query") or "").strip()
    agent_answers = state.get("agent_answers") or []
    evidence_ctx = ""
    if agent_answers:
        last = agent_answers[-1]
        if isinstance(last, dict):
            evidence_ctx = (f"检索置信度：{last.get('confidence_bucket', '')}；"
                            f"证据分：{last.get('evidence_score', '')}")

    sys_msg = SystemMessage(content=get_self_eval_prompt())
    user_payload = (
        f"用户原始问题：{original_query}\n"
        f"{evidence_ctx}\n"
        f"待评回答：\n{answer_body}\n"
    )

    try:
        parser = _structured_output_llm(llm, AnswerSelfEval, max_tokens=config.LLM_STRUCTURED_MAX_TOKENS)
        verdict = parser.invoke([sys_msg, HumanMessage(content=user_payload)])
    except Exception as exc:
        logger.warning("self_eval structured output failed; degrading to neutral: %s", exc)
        return {"self_eval_score": 0.5,
                "self_eval_details": {"degraded": True, "reason": "llm_failure"}}

    dims = {name: _coerce_dim(getattr(verdict, name, None)) for name in _SELF_EVAL_WEIGHTS}
    reason = str(getattr(verdict, "reason", "") or "")
    score = sum(dims[n] * w for n, w in _SELF_EVAL_WEIGHTS.items()) / 5.0

    details = {**dims, "reason": reason, "degraded": False, "caveat_appended": False}

    if score < config.SELF_EVAL_DEGRADE_THRESHOLD:
        caveat = (f"⚠️ 自评提示：本回答在准确性/完整性上置信度较低"
                  f"（自评 {score:.2f}/1.0），建议结合线下医生意见或补充更多症状细节后再判断。")
        details["caveat_appended"] = True
        return {
            "self_eval_score": score,
            "self_eval_details": details,
            "messages": [AIMessage(content=caveat)],
        }

    return {"self_eval_score": score, "self_eval_details": details}
```

Notes:
- `SystemMessage`, `HumanMessage`, `AIMessage` are already imported at the top of rag_nodes.py (line 13).
- `_structured_output_llm` is imported from `.node_helpers` (already used by `supervise`/`decompose_tasks`).
- `logger` is the module logger. `config` is imported (`import config`).
- `config.LLM_STRUCTURED_MAX_TOKENS` exists (used by P3/P4).
- The `messages: [AIMessage(caveat)]` return uses the `add_messages` reducer (MessagesState) → appends a new message; the original answer message is preserved.

(d) Add `"self_eval"` to the `__all__` list, immediately before `"supervise"` (the existing list is not strictly alphabetical — `supervise` precedes `rewrite_query` — so just place `"self_eval"` right before `"supervise"`):
```python
    "should_compress_context",
    "self_eval",
    "supervise",
    "rewrite_query",
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_online_self_eval.TestSelfEvalNode -v
```
Expected: PASS (8 tests). The `test_real_llm_failure_exercises_default_fallback` test passes a bare MagicMock LLM WITHOUT patching `_structured_output_llm`, so the real `_structured_output_llm` + its `_default()` path is genuinely exercised. Because `AnswerSelfEval`'s dims are `Literal[1,2,3,4,5]`, `_default()` sets them to `""` → Pydantic rejects → `_default()` raises → `self_eval`'s try/except catches → degraded path (score 0.5, `degraded=True`, no caveat). This is the critical never-raise + no-false-caveat test (regression guard for the `_default()` Literal mechanism, mirroring P4's `supervise`).

- [ ] **Step 5: Commit**

```bash
git add project/rag_agent/rag_nodes.py tests/test_online_self_eval.py
git commit -m "feat(p5): add self_eval node with weighted scoring and soft-degrade caveat"
```

---

## Task 4: `route_after_self_eval` + `route_after_grounding` rewire

**Files:**
- Modify: `project/rag_agent/edges.py` (import line 5; `route_after_grounding`; append `route_after_self_eval`)
- Modify: `tests/test_online_self_eval.py` (append 2 test classes)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_online_self_eval.py`:

```python
class TestRouteAfterSelfEval(unittest.TestCase):
    def test_to_supervise_when_supervisor_enabled(self):
        import project.rag_agent.edges as edges
        from project.rag_agent.edges import route_after_self_eval
        with unittest.mock.patch.object(edges.config, "ENABLE_MULTI_AGENT_SUPERVISOR", True):
            self.assertEqual(route_after_self_eval(_make_main_state()), "supervise")

    def test_to_end_when_supervisor_disabled(self):
        import project.rag_agent.edges as edges
        from project.rag_agent.edges import route_after_self_eval
        with unittest.mock.patch.object(edges.config, "ENABLE_MULTI_AGENT_SUPERVISOR", False):
            self.assertEqual(route_after_self_eval(_make_main_state()), "__end__")


class TestRouteAfterGroundingSelfEval(unittest.TestCase):
    def test_grounded_routes_to_self_eval_when_enabled(self):
        from project.rag_agent.edges import route_after_grounding
        self.assertEqual(route_after_grounding(_make_main_state(grounding_passed=True)), "self_eval")

    def test_budget_exhausted_routes_to_self_eval_when_enabled(self):
        import config
        from project.rag_agent.edges import route_after_grounding
        state = _make_main_state(grounding_passed=False, grounding_rounds=config.MAX_GROUNDING_ROUNDS)
        self.assertEqual(route_after_grounding(state), "self_eval")

    def test_not_grounded_with_budget_routes_to_revise(self):
        from project.rag_agent.edges import route_after_grounding
        state = _make_main_state(grounding_passed=False, grounding_rounds=0)
        self.assertEqual(route_after_grounding(state), "revise_answer")

    def test_grounded_routes_to_supervise_when_self_eval_disabled(self):
        import project.rag_agent.edges as edges
        from project.rag_agent.edges import route_after_grounding
        with unittest.mock.patch.object(edges.config, "ENABLE_SELF_EVAL", False):
            self.assertEqual(route_after_grounding(_make_main_state(grounding_passed=True)), "supervise")

    def test_grounded_routes_to_end_when_both_disabled(self):
        import project.rag_agent.edges as edges
        from project.rag_agent.edges import route_after_grounding
        with unittest.mock.patch.object(edges.config, "ENABLE_SELF_EVAL", False), \
             unittest.mock.patch.object(edges.config, "ENABLE_MULTI_AGENT_SUPERVISOR", False):
            self.assertEqual(route_after_grounding(_make_main_state(grounding_passed=True)), "__end__")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_online_self_eval.TestRouteAfterSelfEval tests.test_online_self_eval.TestRouteAfterGroundingSelfEval -v
```
Expected: FAIL with `ImportError: cannot import name 'route_after_self_eval'` and `AssertionError: 'supervise' != 'self_eval'`.

- [ ] **Step 3: Implement the edges**

In `project/rag_agent/edges.py`, line 5 currently (after P4):
```python
from config import MAX_ITERATIONS, MAX_TOOL_CALLS, MAX_EVIDENCE_ROUNDS, MAX_GROUNDING_ROUNDS, MAX_SUB_QUESTIONS, ENABLE_MULTI_AGENT_SUPERVISOR
```
Add `ENABLE_SELF_EVAL`:
```python
from config import MAX_ITERATIONS, MAX_TOOL_CALLS, MAX_EVIDENCE_ROUNDS, MAX_GROUNDING_ROUNDS, MAX_SUB_QUESTIONS, ENABLE_MULTI_AGENT_SUPERVISOR, ENABLE_SELF_EVAL
```

Find `route_after_grounding` (P4 version, currently returns `Literal["__end__", "revise_answer", "supervise"]` and computes `_to_supervisor`). Replace the ENTIRE function + add `route_after_self_eval` after it:

```python
def route_after_grounding(state: State) -> Literal["__end__", "revise_answer", "supervise", "self_eval"]:
    """P2/P4/P5: route after the answer grounding check.

    - grounded → self_eval (P5) when on, else supervise (P4) / END
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


def _next_after_grounding() -> Literal["__end__", "supervise", "self_eval"]:
    """P5/P4: terminal target after grounding. self_eval if on, else supervisor if on, else END."""
    if ENABLE_SELF_EVAL:
        return "self_eval"
    return "supervise" if ENABLE_MULTI_AGENT_SUPERVISOR else "__end__"


def route_after_self_eval(state: State) -> str:
    """P5: after self-eval, continue to the P4 supervisor (or END if disabled)."""
    return "supervise" if ENABLE_MULTI_AGENT_SUPERVISOR else "__end__"
```

(`config` is already imported via `import config` from P4. `Literal` is imported at line 1.)

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_online_self_eval.TestRouteAfterSelfEval tests.test_online_self_eval.TestRouteAfterGroundingSelfEval -v
```
Expected: PASS (7 tests).

Also run the full P5 module:
```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_online_self_eval -v 2>&1 | tail -4
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add project/rag_agent/edges.py tests/test_online_self_eval.py
git commit -m "feat(p5): add route_after_self_eval and rewire route_after_grounding to self_eval"
```

---

## Task 5: Graph wiring

**Files:**
- Modify: `project/rag_agent/graph.py`
- Modify: `tests/test_online_self_eval.py` (append source-assertion test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_online_self_eval.py`:

```python
class TestGraphWiring(unittest.TestCase):
    def test_graph_source_references_self_eval_wiring(self):
        import inspect
        import project.rag_agent.graph as graph_mod
        src = inspect.getsource(graph_mod)
        self.assertIn("self_eval", src)
        self.assertIn("route_after_self_eval", src)
        self.assertIn("ENABLE_SELF_EVAL", src)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_online_self_eval.TestGraphWiring -v
```
Expected: FAIL (`'self_eval' not found in src`).

- [ ] **Step 3: Implement the wiring**

In `project/rag_agent/graph.py`:

(a) The `rag_nodes` import block (has `supervise` from P4). Add `self_eval` (alphabetical, before `should_compress_context`). The block currently:
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
    reset_supervisor_state,
    revise_answer,
    rewrite_query,
    should_compress_context,
    supervise,
)
```
Add `self_eval,` before `should_compress_context,`:
```python
    revise_answer,
    rewrite_query,
    self_eval,
    should_compress_context,
    supervise,
)
```

(b) Register the node. Find the P4 block that registers `supervise`:
```python
    # P4: multi-agent supervisor at medical_rag exit
    graph_builder.add_node("supervise", partial(supervise, llm=_light_llm))
    graph_builder.add_node(reset_supervisor_state)
```
Add `self_eval` registration immediately after `supervise`:
```python
    # P4: multi-agent supervisor at medical_rag exit
    graph_builder.add_node("supervise", partial(supervise, llm=_light_llm))
    graph_builder.add_node(reset_supervisor_state)
    # P5: online self-eval between grounding and supervisor
    graph_builder.add_node("self_eval", partial(self_eval, llm=_light_llm))
```

(c) Add the `self_eval` conditional edge. Find the P4 `supervise` conditional edge:
```python
    # P4: supervisor dispatches a peer agent (appointment/triage) or finishes.
    graph_builder.add_conditional_edges("supervise", route_after_supervisor, {
        "handle_appointment_skill": "handle_appointment_skill",
        "recommend_department": "recommend_department",
        "__end__": END,
    })
```
Add the `self_eval` conditional edge immediately AFTER it:
```python
    # P5: after self-eval, continue to the supervisor (or END).
    graph_builder.add_conditional_edges("self_eval", route_after_self_eval, {
        "supervise": "supervise",
        "__end__": END,
    })
```

(d) Update the `answer_grounding_check` conditional-edge map (the P4 `_grounding_map` block). The current block (after P4):
```python
    graph_builder.add_edge("grounded_answer_generation", "answer_grounding_check")
    if config.ENABLE_ANSWER_REFLECTION:
        graph_builder.add_node("revise_answer", partial(revise_answer, llm=_light_llm))
        _grounding_map = {"__end__": END, "revise_answer": "revise_answer"}
        if config.ENABLE_MULTI_AGENT_SUPERVISOR:
            _grounding_map["supervise"] = "supervise"
        graph_builder.add_conditional_edges(
            "answer_grounding_check",
            route_after_grounding,
            _grounding_map,
        )
        graph_builder.add_edge("revise_answer", "answer_grounding_check")
    else:
        if config.ENABLE_MULTI_AGENT_SUPERVISOR:
            graph_builder.add_conditional_edges(
                "answer_grounding_check",
                route_after_grounding,
                {"__end__": END, "supervise": "supervise"},
            )
        else:
            graph_builder.add_edge("answer_grounding_check", END)
```

Replace the `_grounding_map` construction so it adds `"self_eval"` when `ENABLE_SELF_EVAL` is on (and keeps `supervise` when the supervisor is on). Replace the ENTIRE block with:

```python
    graph_builder.add_edge("grounded_answer_generation", "answer_grounding_check")
    if config.ENABLE_ANSWER_REFLECTION:
        graph_builder.add_node("revise_answer", partial(revise_answer, llm=_light_llm))
        _grounding_map = {"__end__": END, "revise_answer": "revise_answer"}
        if config.ENABLE_SELF_EVAL:
            _grounding_map["self_eval"] = "self_eval"
        if config.ENABLE_MULTI_AGENT_SUPERVISOR:
            _grounding_map["supervise"] = "supervise"
        graph_builder.add_conditional_edges(
            "answer_grounding_check",
            route_after_grounding,
            _grounding_map,
        )
        graph_builder.add_edge("revise_answer", "answer_grounding_check")
    else:
        _grounding_map = {"__end__": END}
        if config.ENABLE_SELF_EVAL:
            _grounding_map["self_eval"] = "self_eval"
        if config.ENABLE_MULTI_AGENT_SUPERVISOR:
            _grounding_map["supervise"] = "supervise"
        if config.ENABLE_SELF_EVAL or config.ENABLE_MULTI_AGENT_SUPERVISOR:
            graph_builder.add_conditional_edges(
                "answer_grounding_check",
                route_after_grounding,
                _grounding_map,
            )
        else:
            graph_builder.add_edge("answer_grounding_check", END)
```

> Note: in the reflection-OFF branch, P4 already ensured `route_after_grounding` does not return `revise_answer` when reflection is off (it returns `self_eval`/`supervise`/`__end__`). So the map only needs those keys. The `else: add_edge(..., END)` fallback handles the both-disabled case.

- [ ] **Step 4: Run test to verify it passes + compile check**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_online_self_eval.TestGraphWiring -v
PYTHONPATH=project ./venv/Scripts/python.exe -c "from unittest.mock import MagicMock; import project.rag_agent.graph as g; llm=MagicMock(); llm.bind_tools=MagicMock(return_value=MagicMock()); g.create_agent_graph(llm, [], None, None, {}); print('BUILD OK')"
```
Expected: test PASS; `BUILD OK`.

- [ ] **Step 5: Commit**

```bash
git add project/rag_agent/graph.py tests/test_online_self_eval.py
git commit -m "feat(p5): wire self_eval into the main graph between grounding and supervisor"
```

---

## Task 6: SILENT_NODES

**Files:**
- Modify: `project/core/chat_interface.py` (SILENT_NODES set)
- Modify: `tests/test_chat_interface.py` (assertion)

- [ ] **Step 1: Write the failing test**

In `tests/test_chat_interface.py`, find the SILENT_NODES assertions (around lines 190-194, where P4 added `supervise`/`reset_supervisor_state`). Add after them:
```python
        self.assertIn("self_eval", SILENT_NODES)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_chat_interface -v 2>&1 | grep -i self_eval || echo "self_eval assertion failing or absent"
```

- [ ] **Step 3: Implement**

In `project/core/chat_interface.py`, the `SILENT_NODES` set (P4 added `supervise`/`reset_supervisor_state`). Add `"self_eval"`:
```python
SILENT_NODES = {
    "rewrite_query",
    "intent_router",
    "decompose_tasks",
    "grounded_answer_generation",
    "answer_grounding_check",
    "supervise",
    "reset_supervisor_state",
    "self_eval",
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_chat_interface -v 2>&1 | tail -4
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add project/core/chat_interface.py tests/test_chat_interface.py
git commit -m "feat(p5): silence self_eval node in streaming UI"
```

---

## Task 7: Persistence — route_logs migration + RouteLogStore + finalize_turn

**Files:**
- Modify: `project/db/schema_manager.py` (add migration `006_route_logs_self_eval`)
- Modify: `project/db/route_log_store.py` (`save_log` INSERT adds 2 columns)
- Modify: `project/core/chat_turn_service.py` (`_persist_route_log` payload adds score/details)
- Modify: `tests/test_online_self_eval.py` (append persistence test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_online_self_eval.py`:

```python
class TestSelfEvalPersistence(unittest.TestCase):
    def test_route_log_payload_includes_self_eval_fields(self):
        """_persist_route_log must pass self_eval_score + self_eval_details to the store."""
        from project.core.chat_turn_service import ChatTurnService, TurnArtifacts
        captured = {}

        class _FakeStore:
            def save_log(self, payload):
                captured.update(payload)

        svc = ChatTurnService.__new__(ChatTurnService)
        svc.route_log_store = _FakeStore()
        artifacts = TurnArtifacts(
            latest_values={"primary_intent": "medical_rag", "decision_source": "rule",
                           "self_eval_score": 0.42,
                           "self_eval_details": {"safety": 3, "degraded": False}},
            updated_state={"secondary_intent": "", "topic_focus": "",
                           "deferred_user_question": "", "pending_action_type": ""},
            route_reason="rule_match", had_pending_state=False, secondary_turn_executed=False,
        )
        svc._persist_route_log(
            active_thread_id="t1", request_id="r1", user_message="hi",
            session_state={}, checkpoint_resumed=False, artifacts=artifacts,
        )
        self.assertEqual(captured.get("self_eval_score"), 0.42)
        self.assertEqual(captured.get("self_eval_details"), {"safety": 3, "degraded": False})
```

> **Implementer note:** The exact `TurnArtifacts` constructor signature must match the real one in `project/core/chat_turn_service.py`. Before writing this test, read the `TurnArtifacts` dataclass definition (search `class TurnArtifacts` / `TurnArtifacts = `) and adjust the test's kwargs to match its real fields. If `TurnArtifacts` has more required fields, add them with placeholder values. Do NOT guess — read it first.

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_online_self_eval.TestSelfEvalPersistence -v
```
Expected: FAIL (`captured.get("self_eval_score")` is None — not yet in payload).

- [ ] **Step 3: Implement migration + store + service**

(a) In `project/db/schema_manager.py`, find the migration list (entries like `005_appointment_skill_and_retrieval_quality`). Add a new entry after the last one (use the next number — verify the highest existing number first; if `005` is last, use `006`; if higher exists, use the next). The entry shape (mirror `005`):
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

(b) In `project/db/route_log_store.py`, the `save_log` INSERT (lines 22-53). Add `self_eval_score` and `self_eval_details` to the column list + VALUES. Replace the `cur.execute(...)` block:
```python
                cur.execute(
                    """
                    INSERT INTO route_logs (
                        request_id,
                        thread_id,
                        user_query,
                        primary_intent,
                        secondary_intent,
                        decision_source,
                        route_reason,
                        had_pending_state,
                        extra_metadata,
                        self_eval_score,
                        self_eval_details
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb)
                    """,
                    (
                        item.get("request_id") or None,
                        item.get("thread_id") or None,
                        item.get("user_query") or "",
                        item.get("primary_intent") or "",
                        item.get("secondary_intent") or "",
                        item.get("decision_source") or "",
                        item.get("route_reason") or "",
                        bool(item.get("had_pending_state")),
                        json.dumps(item.get("extra_metadata") or {}, ensure_ascii=False),
                        item.get("self_eval_score"),
                        json.dumps(item.get("self_eval_details") or {}, ensure_ascii=False),
                    ),
                )
```

(c) In `project/core/chat_turn_service.py`, the `_persist_route_log` method (the `save_log({...})` dict around lines 236-254). Add the two fields to the dict, after `"had_pending_state"`:
```python
                    "had_pending_state": artifacts.had_pending_state,
                    "self_eval_score": artifacts.latest_values.get("self_eval_score"),
                    "self_eval_details": artifacts.latest_values.get("self_eval_details") or {},
                    "extra_metadata": {
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_online_self_eval.TestSelfEvalPersistence -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add project/db/schema_manager.py project/db/route_log_store.py project/core/chat_turn_service.py tests/test_online_self_eval.py
git commit -m "feat(p5): persist self-eval score + details to route_logs"
```

---

## Task 8: Compiled-graph integration test

**Files:**
- Modify: `tests/test_online_self_eval.py` (append integration test)

- [ ] **Step 1: Write the test**

Append to `tests/test_online_self_eval.py`:

```python
class TestCompiledSelfEval(unittest.TestCase):
    """Verify self_eval slots into a real compiled graph between grounding-check
    and the supervisor sink, without breaking the P4 chain."""

    def _build_graph(self, fake_eval_returns_caveat):
        from langgraph.graph import StateGraph, START, END
        from project.rag_agent.graph_state import State
        from project.rag_agent.edges import route_after_self_eval

        log = {"self_eval": 0, "supervise": 0}

        def _fake_self_eval(state):
            log["self_eval"] += 1
            if fake_eval_returns_caveat:
                from langchain_core.messages import AIMessage
                return {
                    "self_eval_score": 0.4,
                    "self_eval_details": {"caveat_appended": True},
                    "messages": [AIMessage(content="⚠️ 自评提示：低分")],
                }
            return {"self_eval_score": 0.9, "self_eval_details": {"caveat_appended": False}}

        def _fake_supervise(state):
            log["supervise"] += 1
            return {"supervisor_active": False, "supervisor_rounds": 0, "supervisor_next": "FINISH"}

        def _sink(state):
            return {}

        builder = StateGraph(State)
        builder.add_node("self_eval", _fake_self_eval)
        builder.add_node("supervise", _fake_supervise)
        builder.add_node("end_sink", _sink)
        builder.add_edge(START, "self_eval")
        builder.add_conditional_edges("self_eval", route_after_self_eval, {
            "supervise": "supervise", "__end__": "end_sink",
        })
        builder.add_conditional_edges("supervise", lambda s: "__end__", {"__end__": "end_sink"})
        builder.add_edge("end_sink", END)
        return builder.compile(), log

    def test_self_eval_then_supervise_then_end(self):
        graph, log = self._build_graph(fake_eval_returns_caveat=False)
        final = graph.invoke(_make_main_state())
        self.assertEqual(log["self_eval"], 1)
        self.assertEqual(log["supervise"], 1)
        self.assertAlmostEqual(final["self_eval_score"], 0.9)

    def test_caveat_message_appended(self):
        graph, log = self._build_graph(fake_eval_returns_caveat=True)
        final = graph.invoke(_make_main_state())
        self.assertEqual(log["self_eval"], 1)
        self.assertTrue(any("自评提示" in str(getattr(m, "content", "")) for m in final["messages"]))
```

- [ ] **Step 2: Run the test**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_online_self_eval.TestCompiledSelfEval -v
```
Expected: PASS (2 tests). If a test fails due to a wiring issue (not a test bug), debug; prefer fixing real bugs over weakening tests.

- [ ] **Step 3: Run full P5 module**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_online_self_eval -v 2>&1 | tail -4
```
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_online_self_eval.py
git commit -m "test(p5): compiled-graph integration tests for self_eval slot"
```

---

## Task 9: Regression assertion updates + full sweep + scope-guard verification

**Files:**
- Modify: `tests/test_answer_reflection.py` (route_after_grounding assertions: `supervise`→`self_eval`)
- Modify: `tests/test_multi_agent_supervisor.py` (route_after_grounding assertions: `supervise`→`self_eval`; the `TestRouteAfterGroundingSupervisor` class)

- [ ] **Step 1: Update P2/P4 regression assertions**

With `ENABLE_SELF_EVAL=true` (default), `route_after_grounding` now returns `"self_eval"` (not `"supervise"`) for terminal cases.

In `tests/test_answer_reflection.py`, `TestRouteAfterGrounding`:
- The grounded-case assertion (P4 changed it to `"supervise"`): change `"supervise"` → `"self_eval"`.
- The budget-exhausted-case assertion (P4 changed it to `"supervise"`): change `"supervise"` → `"self_eval"`.
- The not-grounded+budget case stays `"revise_answer"`.
- `TestCompiledGroundingLoop._build_graph` map (P4 added `"supervise": "__end_sink"`): add `"self_eval": "__end_sink"` too (so the compiled loop still terminates regardless of which terminal target fires). The map becomes `{"__end__": "__end_sink", "revise_answer": "revise_answer", "supervise": "__end_sink", "self_eval": "__end_sink"}`.
- Update the P4 explanatory comment to mention P5 self_eval default-on.

In `tests/test_multi_agent_supervisor.py`, `TestRouteAfterGroundingSupervisor`:
- `test_grounded_routes_to_supervise_when_enabled`: assertion `"supervise"` → `"self_eval"` (rename the test to `test_grounded_routes_to_self_eval_when_enabled` too, for accuracy).
- `test_budget_exhausted_routes_to_supervise_when_enabled`: assertion `"supervise"` → `"self_eval"` (rename to `test_budget_exhausted_routes_to_self_eval_when_enabled`).
- `test_grounded_routes_to_end_when_disabled` (supervisor disabled, self_eval enabled): now `route_after_grounding` returns `"self_eval"` (not `__end__`), because self_eval is on. Change the assertion to `"self_eval"`, OR patch `ENABLE_SELF_EVAL=False` too so it returns `__end__`. **Preferred**: patch BOTH flags False to test the true both-disabled END path — update the test to:
  ```python
  def test_grounded_routes_to_end_when_both_disabled(self):
      import project.rag_agent.edges as edges
      from project.rag_agent.edges import route_after_grounding
      with unittest.mock.patch.object(edges.config, "ENABLE_MULTI_AGENT_SUPERVISOR", False), \
           unittest.mock.patch.object(edges.config, "ENABLE_SELF_EVAL", False):
          self.assertEqual(route_after_grounding(_make_main_state(grounding_passed=True)), "__end__")
  ```
- `test_budget_exhausted_routes_to_end_when_disabled`: same treatment — patch both flags False, assert `__end__`.
- `test_not_grounded_with_budget_routes_to_supervise_when_reflection_off` (P4 fix test): with self_eval on, `route_after_grounding` returns `"self_eval"` (not `"supervise"`). Update assertion to `"self_eval"`.
- Run and fix any other `route_after_grounding` assertion that breaks.

- [ ] **Step 2: Run P2 + P4 + P5 regression**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_answer_reflection tests.test_multi_agent_supervisor tests.test_online_self_eval -v 2>&1 | tail -6
```
Expected: all PASS.

- [ ] **Step 3: Run P1 + P3 regression (must be unaffected)**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_agentic_retrieval tests.test_task_decomposition -v 2>&1 | tail -4
```
Expected: all PASS (20 + 19).

- [ ] **Step 4: Full discover + compileall**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest discover -s tests 2>&1 | tail -6
PYTHONPATH=project ./venv/Scripts/python.exe -m compileall project tests 2>&1 | tail -2
```
Expected: no NEW failures vs main baseline (pre-existing mojibake/skill-registry/token failures acceptable); compileall clean.

- [ ] **Step 5: Scope-guard verification**

```bash
git log --oneline main..HEAD
git stash list
git status --short
```
Expected:
- P5 commits touch ONLY: `project/config.py`, `project/rag_agent/{graph_state,schemas,prompts,rag_nodes,edges,graph}.py`, `project/core/{chat_interface,chat_turn_service}.py`, `project/db/{schema_manager,route_log_store}.py`, `tests/test_online_self_eval.py`, `tests/test_answer_reflection.py`, `tests/test_multi_agent_supervisor.py`, `tests/test_chat_interface.py`.
- `stash@{0}` intact.
- Working tree clean.

- [ ] **Step 6: Commit (only the regression-assertion changes)**

```bash
git add tests/test_answer_reflection.py tests/test_multi_agent_supervisor.py
git commit -m "test(p5): update P2/P4 route_after_grounding assertions for self_eval default-on"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- §4.1 self_eval node → Task 3 ✓
- §4.2 AnswerSelfEval schema → Task 2 ✓
- §4.3 get_self_eval_prompt → Task 2 ✓
- §4.4 route_after_self_eval → Task 4 ✓
- §4.5 route_after_grounding rewire → Task 4 ✓
- §4.6 State fields → Task 1 ✓
- §4.7 config → Task 1 ✓
- §4.8 graph wiring → Task 5 ✓
- §4.9 SILENT_NODES → Task 6 ✓
- §4.10 persistence (migration + store + finalize_turn) → Task 7 ✓
- §6 error handling (never-raise, degraded, empty, illegal) → Task 3 tests ✓
- §7 testing (unit + integration + persistence + regression) → Tasks 1-9 ✓
- §10 acceptance 1-11 → all covered ✓

**Placeholder scan:** None — every step has concrete code/commands. The Task 7 `TurnArtifacts` constructor note instructs the implementer to READ the real signature first (not guess) — this is a verification instruction, not a placeholder.

**Type consistency:** `self_eval_score: float | None`, `self_eval_details: dict` consistent across State (Task 1), node (Task 3), persistence (Task 7). `AnswerSelfEval` dims are `Literal[1,2,3,4,5]` (Task 2) — enforces range at schema level AND makes `_default()` raise (→ degraded path, no false caveat), mirroring P4's `SupervisorDecision` mechanism. `_coerce_dim` (Task 3) is defense-in-depth for verdict objects that bypass Pydantic (the `_Bogus` test). `route_after_self_eval` / `_next_after_grounding` return types consistent (Task 4). Weights `0.35/0.30/0.20/0.15` sum to 1.0, score/5.0 → [0,1] (Task 3).

**Never-raise + no-false-caveat contract (Task 3):** the `Literal` dims are load-bearing — they ensure a failed LLM eval (`_default()` fallback) raises → try/except → degraded path (score 0.5, no caveat), NOT a false low-score caveat. Both `test_llm_failure_degrades_neutral_no_caveat` (patched raise) and `test_real_llm_failure_exercises_default_fallback` (bare MagicMock, real `_default()` path) cover this.
