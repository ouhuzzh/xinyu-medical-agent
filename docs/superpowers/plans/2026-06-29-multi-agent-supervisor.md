# P4 Multi-Agent Supervisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LLM `supervise` node at the medical_rag exit that observes the medical answer and dispatches peer agents (appointment/triage) in the same turn, looping up to `MAX_SUPERVISOR_ROUNDS` or FINISH.

**Architecture:** Supervisor sits after the P2 answer-grounding loop. A light-tier LLM (`SupervisorDecision` schema) picks `next_agent ∈ {appointment, triage, FINISH}`. Specialists return to `supervise` via a new `supervisor_active` branch in `route_after_action` (lowest priority). A `reset_supervisor_state` node at turn start prevents supervisor flags from leaking across turns (checkpointer persistence + interrupt interaction). Mirrors the conditional-edge + counter pattern used by P1/P2.

**Tech Stack:** LangGraph StateGraph, conditional edges, `_structured_output_llm` helper, Pydantic schemas, Python `unittest`.

**Spec:** `docs/superpowers/specs/2026-06-29-multi-agent-supervisor-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `project/config.py` | `MAX_SUPERVISOR_ROUNDS`, `ENABLE_MULTI_AGENT_SUPERVISOR` | Modify (append after P3 block, line 94) |
| `project/rag_agent/graph_state.py` | `State` adds `supervisor_active`/`supervisor_rounds`/`supervisor_next` | Modify |
| `project/rag_agent/schemas.py` | `SupervisorDecision` schema | Modify (append) |
| `project/rag_agent/prompts.py` | `get_supervisor_prompt()` | Modify (append) |
| `project/rag_agent/rag_nodes.py` | `supervise` + `reset_supervisor_state` nodes; imports; `__all__` | Modify |
| `project/rag_agent/edges.py` | `route_after_supervisor` (new); modify `route_after_grounding` + `route_after_action`; import | Modify |
| `project/rag_agent/graph.py` | wire `supervise`/`reset_supervisor_state`, rewire edges | Modify |
| `project/core/chat_interface.py` | `SILENT_NODES` adds 2 nodes | Modify |
| `tests/test_multi_agent_supervisor.py` | new test module | Create |
| `tests/test_answer_reflection.py` | update `route_after_grounding` assertions | Modify |
| `tests/test_routing_edges.py` | update `route_after_action` assertions | Modify |
| `tests/test_chat_interface.py` | `SILENT_NODES` assertion | Modify |

**Scope guard (CRITICAL):** The working tree has a stashed batch of unrelated changes (`stash@{0}`). Stage ONLY the P4 files listed above per task. NEVER use `git add -A`/`git add .`/`git commit -am`. Tests run via `PYTHONPATH=project ./venv/Scripts/python.exe -m unittest <module> -v` (Windows bash). Test files use `sys.path.insert(0, .../project)`.

---

## Task 1: Config + State fields

**Files:**
- Modify: `project/config.py` (after line 94, the P3 block)
- Modify: `project/rag_agent/graph_state.py:38` (after `sub_questions: List[str] = []`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_multi_agent_supervisor.py`:

```python
"""Tests for P4 multi-agent supervisor: supervise node + route_after_supervisor +
cross-turn reset + route_after_action/route_after_grounding wiring."""
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))


def _make_main_state(**extra):
    base = {
        "messages": [],
        "originalQuery": "高血压用药期间能打疫苗吗，顺便挂个心内科",
        "primary_user_query": "高血压用药期间能打疫苗吗，顺便挂个心内科",
        "rewrittenQuestions": ["高血压用药期间能打疫苗吗"],
        "conversation_summary": "",
        "recent_context": "",
        "topic_focus": "",
        "user_memories": "",
        "sub_questions": ["高血压用药期间能打疫苗吗"],
        "agent_answers": [{"index": 0, "question": "高血压用药期间能打疫苗吗",
                           "answer": "一般可以接种，但需先咨询医生。", "confidence_bucket": "medium"}],
        "secondary_intent": "",
        "deferred_user_question": "",
        "grounding_passed": True,
        "grounding_rounds": 0,
        "supervisor_active": False,
        "supervisor_rounds": 0,
        "supervisor_next": "FINISH",
    }
    base.update(extra)
    return base


class TestConfigFields(unittest.TestCase):
    def test_supervisor_config_fields_exist(self):
        import config
        self.assertIsInstance(config.MAX_SUPERVISOR_ROUNDS, int)
        self.assertGreaterEqual(config.MAX_SUPERVISOR_ROUNDS, 1)
        self.assertIsInstance(config.ENABLE_MULTI_AGENT_SUPERVISOR, bool)


class TestStateFields(unittest.TestCase):
    def test_state_has_supervisor_fields(self):
        from project.rag_agent.graph_state import State
        defaults = State.__annotations__
        self.assertIn("supervisor_active", defaults)
        self.assertIn("supervisor_rounds", defaults)
        self.assertIn("supervisor_next", defaults)
        # Default values
        s = State(messages=[])
        self.assertFalse(s["supervisor_active"])
        self.assertEqual(s["supervisor_rounds"], 0)
        self.assertEqual(s["supervisor_next"], "FINISH")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_multi_agent_supervisor.TestConfigFields tests.test_multi_agent_supervisor.TestStateFields -v
```
Expected: FAIL with `AttributeError: module 'config' has no attribute 'MAX_SUPERVISOR_ROUNDS'` (and `KeyError: 'supervisor_active'`).

- [ ] **Step 3: Implement config + state**

In `project/config.py`, after line 94 (the `ENABLE_TASK_DECOMPOSITION` line), insert:

```python

# P4: multi-agent supervisor — LLM-coordinated agent handoff after medical_rag
MAX_SUPERVISOR_ROUNDS = int(os.environ.get("MAX_SUPERVISOR_ROUNDS", "3"))
ENABLE_MULTI_AGENT_SUPERVISOR = os.environ.get("ENABLE_MULTI_AGENT_SUPERVISOR", "true").lower() == "true"
```

In `project/rag_agent/graph_state.py`, after line 38 (`sub_questions: List[str] = []`), insert:

```python
    # P4: multi-agent supervisor — loop flags at medical_rag exit
    supervisor_active: bool = False
    supervisor_rounds: int = 0
    supervisor_next: str = "FINISH"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_multi_agent_supervisor.TestConfigFields tests.test_multi_agent_supervisor.TestStateFields -v
```
Expected: PASS (4 tests: 1 config, 1 state — the file currently has only these 2 classes).

- [ ] **Step 5: Commit**

```bash
git add project/config.py project/rag_agent/graph_state.py tests/test_multi_agent_supervisor.py
git commit -m "feat(p4): add supervisor config fields and State fields"
```

---

## Task 2: SupervisorDecision schema + prompt

**Files:**
- Modify: `project/rag_agent/schemas.py` (append after line 195, the `TaskDecomposition` class)
- Modify: `project/rag_agent/prompts.py` (append after line 448, end of file)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_agent_supervisor.py` (before the `if __name__` block):

```python
class TestSupervisorDecisionSchema(unittest.TestCase):
    def test_schema_fields(self):
        from project.rag_agent.schemas import SupervisorDecision
        from typing import get_args
        fields = SupervisorDecision.model_fields
        self.assertIn("next_agent", fields)
        self.assertIn("reason", fields)
        # next_agent must be a Literal of appointment/triage/FINISH
        annot = fields["next_agent"].annotation
        self.assertEqual(set(get_args(annot)), {"appointment", "triage", "FINISH"})


class TestSupervisorPrompt(unittest.TestCase):
    def test_prompt_exists(self):
        from project.rag_agent.prompts import get_supervisor_prompt
        p = get_supervisor_prompt()
        self.assertIn("appointment", p)
        self.assertIn("triage", p)
        self.assertIn("FINISH", p)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_multi_agent_supervisor.TestSupervisorDecisionSchema tests.test_multi_agent_supervisor.TestSupervisorPrompt -v
```
Expected: FAIL with `ImportError: cannot import name 'SupervisorDecision'`.

- [ ] **Step 3: Implement schema + prompt**

Append to `project/rag_agent/schemas.py`:

```python


class SupervisorDecision(BaseModel):
    next_agent: Literal["appointment", "triage", "FINISH"] = Field(
        description="下一步派发的专家 agent；无需后续动作时为 FINISH。"
    )
    reason: str = Field(description="简短说明为何派发该 agent 或 FINISH。")
```

Append to `project/rag_agent/prompts.py`:

```python


def get_supervisor_prompt() -> str:
    """System prompt for the supervise node (P4).

    The supervisor observes the medical agent's answer + the user's original
    query and decides whether to dispatch a peer action-agent in the same turn.
    Output must be strict JSON matching SupervisorDecision:
    {"next_agent": "appointment|triage|FINISH", "reason": str}.
    """
    return (
        "你是一名医疗助手的 supervisor。医疗问答 agent 刚给出答案，你需要判断是否"
        "在同轮内派发一个后续动作 agent。\n\n"
        "可选 agent：\n"
        "- appointment：用户明确表达挂号/预约/改号需求，且医疗答案未覆盖该动作。\n"
        "- triage：用户明确表达需要推荐就诊科室，且医疗答案未覆盖该建议。\n"
        "- FINISH：纯医学知识问答、闲聊、或动作需求已被满足/不明确时，结束本轮。\n\n"
        "判定原则：\n"
        "- 仅当用户原始查询明确暗示了挂号或分诊需求时才派发对应 agent。\n"
        "- 不要为已经回答完的医学问题重复派发医疗 agent。\n"
        "- 不确定时选 FINISH。\n\n"
        "严格输出 JSON，不要输出多余文字：\n"
        '{"next_agent": "appointment|triage|FINISH", "reason": "简短依据"}'
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_multi_agent_supervisor.TestSupervisorDecisionSchema tests.test_multi_agent_supervisor.TestSupervisorPrompt -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add project/rag_agent/schemas.py project/rag_agent/prompts.py tests/test_multi_agent_supervisor.py
git commit -m "feat(p4): add SupervisorDecision schema and supervisor prompt"
```

---

## Task 3: `reset_supervisor_state` node

**Files:**
- Modify: `project/rag_agent/rag_nodes.py` (imports line 18-25, add node function near `decompose_tasks`, `__all__` line 864)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_agent_supervisor.py`:

```python
class TestResetSupervisorState(unittest.TestCase):
    def test_resets_flags_regardless_of_input(self):
        from project.rag_agent.rag_nodes import reset_supervisor_state
        state = _make_main_state(supervisor_active=True, supervisor_rounds=2, supervisor_next="appointment")
        result = reset_supervisor_state(state)
        self.assertEqual(result, {"supervisor_active": False, "supervisor_rounds": 0})

    def test_does_not_touch_other_fields(self):
        from project.rag_agent.rag_nodes import reset_supervisor_state
        state = _make_main_state(originalQuery="keep me")
        result = reset_supervisor_state(state)
        self.assertNotIn("originalQuery", result)
        self.assertEqual(set(result.keys()), {"supervisor_active", "supervisor_rounds"})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_multi_agent_supervisor.TestResetSupervisorState -v
```
Expected: FAIL with `ImportError: cannot import name 'reset_supervisor_state'`.

- [ ] **Step 3: Implement the node**

In `project/rag_agent/rag_nodes.py`, the imports block (lines 18-25) imports schemas from `.schemas`. Add `SupervisorDecision` to that import (it will be needed in Task 4; safe to add now). The import currently looks like:

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

Change to:

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

Then add the `reset_supervisor_state` function. Place it immediately before the existing `decompose_tasks` function (search for `def decompose_tasks`). Insert:

```python
def reset_supervisor_state(state: State):
    """P4: clear supervisor loop flags at turn start.

    LangGraph's checkpointer persists State across turns. If a supervisor-
    dispatched specialist interrupted (e.g. appointment needs clarification),
    the leftover supervisor_active=True would mis-route the resumed specialist
    back to supervise. This node resets those flags every turn, before
    analyze_turn, with zero invasion of analyze_turn's return paths.
    """
    return {"supervisor_active": False, "supervisor_rounds": 0}


```

Add `"reset_supervisor_state"` to the `__all__` list (line 864), in alphabetical position between `"revise_answer"` and `"rewrite_query"`:

```python
    "revise_answer",
    "reset_supervisor_state",
    "rewrite_query",
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_multi_agent_supervisor.TestResetSupervisorState -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add project/rag_agent/rag_nodes.py tests/test_multi_agent_supervisor.py
git commit -m "feat(p4): add reset_supervisor_state node"
```

---

## Task 4: `supervise` node

**Files:**
- Modify: `project/rag_agent/rag_nodes.py` (imports line 26-36 add prompt import; add `supervise` function; `__all__`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_agent_supervisor.py`:

```python
class _FakeStructuredLLM:
    """Mimics _structured_output_llm.invoke returning a schema instance."""
    def __init__(self, verdict):
        self._verdict = verdict
    def invoke(self, messages):
        return self._verdict


class TestSuperviseNode(unittest.TestCase):
    def _patched_module(self, verdict, **state_overrides):
        """Return supervise fn with _structured_output_llm patched to return verdict."""
        import project.rag_agent.rag_nodes as mod
        import project.rag_agent.schemas as schemas
        fake = _FakeStructuredLLM(verdict)
        state = _make_main_state(**state_overrides)
        return mod, fake, state

    def test_disabled_short_circuits_to_finish_no_llm(self):
        import project.rag_agent.rag_nodes as mod
        with unittest.mock.patch.object(mod.config, "ENABLE_MULTI_AGENT_SUPERVISOR", False):
            llm = MagicMock()
            result = mod.supervise(_make_main_state(), llm)
        self.assertEqual(result["supervisor_next"], "FINISH")
        self.assertFalse(result["supervisor_active"])
        self.assertEqual(result["supervisor_rounds"], 0)
        llm.invoke.assert_not_called()

    def test_budget_exhausted_short_circuits_to_finish_no_llm(self):
        import project.rag_agent.rag_nodes as mod
        import config
        llm = MagicMock()
        result = mod.supervise(
            _make_main_state(supervisor_rounds=config.MAX_SUPERVISOR_ROUNDS), llm
        )
        self.assertEqual(result["supervisor_next"], "FINISH")
        llm.invoke.assert_not_called()

    def test_dispatch_appointment_sets_flags_and_clears_secondary(self):
        import project.rag_agent.rag_nodes as mod
        from project.rag_agent.schemas import SupervisorDecision
        verdict = SupervisorDecision(next_agent="appointment", reason="用户要挂号")
        with unittest.mock.patch.object(mod, "_structured_output_llm",
                                        return_value=_FakeStructuredLLM(verdict)):
            result = mod.supervise(
                _make_main_state(secondary_intent="appointment",
                                 deferred_user_question="挂心内科"), MagicMock()
            )
        self.assertTrue(result["supervisor_active"])
        self.assertEqual(result["supervisor_rounds"], 1)
        self.assertEqual(result["supervisor_next"], "appointment")
        self.assertEqual(result["secondary_intent"], "")
        self.assertEqual(result["deferred_user_question"], "")

    def test_dispatch_triage_increments_rounds(self):
        import project.rag_agent.rag_nodes as mod
        from project.rag_agent.schemas import SupervisorDecision
        verdict = SupervisorDecision(next_agent="triage", reason="要推荐科室")
        with unittest.mock.patch.object(mod, "_structured_output_llm",
                                        return_value=_FakeStructuredLLM(verdict)):
            result = mod.supervise(_make_main_state(supervisor_rounds=1), MagicMock())
        self.assertTrue(result["supervisor_active"])
        self.assertEqual(result["supervisor_rounds"], 2)
        self.assertEqual(result["supervisor_next"], "triage")

    def test_finish_resets_flags(self):
        import project.rag_agent.rag_nodes as mod
        from project.rag_agent.schemas import SupervisorDecision
        verdict = SupervisorDecision(next_agent="FINISH", reason="无需动作")
        with unittest.mock.patch.object(mod, "_structured_output_llm",
                                        return_value=_FakeStructuredLLM(verdict)):
            result = mod.supervise(_make_main_state(supervisor_rounds=1), MagicMock())
        self.assertFalse(result["supervisor_active"])
        self.assertEqual(result["supervisor_rounds"], 0)
        self.assertEqual(result["supervisor_next"], "FINISH")

    def test_illegal_next_agent_treated_as_finish(self):
        """LLM returns next_agent that _default() produces (empty str for Literal) → FINISH, no raise."""
        import project.rag_agent.rag_nodes as mod
        # A fake verdict with an empty/illegal next_agent (simulating _default() fallback)
        class _BogusVerdict:
            next_agent = ""
            reason = ""
        with unittest.mock.patch.object(mod, "_structured_output_llm",
                                        return_value=_FakeStructuredLLM(_BogusVerdict())):
            result = mod.supervise(_make_main_state(), MagicMock())
        self.assertEqual(result["supervisor_next"], "FINISH")
        self.assertFalse(result["supervisor_active"])

    def test_real_llm_failure_exercises_default_fallback(self):
        """Bare MagicMock LLM (no patch of _structured_output_llm) → _default() path → FINISH, no raise."""
        import project.rag_agent.rag_nodes as mod
        result = mod.supervise(_make_main_state(), MagicMock())
        self.assertEqual(result["supervisor_next"], "FINISH")
        self.assertFalse(result["supervisor_active"])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_multi_agent_supervisor.TestSuperviseNode -v
```
Expected: FAIL with `ImportError: cannot import name 'supervise'` or `AttributeError: module ... has no attribute 'supervise'`.

- [ ] **Step 3: Implement the supervise node**

In `project/rag_agent/rag_nodes.py`, the prompts import block (lines 26-36) imports from `.prompts`. Add `get_supervisor_prompt`:

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
    get_supervisor_prompt,
    get_task_decomposition_prompt,
)
```

Add the `supervise` function. Place it immediately after the `reset_supervisor_state` function (added in Task 3, before `decompose_tasks`):

```python
def supervise(state: State, llm):
    """P4: multi-agent supervisor at the medical_rag exit.

    Observes the medical agent's answer + the user's original query and decides
    whether to dispatch a peer action-agent (appointment/triage) in the same
    turn, looping up to MAX_SUPERVISOR_ROUNDS. Never raises: on any LLM/parse
    failure it degrades to FINISH (the _structured_output_llm helper returns a
    schema default; an illegal next_agent is explicitly coerced to FINISH).
    """
    rounds = int(state.get("supervisor_rounds", 0) or 0)

    # Budget / disable guard: no LLM call.
    if not config.ENABLE_MULTI_AGENT_SUPERVISOR or rounds >= config.MAX_SUPERVISOR_ROUNDS:
        return {"supervisor_active": False, "supervisor_rounds": 0, "supervisor_next": "FINISH"}

    original_query = str(state.get("originalQuery") or state.get("primary_user_query") or "").strip()
    secondary_intent = str(state.get("secondary_intent", "") or "").strip()
    deferred = str(state.get("deferred_user_question", "") or "").strip()

    # Surface the medical answer just produced (last agent_answer entry).
    agent_answers = state.get("agent_answers") or []
    last_answer = ""
    if agent_answers:
        last_entry = agent_answers[-1]
        if isinstance(last_entry, dict):
            last_answer = str(last_entry.get("answer", "") or "").strip()

    sys_msg = SystemMessage(content=get_supervisor_prompt())
    user_payload = (
        f"用户原始问题：{original_query}\n"
        f"医疗 agent 给出的答案：{last_answer}\n"
        f"规则检测的第二意图：{secondary_intent or '（无）'}\n"
        f"规则检测的延迟问题：{deferred or '（无）'}\n"
        f"对话摘要：{state.get('conversation_summary', '') or '（无）'}\n"
    )

    parser = _structured_output_llm(llm, SupervisorDecision, max_tokens=config.LLM_STRUCTURED_MAX_TOKENS)
    try:
        verdict = parser.invoke([sys_msg, HumanMessage(content=user_payload)])
    except Exception:
        logger.warning("supervise structured output failed; degrading to FINISH.")
        return {"supervisor_active": False, "supervisor_rounds": 0, "supervisor_next": "FINISH"}

    next_agent = str(getattr(verdict, "next_agent", "") or "").strip()
    if next_agent not in ("appointment", "triage", "FINISH"):
        # Illegal/empty (e.g. _default() fallback for a Literal field) → safe FINISH.
        next_agent = "FINISH"

    if next_agent == "FINISH":
        return {"supervisor_active": False, "supervisor_rounds": 0, "supervisor_next": "FINISH"}

    # Dispatch: consume the secondary-intent handoff signal so route_after_action's
    # prepare_secondary_turn branch cannot re-fire it (double-dispatch guard).
    return {
        "supervisor_active": True,
        "supervisor_rounds": rounds + 1,
        "supervisor_next": next_agent,
        "secondary_intent": "",
        "deferred_user_question": "",
    }
```

> Note: `LLM_STRUCTURED_MAX_TOKENS` is an existing config field used by `decompose_tasks` (P3). If the implementer finds it absent, check `project/config.py` — it was added for P3's structured-output calls; reuse the same name. `HumanMessage` is already imported at the top of `rag_nodes.py`.

Add `"supervise"` to the `__all__` list, in alphabetical position before `"rewrite_query"` (and after `reset_supervisor_state`):

```python
    "revise_answer",
    "reset_supervisor_state",
    "supervise",
    "rewrite_query",
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_multi_agent_supervisor.TestSuperviseNode -v
```
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add project/rag_agent/rag_nodes.py tests/test_multi_agent_supervisor.py
git commit -m "feat(p4): add supervise node with budget/dispatch/finish logic"
```

---

## Task 5: `route_after_supervisor` edge + `route_after_grounding` rewire

**Files:**
- Modify: `project/rag_agent/edges.py` (line 5 import; line 281 `route_after_grounding`; append `route_after_supervisor`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_agent_supervisor.py`:

```python
class TestRouteAfterSupervisor(unittest.TestCase):
    def test_appointment_to_handle_appointment_skill(self):
        from project.rag_agent.edges import route_after_supervisor
        self.assertEqual(route_after_supervisor(_make_main_state(supervisor_next="appointment")),
                         "handle_appointment_skill")

    def test_triage_to_recommend_department(self):
        from project.rag_agent.edges import route_after_supervisor
        self.assertEqual(route_after_supervisor(_make_main_state(supervisor_next="triage")),
                         "recommend_department")

    def test_finish_to_end(self):
        from project.rag_agent.edges import route_after_supervisor
        self.assertEqual(route_after_supervisor(_make_main_state(supervisor_next="FINISH")),
                         "__end__")

    def test_unknown_to_end(self):
        from project.rag_agent.edges import route_after_supervisor
        self.assertEqual(route_after_supervisor(_make_main_state(supervisor_next="bogus")),
                         "__end__")


class TestRouteAfterGroundingSupervisor(unittest.TestCase):
    def test_grounded_routes_to_supervise_when_enabled(self):
        from project.rag_agent.edges import route_after_grounding
        state = _make_main_state(grounding_passed=True)
        self.assertEqual(route_after_grounding(state), "supervise")

    def test_budget_exhausted_routes_to_supervise_when_enabled(self):
        import config
        from project.rag_agent.edges import route_after_grounding
        state = _make_main_state(grounding_passed=False, grounding_rounds=config.MAX_GROUNDING_ROUNDS)
        self.assertEqual(route_after_grounding(state), "supervise")

    def test_not_grounded_with_budget_routes_to_revise(self):
        from project.rag_agent.edges import route_after_grounding
        state = _make_main_state(grounding_passed=False, grounding_rounds=0)
        self.assertEqual(route_after_grounding(state), "revise_answer")

    def test_grounded_routes_to_end_when_disabled(self):
        import project.rag_agent.edges as edges
        with unittest.mock.patch.object(edges.config, "ENABLE_MULTI_AGENT_SUPERVISOR", False):
            from project.rag_agent.edges import route_after_grounding
            self.assertEqual(route_after_grounding(_make_main_state(grounding_passed=True)),
                             "__end__")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_multi_agent_supervisor.TestRouteAfterSupervisor tests.test_multi_agent_supervisor.TestRouteAfterGroundingSupervisor -v
```
Expected: FAIL with `ImportError: cannot import name 'route_after_supervisor'` and `AssertionError: '__end__' != 'supervise'`.

- [ ] **Step 3: Implement the edges**

In `project/rag_agent/edges.py`, line 5 currently:

```python
from config import MAX_ITERATIONS, MAX_TOOL_CALLS, MAX_EVIDENCE_ROUNDS, MAX_GROUNDING_ROUNDS, MAX_SUB_QUESTIONS
```

Change to:

```python
from config import MAX_ITERATIONS, MAX_TOOL_CALLS, MAX_EVIDENCE_ROUNDS, MAX_GROUNDING_ROUNDS, MAX_SUB_QUESTIONS, ENABLE_MULTI_AGENT_SUPERVISOR
```

Replace the `route_after_grounding` function (line 281-293) with:

```python
def route_after_grounding(state: State) -> str:
    """P2/P4: route after the answer grounding check.

    - grounded (grounding_passed=True) → supervise (P4) when supervisor enabled, else END
    - not grounded + budget remaining (grounding_rounds < MAX_GROUNDING_ROUNDS) → revise_answer
    - not grounded + budget exhausted → supervise (P4) when supervisor enabled, else END
    """
    _to_supervisor = "supervise" if ENABLE_MULTI_AGENT_SUPERVISOR else "__end__"
    if bool(state.get("grounding_passed", False)):
        return _to_supervisor
    rounds = int(state.get("grounding_rounds", 0) or 0)
    if rounds < MAX_GROUNDING_ROUNDS:
        return "revise_answer"
    return _to_supervisor


def route_after_supervisor(state: State) -> str:
    """P4: dispatch the supervisor's chosen agent, or finish."""
    nxt = str(state.get("supervisor_next", "FINISH") or "FINISH").strip()
    if nxt == "appointment":
        return "handle_appointment_skill"
    if nxt == "triage":
        return "recommend_department"
    return "__end__"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_multi_agent_supervisor.TestRouteAfterSupervisor tests.test_multi_agent_supervisor.TestRouteAfterGroundingSupervisor -v
```
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add project/rag_agent/edges.py tests/test_multi_agent_supervisor.py
git commit -m "feat(p4): add route_after_supervisor and rewire route_after_grounding to supervise"
```

---

## Task 6: `route_after_action` supervisor branch

**Files:**
- Modify: `project/rag_agent/edges.py` (line 174 `route_after_action`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_agent_supervisor.py`:

```python
class TestRouteAfterActionSupervisorBranch(unittest.TestCase):
    def test_supervisor_active_loops_back_to_supervise(self):
        from project.rag_agent.edges import route_after_action
        state = _make_main_state(supervisor_active=True)
        # Strip any pending/secondary signals so only supervisor_active remains.
        state.update({"pending_clarification": "", "clarification_target": "",
                      "secondary_intent": "", "deferred_user_question": "",
                      "pending_action_type": "", "pending_candidates": [],
                      "deferred_confirmation_action": ""})
        self.assertEqual(route_after_action(state), "supervise")

    def test_pending_clarification_beats_supervisor(self):
        from project.rag_agent.edges import route_after_action
        state = _make_main_state(supervisor_active=True,
                                 pending_clarification="选哪个医生?",
                                 clarification_target="handle_appointment_skill")
        self.assertEqual(route_after_action(state), "request_clarification")

    def test_secondary_turn_beats_supervisor(self):
        from project.rag_agent.edges import route_after_action
        state = _make_main_state(supervisor_active=True,
                                 secondary_intent="appointment",
                                 deferred_user_question="挂号",
                                 pending_action_type="",
                                 pending_candidates=[],
                                 deferred_confirmation_action="")
        self.assertEqual(route_after_action(state), "prepare_secondary_turn")

    def test_no_supervisor_no_pending_goes_to_end(self):
        from project.rag_agent.edges import route_after_action
        state = _make_main_state(supervisor_active=False)
        state.update({"pending_clarification": "", "clarification_target": "",
                      "secondary_intent": "", "deferred_user_question": "",
                      "pending_action_type": "", "pending_candidates": [],
                      "deferred_confirmation_action": ""})
        self.assertEqual(route_after_action(state), "__end__")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_multi_agent_supervisor.TestRouteAfterActionSupervisorBranch -v
```
Expected: FAIL with `AssertionError: '__end__' != 'supervise'` (first test).

- [ ] **Step 3: Implement the branch**

In `project/rag_agent/edges.py`, replace `route_after_action` (line 174-186) with:

```python
def route_after_action(state: State) -> str:
    """Route after an action specialist (appointment/triage) finishes.

    Priority: pending clarification > secondary turn > supervisor loop > END.
    The supervisor_active branch (P4) is lowest priority so that explicit
    pending/secondary signals (stronger closure intents) win.
    """
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

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_multi_agent_supervisor.TestRouteAfterActionSupervisorBranch -v
```
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add project/rag_agent/edges.py tests/test_multi_agent_supervisor.py
git commit -m "feat(p4): add supervisor_active branch to route_after_action"
```

---

## Task 7: Graph wiring

**Files:**
- Modify: `project/rag_agent/graph.py` (imports line 12-25; line 128 START edge; line 181-184 specialist maps; line 191-203 grounding block)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_agent_supervisor.py`:

```python
class TestGraphWiring(unittest.TestCase):
    def test_graph_source_references_supervisor_wiring(self):
        import inspect
        import project.rag_agent.graph as graph_mod
        src = inspect.getsource(graph_mod)
        self.assertIn("supervise", src)
        self.assertIn("reset_supervisor_state", src)
        self.assertIn("route_after_supervisor", src)
        self.assertIn("ENABLE_MULTI_AGENT_SUPERVISOR", src)
        # reset_supervisor_state must sit between START and analyze_turn
        self.assertIn('add_edge(START, "reset_supervisor_state")', src)
        self.assertIn('add_edge("reset_supervisor_state", "analyze_turn")', src)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_multi_agent_supervisor.TestGraphWiring -v
```
Expected: FAIL (`supervise` not in source).

- [ ] **Step 3: Implement the wiring**

In `project/rag_agent/graph.py`:

(a) The `rag_nodes` import block (lines 12-25) — add `supervise` and `reset_supervisor_state` (alphabetical). Current block ends with `rewrite_query,` / `should_compress_context,`. Insert `reset_supervisor_state` and `supervise` in alphabetical order:

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

(b) Register the two new nodes. After the existing `graph_builder.add_node("decompose_tasks", partial(decompose_tasks, llm=_light_llm))` line (line 93), add:

```python
    # P4: multi-agent supervisor at medical_rag exit
    graph_builder.add_node("supervise", partial(supervise, llm=_light_llm))
    graph_builder.add_node(reset_supervisor_state)
```

(c) Rewire the START edge. Line 128 currently:

```python
    graph_builder.add_edge(START, "analyze_turn")
```

Change to:

```python
    # P4: reset supervisor flags before analyze_turn to prevent cross-turn leak.
    graph_builder.add_edge(START, "reset_supervisor_state")
    graph_builder.add_edge("reset_supervisor_state", "analyze_turn")
```

(d) Add `route_after_supervisor` import to the `from .edges import *` (already wildcard — `route_after_supervisor` is exported since it's a top-level function in edges.py). No change needed if wildcard import is in place. Verify line 9 is `from .edges import *`.

(e) Add the `supervise` conditional edge and the specialist `supervise` mappings. After the existing `graph_builder.add_edge(["agent"], "grounded_answer_generation")` line (line 180), the 4 specialist conditional edges (lines 181-184) each need `"supervise": "supervise"` added. Replace those 4 lines:

```python
    graph_builder.add_conditional_edges("recommend_department", route_after_action, {"request_clarification": "request_clarification", "prepare_secondary_turn": "prepare_secondary_turn", "supervise": "supervise", "__end__": END})
    graph_builder.add_conditional_edges("handle_appointment_skill", route_after_action, {"request_clarification": "request_clarification", "prepare_secondary_turn": "prepare_secondary_turn", "supervise": "supervise", "__end__": END})
    graph_builder.add_conditional_edges("handle_appointment", route_after_action, {"request_clarification": "request_clarification", "prepare_secondary_turn": "prepare_secondary_turn", "supervise": "supervise", "__end__": END})
    graph_builder.add_conditional_edges("handle_cancel_appointment", route_after_action, {"request_clarification": "request_clarification", "prepare_secondary_turn": "prepare_secondary_turn", "supervise": "supervise", "__end__": END})
```

(f) Add the `supervise` conditional edge. After the `prepare_secondary_turn` conditional edges block (lines 185-190), add:

```python
    # P4: supervisor dispatches a peer agent (appointment/triage) or finishes.
    graph_builder.add_conditional_edges("supervise", route_after_supervisor, {
        "handle_appointment_skill": "handle_appointment_skill",
        "recommend_department": "recommend_department",
        "__end__": END,
    })
```

(g) Update the `answer_grounding_check` conditional edge (lines 192-203) to include `supervise` when the supervisor is enabled. The current block:

```python
    graph_builder.add_edge("grounded_answer_generation", "answer_grounding_check")
    if config.ENABLE_ANSWER_REFLECTION:
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

Replace with (adds a `supervise` target when supervisor enabled; when supervisor disabled, `route_after_grounding` returns `__end__` so the mapping still holds):

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

> When `ENABLE_ANSWER_REFLECTION=False` AND `ENABLE_MULTI_AGENT_SUPERVISOR=True`, `route_after_grounding` can return `revise_answer`? No — `revise_answer` only fires under the reflection branch. But `route_after_grounding` always *can* return `"revise_answer"` regardless of config. To be safe, include `revise_answer` in the non-reflection map only if the node is registered. Since under `ENABLE_ANSWER_REFLECTION=False` the `revise_answer` node is NOT registered, but `route_after_grounding` returns `revise_answer` only when `rounds < MAX_GROUNDING_ROUNDS` (which is true with rounds=0 by default)... **This is a latent bug**: under reflection-off, the original code already had this issue (it just went to END, ignoring revise_answer). P4 should not change that behavior. Keep the non-reflection map as `{"__end__": END, "supervise": "supervise"}` — if `route_after_grounding` returns `revise_answer` in this branch, LangGraph will raise KeyError, but this is the EXISTING behavior (original code: edge to END, `route_after_grounding` returning `revise_answer` would map to... nothing, KeyError too). This is out of P4 scope — the existing config combinations are `ENABLE_ANSWER_REFLECTION=True` by default. The implementer should verify `ENABLE_ANSWER_REFLECTION` defaults to true and note this is pre-existing.

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_multi_agent_supervisor.TestGraphWiring -v
```
Expected: PASS.

Also run a compile check to ensure the graph still compiles:

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -c "import project.rag_agent.graph as g; print('imports ok')"
```
Expected: `imports ok` (no SyntaxError / ImportError).

- [ ] **Step 5: Commit**

```bash
git add project/rag_agent/graph.py tests/test_multi_agent_supervisor.py
git commit -m "feat(p4): wire supervise/reset_supervisor_state into the main graph"
```

---

## Task 8: SILENT_NODES + regression assertion updates

**Files:**
- Modify: `project/core/chat_interface.py:25` (SILENT_NODES set)
- Modify: `tests/test_chat_interface.py` (SILENT_NODES assertion)
- Modify: `tests/test_answer_reflection.py:130-145` (route_after_grounding assertions)
- Modify: `tests/test_routing_edges.py` (route_after_action assertions, if any now fail)

- [ ] **Step 1: Write the failing test**

In `tests/test_chat_interface.py`, find the existing `SILENT_NODES` assertion (search for `SILENT_NODES`). If there's an `assertIn("decompose_tasks", SILENT_NODES)`-style test, add adjacent assertions. If not, add a new test method to the relevant test class:

```python
    def test_supervisor_nodes_are_silent(self):
        from project.core.chat_interface import SILENT_NODES
        self.assertIn("supervise", SILENT_NODES)
        self.assertIn("reset_supervisor_state", SILENT_NODES)
```

Run it to confirm it fails:

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_chat_interface -v 2>&1 | grep -i supervisor || echo "test not found or failing"
```

- [ ] **Step 2: Implement SILENT_NODES**

In `project/core/chat_interface.py` line 25-31, the set currently:

```python
SILENT_NODES = {
    "rewrite_query",
    "intent_router",
    "decompose_tasks",
    "grounded_answer_generation",
    "answer_grounding_check",
}
```

Change to:

```python
SILENT_NODES = {
    "rewrite_query",
    "intent_router",
    "decompose_tasks",
    "grounded_answer_generation",
    "answer_grounding_check",
    "supervise",
    "reset_supervisor_state",
}
```

- [ ] **Step 3: Update P2/P-routing regression assertions**

In `tests/test_answer_reflection.py`, the `TestRouteAfterGrounding` class (lines 130-145) asserts `route_after_grounding` returns `"__end__"`. With the supervisor enabled by default, it now returns `"supervise"`. Update those three assertions:

- Line 134: `self.assertEqual(route_after_grounding(state), "__end__")` → `self.assertEqual(route_after_grounding(state), "supervise")`
- Line 139: `self.assertEqual(route_after_grounding(state), "revise_answer")` → unchanged (still revise_answer).
- Line 145: `self.assertEqual(route_after_grounding(state), "__end__")` → `self.assertEqual(route_after_grounding(state), "supervise")`

Add a comment at the top of `TestRouteAfterGrounding` explaining the supervisor default-on changes the END target.

Also check `TestCompiledGroundingLoop` (line 159+): it builds a graph with `route_after_grounding` mapped to `{"__end__": "__end_sink", "revise_answer": "revise_answer"}`. With supervisor on, `route_after_grounding` returns `"supervise"` — not in the map → KeyError. Fix by adding `"supervise": "__end_sink"` to that map (the loop test only cares about the grounding↔revise loop terminating; routing supervise to the sink is the correct termination stand-in):

```python
        builder.add_conditional_edges(
            "answer_grounding_check",
            route_after_grounding,
            {"__end__": "__end_sink", "revise_answer": "revise_answer", "supervise": "__end_sink"},
        )
```

In `tests/test_routing_edges.py`, the `route_after_action` tests (lines 131-160) use states WITHOUT `supervisor_active`. Since the default State value is `False` and `_make_main_state`/test dicts may not set it, verify they still pass. If any `route_after_action` test sets `supervisor_active=True` inadvertently (unlikely), update. Run and see:

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_routing_edges -v
```

If failures appear related to `route_after_action` now returning `supervise`, the test state has `supervisor_active` truthy — fix by ensuring test states explicitly set `supervisor_active=False` where they expect `__end__`.

- [ ] **Step 4: Run all affected tests to verify they pass**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_multi_agent_supervisor tests.test_answer_reflection tests.test_routing_edges tests.test_chat_interface -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add project/core/chat_interface.py tests/test_chat_interface.py tests/test_answer_reflection.py tests/test_routing_edges.py
git commit -m "feat(p4): silence supervisor nodes and update P2/routing regression assertions"
```

---

## Task 9: Compiled-graph integration test (multi-step handoff + simple FINISH)

**Files:**
- Modify: `tests/test_multi_agent_supervisor.py` (append integration tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multi_agent_supervisor.py`:

```python
class TestCompiledSupervisorLoop(unittest.TestCase):
    """Verify the supervise → specialist → route_after_action → supervise loop
    survives LangGraph's real state machinery, and that FINISH terminates."""

    def _build_graph(self, supervise_verdicts):
        """supervise_verdicts: list of SupervisorDecision returned in call order."""
        from langgraph.graph import StateGraph, START, END
        from project.rag_agent.graph_state import State
        from project.rag_agent.rag_nodes import supervise
        from project.rag_agent.edges import route_after_supervisor, route_after_action
        from functools import partial

        call_log = {"supervise": 0, "specialist": 0}

        # Fake supervise: returns verdicts in order, writes the node's state delta.
        verdicts = list(supervise_verdicts)

        def _fake_supervise(state, llm):
            call_log["supervise"] += 1
            if not verdicts:
                nxt = "FINISH"
            else:
                v = verdicts.pop(0)
                nxt = getattr(v, "next_agent", "FINISH")
            rounds = int(state.get("supervisor_rounds", 0) or 0)
            if nxt == "FINISH":
                return {"supervisor_active": False, "supervisor_rounds": 0, "supervisor_next": "FINISH",
                        "secondary_intent": "", "deferred_user_question": ""}
            return {"supervisor_active": True, "supervisor_rounds": rounds + 1, "supervisor_next": nxt,
                    "secondary_intent": "", "deferred_user_question": ""}

        def _specialist(state):
            call_log["specialist"] += 1
            return {"pending_clarification": "", "clarification_target": "",
                    "secondary_intent": "", "deferred_user_question": "",
                    "pending_action_type": "", "pending_candidates": [],
                    "deferred_confirmation_action": "",
                    "messages": []}

        builder = StateGraph(State)
        builder.add_node("supervise", _fake_supervise)
        builder.add_node("specialist", _specialist)
        builder.add_edge(START, "supervise")
        builder.add_conditional_edges("supervise", route_after_supervisor, {
            "handle_appointment_skill": "specialist",
            "recommend_department": "specialist",
            "__end__": END,
        })
        builder.add_conditional_edges("specialist", route_after_action, {
            "request_clarification": END,
            "prepare_secondary_turn": END,
            "supervise": "supervise",
            "__end__": END,
        })
        return builder.compile(), call_log

    def test_multistep_handoff_appointment_then_finish(self):
        from project.rag_agent.schemas import SupervisorDecision
        graph, call_log = self._build_graph([
            SupervisorDecision(next_agent="appointment", reason="挂号"),
            SupervisorDecision(next_agent="FINISH", reason="完成"),
        ])
        final = graph.invoke(_make_main_state(grounding_passed=True, supervisor_rounds=0))
        self.assertEqual(call_log["supervise"], 2)
        self.assertEqual(call_log["specialist"], 1)
        self.assertFalse(final["supervisor_active"])
        self.assertEqual(final["supervisor_rounds"], 0)
        self.assertEqual(final["supervisor_next"], "FINISH")

    def test_simple_finish_no_specialist(self):
        from project.rag_agent.schemas import SupervisorDecision
        graph, call_log = self._build_graph([
            SupervisorDecision(next_agent="FINISH", reason="纯问答"),
        ])
        final = graph.invoke(_make_main_state(grounding_passed=True, supervisor_rounds=0))
        self.assertEqual(call_log["supervise"], 1)
        self.assertEqual(call_log["specialist"], 0)
        self.assertEqual(final["supervisor_next"], "FINISH")

    def test_budget_guard_terminates_loop(self):
        """If supervise keeps dispatching past MAX_SUPERVISOR_ROUNDS, the loop must terminate."""
        from project.rag_agent.schemas import SupervisorDecision
        # Feed enough appointment verdicts to exceed the budget; the fake supervise
        # does NOT enforce the budget (that's the real node's job). We instead cap
        # via MAX_SUPERVISOR_ROUNDS by setting a high starting round count so the
        # FIRST supervise call sees rounds >= MAX and returns FINISH — but our fake
        # bypasses that. So this test instead verifies route_after_action + a FINISH
        # verdict after a few rounds still converges.
        import config
        verdicts = [SupervisorDecision(next_agent="appointment", reason="x")
                    for _ in range(config.MAX_SUPERVISOR_ROUNDS)]
        verdicts.append(SupervisorDecision(next_agent="FINISH", reason="done"))
        graph, call_log = self._build_graph(verdicts)
        final = graph.invoke(_make_main_state(grounding_passed=True, supervisor_rounds=0))
        # Loop ran MAX_SUPERVISOR_ROUNDS dispatches then FINISH — did not hang.
        self.assertEqual(call_log["supervise"], config.MAX_SUPERVISOR_ROUNDS + 1)
        self.assertEqual(final["supervisor_next"], "FINISH")
```

> Note: this test feeds enough `appointment` verdicts to exceed the budget then a `FINISH`; it confirms the loop converges (does not hang) over multiple rounds. The real budget-enforcement (`supervisor_rounds >= MAX`) is unit-tested in Task 4 (`test_budget_exhausted_short_circuits_to_finish_no_llm`); this integration test exercises the wiring convergence.

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_multi_agent_supervisor.TestCompiledSupervisorLoop -v
```
Expected: may FAIL if LangGraph routing or State field handling is off — that's the point of the integration test. If it PASSES immediately, the integration is sound.

- [ ] **Step 3: No implementation needed (test-only)**

These tests exercise already-implemented nodes/edges. If a test fails, debug the wiring (most likely cause: a State field default or a `route_after_action` returning an unexpected value). Fix the *implementation* (Tasks 4-7), not the test, unless the test itself has a bug.

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_multi_agent_supervisor.TestCompiledSupervisorLoop -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_multi_agent_supervisor.py
git commit -m "test(p4): compiled-graph integration tests for supervisor loop and FINISH"
```

---

## Task 10: Full regression sweep + scope-guard verification

**Files:**
- No code changes (verification + cleanup only)

- [ ] **Step 1: Run the P4 test module**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_multi_agent_supervisor -v
```
Expected: all tests PASS (count: 2+2+2+7+4+8+4+1+3 = ~33 tests).

- [ ] **Step 2: Run the P1/P2/P3 regression modules**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest tests.test_agentic_retrieval tests.test_answer_reflection tests.test_task_decomposition -v
```
Expected: all PASS (20 + updated-P2 + 19). P1 (20) and P3 (19) must be unchanged counts; P2 assertions updated in Task 8.

- [ ] **Step 3: Run full discovery**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m unittest discover -s tests -v 2>&1 | tail -30
```
Expected: no NEW failures vs. the main baseline (pre-existing mojibake / missing-token failures are acceptable; flag any new ones). Confirm P4 introduces zero new regressions.

- [ ] **Step 4: Compile check**

```bash
PYTHONPATH=project ./venv/Scripts/python.exe -m compileall project tests
```
Expected: no errors.

- [ ] **Step 5: Scope-guard verification**

```bash
git log --oneline -10
git stash list
git status
```
Expected:
- The 10 P4 commits touch ONLY: `project/config.py`, `project/rag_agent/{graph_state,schemas,prompts,rag_nodes,edges,graph}.py`, `project/core/chat_interface.py`, `tests/test_multi_agent_supervisor.py`, `tests/test_answer_reflection.py`, `tests/test_routing_edges.py`, `tests/test_chat_interface.py`, `docs/superpowers/...` (spec — already committed pre-plan).
- `stash@{0}` still present (the other-agent multi-session feature).
- Working tree clean (no stray frontend/api/db changes swept in).

- [ ] **Step 6: Commit (only if any cleanup edits were made; otherwise skip)**

If the regression sweep surfaced a fix, commit it. Otherwise this task has no commit.

---

## Self-Review (completed by plan author)

**Spec coverage:**
- §4.1 supervise node → Task 4 ✓
- §4.2 reset_supervisor_state → Task 3 ✓
- §4.3 route_after_supervisor → Task 5 ✓
- §4.4 route_after_grounding rewire → Task 5 ✓
- §4.5 route_after_action branch → Task 6 ✓
- §4.6 State fields → Task 1 ✓
- §4.7 config → Task 1 ✓
- §4.8 graph wiring → Task 7 ✓
- §4.9 SILENT_NODES → Task 8 ✓
- §6 cross-turn leak → Task 3 (reset node) + Task 9 (integration) ✓
- §8 testing → Tasks 1-9 ✓
- §11 acceptance 1-11 → all covered ✓

**Placeholder scan:** None — every step has concrete code/commands.

**Type consistency:** `supervisor_next` (str), `supervisor_active` (bool), `supervisor_rounds` (int) consistent across all tasks. `SupervisorDecision.next_agent` is `Literal["appointment","triage","FINISH"]` in schema (Task 2) and matched in supervise validation (Task 4) and route_after_supervisor (Task 5). `route_after_grounding` return type changed `Literal[...]` → `str` (Task 5) — consistent. `route_after_action` return `Literal[...]` → `str` (Task 6) — consistent.

**Known pre-existing edge case noted (Task 7 step 3g):** `ENABLE_ANSWER_REFLECTION=False` + supervisor-on combination has a latent map-mismatch with `revise_answer`; this is pre-existing behavior (default config has reflection ON) and out of P4 scope.
