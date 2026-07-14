"""Tests for Phase 2 unified turn planner: plan_tasks / dispatch_next_task /
advance_task / completeness_gate, plus the planner routing edges.

plan_tasks is exercised with a mocked structured-output LLM; the drain loop is
stepped through manually (applying the accumulate_or_reset reducer) so the test
doesn't need a live model or skill registry.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from langchain_core.messages import HumanMessage, AIMessage  # noqa: E402


class TestConfigFields(unittest.TestCase):
    def test_planner_config_fields_exist(self):
        import config
        self.assertTrue(hasattr(config, "MAX_PLANNED_TASKS"))
        self.assertGreaterEqual(config.MAX_PLANNED_TASKS, 3)


class TestStateFields(unittest.TestCase):
    def test_planner_state_fields_exist(self):
        from project.rag_agent.graph_state import State
        self.assertIn("planned_tasks", State.__annotations__)
        self.assertIn("task_results", State.__annotations__)


class TestSchemas(unittest.TestCase):
    def test_build_turn_plan_schema_constrains_intent(self):
        from project.rag_agent.schemas import build_turn_plan_schema
        schema = build_turn_plan_schema(["medical_rag", "appointment", "triage"])
        # The dynamic task model's intent field should be a Literal.
        task_field = schema.model_fields["tasks"]
        # tasks is List[DynamicPlannedTask]; inspect the inner model's intent.
        inner = task_field.annotation.__args__[0]
        intent_field = inner.model_fields["intent"]
        self.assertIsNotNone(intent_field.annotation)

    def test_turn_plan_prompt_mentions_json_and_tasks(self):
        from project.rag_agent.prompts import get_turn_planner_prompt
        text = get_turn_planner_prompt([("medical_rag", "health questions")])
        self.assertIn("tasks", text)
        self.assertIn("JSON", text)
        self.assertIn("intent", text)


def _verdict(tasks):
    """Build a TurnPlan-like object the parser.invoke() would return."""
    return SimpleNamespace(reason="test", tasks=[SimpleNamespace(intent=i, query=q) for i, q in tasks])


def _make_state(**extra):
    base = {
        "messages": [HumanMessage(content="挂号皮肤科，顺便问湿疹护理")],
        "primary_user_query": "挂号皮肤科，顺便问湿疹护理",
        "originalQuery": "挂号皮肤科，顺便问湿疹护理",
        "conversation_summary": "",
        "recent_context": "",
        "topic_focus": "",
        "user_memories": "",
        "planned_tasks": [],
        "task_results": [],
    }
    base.update(extra)
    return base


def _apply(state, update):
    """Apply a node's update dict, honoring the accumulate_or_reset reducer for
    list fields (task_results / agent_answers)."""
    if not update:
        return
    for k, v in update.items():
        if k in ("task_results", "agent_answers") and isinstance(v, list):
            if v and any(isinstance(item, dict) and item.get("__reset__") for item in v):
                state[k] = []
            else:
                state[k] = list(state.get(k, [])) + v
        elif k == "messages":
            state[k] = list(state.get(k, [])) + v
        else:
            state[k] = v


class TestPlanTasks(unittest.TestCase):
    def test_compound_yields_multiple_tasks(self):
        from project.rag_agent.planner_nodes import plan_tasks
        parser = MagicMock()
        parser.invoke.return_value = _verdict([
            ("appointment", "挂号皮肤科"),
            ("medical_rag", "湿疹如何护理"),
        ])
        with patch("project.rag_agent.planner_nodes._structured_output_llm", return_value=parser):
            result = plan_tasks(_make_state(), MagicMock())
        self.assertEqual(len(result["planned_tasks"]), 2)
        self.assertEqual([t["id"] for t in result["planned_tasks"]], [0, 1])
        self.assertEqual(result["planned_tasks"][0]["intent"], "appointment")
        self.assertEqual(result["planned_tasks"][1]["query"], "湿疹如何护理")

    def test_single_intent_yields_one_task(self):
        from project.rag_agent.planner_nodes import plan_tasks
        parser = MagicMock()
        parser.invoke.return_value = _verdict([("medical_rag", "高血压怎么控制")])
        with patch("project.rag_agent.planner_nodes._structured_output_llm", return_value=parser):
            result = plan_tasks(_make_state(primary_user_query="高血压怎么控制"), MagicMock())
        self.assertEqual(len(result["planned_tasks"]), 1)

    def test_llm_failure_falls_back_to_single_task(self):
        from project.rag_agent.planner_nodes import plan_tasks
        parser = MagicMock()
        parser.invoke.side_effect = RuntimeError("LLM down")
        with patch("project.rag_agent.planner_nodes._structured_output_llm", return_value=parser):
            result = plan_tasks(_make_state(), MagicMock())
        self.assertEqual(len(result["planned_tasks"]), 1)
        self.assertEqual(result["planned_tasks"][0]["intent"], "medical_rag")

    def test_empty_verdict_falls_back_to_single_task(self):
        from project.rag_agent.planner_nodes import plan_tasks
        parser = MagicMock()
        parser.invoke.return_value = _verdict([])
        with patch("project.rag_agent.planner_nodes._structured_output_llm", return_value=parser):
            result = plan_tasks(_make_state(), MagicMock())
        self.assertEqual(len(result["planned_tasks"]), 1)

    def test_respects_max_planned_tasks(self):
        import config
        from project.rag_agent.planner_nodes import plan_tasks
        parser = MagicMock()
        parser.invoke.return_value = _verdict([
            ("medical_rag", f"问题{i}") for i in range(10)
        ])
        with patch("project.rag_agent.planner_nodes._structured_output_llm", return_value=parser):
            result = plan_tasks(_make_state(), MagicMock())
        self.assertEqual(len(result["planned_tasks"]), config.MAX_PLANNED_TASKS)


class TestDispatchNextTask(unittest.TestCase):
    def _planned(self):
        return [
            {"id": 0, "intent": "appointment", "query": "挂号皮肤科"},
            {"id": 1, "intent": "medical_rag", "query": "湿疹护理"},
            {"id": 2, "intent": "medical_rag", "query": "用药注意"},
        ]

    def test_first_task_staged_without_message_injection(self):
        from project.rag_agent.planner_nodes import dispatch_next_task
        state = _make_state(planned_tasks=self._planned(), task_results=[])
        update = dispatch_next_task(state)
        self.assertEqual(update["intent"], "appointment")
        self.assertEqual(update["primary_user_query"], "挂号皮肤科")
        self.assertEqual(update["originalQuery"], "挂号皮肤科")
        # First task: original message already in history, no injection.
        self.assertNotIn("messages", update)
        # Per-task medical fields reset.
        self.assertEqual(update["sub_questions"], [])
        self.assertEqual(update["agent_answers"], [{"__reset__": True}])

    def test_subsequent_task_staged_with_message_injection(self):
        from project.rag_agent.planner_nodes import dispatch_next_task
        state = _make_state(planned_tasks=self._planned(), task_results=[{"id": 0, "status": "done"}])
        update = dispatch_next_task(state)
        self.assertEqual(update["intent"], "medical_rag")
        self.assertEqual(update["primary_user_query"], "湿疹护理")
        self.assertIn("messages", update)
        self.assertTrue(any(isinstance(m, HumanMessage) and m.content == "湿疹护理" for m in update["messages"]))

    def test_no_undone_task_returns_empty(self):
        from project.rag_agent.planner_nodes import dispatch_next_task
        state = _make_state(planned_tasks=self._planned(),
                            task_results=[{"id": 0}, {"id": 1}, {"id": 2}])
        self.assertEqual(dispatch_next_task(state), {})


class TestAdvanceTask(unittest.TestCase):
    def test_records_lowest_undone_id(self):
        from project.rag_agent.planner_nodes import advance_task
        planned = [{"id": 0, "intent": "appointment"}, {"id": 1, "intent": "medical_rag"}]
        state = _make_state(planned_tasks=planned, task_results=[])
        result = advance_task(state)
        self.assertEqual(result["task_results"], [{"id": 0, "intent": "appointment", "status": "done"}])

    def test_skips_done_records_next(self):
        from project.rag_agent.planner_nodes import advance_task
        planned = [{"id": 0, "intent": "appointment"}, {"id": 1, "intent": "medical_rag"}]
        state = _make_state(planned_tasks=planned, task_results=[{"id": 0, "status": "done"}])
        result = advance_task(state)
        self.assertEqual(result["task_results"], [{"id": 1, "intent": "medical_rag", "status": "done"}])

    def test_no_undone_returns_empty(self):
        from project.rag_agent.planner_nodes import advance_task
        planned = [{"id": 0, "intent": "appointment"}]
        state = _make_state(planned_tasks=planned, task_results=[{"id": 0}])
        self.assertEqual(advance_task(state), {})


class TestCompletenessGate(unittest.TestCase):
    def test_all_done_no_caveat(self):
        from project.rag_agent.planner_nodes import completeness_gate
        planned = [{"id": 0, "query": "A"}, {"id": 1, "query": "B"}]
        state = _make_state(planned_tasks=planned, task_results=[{"id": 0}, {"id": 1}])
        self.assertEqual(completeness_gate(state), {})

    def test_missing_task_appends_caveat(self):
        from project.rag_agent.planner_nodes import completeness_gate
        planned = [{"id": 0, "query": "挂号皮肤科"}, {"id": 1, "query": "湿疹护理"}]
        state = _make_state(planned_tasks=planned, task_results=[{"id": 0}])
        result = completeness_gate(state)
        self.assertEqual(len(result["messages"]), 1)
        self.assertIsInstance(result["messages"][0], AIMessage)
        self.assertIn("湿疹护理", result["messages"][0].content)
        self.assertIn("⚠️", result["messages"][0].content)

    def test_no_planned_tasks_no_caveat(self):
        from project.rag_agent.planner_nodes import completeness_gate
        self.assertEqual(completeness_gate(_make_state(planned_tasks=[])), {})


class TestPlannerEdges(unittest.TestCase):
    def test_route_after_plan_tasks(self):
        from project.rag_agent.edges import route_after_plan_tasks
        self.assertEqual(route_after_plan_tasks({"planned_tasks": [{"id": 0}]}), "dispatch_next_task")
        self.assertEqual(route_after_plan_tasks({"planned_tasks": []}), "completeness_gate")

    def test_route_after_dispatch_by_intent(self):
        from project.rag_agent.edges import route_after_dispatch
        self.assertEqual(route_after_dispatch({"intent": "medical_rag"}), "rewrite_query")
        self.assertEqual(route_after_dispatch({"intent": "appointment"}), "handle_appointment_skill")
        self.assertEqual(route_after_dispatch({"intent": "cancel_appointment"}), "handle_appointment_skill")
        self.assertEqual(route_after_dispatch({"intent": "triage"}), "recommend_department")
        self.assertEqual(route_after_dispatch({"intent": "greeting"}), "__end__")

    def test_route_to_next_or_gate(self):
        from project.rag_agent.edges import route_to_next_or_gate
        planned = [{"id": 0}, {"id": 1}]
        self.assertEqual(route_to_next_or_gate({"planned_tasks": planned, "task_results": [{"id": 0}]}), "dispatch_next_task")
        self.assertEqual(route_to_next_or_gate({"planned_tasks": planned, "task_results": [{"id": 0}, {"id": 1}]}), "completeness_gate")
        self.assertEqual(route_to_next_or_gate({"planned_tasks": [], "task_results": []}), "completeness_gate")


class TestEndToEndDrain(unittest.TestCase):
    """Step through the full drain loop manually: dispatch -> (handler) -> advance
    -> route_to_next_or_gate -> dispatch ... -> completeness_gate."""

    def _planned(self):
        return [
            {"id": 0, "intent": "appointment", "query": "挂号皮肤科"},
            {"id": 1, "intent": "medical_rag", "query": "湿疹护理"},
            {"id": 2, "intent": "medical_rag", "query": "用药注意"},
        ]

    def test_three_tasks_drain_to_completion(self):
        from project.rag_agent.planner_nodes import dispatch_next_task, advance_task, completeness_gate
        from project.rag_agent.edges import route_to_next_or_gate
        state = _make_state(planned_tasks=self._planned(), task_results=[])

        # Task 0
        _apply(state, dispatch_next_task(state))
        self.assertEqual(state["intent"], "appointment")
        _apply(state, advance_task(state))
        self.assertEqual([r["id"] for r in state["task_results"]], [0])
        self.assertEqual(route_to_next_or_gate(state), "dispatch_next_task")

        # Task 1
        _apply(state, dispatch_next_task(state))
        self.assertEqual(state["intent"], "medical_rag")
        self.assertEqual(state["primary_user_query"], "湿疹护理")
        _apply(state, advance_task(state))
        self.assertEqual([r["id"] for r in state["task_results"]], [0, 1])

        # Task 2
        _apply(state, dispatch_next_task(state))
        self.assertEqual(state["primary_user_query"], "用药注意")
        _apply(state, advance_task(state))
        self.assertEqual([r["id"] for r in state["task_results"]], [0, 1, 2])

        # All done -> gate, no caveat.
        self.assertEqual(route_to_next_or_gate(state), "completeness_gate")
        self.assertEqual(completeness_gate(state), {})

    def test_drain_then_gate_flags_missing_task(self):
        """If a handler is skipped (advance never records its task), the gate names it."""
        from project.rag_agent.planner_nodes import dispatch_next_task, advance_task, completeness_gate
        from project.rag_agent.edges import route_to_next_or_gate
        planned = [{"id": 0, "intent": "appointment", "query": "挂号"},
                   {"id": 1, "intent": "medical_rag", "query": "湿疹护理"}]
        state = _make_state(planned_tasks=planned, task_results=[])
        # Task 0 drains normally.
        _apply(state, dispatch_next_task(state))
        _apply(state, advance_task(state))
        # Simulate task 1 being skipped (e.g. handler errored) -> jump to gate.
        gate = completeness_gate(state)
        self.assertEqual(len(gate["messages"]), 1)
        self.assertIn("湿疹护理", gate["messages"][0].content)


class TestResetSupervisorStatePlanner(unittest.TestCase):
    def test_clears_planner_state(self):
        from project.rag_agent.rag_nodes import reset_turn_state
        result = reset_turn_state({"planned_tasks": [{"id": 0}], "task_results": [{"id": 0}]})
        self.assertEqual(result["planned_tasks"], [])
        # task_results cleared via __reset__ sentinel.
        self.assertEqual(result["task_results"], [{"__reset__": True}])


if __name__ == "__main__":
    unittest.main()
