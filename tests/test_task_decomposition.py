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


class TestStateFields(unittest.TestCase):
    def test_sub_questions_field_exists(self):
        from project.rag_agent.graph_state import State
        self.assertIn("sub_questions", State.__annotations__)


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

    def test_empty_primary_still_emits_one_send(self):
        """Degenerate: no sub_questions AND no primary → still one Send (question=''), not [].

        Restores the old always-emit-one-Send contract so the graph produces an
        answer rather than silently terminating.
        """
        from project.rag_agent.edges import route_after_query_plan
        state = _make_main_state(rewrittenQuestions=[], originalQuery="", sub_questions=[])
        sends = route_after_query_plan(state)
        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0].arg["question"], "")
        self.assertEqual(sends[0].arg["question_index"], 0)

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
