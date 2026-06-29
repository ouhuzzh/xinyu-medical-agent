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
