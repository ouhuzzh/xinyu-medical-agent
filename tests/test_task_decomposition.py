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
