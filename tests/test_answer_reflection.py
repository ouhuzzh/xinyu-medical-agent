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


class TestStateFields(unittest.TestCase):
    def test_grounding_reflection_fields_exist(self):
        from project.rag_agent.graph_state import State
        defaults = State.__annotations__
        for field in ("grounding_passed", "grounding_critique", "grounding_rounds"):
            self.assertIn(field, defaults, f"State missing field: {field}")


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
