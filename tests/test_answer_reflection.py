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
