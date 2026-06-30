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
        """dims out of [1,5] coerced into range (defense-in-depth for non-Pydantic verdicts)."""
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


if __name__ == "__main__":
    unittest.main()
