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


if __name__ == "__main__":
    unittest.main()
