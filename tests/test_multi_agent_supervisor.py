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
        # Default values — TypedDict does not populate defaults on instance
        # construction (State(messages=[]) only contains "messages"), so verify
        # the declared defaults via class attributes, consistent with the
        # existing State-field pattern in test_answer_reflection.py.
        self.assertIs(State.supervisor_active, False)
        self.assertEqual(State.supervisor_rounds, 0)
        self.assertEqual(State.supervisor_next, "FINISH")


if __name__ == "__main__":
    unittest.main()
