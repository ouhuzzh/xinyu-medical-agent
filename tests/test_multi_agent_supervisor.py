"""Tests for P4 multi-agent supervisor: supervise node + route_after_supervisor +
cross-turn reset + route_after_action/route_after_grounding wiring."""
import os
import sys
import unittest
from typing import get_args
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


class TestSupervisorDecisionSchema(unittest.TestCase):
    def test_schema_fields(self):
        from project.rag_agent.schemas import SupervisorDecision
        fields = SupervisorDecision.model_fields
        self.assertIn("next_agent", fields)
        self.assertIn("reason", fields)
        # next_agent must be a Literal of appointment/triage/FINISH
        annot = fields["next_agent"].annotation
        self.assertEqual(set(get_args(annot)), {"appointment", "triage", "FINISH"})

    def test_schema_rejects_invalid_next_agent(self):
        from project.rag_agent.schemas import SupervisorDecision
        from pydantic import ValidationError
        # Valid construction succeeds
        SupervisorDecision(next_agent="appointment", reason="x")
        SupervisorDecision(next_agent="triage", reason="x")
        SupervisorDecision(next_agent="FINISH", reason="x")
        # Invalid value rejected
        with self.assertRaises(ValidationError):
            SupervisorDecision(next_agent="medical_rag", reason="x")


class TestSupervisorPrompt(unittest.TestCase):
    def test_prompt_exists(self):
        from project.rag_agent.prompts import get_supervisor_prompt
        p = get_supervisor_prompt()
        self.assertIn("appointment", p)
        self.assertIn("triage", p)
        self.assertIn("FINISH", p)
        # JSON field names must match the schema (catches prompt/schema drift)
        self.assertIn("next_agent", p)
        self.assertIn("reason", p)
        # Strict-JSON marker present
        self.assertIn("JSON", p)


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


if __name__ == "__main__":
    unittest.main()
