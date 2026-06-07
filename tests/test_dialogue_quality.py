"""Dialogue quality tests for rule-based code paths (no LLM required).

Tests intent routing, follow-up detection, pending state, memory parsing,
context building, compound query splitting.
"""

import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))


class TestIntentRouting(unittest.TestCase):
    """Rule-based intent classification — 20 scenarios covering known routing edges."""

    @staticmethod
    def _clf(q, summary="", recent="", topic=""):
        from rag_agent.routing_nodes import _classify_query_by_rules
        intent, _ = _classify_query_by_rules(q, conversation_summary=summary,
                                              recent_context=recent, topic_focus=topic)
        return intent

    def test_greeting(self):
        self.assertEqual(self._clf("hi"), "greeting")
        self.assertEqual(self._clf("hello"), "greeting")
        self.assertEqual(self._clf("thank you"), "greeting")

    def test_cancel(self):
        # Cancel keywords are Chinese; English falls to LLM
        self.assertEqual(self._clf("cancel my appointment"), "")

    def test_department_question(self):
        self.assertEqual(self._clf("which department for cough"), "triage")

    def test_explicit_appointment(self):
        # Chinese keywords required for appointment detection
        self.assertEqual(self._clf("book an appointment"), "")  # English → LLM

    def test_short_followups_to_llm(self):
        self.assertEqual(self._clf("will it be ok"), "")
        self.assertEqual(self._clf("what should i do"), "")
        self.assertEqual(self._clf("how long"), "")

    def test_symptoms_to_llm(self):
        self.assertEqual(self._clf("headache"), "")
        self.assertEqual(self._clf("coughing for days"), "")

    def test_medication_to_llm(self):
        self.assertEqual(self._clf("what medicine to take"), "")

    def test_short_reply_to_llm(self):
        self.assertEqual(self._clf("ok"), "")
        self.assertEqual(self._clf("yes"), "")


class TestClarificationDetection(unittest.TestCase):
    """Detection of user responses to assistant clarification questions."""

    @staticmethod
    def _check(q):
        from rag_agent.routing_nodes import _looks_like_clarification_response
        return _looks_like_clarification_response(q)

    def test_short_is_clarification(self):
        self.assertTrue(self._check("cardiology"))
        self.assertTrue(self._check("tomorrow"))
        self.assertTrue(self._check("morning"))
        self.assertTrue(self._check("yes"))
        self.assertTrue(self._check("ok"))

    def test_long_question_not_clarification(self):
        self.assertFalse(self._check("what should i do about my headache"))

    def test_greeting_not_clarification(self):
        self.assertFalse(self._check("hello"))

    def test_date_formats(self):
        self.assertTrue(self._check("2026-06-08"))


class TestMemoryParsing(unittest.TestCase):
    """LLM JSON response parsing for memory extraction."""

    @staticmethod
    def _parse(raw):
        from memory.memory_extractor import MemoryExtractor
        return MemoryExtractor._parse_extraction_response(raw)

    def test_valid_json(self):
        r = self._parse('[{"memory_type":"medical","content":"allergy","importance":10}]')
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0]["memory_type"], "medical")

    def test_empty_array(self):
        self.assertEqual(self._parse("[]"), [])

    def test_code_block(self):
        r = self._parse('```json\n[{"memory_type":"fact","content":"teacher","importance":5}]\n```')
        self.assertEqual(len(r), 1)

    def test_plain_text(self):
        self.assertEqual(self._parse("nothing here"), [])

    def test_embedded_array(self):
        r = self._parse('prefix [{"memory_type":"preference","content":"short","importance":4}] suffix')
        self.assertEqual(len(r), 1)

    def test_multiple(self):
        r = self._parse('[{"type":"medical","content":"A","imp":9},{"type":"fact","content":"B","imp":5}]')
        self.assertEqual(len(r), 2)

    def test_malformed(self):
        self.assertEqual(self._parse('{"not": "array"}'), [])

    def test_empty_string(self):
        self.assertEqual(self._parse(""), [])


class TestContextBuilding(unittest.TestCase):
    """Recent context and topic focus helpers."""

    def test_recent_context(self):
        from rag_agent.node_helpers import _build_recent_context
        from langchain_core.messages import HumanMessage, AIMessage
        # 5 messages: builds context from last few exchanges
        ctx = _build_recent_context([
            HumanMessage(content="hello"), AIMessage(content="hi"),
            HumanMessage(content="headache"), AIMessage(content="where"),
            HumanMessage(content="forehead"),
        ])
        self.assertIn("headache", ctx)

    def test_recent_context_two_turns(self):
        from rag_agent.node_helpers import _build_recent_context
        from langchain_core.messages import HumanMessage, AIMessage
        ctx = _build_recent_context([HumanMessage(content="hello"), AIMessage(content="hi")])
        self.assertIn("hello", ctx)

    def test_recent_context_empty(self):
        from rag_agent.node_helpers import _build_recent_context
        self.assertEqual(_build_recent_context([]), "")

    def test_topic_focus_medical(self):
        from rag_agent.node_helpers import _extract_topic_focus
        f = _extract_topic_focus("headache and hypertension", "", {}, "")
        self.assertTrue(f)

    def test_topic_focus_keeps_prior(self):
        from rag_agent.node_helpers import _extract_topic_focus
        f = _extract_topic_focus("ok thanks", "hypertension", {}, "")
        self.assertEqual(f, "hypertension")


class TestCompoundSplitting(unittest.TestCase):
    """Compound request detection."""

    @staticmethod
    def _split(q):
        from rag_agent.routing_nodes import _split_compound_request
        return _split_compound_request(q)

    def test_single(self):
        self.assertEqual(self._split("headache"), ["headache"])

    def test_empty(self):
        self.assertEqual(self._split(""), [])


class TestPendingAction(unittest.TestCase):
    """Pending action state transitions."""

    def test_reset_pending(self):
        from rag_agent.node_helpers import _reset_pending_action_if_needed
        state = {"pending_action_type": "appointment", "pending_candidates": [{}],
                 "pending_confirmation_id": "c1", "pending_action_payload": {"dept": "cardio"}}
        r = _reset_pending_action_if_needed(state)
        self.assertEqual(r["pending_action_type"], "")

    def test_continue_pending(self):
        from rag_agent.routing_nodes import _should_continue_pending_action
        state = {"pending_action_type": "appointment", "pending_candidates": [
            {"schedule_id": "s1", "department": "cardio", "doctor_name": "Dr.Zhang",
             "schedule_date": "2026-06-08", "time_slot": "morning"}],
            "pending_confirmation_id": "c1", "pending_action_payload": {"dept": "cardio"}}
        # appointment update keywords trigger continue
        self.assertTrue(_should_continue_pending_action(state, "book Dr.Zhang tomorrow"))


class TestMemoryPrefilter(unittest.TestCase):
    """Memory extraction pre-filter (avoids LLM calls for trivial messages)."""

    @staticmethod
    def _should_skip(user_msg, assistant_msg=""):
        from unittest.mock import MagicMock
        from memory.memory_extractor import MemoryExtractor
        e = MemoryExtractor(MagicMock(), MagicMock())
        return e._should_skip_extraction(user_msg, assistant_msg)

    def test_skip_greetings(self):
        # Prefilter regex is Chinese-focused; short English greetings match via length/pattern
        self.assertTrue(self._should_skip("hi", "hello"))
        self.assertTrue(self._should_skip("ok", ""))

    def test_skip_short_confirmations(self):
        self.assertTrue(self._should_skip("ok", ""))
        self.assertTrue(self._should_skip("good", ""))

    def test_dont_skip_medical(self):
        self.assertFalse(self._should_skip("headache", "it could be..."))
        self.assertFalse(self._should_skip("i have hypertension", ""))

    def test_dont_skip_appointment(self):
        self.assertFalse(self._should_skip("book cardiology", ""))

    def test_skip_very_short(self):
        self.assertTrue(self._should_skip("hmm", ""))
        self.assertTrue(self._should_skip(".", ""))


class TestRewritePlan(unittest.TestCase):
    """Query planning for retrieval."""

    def test_plan_queries_with_context(self):
        from rag_agent.tools import plan_queries
        q = plan_queries("headache", topic_focus="hypertension")
        self.assertGreater(len(q), 1)

    def test_plan_queries_empty(self):
        from rag_agent.tools import plan_queries
        self.assertEqual(plan_queries(""), [])

    def test_plan_queries_followup(self):
        from rag_agent.tools import plan_queries
        q = plan_queries("then what", recent_context="user has hypertension")
        self.assertGreater(len(q), 1)


class TestSkillRegistry(unittest.TestCase):
    """Skill framework integration."""

    def test_registry_singleton(self):
        from skills.registry import get_skill_registry, reset_skill_registry
        reset_skill_registry()
        r1 = get_skill_registry()
        r2 = get_skill_registry()
        self.assertIs(r1, r2)

    def test_no_match_returns_none(self):
        from skills.registry import get_skill_registry, reset_skill_registry
        reset_skill_registry()
        r = get_skill_registry()
        self.assertIsNone(r.classify_intent("anything", context={"conversation_summary": ""}))


if __name__ == "__main__":
    unittest.main()
