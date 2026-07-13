"""Tests for Phase 1 compound-request queue: connector expansion, whitelist
removal, multi-segment drain queue, terminal drain points, and the missing
sub-question caveat.

These tests exercise the pure routing helpers directly (no skill registry /
embedding model required), plus analyze_turn's queue filling with a mocked
classifier.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from langchain_core.messages import HumanMessage  # noqa: E402


class TestConfigFields(unittest.TestCase):
    def test_compound_queue_config_fields_exist(self):
        import config
        self.assertTrue(hasattr(config, "ENABLE_COMPOUND_QUEUE"))
        self.assertTrue(config.ENABLE_COMPOUND_QUEUE)  # default on
        self.assertTrue(hasattr(config, "MAX_COMPOUND_SEGMENTS"))
        self.assertGreaterEqual(config.MAX_COMPOUND_SEGMENTS, 3)


class TestStateField(unittest.TestCase):
    def test_deferred_extra_tasks_field_exists(self):
        from project.rag_agent.graph_state import State
        self.assertIn("deferred_extra_tasks", State.__annotations__)


class TestSplitCompoundRequest(unittest.TestCase):
    def test_new_connectors_split(self):
        from project.rag_agent.routing_nodes import _split_compound_request
        # 还有 / 对了 / 问一下 are new connectors.
        self.assertEqual(_split_compound_request("挂号皮肤科，还有问湿疹"), ["挂号皮肤科", "问湿疹"])
        self.assertEqual(_split_compound_request("问A，对了问B"), ["问A", "问B"])
        self.assertEqual(_split_compound_request("问A，问一下问B"), ["问A", "问B"])

    def test_three_segments_not_truncated_when_queue_enabled(self):
        """New mode: 3+ segments are all kept (up to MAX_COMPOUND_SEGMENTS)."""
        from project.rag_agent.routing_nodes import _split_compound_request
        segs = _split_compound_request("挂号皮肤科，另外问湿疹，还有问用药")
        self.assertEqual(len(segs), 3)
        self.assertEqual(segs[0], "挂号皮肤科")
        self.assertEqual(segs[2], "问用药")

    def test_legacy_mode_truncates_at_two(self):
        """ENABLE_COMPOUND_QUEUE=false -> original [:2] cap."""
        import config
        from project.rag_agent.routing_nodes import _split_compound_request
        with patch.object(config, "ENABLE_COMPOUND_QUEUE", False):
            segs = _split_compound_request("挂号皮肤科，另外问湿疹，还有问用药")
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs, ["挂号皮肤科", "问湿疹"])

    def test_ran_hou_zai_not_mis_split(self):
        """Regression: '然后再' must match before '然后' (length-desc alternation).

        The old fixed-order regex matched '然后' first, leaving a stray '再'.
        """
        from project.rag_agent.routing_nodes import _split_compound_request
        segs = _split_compound_request("买药然后再问护理")
        self.assertEqual(segs, ["买药", "问护理"])

    def test_single_segment_returned_as_is(self):
        from project.rag_agent.routing_nodes import _split_compound_request
        self.assertEqual(_split_compound_request("高血压怎么办"), ["高血压怎么办"])
        self.assertEqual(_split_compound_request(""), [])


class TestChooseCompoundIntents(unittest.TestCase):
    def test_new_mode_keeps_action_plus_medical(self):
        from project.rag_agent.routing_nodes import _choose_compound_intents
        self.assertEqual(_choose_compound_intents("appointment", "medical_rag"), ("appointment", "medical_rag"))
        self.assertEqual(_choose_compound_intents("cancel_appointment", "medical_rag"), ("cancel_appointment", "medical_rag"))

    def test_new_mode_keeps_medical_plus_action_in_order(self):
        """medical_rag + action: kept in user order (supervisor dispatches action)."""
        from project.rag_agent.routing_nodes import _choose_compound_intents
        self.assertEqual(_choose_compound_intents("medical_rag", "appointment"), ("medical_rag", "appointment"))
        self.assertEqual(_choose_compound_intents("medical_rag", "triage"), ("medical_rag", "triage"))

    def test_new_mode_does_not_split_medical_plus_medical(self):
        """medical_rag + medical_rag -> don't split (decompose handles facets)."""
        from project.rag_agent.routing_nodes import _choose_compound_intents
        self.assertEqual(_choose_compound_intents("medical_rag", "medical_rag"), ("medical_rag", ""))

    def test_new_mode_keeps_previously_dropped_pairs(self):
        """Pairs the legacy whitelist dropped are now kept when action-primary."""
        from project.rag_agent.routing_nodes import _choose_compound_intents
        # appointment + triage was NOT in the legacy whitelist -> would drop triage.
        self.assertEqual(_choose_compound_intents("appointment", "triage"), ("appointment", "triage"))

    def test_legacy_mode_whitelist_drops_non_whitelisted_pair(self):
        import config
        from project.rag_agent.routing_nodes import _choose_compound_intents
        with patch.object(config, "ENABLE_COMPOUND_QUEUE", False):
            # appointment + triage not whitelisted -> second dropped.
            self.assertEqual(_choose_compound_intents("appointment", "triage"), ("appointment", ""))
            # appointment + medical_rag whitelisted -> kept.
            self.assertEqual(_choose_compound_intents("appointment", "medical_rag"), ("appointment", "medical_rag"))
            # medical_rag + appointment -> reordered to appointment-first (legacy).
            self.assertEqual(_choose_compound_intents("medical_rag", "appointment"), ("appointment", "medical_rag"))


class TestPrepareSecondaryTurnDrain(unittest.TestCase):
    def test_drains_first_extra_when_no_immediate_secondary(self):
        from project.rag_agent.routing_nodes import prepare_secondary_turn
        state = {
            "secondary_intent": "",
            "deferred_user_question": "",
            "deferred_extra_tasks": [
                {"intent": "medical_rag", "query": "问用药"},
                {"intent": "medical_rag", "query": "问护理"},
            ],
        }
        update = prepare_secondary_turn(state)
        self.assertEqual(update["intent"], "medical_rag")
        self.assertEqual(update["primary_user_query"], "问用药")
        self.assertEqual(update["secondary_intent"], "")  # consumed
        self.assertEqual(update["deferred_user_question"], "")
        # Queue shortened by one.
        self.assertEqual(len(update["deferred_extra_tasks"]), 1)
        self.assertEqual(update["deferred_extra_tasks"][0]["query"], "问护理")
        # A HumanMessage was injected for the drained question.
        self.assertTrue(any(isinstance(m, HumanMessage) and m.content == "问用药" for m in update["messages"]))

    def test_drain_defaults_empty_intent_to_medical_rag(self):
        from project.rag_agent.routing_nodes import prepare_secondary_turn
        state = {"secondary_intent": "", "deferred_user_question": "",
                 "deferred_extra_tasks": [{"intent": "", "query": "随便问个问题"}]}
        update = prepare_secondary_turn(state)
        self.assertEqual(update["intent"], "medical_rag")

    def test_empty_state_returns_empty(self):
        from project.rag_agent.routing_nodes import prepare_secondary_turn
        self.assertEqual(prepare_secondary_turn({"secondary_intent": "", "deferred_user_question": "", "deferred_extra_tasks": []}), {})

    def test_immediate_secondary_still_works(self):
        """Existing single-secondary path is unchanged."""
        from project.rag_agent.routing_nodes import prepare_secondary_turn
        state = {"secondary_intent": "appointment", "deferred_user_question": "挂呼吸内科"}
        update = prepare_secondary_turn(state)
        self.assertEqual(update["intent"], "appointment")
        self.assertEqual(update["primary_user_query"], "挂呼吸内科")
        self.assertEqual(update["deferred_user_question"], "")
        self.assertNotIn("deferred_extra_tasks", update)  # untouched


class TestRouteAfterActionDrain(unittest.TestCase):
    def test_queue_triggers_prepare_secondary_turn(self):
        from project.rag_agent.edges import route_after_action
        decision = route_after_action({
            "secondary_intent": "",
            "deferred_user_question": "",
            "deferred_extra_tasks": [{"intent": "medical_rag", "query": "问用药"}],
            "pending_clarification": "",
            "pending_action_type": "",
            "pending_candidates": [],
            "deferred_confirmation_action": "",
        })
        self.assertEqual(decision, "prepare_secondary_turn")

    def test_queue_blocked_by_pending_candidates(self):
        from project.rag_agent.edges import route_after_action
        decision = route_after_action({
            "secondary_intent": "",
            "deferred_user_question": "",
            "deferred_extra_tasks": [{"intent": "medical_rag", "query": "问用药"}],
            "pending_clarification": "",
            "pending_action_type": "",
            "pending_candidates": [{"appointment_no": "APT001"}],
            "deferred_confirmation_action": "",
        })
        self.assertEqual(decision, "__end__")

    def test_no_queue_no_secondary_ends(self):
        from project.rag_agent.edges import route_after_action
        decision = route_after_action({
            "secondary_intent": "",
            "deferred_user_question": "",
            "deferred_extra_tasks": [],
            "pending_clarification": "",
            "pending_action_type": "",
            "pending_candidates": [],
            "deferred_confirmation_action": "",
        })
        self.assertEqual(decision, "__end__")


class TestRouteAfterSupervisorDrain(unittest.TestCase):
    def test_finish_with_extras_drains(self):
        from project.rag_agent.edges import route_after_supervisor
        self.assertEqual(
            route_after_supervisor({"supervisor_next": "FINISH", "deferred_extra_tasks": [{"intent": "medical_rag", "query": "q"}]}),
            "prepare_secondary_turn",
        )

    def test_finish_without_extras_ends(self):
        from project.rag_agent.edges import route_after_supervisor
        self.assertEqual(route_after_supervisor({"supervisor_next": "FINISH", "deferred_extra_tasks": []}), "__end__")

    def test_appointment_dispatch_still_works(self):
        from project.rag_agent.edges import route_after_supervisor
        self.assertEqual(route_after_supervisor({"supervisor_next": "appointment", "deferred_extra_tasks": []}), "handle_appointment_skill")


class TestRouteAfterSelfEvalDrain(unittest.TestCase):
    def test_supervisor_off_drains_extras(self):
        import config
        from project.rag_agent.edges import route_after_self_eval
        with patch.object(config, "ENABLE_MULTI_AGENT_SUPERVISOR", False):
            self.assertEqual(
                route_after_self_eval({"deferred_extra_tasks": [{"intent": "medical_rag", "query": "q"}]}),
                "prepare_secondary_turn",
            )

    def test_supervisor_off_no_extras_ends(self):
        import config
        from project.rag_agent.edges import route_after_self_eval
        with patch.object(config, "ENABLE_MULTI_AGENT_SUPERVISOR", False):
            self.assertEqual(route_after_self_eval({"deferred_extra_tasks": []}), "__end__")

    def test_supervisor_on_goes_to_supervise(self):
        import config
        from project.rag_agent.edges import route_after_self_eval
        with patch.object(config, "ENABLE_MULTI_AGENT_SUPERVISOR", True):
            # Even with extras, supervisor-on routes to supervise (drain happens at route_after_supervisor).
            self.assertEqual(
                route_after_self_eval({"deferred_extra_tasks": [{"intent": "medical_rag", "query": "q"}]}),
                "supervise",
            )


class TestMissingSubQuestionCaveat(unittest.TestCase):
    def _helper(self, subs, answers):
        from project.rag_agent.rag_nodes import _build_missing_subquestion_caveat
        return _build_missing_subquestion_caveat({"sub_questions": subs}, answers)

    def test_all_answered_no_caveat(self):
        caveat = self._helper(["问A", "问B"], [{"index": 0, "answer": "ansA"}, {"index": 1, "answer": "ansB"}])
        self.assertEqual(caveat, "")

    def test_missing_index_named_in_caveat(self):
        caveat = self._helper(["问A", "问B", "问C"], [{"index": 0, "answer": "ansA"}, {"index": 2, "answer": "ansC"}])
        self.assertIn("问B", caveat)
        self.assertIn("⚠️", caveat)

    def test_fallback_answer_named_in_caveat(self):
        caveat = self._helper(["问A", "问B"], [{"index": 0, "answer": "ansA"}, {"index": 1, "answer": "Unable to generate an answer."}])
        self.assertIn("问B", caveat)

    def test_empty_answer_named_in_caveat(self):
        caveat = self._helper(["问A", "问B"], [{"index": 0, "answer": "ansA"}, {"index": 1, "answer": ""}])
        self.assertIn("问B", caveat)

    def test_single_subquestion_no_caveat(self):
        """A single (non-compound) question never gets the 'multiple parts' caveat."""
        caveat = self._helper(["问A"], [{"index": 0, "answer": "Unable to generate an answer."}])
        self.assertEqual(caveat, "")

    def test_no_subquestions_no_caveat(self):
        caveat = self._helper([], [{"index": 0, "answer": "ansA"}])
        self.assertEqual(caveat, "")


class TestAnalyzeTurnFillsDrainQueue(unittest.TestCase):
    """analyze_turn fills deferred_extra_tasks for 3+ segment compounds."""

    def _base_state(self, query):
        return {
            "messages": [HumanMessage(content=query)],
            "conversation_summary": "",
            "pending_action_type": "",
            "pending_candidates": [],
            "pending_clarification": "",
            "clarification_target": "",
            "appointment_context": {},
            "recommended_department": "",
            "topic_focus": "",
        }

    def test_three_segment_compound_fills_queue(self):
        from project.rag_agent.routing_nodes import analyze_turn
        # "挂号皮肤科，另外问湿疹，还有问用药" -> 3 segments.
        # first=appointment (L1), second & extra = medical_rag (L1/L2 miss -> default).
        with patch("project.rag_agent.routing_nodes._classify_query_pipeline",
                   side_effect=[
                       ("appointment", 1.0, "l1_keyword"),
                       ("", 0.0, "need_llm"),
                       ("", 0.0, "need_llm"),
                   ]):
            result = analyze_turn(self._base_state("挂号皮肤科，另外问湿疹，还有问用药"))
        self.assertEqual(result["primary_intent"], "appointment")
        self.assertEqual(result["secondary_intent"], "medical_rag")
        self.assertEqual(result["deferred_user_question"], "问湿疹")
        self.assertEqual(len(result["deferred_extra_tasks"]), 1)
        self.assertEqual(result["deferred_extra_tasks"][0]["intent"], "medical_rag")
        self.assertEqual(result["deferred_extra_tasks"][0]["query"], "问用药")

    def test_two_segment_compound_has_empty_queue(self):
        from project.rag_agent.routing_nodes import analyze_turn
        with patch("project.rag_agent.routing_nodes._classify_query_pipeline",
                   side_effect=[("appointment", 1.0, "l1_keyword"), ("", 0.0, "need_llm")]):
            result = analyze_turn(self._base_state("挂号皮肤科，另外问湿疹"))
        self.assertEqual(result["primary_intent"], "appointment")
        self.assertEqual(result["secondary_intent"], "medical_rag")
        self.assertEqual(result["deferred_extra_tasks"], [])


if __name__ == "__main__":
    unittest.main()
