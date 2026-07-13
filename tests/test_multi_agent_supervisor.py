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
        self.assertEqual(result, {"supervisor_active": False, "supervisor_rounds": 0, "deferred_extra_tasks": []})

    def test_does_not_touch_other_fields(self):
        from project.rag_agent.rag_nodes import reset_supervisor_state
        state = _make_main_state(originalQuery="keep me")
        result = reset_supervisor_state(state)
        self.assertNotIn("originalQuery", result)
        # deferred_extra_tasks is cleared every fresh turn so stale compound-drain
        # queues don't bleed into a new user message.
        self.assertEqual(set(result.keys()), {"supervisor_active", "supervisor_rounds", "deferred_extra_tasks"})

    def test_clears_stale_deferred_extras(self):
        """A leftover drain queue from a previous turn is cleared at turn start."""
        from project.rag_agent.rag_nodes import reset_supervisor_state
        state = _make_main_state(deferred_extra_tasks=[{"intent": "medical_rag", "query": "stale"}])
        result = reset_supervisor_state(state)
        self.assertEqual(result["deferred_extra_tasks"], [])


class _FakeStructuredLLM:
    """Mimics _structured_output_llm.invoke returning a schema instance."""
    def __init__(self, verdict):
        self._verdict = verdict
    def invoke(self, messages):
        return self._verdict


class TestSuperviseNode(unittest.TestCase):
    def test_disabled_short_circuits_to_finish_no_llm(self):
        import project.rag_agent.rag_nodes as mod
        # If the guard fires, _structured_output_llm is never called.
        # side_effect=AssertionError proves the guard short-circuited (a missing
        # guard would call _structured_output_llm and fail this test loudly).
        with unittest.mock.patch.object(mod, "_structured_output_llm",
                                        side_effect=AssertionError("guard should have short-circuited")):
            with unittest.mock.patch.object(mod.config, "ENABLE_MULTI_AGENT_SUPERVISOR", False):
                result = mod.supervise(_make_main_state(), MagicMock())
        self.assertEqual(result["supervisor_next"], "FINISH")
        self.assertFalse(result["supervisor_active"])
        self.assertEqual(result["supervisor_rounds"], 0)

    def test_budget_exhausted_short_circuits_to_finish_no_llm(self):
        import project.rag_agent.rag_nodes as mod
        import config
        with unittest.mock.patch.object(mod, "_structured_output_llm",
                                        side_effect=AssertionError("guard should have short-circuited")):
            result = mod.supervise(
                _make_main_state(supervisor_rounds=config.MAX_SUPERVISOR_ROUNDS), MagicMock()
            )
        self.assertEqual(result["supervisor_next"], "FINISH")
        self.assertFalse(result["supervisor_active"])
        self.assertEqual(result["supervisor_rounds"], 0)

    def test_dispatch_appointment_sets_flags_and_clears_secondary(self):
        import project.rag_agent.rag_nodes as mod
        from project.rag_agent.schemas import SupervisorDecision
        verdict = SupervisorDecision(next_agent="appointment", reason="用户要挂号")
        with unittest.mock.patch.object(mod, "_structured_output_llm",
                                        return_value=_FakeStructuredLLM(verdict)):
            result = mod.supervise(
                _make_main_state(secondary_intent="appointment",
                                 deferred_user_question="挂心内科"), MagicMock()
            )
        self.assertTrue(result["supervisor_active"])
        self.assertEqual(result["supervisor_rounds"], 1)
        self.assertEqual(result["supervisor_next"], "appointment")
        self.assertEqual(result["secondary_intent"], "")
        self.assertEqual(result["deferred_user_question"], "")

    def test_dispatch_triage_increments_rounds(self):
        import project.rag_agent.rag_nodes as mod
        from project.rag_agent.schemas import SupervisorDecision
        verdict = SupervisorDecision(next_agent="triage", reason="要推荐科室")
        with unittest.mock.patch.object(mod, "_structured_output_llm",
                                        return_value=_FakeStructuredLLM(verdict)):
            result = mod.supervise(_make_main_state(supervisor_rounds=1), MagicMock())
        self.assertTrue(result["supervisor_active"])
        self.assertEqual(result["supervisor_rounds"], 2)
        self.assertEqual(result["supervisor_next"], "triage")

    def test_finish_resets_flags(self):
        import project.rag_agent.rag_nodes as mod
        from project.rag_agent.schemas import SupervisorDecision
        verdict = SupervisorDecision(next_agent="FINISH", reason="无需动作")
        with unittest.mock.patch.object(mod, "_structured_output_llm",
                                        return_value=_FakeStructuredLLM(verdict)):
            result = mod.supervise(_make_main_state(supervisor_rounds=1), MagicMock())
        self.assertFalse(result["supervisor_active"])
        self.assertEqual(result["supervisor_rounds"], 0)
        self.assertEqual(result["supervisor_next"], "FINISH")

    def test_illegal_next_agent_treated_as_finish(self):
        """LLM returns next_agent that _default() produces (empty str for Literal) → FINISH, no raise."""
        import project.rag_agent.rag_nodes as mod
        class _BogusVerdict:
            next_agent = ""
            reason = ""
        with unittest.mock.patch.object(mod, "_structured_output_llm",
                                        return_value=_FakeStructuredLLM(_BogusVerdict())):
            result = mod.supervise(_make_main_state(), MagicMock())
        self.assertEqual(result["supervisor_next"], "FINISH")
        self.assertFalse(result["supervisor_active"])

    def test_real_llm_failure_exercises_default_fallback(self):
        """Bare MagicMock LLM (no patch of _structured_output_llm) → _default() path → FINISH, no raise."""
        import project.rag_agent.rag_nodes as mod
        result = mod.supervise(_make_main_state(), MagicMock())
        self.assertEqual(result["supervisor_next"], "FINISH")
        self.assertFalse(result["supervisor_active"])


class TestRouteAfterSupervisor(unittest.TestCase):
    def test_appointment_to_handle_appointment_skill(self):
        from project.rag_agent.edges import route_after_supervisor
        self.assertEqual(route_after_supervisor(_make_main_state(supervisor_next="appointment")),
                         "handle_appointment_skill")

    def test_triage_to_recommend_department(self):
        from project.rag_agent.edges import route_after_supervisor
        self.assertEqual(route_after_supervisor(_make_main_state(supervisor_next="triage")),
                         "recommend_department")

    def test_finish_to_end(self):
        from project.rag_agent.edges import route_after_supervisor
        self.assertEqual(route_after_supervisor(_make_main_state(supervisor_next="FINISH")),
                         "__end__")

    def test_unknown_to_end(self):
        from project.rag_agent.edges import route_after_supervisor
        self.assertEqual(route_after_supervisor(_make_main_state(supervisor_next="bogus")),
                         "__end__")


class TestRouteAfterGroundingSupervisor(unittest.TestCase):
    def test_grounded_routes_to_self_eval_when_enabled(self):
        from project.rag_agent.edges import route_after_grounding
        state = _make_main_state(grounding_passed=True)
        self.assertEqual(route_after_grounding(state), "self_eval")

    def test_budget_exhausted_routes_to_self_eval_when_enabled(self):
        import config
        from project.rag_agent.edges import route_after_grounding
        state = _make_main_state(grounding_passed=False, grounding_rounds=config.MAX_GROUNDING_ROUNDS)
        self.assertEqual(route_after_grounding(state), "self_eval")

    def test_not_grounded_with_budget_routes_to_revise(self):
        from project.rag_agent.edges import route_after_grounding
        state = _make_main_state(grounding_passed=False, grounding_rounds=0)
        self.assertEqual(route_after_grounding(state), "revise_answer")

    def test_grounded_routes_to_end_when_both_disabled(self):
        import project.rag_agent.edges as edges
        from project.rag_agent.edges import route_after_grounding
        with unittest.mock.patch.object(edges.config, "ENABLE_MULTI_AGENT_SUPERVISOR", False), \
             unittest.mock.patch.object(edges.config, "ENABLE_SELF_EVAL", False):
            self.assertEqual(route_after_grounding(_make_main_state(grounding_passed=True)), "__end__")

    def test_budget_exhausted_routes_to_end_when_both_disabled(self):
        import config
        import project.rag_agent.edges as edges
        from project.rag_agent.edges import route_after_grounding
        state = _make_main_state(grounding_passed=False, grounding_rounds=config.MAX_GROUNDING_ROUNDS)
        with unittest.mock.patch.object(edges.config, "ENABLE_MULTI_AGENT_SUPERVISOR", False), \
             unittest.mock.patch.object(edges.config, "ENABLE_SELF_EVAL", False):
            self.assertEqual(route_after_grounding(state), "__end__")

    def test_not_grounded_with_budget_routes_to_self_eval_when_reflection_off(self):
        """Regression: reflection-off + supervisor-on must not return revise_answer
        (the revise_answer node isn't registered in that branch). With self_eval on,
        falls through to self_eval (not supervise)."""
        import project.rag_agent.edges as edges
        from project.rag_agent.edges import route_after_grounding
        state = _make_main_state(grounding_passed=False, grounding_rounds=0)
        with unittest.mock.patch.object(edges.config, "ENABLE_ANSWER_REFLECTION", False):
            self.assertEqual(route_after_grounding(state), "self_eval")


class TestRouteAfterActionSupervisorBranch(unittest.TestCase):
    def test_supervisor_active_loops_back_to_supervise(self):
        from project.rag_agent.edges import route_after_action
        state = _make_main_state(supervisor_active=True)
        # Strip any pending/secondary signals so only supervisor_active remains.
        state.update({"pending_clarification": "", "clarification_target": "",
                      "secondary_intent": "", "deferred_user_question": "",
                      "pending_action_type": "", "pending_candidates": [],
                      "deferred_confirmation_action": ""})
        self.assertEqual(route_after_action(state), "supervise")

    def test_pending_clarification_beats_supervisor(self):
        from project.rag_agent.edges import route_after_action
        state = _make_main_state(supervisor_active=True,
                                 pending_clarification="选哪个医生?",
                                 clarification_target="handle_appointment_skill")
        self.assertEqual(route_after_action(state), "request_clarification")

    def test_secondary_turn_beats_supervisor(self):
        # Tests router priority in isolation; supervise clears these signals on
        # dispatch, so this combination only arises if a specialist re-populates them.
        from project.rag_agent.edges import route_after_action
        state = _make_main_state(supervisor_active=True,
                                 secondary_intent="appointment",
                                 deferred_user_question="挂号",
                                 pending_action_type="",
                                 pending_candidates=[],
                                 deferred_confirmation_action="")
        self.assertEqual(route_after_action(state), "prepare_secondary_turn")

    def test_no_supervisor_no_pending_goes_to_end(self):
        from project.rag_agent.edges import route_after_action
        state = _make_main_state(supervisor_active=False)
        state.update({"pending_clarification": "", "clarification_target": "",
                      "secondary_intent": "", "deferred_user_question": "",
                      "pending_action_type": "", "pending_candidates": [],
                      "deferred_confirmation_action": ""})
        self.assertEqual(route_after_action(state), "__end__")


class TestGraphWiring(unittest.TestCase):
    def test_graph_source_references_supervisor_wiring(self):
        import inspect
        import project.rag_agent.graph as graph_mod
        src = inspect.getsource(graph_mod)
        # New nodes registered
        self.assertIn("supervise", src)
        self.assertIn("reset_supervisor_state", src)
        # New edge function used
        self.assertIn("route_after_supervisor", src)
        # Supervisor config flag referenced (gating the wiring)
        self.assertIn("ENABLE_MULTI_AGENT_SUPERVISOR", src)
        # reset_supervisor_state must sit between START and analyze_turn
        self.assertIn('add_edge(START, "reset_supervisor_state")', src)
        self.assertIn('add_edge("reset_supervisor_state", "analyze_turn")', src)
        # The supervise conditional edge maps to the two specialists + END
        self.assertIn('"handle_appointment_skill": "handle_appointment_skill"', src)
        self.assertIn('"recommend_department": "recommend_department"', src)


class TestCompiledSupervisorLoop(unittest.TestCase):
    """Verify the supervise → specialist → route_after_action → supervise loop
    survives LangGraph's real state machinery, and that FINISH terminates."""

    def _build_graph(self, supervise_verdicts):
        """supervise_verdicts: list of SupervisorDecision returned in call order."""
        from langgraph.graph import StateGraph, START, END
        from project.rag_agent.graph_state import State
        from project.rag_agent.edges import route_after_supervisor, route_after_action

        call_log = {"supervise": 0, "specialist": 0}

        # Fake supervise: returns verdicts in order, writes the node's state delta.
        verdicts = list(supervise_verdicts)

        def _fake_supervise(state):
            call_log["supervise"] += 1
            if not verdicts:
                nxt = "FINISH"
            else:
                v = verdicts.pop(0)
                nxt = getattr(v, "next_agent", "FINISH")
            rounds = int(state.get("supervisor_rounds", 0) or 0)
            if nxt == "FINISH":
                return {"supervisor_active": False, "supervisor_rounds": 0, "supervisor_next": "FINISH",
                        "secondary_intent": "", "deferred_user_question": ""}
            return {"supervisor_active": True, "supervisor_rounds": rounds + 1, "supervisor_next": nxt,
                    "secondary_intent": "", "deferred_user_question": ""}

        def _specialist(state):
            call_log["specialist"] += 1
            return {"pending_clarification": "", "clarification_target": "",
                    "secondary_intent": "", "deferred_user_question": "",
                    "pending_action_type": "", "pending_candidates": [],
                    "deferred_confirmation_action": "",
                    "messages": []}

        builder = StateGraph(State)
        builder.add_node("supervise", _fake_supervise)
        builder.add_node("specialist", _specialist)
        builder.add_edge(START, "supervise")
        builder.add_conditional_edges("supervise", route_after_supervisor, {
            "handle_appointment_skill": "specialist",
            "recommend_department": "specialist",
            "__end__": END,
        })
        builder.add_conditional_edges("specialist", route_after_action, {
            "request_clarification": END,
            "prepare_secondary_turn": END,
            "supervise": "supervise",
            "__end__": END,
        })
        return builder.compile(), call_log

    def test_multistep_handoff_appointment_then_finish(self):
        from project.rag_agent.schemas import SupervisorDecision
        graph, call_log = self._build_graph([
            SupervisorDecision(next_agent="appointment", reason="挂号"),
            SupervisorDecision(next_agent="FINISH", reason="完成"),
        ])
        final = graph.invoke(_make_main_state(grounding_passed=True, supervisor_rounds=0))
        self.assertEqual(call_log["supervise"], 2)
        self.assertEqual(call_log["specialist"], 1)
        self.assertFalse(final["supervisor_active"])
        self.assertEqual(final["supervisor_rounds"], 0)
        self.assertEqual(final["supervisor_next"], "FINISH")

    def test_simple_finish_no_specialist(self):
        from project.rag_agent.schemas import SupervisorDecision
        graph, call_log = self._build_graph([
            SupervisorDecision(next_agent="FINISH", reason="纯问答"),
        ])
        final = graph.invoke(_make_main_state(grounding_passed=True, supervisor_rounds=0))
        self.assertEqual(call_log["supervise"], 1)
        self.assertEqual(call_log["specialist"], 0)
        self.assertEqual(final["supervisor_next"], "FINISH")

    def test_long_loop_terminates(self):
        """Verify the compiled loop converges (does not hang) when FINISH is returned after
        MAX_SUPERVISOR_ROUNDS dispatches. The budget guard itself is unit-tested in
        TestSuperviseNode.test_budget_exhausted_short_circuits_to_finish_no_llm."""
        import config
        from project.rag_agent.schemas import SupervisorDecision
        verdicts = [SupervisorDecision(next_agent="appointment", reason="x")
                    for _ in range(config.MAX_SUPERVISOR_ROUNDS)]
        verdicts.append(SupervisorDecision(next_agent="FINISH", reason="done"))
        graph, call_log = self._build_graph(verdicts)
        final = graph.invoke(_make_main_state(grounding_passed=True, supervisor_rounds=0))
        # Loop ran MAX_SUPERVISOR_ROUNDS dispatches then FINISH — did not hang.
        self.assertEqual(call_log["supervise"], config.MAX_SUPERVISOR_ROUNDS + 1)
        self.assertEqual(final["supervisor_next"], "FINISH")


class TestRequestClarificationClearsSupervisor(unittest.TestCase):
    def test_request_clarification_clears_supervisor_flags(self):
        """P4: on clarification-resume, request_clarification must clear supervisor
        flags so the resumed specialist routes to END, not back to supervise."""
        from project.rag_agent.routing_nodes import request_clarification
        state = _make_main_state(supervisor_active=True, supervisor_rounds=2, supervisor_next="appointment")
        result = request_clarification(state)
        self.assertFalse(result["supervisor_active"])
        self.assertEqual(result["supervisor_rounds"], 0)
        # Only clears supervisor flags — no other state touched.
        self.assertEqual(set(result.keys()), {"supervisor_active", "supervisor_rounds"})


if __name__ == "__main__":
    unittest.main()
