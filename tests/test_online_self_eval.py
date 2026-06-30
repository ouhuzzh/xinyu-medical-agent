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


class TestExtractAnswerBody(unittest.TestCase):
    def test_strips_citation_and_confidence_tail(self):
        from project.rag_agent.rag_nodes import _extract_answer_body
        answer = ("一般可以接种，但需先咨询医生。\n\n"
                  "证据强度：`中等`。检索证据为中等强度。\n\n"
                  "参考来源：\n[1] 来源A\n[2] 来源B")
        body = _extract_answer_body(answer)
        self.assertEqual(body, "一般可以接种，但需先咨询医生。")

    def test_strips_version_reminder(self):
        from project.rag_agent.rag_nodes import _extract_answer_body
        answer = "答案是X。\n\n版本提醒：当前命中了较旧资料。"
        self.assertEqual(_extract_answer_body(answer), "答案是X。")

    def test_no_markers_returns_answer_unchanged(self):
        from project.rag_agent.rag_nodes import _extract_answer_body
        answer = "纯回答正文，没有任何附加块。"
        self.assertEqual(_extract_answer_body(answer), answer)

    def test_empty_or_none_returns_empty(self):
        from project.rag_agent.rag_nodes import _extract_answer_body
        self.assertEqual(_extract_answer_body(""), "")
        self.assertEqual(_extract_answer_body(None), "")


class TestRouteAfterSelfEval(unittest.TestCase):
    def test_to_supervise_when_supervisor_enabled(self):
        import project.rag_agent.edges as edges
        from project.rag_agent.edges import route_after_self_eval
        with unittest.mock.patch.object(edges.config, "ENABLE_MULTI_AGENT_SUPERVISOR", True):
            self.assertEqual(route_after_self_eval(_make_main_state()), "supervise")

    def test_to_end_when_supervisor_disabled(self):
        import project.rag_agent.edges as edges
        from project.rag_agent.edges import route_after_self_eval
        with unittest.mock.patch.object(edges.config, "ENABLE_MULTI_AGENT_SUPERVISOR", False):
            self.assertEqual(route_after_self_eval(_make_main_state()), "__end__")


class TestRouteAfterGroundingSelfEval(unittest.TestCase):
    def test_grounded_routes_to_self_eval_when_enabled(self):
        from project.rag_agent.edges import route_after_grounding
        self.assertEqual(route_after_grounding(_make_main_state(grounding_passed=True)), "self_eval")

    def test_budget_exhausted_routes_to_self_eval_when_enabled(self):
        import config
        from project.rag_agent.edges import route_after_grounding
        state = _make_main_state(grounding_passed=False, grounding_rounds=config.MAX_GROUNDING_ROUNDS)
        self.assertEqual(route_after_grounding(state), "self_eval")

    def test_not_grounded_with_budget_routes_to_revise(self):
        from project.rag_agent.edges import route_after_grounding
        state = _make_main_state(grounding_passed=False, grounding_rounds=0)
        self.assertEqual(route_after_grounding(state), "revise_answer")

    def test_grounded_routes_to_supervise_when_self_eval_disabled(self):
        import project.rag_agent.edges as edges
        from project.rag_agent.edges import route_after_grounding
        with unittest.mock.patch.object(edges.config, "ENABLE_SELF_EVAL", False):
            self.assertEqual(route_after_grounding(_make_main_state(grounding_passed=True)), "supervise")

    def test_grounded_routes_to_end_when_both_disabled(self):
        import project.rag_agent.edges as edges
        from project.rag_agent.edges import route_after_grounding
        with unittest.mock.patch.object(edges.config, "ENABLE_SELF_EVAL", False), \
             unittest.mock.patch.object(edges.config, "ENABLE_MULTI_AGENT_SUPERVISOR", False):
            self.assertEqual(route_after_grounding(_make_main_state(grounding_passed=True)), "__end__")


class TestGraphWiring(unittest.TestCase):
    def test_graph_source_references_self_eval_wiring(self):
        import inspect
        import project.rag_agent.graph as graph_mod
        src = inspect.getsource(graph_mod)
        self.assertIn("self_eval", src)
        self.assertIn("route_after_self_eval", src)
        self.assertIn("ENABLE_SELF_EVAL", src)


class TestSelfEvalPersistence(unittest.TestCase):
    def test_route_log_payload_includes_self_eval_fields(self):
        """_persist_route_log must pass self_eval_score + self_eval_details to the store."""
        from project.core.chat_turn_service import ChatTurnService, TurnArtifacts
        captured = {}

        class _FakeStore:
            def save_log(self, payload):
                captured.update(payload)

        svc = ChatTurnService.__new__(ChatTurnService)
        svc.route_log_store = _FakeStore()
        artifacts = TurnArtifacts(
            response_messages=[],
            latest_values={"primary_intent": "medical_rag", "decision_source": "rule",
                           "self_eval_score": 0.42,
                           "self_eval_details": {"safety": 3, "degraded": False}},
            final_assistant="回答",
            combined_assistant_text="回答",
            clarification_text="",
            updated_state={"secondary_intent": "", "topic_focus": "",
                           "deferred_user_question": "", "pending_action_type": ""},
            had_pending_state=False,
            route_reason="rule_match",
            secondary_turn_executed=False,
            response_messages_changed=False,
        )
        svc._persist_route_log(
            active_thread_id="t1", request_id="r1", user_message="hi",
            session_state={}, checkpoint_resumed=False, artifacts=artifacts,
        )
        self.assertEqual(captured.get("self_eval_score"), 0.42)
        self.assertEqual(captured.get("self_eval_details"), {"safety": 3, "degraded": False})


class TestCompiledSelfEval(unittest.TestCase):
    """Verify self_eval slots into a real compiled graph between grounding-check
    and the supervisor sink, without breaking the P4 chain."""

    def _build_graph(self, fake_eval_returns_caveat):
        from langgraph.graph import StateGraph, START, END
        from project.rag_agent.graph_state import State
        from project.rag_agent.edges import route_after_self_eval

        log = {"self_eval": 0, "supervise": 0}

        def _fake_self_eval(state):
            log["self_eval"] += 1
            if fake_eval_returns_caveat:
                from langchain_core.messages import AIMessage
                return {
                    "self_eval_score": 0.4,
                    "self_eval_details": {"caveat_appended": True},
                    "messages": [AIMessage(content="⚠️ 自评提示：低分")],
                }
            return {"self_eval_score": 0.9, "self_eval_details": {"caveat_appended": False}}

        def _fake_supervise(state):
            log["supervise"] += 1
            return {"supervisor_active": False, "supervisor_rounds": 0, "supervisor_next": "FINISH"}

        def _sink(state):
            return {}

        builder = StateGraph(State)
        builder.add_node("self_eval", _fake_self_eval)
        builder.add_node("supervise", _fake_supervise)
        builder.add_node("end_sink", _sink)
        builder.add_edge(START, "self_eval")
        builder.add_conditional_edges("self_eval", route_after_self_eval, {
            "supervise": "supervise", "__end__": "end_sink",
        })
        builder.add_conditional_edges("supervise", lambda s: "__end__", {"__end__": "end_sink"})
        builder.add_edge("end_sink", END)
        return builder.compile(), log

    def test_self_eval_then_supervise_then_end(self):
        graph, log = self._build_graph(fake_eval_returns_caveat=False)
        final = graph.invoke(_make_main_state())
        self.assertEqual(log["self_eval"], 1)
        self.assertEqual(log["supervise"], 1)
        self.assertAlmostEqual(final["self_eval_score"], 0.9)

    def test_caveat_message_appended(self):
        graph, log = self._build_graph(fake_eval_returns_caveat=True)
        final = graph.invoke(_make_main_state())
        self.assertEqual(log["self_eval"], 1)
        self.assertTrue(any("自评提示" in str(getattr(m, "content", "")) for m in final["messages"]))


class TestCaveatSurfacing(unittest.TestCase):
    """P5: the soft-degrade caveat must surface in response_messages (it can't
    stream because self_eval is a SILENT_NODE returning a plain AIMessage)."""

    def _make_service(self, latest_values):
        from project.core.chat_turn_service import ChatTurnService
        from project.core.chat_interface import ChatInterface

        class _FakeGraph:
            def get_state(self, config):
                class _S:
                    values = latest_values
                return _S()

        class _FakeRag:
            agent_graph = _FakeGraph()

        svc = ChatTurnService.__new__(ChatTurnService)
        svc.rag_system = _FakeRag()
        svc.route_log_store = None
        svc._extract_final_assistant_text = ChatInterface._extract_final_assistant_text
        svc._extract_all_visible_assistant_texts = ChatInterface._extract_all_visible_assistant_texts
        svc._extract_clarification_text = ChatInterface._extract_clarification_text
        svc._extract_latest_state_assistant = ChatInterface._extract_latest_state_assistant
        svc._build_chat_failure_fallback = lambda user_message: "fallback"
        svc._resolved_session_state = lambda lv, ss, um, ct: {}
        svc._invalidate_memory_cache = lambda *a, **k: None
        return svc

    def test_caveat_appended_to_response_messages(self):
        from langchain_core.messages import AIMessage
        from project.core.chat_turn_service import TurnArtifacts
        # State: answer present (grounded_answer_generation output), then caveat.
        latest_values = {
            "messages": [
                AIMessage(content="最终答案+引用", name="grounded_answer_generation"),
                AIMessage(content="⚠️ 自评提示：低分", name="self_eval_caveat"),
            ],
            "self_eval_details": {"caveat_appended": True},
            "route_reason": "rule_match",
        }
        svc = self._make_service(latest_values)
        # response_messages already has the streamed answer
        response_messages = [{"role": "assistant", "content": "最终答案+引用"}]
        artifacts = svc.prepare_turn_artifacts(
            active_thread_id="t1", graph_config={},
            response_messages=response_messages, user_message="q",
            session_state={},
        )
        # The caveat was appended
        contents = [m["content"] for m in artifacts.response_messages]
        self.assertIn("⚠️ 自评提示：低分", contents)
        self.assertTrue(artifacts.response_messages_changed)
        # final_assistant remains the ANSWER (not the caveat)
        self.assertEqual(artifacts.final_assistant, "最终答案+引用")

    def test_no_caveat_leaves_response_messages_unchanged(self):
        from langchain_core.messages import AIMessage
        latest_values = {
            "messages": [AIMessage(content="最终答案", name="grounded_answer_generation")],
            "route_reason": "rule_match",
        }
        svc = self._make_service(latest_values)
        response_messages = [{"role": "assistant", "content": "最终答案"}]
        artifacts = svc.prepare_turn_artifacts(
            active_thread_id="t1", graph_config={},
            response_messages=response_messages, user_message="q",
            session_state={},
        )
        contents = [m["content"] for m in artifacts.response_messages]
        self.assertFalse(any("自评提示" in c for c in contents))
        # No caveat → no change from the caveat-surfacing block
        self.assertEqual(len(artifacts.response_messages), 1)

    def test_state_fallback_returns_answer_not_caveat(self):
        """When final_assistant is empty (no streamed answer), the state-fallback
        must return the ANSWER, not the caveat (which is last in state)."""
        from langchain_core.messages import AIMessage
        from project.core.chat_interface import ChatInterface
        latest_values = {
            "messages": [
                AIMessage(content="最终答案+引用", name="grounded_answer_generation"),
                AIMessage(content="⚠️ 自评提示：低分", name="self_eval_caveat"),
            ],
        }
        # Directly test the static helper
        result = ChatInterface._extract_latest_state_assistant(latest_values)
        self.assertEqual(result, "最终答案+引用")


if __name__ == "__main__":
    unittest.main()
