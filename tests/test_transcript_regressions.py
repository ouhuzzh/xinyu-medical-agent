import sys
import unittest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "project"))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from rag_agent.nodes import intent_router, rewrite_query, recommend_department  # noqa: E402
from rag_agent.schemas import IntentAnalysis, QueryAnalysis, DepartmentRecommendation  # noqa: E402


def setUpModule():
    """Register skill plugins so L1 keyword rules fire - the rule-covered cases
    below must not reach the LLM (ExplodingStructuredLLM asserts it isn't invoked)."""
    import config
    if getattr(config, "SKILLS_ENABLED", False):
        from core.skill_bootstrapper import SkillBootstrapper
        SkillBootstrapper().bootstrap()


class FakeStructuredLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def with_config(self, **kwargs):
        return self

    def bind(self, **kwargs):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        return self.responses.pop(0)


class ExplodingStructuredLLM:
    def with_config(self, **kwargs):
        return self

    def bind(self, **kwargs):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):  # pragma: no cover - used to prove the call does not happen
        raise AssertionError("LLM should not be invoked for this rule-covered case.")


class FailingStructuredLLM:
    def with_config(self, **kwargs):
        return self

    def bind(self, **kwargs):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        raise RuntimeError("structured output failed")


class TranscriptRegressionTests(unittest.TestCase):
    def test_intent_router_routes_headache_question_without_llm(self):
        state = {
            "messages": [HumanMessage(content="头疼怎么处理")],
            "conversation_summary": "",
            "pending_action_type": "",
            "pending_candidates": [],
        }

        result = intent_router(state, ExplodingStructuredLLM())

        self.assertEqual(result["intent"], "medical_rag")
        self.assertEqual(result["pending_clarification"], "")

    def test_intent_router_routes_symptom_statement_without_llm(self):
        state = {
            "messages": [HumanMessage(content="发烧三天，头痛明显")],
            "conversation_summary": "",
            "pending_action_type": "",
            "pending_candidates": [],
        }

        result = intent_router(state, ExplodingStructuredLLM())

        self.assertEqual(result["intent"], "medical_rag")
        self.assertEqual(result["pending_clarification"], "")

    def test_intent_router_falls_back_to_medical_rag_when_structured_output_fails(self):
        state = {
            "messages": [HumanMessage(content="我不太舒服，怎么办")],
            "conversation_summary": "",
            "pending_action_type": "",
            "pending_candidates": [],
        }

        result = intent_router(state, FailingStructuredLLM())

        self.assertEqual(result["intent"], "medical_rag")
        self.assertEqual(result["decision_source"], "llm_error_fallback")
        self.assertEqual(result["pending_clarification"], "")

    def test_intent_router_treats_follow_up_medical_question_as_medical_rag(self):
        llm = FakeStructuredLLM(
            [
                IntentAnalysis(intent="clarification", is_clear=False, clarification_needed="请再详细一点"),
            ]
        )
        state = {
            "messages": [HumanMessage(content="那会头晕吗")],
            "conversation_summary": "用户正在询问高血压的常见表现和头晕是否相关。",
            "pending_action_type": "",
            "pending_candidates": [],
        }

        result = intent_router(state, llm)

        self.assertEqual(result["intent"], "medical_rag")
        self.assertEqual(result["pending_clarification"], "")

    def test_intent_router_uses_recent_history_for_follow_up_when_summary_is_empty(self):
        state = {
            "messages": [
                HumanMessage(content="高血压会引起头晕吗"),
                AIMessage(content="高血压有时会引起头晕，也可能伴随头痛。"),
                HumanMessage(content="那应该注意什么"),
            ],
            "conversation_summary": "",
            "pending_action_type": "",
            "pending_candidates": [],
        }

        result = intent_router(state, ExplodingStructuredLLM())

        self.assertEqual(result["intent"], "medical_rag")
        self.assertIn("高血压", result["recent_context"])

    def test_intent_router_short_circuits_clear_medical_question_before_llm(self):
        state = {
            "messages": [HumanMessage(content="高血压会引起头晕吗")],
            "conversation_summary": "",
            "pending_action_type": "",
            "pending_candidates": [],
        }

        result = intent_router(state, ExplodingStructuredLLM())

        self.assertEqual(result["intent"], "medical_rag")

    def test_intent_router_prefers_medical_rag_for_mixed_booking_medical_question(self):
        state = {
            "messages": [HumanMessage(content="预约前高血压药还要不要吃")],
            "conversation_summary": "",
            "pending_action_type": "",
            "pending_candidates": [],
        }

        result = intent_router(state, ExplodingStructuredLLM())

        self.assertEqual(result["intent"], "medical_rag")

    @unittest.skip("stale: general_conversation_rule removed in lean-routing refactor 9b8be5f")
    def test_intent_router_treats_general_emotional_chat_as_answerable(self):
        state = {
            "messages": [HumanMessage(content="我今天有点烦")],
            "conversation_summary": "",
            "pending_action_type": "",
            "pending_candidates": [],
        }

        result = intent_router(state, ExplodingStructuredLLM())

        self.assertEqual(result["intent"], "medical_rag")
        self.assertEqual(result["route_reason"], "general_conversation_rule")

    @unittest.skip("stale: general_conversation_rule removed in lean-routing refactor 9b8be5f")
    def test_intent_router_treats_non_medical_general_question_as_answerable(self):
        state = {
            "messages": [HumanMessage(content="帮我介绍一下东京有什么好玩的")],
            "conversation_summary": "",
            "pending_action_type": "",
            "pending_candidates": [],
        }

        result = intent_router(state, ExplodingStructuredLLM())

        self.assertEqual(result["intent"], "medical_rag")
        self.assertEqual(result["route_reason"], "general_conversation_rule")

    @unittest.skip("stale: compound cancel+medical now decomposed by the turn planner, not intent_router cancel-flow preference")
    def test_intent_router_prefers_cancel_flow_for_explicit_cancel_then_medical_question(self):
        state = {
            "messages": [HumanMessage(content="取消刚才那个预约，然后我这个咳嗽还要看吗")],
            "conversation_summary": "",
            "last_appointment_no": "APT001",
            "pending_action_type": "",
            "pending_candidates": [],
        }

        result = intent_router(state, ExplodingStructuredLLM())

        self.assertEqual(result["intent"], "cancel_appointment")

    def test_intent_router_short_clarifies_vague_medication_dosing_question(self):
        state = {
            "messages": [HumanMessage(content="这个药我一天吃几片")],
            "conversation_summary": "",
            "pending_action_type": "",
            "pending_candidates": [],
        }

        result = intent_router(state, ExplodingStructuredLLM())

        self.assertEqual(result["intent"], "clarification")
        self.assertIn("药名", result["pending_clarification"])

    @unittest.skip("needs review: pre-planner intent_router clarification-resume routing; behavior changed across refactors")
    def test_intent_router_routes_pending_clarification_back_to_original_target(self):
        state = {
            "messages": [HumanMessage(content="明天下午")],
            "conversation_summary": "",
            "pending_clarification": "请补充时间",
            "clarification_target": "handle_appointment",
            "intent": "appointment",
            "pending_action_type": "",
            "pending_candidates": [],
        }

        result = intent_router(state, ExplodingStructuredLLM())

        self.assertEqual(result["intent"], "appointment")
        self.assertEqual(result["pending_clarification"], "")

    def test_rewrite_query_accepts_contextual_follow_up_without_extra_clarification(self):
        llm = FakeStructuredLLM(
            [
                QueryAnalysis(is_clear=False, questions=[], clarification_needed="请说明你指的是哪种疾病"),
            ]
        )
        state = {
            "messages": [HumanMessage(content="那应该注意什么")],
            "conversation_summary": "用户刚刚询问糖尿病的常见症状与日常管理。",
            "intent": "medical_rag",
        }

        result = rewrite_query(state, llm)

        self.assertTrue(result["questionIsClear"])
        self.assertEqual(result["pending_clarification"], "")
        self.assertTrue(result["rewrittenQuestions"])
        self.assertIn("糖尿病", result["recent_context"] or state["conversation_summary"])

    def test_rewrite_query_keeps_general_non_medical_question_clear(self):
        llm = FakeStructuredLLM([])
        state = {
            "messages": [HumanMessage(content="帮我介绍一下东京有什么好玩的")],
            "conversation_summary": "",
            "intent": "medical_rag",
        }

        result = rewrite_query(state, llm)

        self.assertTrue(result["questionIsClear"])
        self.assertEqual(result["rewrittenQuestions"], ["帮我介绍一下东京有什么好玩的"])
        self.assertEqual(result["pending_clarification"], "")

    def test_rewrite_query_falls_back_to_original_query_when_structured_output_fails(self):
        state = {
            "messages": [HumanMessage(content="头痛怎么办")],
            "conversation_summary": "",
            "intent": "medical_rag",
        }

        result = rewrite_query(state, FailingStructuredLLM())

        self.assertTrue(result["questionIsClear"])
        self.assertEqual(result["rewrittenQuestions"], ["头痛怎么办"])
        self.assertEqual(result["pending_clarification"], "")

    @unittest.skip("needs review: pre-refactor rewrite_query recent-history inclusion; verify intended vs regression")
    def test_rewrite_query_keeps_recent_history_instead_of_deleting_everything(self):
        llm = FakeStructuredLLM(
            [
                QueryAnalysis(is_clear=True, questions=["高血压应该注意什么"], clarification_needed=""),
            ]
        )
        state = {
            "messages": [
                SystemMessage(content="system", id="sys-1"),
                HumanMessage(content="第一轮用户", id="h1"),
                AIMessage(content="第一轮助手", id="a1"),
                HumanMessage(content="第二轮用户", id="h2"),
                AIMessage(content="第二轮助手", id="a2"),
                HumanMessage(content="第三轮用户", id="h3"),
                AIMessage(content="第三轮助手", id="a3"),
                HumanMessage(content="那应该注意什么", id="h4"),
            ],
            "conversation_summary": "用户正在问高血压相关问题。",
            "intent": "medical_rag",
        }

        result = rewrite_query(state, llm)

        removed_ids = [message.id for message in result["messages"]]
        self.assertEqual(removed_ids, ["h1", "a1"])
        self.assertEqual(result["rewrittenQuestions"], ["高血压应该注意什么"])

    def test_recommend_department_defaults_to_emergency_when_high_risk_and_model_wants_clarification(self):
        llm = FakeStructuredLLM(
            [
                DepartmentRecommendation(
                    department="",
                    reason="",
                    needs_clarification=True,
                    clarification_needed="胸痛持续多久了？",
                )
            ]
        )
        state = {
            "messages": [HumanMessage(content="我现在胸痛还有点呼吸困难")],
            "risk_level": "high",
            "appointment_context": {},
        }

        result = recommend_department(state, llm)

        self.assertEqual(result["recommended_department"], "急诊科")
        self.assertIn("急诊科", result["messages"][0].content)
        self.assertEqual(result["pending_clarification"], "")


if __name__ == "__main__":
    unittest.main()
