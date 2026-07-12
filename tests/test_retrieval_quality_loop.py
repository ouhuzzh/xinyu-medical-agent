import sys
import unittest

from langchain_core.messages import AIMessage
from langchain_core.documents import Document

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "project"))

from rag_agent.nodes import answer_grounding_check, grounded_answer_generation  # noqa: E402
from rag_agent.tools import check_sufficiency, grade_documents, ground_answer, plan_queries  # noqa: E402
import config  # noqa: E402


class _StubLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, _messages):
        return AIMessage(content=self._content)


class RetrievalQualityLoopTests(unittest.TestCase):
    def test_plan_queries_expands_follow_up_with_topic_and_context(self):
        queries = plan_queries(
            "那应该注意什么",
            topic_focus="高血压",
            recent_context="User: 高血压会头晕吗\nAssistant: 有时会。",
        )

        self.assertGreaterEqual(len(queries), 3)
        self.assertEqual(queries[0], "那应该注意什么")
        self.assertTrue(any("高血压" in item for item in queries))
        self.assertTrue(any("症状 治疗 注意事项" in item or "高血压会头晕吗" in item for item in queries))

    def test_grade_documents_filters_weakly_related_chunks(self):
        docs = [
            Document(page_content="高血压患者平时要注意低盐饮食。", metadata={"score": 0.88}),
            Document(page_content="This document is about gardening soil and flowers.", metadata={"score": 0.55}),
        ]

        graded = grade_documents("高血压注意事项", docs)

        self.assertEqual(len(graded), 1)
        self.assertEqual(graded[0].metadata["relevance_grade"], "high")
        self.assertTrue(graded[0].metadata["keep"])

    def test_check_sufficiency_requests_retry_for_sparse_evidence(self):
        docs = [
            Document(page_content="高血压。", metadata={"score": 0.68, "relevance_grade": "medium"}),
        ]

        result = check_sufficiency("高血压需要注意什么", docs)

        self.assertFalse(result["is_sufficient"])
        self.assertTrue(result["retry_query"])

    def test_check_sufficiency_uses_configured_direct_evidence_threshold(self):
        docs = [
            Document(page_content="高血压患者平时要注意低盐饮食。", metadata={"score": 0.83, "relevance_grade": "high"}),
        ]

        original_threshold = config.RAG_DIRECT_EVIDENCE_SCORE
        try:
            config.RAG_DIRECT_EVIDENCE_SCORE = 0.82
            result = check_sufficiency("高血压需要注意什么", docs)
        finally:
            config.RAG_DIRECT_EVIDENCE_SCORE = original_threshold

        self.assertTrue(result["is_sufficient"])
        self.assertEqual(result["reason"], "direct_evidence")

    def test_ground_answer_adds_guardrail_when_evidence_is_low(self):
        docs = [
            Document(page_content="有限证据", metadata={"score": 0.71, "relevance_grade": "medium"}),
        ]

        grounded = ground_answer("可以先观察，但最好结合门诊复诊。", docs, medical_mode=True)

        self.assertFalse(grounded["grounded"])
        self.assertIn("知识库证据有限", grounded["revised_answer"])

    def test_ground_answer_allows_generic_medical_fallback_without_docs(self):
        grounded = ground_answer(
            "感冒发烧时可以先注意休息、补充水分，并观察体温变化。",
            [],
            question="感冒发烧怎么办",
            medical_mode=True,
            high_risk=False,
        )

        self.assertFalse(grounded["grounded"])
        self.assertIn("通用医学信息回答", grounded["revised_answer"])
        self.assertIn("不能替代专业医生", grounded["revised_answer"])

    def test_ground_answer_keeps_non_medical_reply_when_no_docs(self):
        grounded = ground_answer(
            "如果你想去东京玩，可以先看浅草寺、上野和银座。",
            [],
            question="东京有什么好玩的",
            medical_mode=False,
        )

        self.assertTrue(grounded["grounded"])
        self.assertIn("东京", grounded["revised_answer"])
        self.assertNotIn("通用医学信息回答", grounded["revised_answer"])

    def test_grounded_answer_generation_hides_internal_artifacts_and_formats_metadata(self):
        llm = _StubLLM(
            '```json\n{"queries": ["高血压注意事项"]}\n```\n'
            "高血压患者应注意低盐饮食。\n\n---\n**Sources:**\n- internal.pdf"
        )
        state = {
            "agent_answers": [
                {
                    "index": 0,
                    "answer": "高血压患者应注意低盐饮食。",
                    "confidence_bucket": "high",
                    "sources": [
                        {
                            "title": "高血压管理指南.txt",
                            "source_type": "clinical_guideline",
                            "freshness_bucket": "outdated",
                            "original_url": "https://example.com/guide",
                        }
                    ],
                }
            ],
            "originalQuery": "高血压应该注意什么？",
            "conversation_summary": "",
            "recent_context": "",
            "topic_focus": "高血压",
            "risk_level": "normal",
        }

        result = grounded_answer_generation(state, llm)
        content = result["messages"][0].content

        self.assertNotIn('{"queries"', content)
        self.assertNotIn("**Sources:**", content)
        self.assertIn("证据强度：`高`", content)
        self.assertIn("较直接、较匹配的资料", content)
        self.assertIn("参考来源：", content)
        self.assertIn("高血压管理指南.txt（临床指南，时效：较旧）", content)

    def test_answer_grounding_check_prefers_real_evidence_score_over_fixed_bucket_defaults(self):
        state = {
            "messages": [AIMessage(content="高血压患者可以先注意低盐饮食。")],
            "agent_answers": [{"confidence_bucket": "high"}],
            "grounding_evidence_score": 0.71,
            "originalQuery": "高血压应该注意什么？",
            "conversation_summary": "",
            "recent_context": "",
            "topic_focus": "高血压",
            "risk_level": "normal",
        }

        result = answer_grounding_check(state, llm=None)

        self.assertIn("知识库证据有限", result["messages"][0].content)


if __name__ == "__main__":
    unittest.main()
