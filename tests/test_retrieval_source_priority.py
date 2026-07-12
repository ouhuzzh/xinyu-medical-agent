import sys
import unittest

from langchain_core.documents import Document

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "project"))

from rag_agent.tools import ToolFactory, reset_retrieval_context, set_retrieval_context  # noqa: E402


class FakeCollection:
    def __init__(self, docs=None, docs_by_source=None):
        self.docs = docs or []
        self.docs_by_source = docs_by_source or {}
        self.calls = []
        self.keyword_calls = []
        self.logged = []

    def similarity_search(self, query, k=4, score_threshold=None, source_types=None, rerank=True):
        self.calls.append(
            {
                "query": query,
                "k": k,
                "score_threshold": score_threshold,
                "source_types": list(source_types or []),
                "rerank": rerank,
            }
        )
        query_key = f"query:{query}"
        if source_types:
            source_type = source_types[0]
            scoped_key = f"{query_key}:{source_type}"
            if scoped_key in self.docs_by_source:
                return list(self.docs_by_source.get(scoped_key, []))
        if source_types:
            source_type = source_types[0]
            return list(self.docs_by_source.get(source_type, []))
        if query_key in self.docs_by_source:
            return list(self.docs_by_source.get(query_key, []))
        return list(self.docs)

    def rerank_candidates(self, query, candidates, top_n):
        return candidates[:top_n]

    def keyword_search(self, query, k=4, source_types=None):
        self.keyword_calls.append(
            {
                "query": query,
                "k": k,
                "source_types": list(source_types or []),
            }
        )
        query_key = f"query:{query}"
        if source_types:
            source_type = source_types[0]
            scoped_key = f"keyword:{query}:{source_type}"
            if scoped_key in self.docs_by_source:
                return list(self.docs_by_source.get(scoped_key, []))
        if source_types:
            source_type = source_types[0]
            return list(self.docs_by_source.get(f"keyword:{source_type}", []))
        if f"keyword:{query}" in self.docs_by_source:
            return list(self.docs_by_source.get(f"keyword:{query}", []))
        return list(self.docs_by_source.get("keyword", []))

    def log_retrieval(self, **payload):
        self.logged.append(payload)


class RetrievalSourcePriorityTests(unittest.TestCase):
    def test_search_child_chunks_prefers_patient_facing_sources(self):
        docs = [
            Document(
                page_content="Clinical guideline content about asthma treatment targets.",
                metadata={"parent_id": "p3", "source": "guideline.pdf", "source_type": "clinical_guideline", "score": 0.99},
            ),
            Document(
                page_content="Public health content about asthma triggers and prevention.",
                metadata={"parent_id": "p2", "source": "who.pdf", "source_type": "public_health", "score": 0.95},
            ),
            Document(
                page_content="Patient education content explaining asthma symptoms.",
                metadata={"parent_id": "p1", "source": "medlineplus.pdf", "source_type": "patient_education", "score": 0.90},
            ),
        ]
        tool_factory = ToolFactory(FakeCollection(docs))

        result = tool_factory._search_child_chunks("asthma", limit=3)

        first_index = result.find("medlineplus.pdf")
        second_index = result.find("who.pdf")
        third_index = result.find("guideline.pdf")
        self.assertNotEqual(first_index, -1)
        self.assertNotEqual(second_index, -1)
        self.assertNotEqual(third_index, -1)
        self.assertLess(first_index, second_index)
        self.assertLess(second_index, third_index)
        self.assertIn("Source Type: patient_education", result)

    def test_sort_docs_by_source_priority_uses_score_within_same_tier(self):
        docs = [
            Document(page_content="A", metadata={"source_type": "patient_education", "score": 0.81}),
            Document(page_content="B", metadata={"source_type": "patient_education", "score": 0.93}),
        ]

        sorted_docs = ToolFactory._sort_docs_by_source_priority(docs)

        self.assertEqual(sorted_docs[0].page_content, "B")
        self.assertEqual(sorted_docs[1].page_content, "A")

    def test_layered_similarity_search_queries_tiers_before_fallback(self):
        docs_by_source = {
            "patient_education": [
                Document(page_content="patient", metadata={"parent_id": "p1", "source": "medlineplus.pdf", "source_type": "patient_education", "score": 0.80}),
            ],
            "public_health": [
                Document(page_content="public", metadata={"parent_id": "p2", "source": "who.pdf", "source_type": "public_health", "score": 0.82}),
            ],
            "clinical_guideline": [
                Document(page_content="clinical", metadata={"parent_id": "p3", "source": "guideline.pdf", "source_type": "clinical_guideline", "score": 0.95}),
            ],
        }
        collection = FakeCollection(docs_by_source=docs_by_source)
        tool_factory = ToolFactory(collection)

        results = tool_factory._layered_similarity_search("hypertension", limit=3, score_threshold=0.7)

        self.assertEqual([doc.metadata["source_type"] for doc in results], ["patient_education", "public_health", "clinical_guideline"])
        self.assertEqual(
            [call["source_types"] for call in collection.calls[:3]],
            [["patient_education"], ["public_health"], ["clinical_guideline"]],
        )

    def test_query_type_can_prioritize_public_health_first(self):
        collection = FakeCollection(docs_by_source={})
        tool_factory = ToolFactory(collection)

        tool_factory._layered_similarity_search("如何预防流感传播", limit=2, score_threshold=0.7)

        self.assertEqual(
            [call["source_types"] for call in collection.calls[:3]],
            [["public_health"], ["patient_education"], ["clinical_guideline"]],
        )

    def test_query_type_can_prioritize_clinical_guideline_first(self):
        collection = FakeCollection(docs_by_source={})
        tool_factory = ToolFactory(collection)

        tool_factory._layered_similarity_search("高血压诊疗指南 第十版 剂量标准", limit=2, score_threshold=0.7)

        self.assertEqual(
            [call["source_types"] for call in collection.calls[:3]],
            [["clinical_guideline"], ["patient_education"], ["public_health"]],
        )

    def test_hybrid_search_can_promote_exact_keyword_hit_within_same_tier(self):
        exact_doc = Document(
            page_content="Contains exact rareterm evidence.",
            metadata={"parent_id": "p-exact", "source": "exact.pdf", "source_type": "patient_education", "score": 0.62},
        )
        noisy_doc = Document(
            page_content="Semantically similar but not exact.",
            metadata={"parent_id": "p-noisy", "source": "noisy.pdf", "source_type": "patient_education", "score": 0.92},
        )
        collection = FakeCollection(
            docs_by_source={
                "patient_education": [noisy_doc, exact_doc],
                "keyword:patient_education": [exact_doc],
            }
        )
        tool_factory = ToolFactory(collection)

        results = tool_factory.search_documents("rareterm", limit=2, score_threshold=0.0)

        self.assertEqual(results[0].metadata["source"], "exact.pdf")
        self.assertTrue(collection.keyword_calls)

    def test_search_child_chunks_returns_safe_no_evidence_message(self):
        collection = FakeCollection(docs_by_source={})
        tool_factory = ToolFactory(collection)

        result = tool_factory._search_child_chunks("totally missing query", limit=2)

        self.assertIn("知识库中暂无相关信息", result)

    def test_search_child_chunks_logs_retrieval_context(self):
        collection = FakeCollection(
            docs_by_source={
                "patient_education": [
                    Document(
                        page_content="patient result",
                        metadata={"parent_id": "p1", "source": "medlineplus.pdf", "source_type": "patient_education", "score": 0.9},
                    )
                ]
            }
        )
        tool_factory = ToolFactory(collection)
        token = set_retrieval_context(thread_id="thread-log", original_query="高血压怎么办")

        try:
            tool_factory._search_child_chunks("高血压 怎么办", limit=1)
        finally:
            reset_retrieval_context(token)

        self.assertEqual(len(collection.logged), 1)
        self.assertEqual(collection.logged[0]["thread_id"], "thread-log")
        self.assertEqual(collection.logged[0]["query_text"], "高血压怎么办")

    def test_search_child_chunks_uses_query_plan_to_recover_follow_up_query(self):
        resolved_doc = Document(
            page_content="高血压患者应注意低盐饮食、规律监测血压。",
            metadata={"parent_id": "p1", "source": "hypertension.md", "source_type": "patient_education", "score": 0.91},
        )
        collection = FakeCollection(
            docs_by_source={
                "query:高血压应该注意什么:patient_education": [resolved_doc],
            }
        )
        tool_factory = ToolFactory(collection)

        result = tool_factory._search_child_chunks(
            "那应该注意什么",
            limit=1,
            query_plan=["那应该注意什么", "高血压应该注意什么"],
        )

        self.assertIn("hypertension.md", result)
        self.assertIn("Matched Query: 高血压应该注意什么", result)
        self.assertIn("高血压应该注意什么", collection.logged[0]["query_plan"])


if __name__ == "__main__":
    unittest.main()
