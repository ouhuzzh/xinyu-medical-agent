import sys
import unittest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "project"))

from db.retrieval_log_store import RetrievalLogStore  # noqa: E402
from benchmarks.evaluate_retrieval_quality import _render_markdown_report, _render_text_report  # noqa: E402


class RetrievalLogReportTests(unittest.TestCase):
    def test_summarize_recent_aggregates_retry_and_confidence_metrics(self):
        store = RetrievalLogStore()
        store.list_recent = lambda limit=20: [
            {
                "thread_id": "t1",
                "query_text": "那应该注意什么",
                "rewritten_query": "那应该注意什么",
                "retrieval_mode": "hybrid_layered",
                "top_k": 3,
                "result_count": 2,
                "selected_parent_ids": ["p1", "p2"],
                "query_plan": ["那应该注意什么", "高血压应该注意什么"],
                "graded_doc_count": 2,
                "sufficiency_result": "limited_but_usable",
                "retry_count": 1,
                "final_confidence_bucket": "medium",
                "timestamp": "2026-04-19 10:00:00",
            },
            {
                "thread_id": "t2",
                "query_text": "罕见病 xyz",
                "rewritten_query": "罕见病 xyz",
                "retrieval_mode": "hybrid_layered",
                "top_k": 3,
                "result_count": 0,
                "selected_parent_ids": [],
                "query_plan": ["罕见病 xyz"],
                "graded_doc_count": 0,
                "sufficiency_result": "no_relevant_documents",
                "retry_count": 0,
                "final_confidence_bucket": "no_evidence",
                "timestamp": "2026-04-19 10:01:00",
            },
        ]

        summary = store.summarize_recent(limit=10)

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["retry_rate"], 0.5)
        self.assertEqual(summary["multi_query_rate"], 0.5)
        self.assertEqual(summary["no_evidence_rate"], 0.5)
        self.assertEqual(summary["low_confidence_rate"], 0.5)
        self.assertEqual(summary["avg_query_plan_size"], 1.5)
        self.assertEqual(summary["avg_result_count"], 1.0)
        self.assertEqual(summary["avg_retry_count"], 0.5)

    def test_render_reports_include_retrieval_metrics_and_events(self):
        report = {
            "summary": {
                "sample_count": 2,
                "retry_rate": 0.5,
                "multi_query_rate": 0.5,
                "no_evidence_rate": 0.5,
                "low_confidence_rate": 0.5,
                "avg_query_plan_size": 1.5,
                "avg_result_count": 1.0,
                "avg_retry_count": 0.5,
                "confidence_distribution": {"medium": 1, "no_evidence": 1},
                "sufficiency_distribution": {"limited_but_usable": 1, "no_relevant_documents": 1},
            },
            "events": [
                {
                    "timestamp": "2026-04-19 10:00:00",
                    "query_text": "那应该注意什么",
                    "rewritten_query": "那应该注意什么",
                    "retrieval_mode": "hybrid_layered",
                    "top_k": 3,
                    "result_count": 2,
                    "selected_parent_ids": ["p1", "p2"],
                    "query_plan": ["那应该注意什么", "高血压应该注意什么"],
                    "graded_doc_count": 2,
                    "sufficiency_result": "limited_but_usable",
                    "retry_count": 1,
                    "final_confidence_bucket": "medium",
                    "timestamp": "2026-04-19 10:00:00",
                }
            ],
        }

        text = _render_text_report(report)
        markdown = _render_markdown_report(report)

        self.assertIn("Retrieval Quality Summary", text)
        self.assertIn("retry_rate", text)
        self.assertIn("# Retrieval Quality Report", markdown)
        self.assertIn("## Recent Retrieval Events", markdown)
        self.assertIn("那应该注意什么", markdown)


if __name__ == "__main__":
    unittest.main()
