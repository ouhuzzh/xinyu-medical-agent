import sys
import unittest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "project"))

from db.route_log_store import RouteLogStore  # noqa: E402
from benchmarks.evaluate_route_quality import _render_markdown_report, _render_text_report  # noqa: E402


class RouteQualityReportTests(unittest.TestCase):
    def test_summarize_recent_aggregates_compound_and_resume_rates(self):
        store = RouteLogStore()
        store.list_recent = lambda limit=20: [
            {
                "thread_id": "t1",
                "user_query": "取消刚才那个预约，然后我这个咳嗽还要看吗",
                "primary_intent": "cancel_appointment",
                "secondary_intent": "medical_rag",
                "decision_source": "rule",
                "route_reason": "explicit_cancel_rule+medical_question_rule",
                "had_pending_state": True,
                "extra_metadata": {"topic_focus": "咳嗽", "deferred_user_question": "我这个咳嗽还要看吗"},
                "timestamp": "2026-04-18 10:00:00",
            },
            {
                "thread_id": "t2",
                "user_query": "高血压一般有什么症状",
                "primary_intent": "medical_rag",
                "secondary_intent": "",
                "decision_source": "rule",
                "route_reason": "medical_question_rule",
                "had_pending_state": False,
                "extra_metadata": {"topic_focus": "高血压"},
                "timestamp": "2026-04-18 10:01:00",
            },
        ]

        summary = store.summarize_recent(limit=10)

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["compound_request_rate"], 0.5)
        self.assertEqual(summary["pending_resume_rate"], 0.5)
        self.assertEqual(summary["checkpoint_resume_rate"], 0.0)
        self.assertEqual(summary["secondary_turn_completion_rate"], 0.0)
        self.assertEqual(summary["deferred_question_rate"], 0.5)
        self.assertEqual(summary["intent_distribution"]["medical_rag"], 1)
        self.assertEqual(summary["intent_distribution"]["cancel_appointment"], 1)

    def test_render_reports_include_route_distributions_and_recent_events(self):
        report = {
            "summary": {
                "sample_count": 2,
                "compound_request_rate": 0.5,
                "pending_resume_rate": 0.5,
                "checkpoint_resume_rate": 0.5,
                "secondary_turn_completion_rate": 0.5,
                "deferred_question_rate": 0.5,
                "intent_distribution": {"medical_rag": 1, "cancel_appointment": 1},
                "secondary_intent_distribution": {"none": 1, "medical_rag": 1},
                "decision_source_distribution": {"rule": 2},
                "route_reason_distribution": {"medical_question_rule": 1, "explicit_cancel_rule+medical_question_rule": 1},
            },
            "events": [
                {
                    "timestamp": "2026-04-18 10:00:00",
                    "user_query": "取消刚才那个预约，然后我这个咳嗽还要看吗",
                    "primary_intent": "cancel_appointment",
                    "secondary_intent": "medical_rag",
                    "decision_source": "rule",
                    "route_reason": "explicit_cancel_rule+medical_question_rule",
                    "had_pending_state": True,
                    "extra_metadata": {"topic_focus": "咳嗽", "deferred_user_question": "我这个咳嗽还要看吗", "checkpoint_resumed": True, "secondary_turn_executed": True},
                }
            ],
        }

        text = _render_text_report(report)
        markdown = _render_markdown_report(report)

        self.assertIn("Route Quality Summary", text)
        self.assertIn("compound_request_rate", text)
        self.assertIn("checkpoint_resume_rate", text)
        self.assertIn("# Route Quality Report", markdown)
        self.assertIn("## Recent Route Events", markdown)
        self.assertIn("cancel_appointment", markdown)


if __name__ == "__main__":
    unittest.main()
