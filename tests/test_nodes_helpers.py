import sys
import unittest
from datetime import date, timedelta

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "project"))

import config  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from rag_agent.nodes import (  # noqa: E402
    _build_recent_context,
    _confidence_bucket_explanation,
    _format_reference_lines,
    _sanitize_final_answer_text,
    _normalize_date,
    _normalize_time_slot,
    _is_explicit_confirmation,
    _looks_like_appointment_discovery_query,
    _pick_candidate_from_text,
    _strip_leading_query_plan_blob,
    _should_use_last_appointment,
)


class NodesHelperTests(unittest.TestCase):
    def test_normalize_date_supports_relative_weekday(self):
        today = date.today()
        expected = today + timedelta(days=(0 - today.weekday()) % 7 + 7)
        self.assertEqual(_normalize_date("下周一上午"), expected.isoformat())

    def test_normalize_date_supports_weekend(self):
        today = date.today()
        expected = today + timedelta(days=(5 - today.weekday()) % 7)
        self.assertEqual(_normalize_date("这个周末"), expected.isoformat())

    def test_normalize_date_rejects_invalid_date(self):
        self.assertEqual(_normalize_date("2026年13月1日"), "")

    def test_normalize_time_slot_supports_noon_and_embedded_phrase(self):
        self.assertEqual(_normalize_time_slot("中午12点"), "afternoon")
        self.assertEqual(_normalize_time_slot("周三上午"), "morning")
        self.assertEqual(_normalize_time_slot("morning"), "morning")

    def test_explicit_confirmation_is_strict(self):
        self.assertTrue(_is_explicit_confirmation("确认预约", "appointment"))
        self.assertTrue(_is_explicit_confirmation("确认取消", "cancel_appointment"))
        self.assertFalse(_is_explicit_confirmation("可以", "appointment"))

    def test_pick_candidate_from_text_supports_appointment_number_and_ordinal(self):
        candidates = [
            {"appointment_id": 1, "appointment_no": "APT111AAA"},
            {"appointment_id": 2, "appointment_no": "APT222BBB"},
        ]
        self.assertEqual(_pick_candidate_from_text("帮我取消 APT222BBB", candidates), candidates[1])
        self.assertEqual(_pick_candidate_from_text("取消第 1 个", candidates), candidates[0])

    def test_should_use_last_appointment_requires_explicit_recent_reference(self):
        self.assertTrue(_should_use_last_appointment("帮我取消最近的那个预约"))
        self.assertTrue(_should_use_last_appointment("取消上次那个"))
        self.assertFalse(_should_use_last_appointment("帮我取消预约"))

    def test_appointment_discovery_query_covers_existing_booking_lookup(self):
        self.assertTrue(_looks_like_appointment_discovery_query("我之前挂了谁的号"))
        self.assertTrue(_looks_like_appointment_discovery_query("我现在挂了谁的号"))

    def test_strip_leading_query_plan_blob_removes_json_prefix(self):
        text = '{"queries": ["呼吸内科", "呼吸内科挂号"]}呼吸内科主要诊治呼吸系统疾病。'
        self.assertEqual(_strip_leading_query_plan_blob(text), "呼吸内科主要诊治呼吸系统疾病。")

    def test_sanitize_final_answer_text_removes_query_plan_and_sources_block(self):
        text = (
            '```json\n{"queries": ["高血压注意事项"]}\n```\n'
            "高血压患者要注意低盐饮食。\n\n---\n**Sources:**\n- file1.pdf\n- file2.txt"
        )
        self.assertEqual(_sanitize_final_answer_text(text), "高血压患者要注意低盐饮食。")

    def test_format_reference_lines_uses_user_friendly_labels(self):
        lines = _format_reference_lines(
            [
                {
                    "title": "高血压管理指南.txt",
                    "source_type": "clinical_guideline",
                    "freshness_bucket": "outdated",
                    "original_url": "https://example.com/guide",
                }
            ]
        )
        self.assertEqual(
            lines[0],
            "- 高血压管理指南.txt（临床指南，时效：较旧） [链接](https://example.com/guide)",
        )

    def test_confidence_bucket_explanation_is_patient_friendly(self):
        self.assertIn("较直接", _confidence_bucket_explanation("high", is_medical_request=True))
        self.assertIn("初步参考", _confidence_bucket_explanation("medium", is_medical_request=True))
        self.assertIn("通用医学信息", _confidence_bucket_explanation("no_evidence", is_medical_request=True))

    def test_build_recent_context_uses_configured_turn_window(self):
        original_turns = config.RECENT_CONTEXT_TURNS
        config.RECENT_CONTEXT_TURNS = 3
        try:
            messages = [
                HumanMessage(content="第1轮用户"),
                AIMessage(content="第1轮助手"),
                HumanMessage(content="第2轮用户"),
                AIMessage(content="第2轮助手"),
                HumanMessage(content="第3轮用户"),
                AIMessage(content="第3轮助手"),
                HumanMessage(content="当前问题"),
            ]
            recent = _build_recent_context(messages)
        finally:
            config.RECENT_CONTEXT_TURNS = original_turns

        self.assertIn("第1轮用户", recent)
        self.assertIn("第3轮助手", recent)
        self.assertNotIn("当前问题", recent)


if __name__ == "__main__":
    unittest.main()
