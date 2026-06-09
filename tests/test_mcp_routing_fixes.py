"""Tests for MCP backend integration and routing fixes."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))


class TestStrictConfirmation(unittest.TestCase):
    """Confirm _is_explicit_confirmation rejects vague words."""

    def setUp(self):
        from rag_agent.node_helpers import _is_explicit_confirmation
        self._confirm = _is_explicit_confirmation

    def test_explicit_booking_confirmed(self):
        self.assertTrue(self._confirm("确认预约", "appointment"))
        self.assertTrue(self._confirm("确认挂号", "appointment"))

    def test_explicit_cancel_confirmed(self):
        self.assertTrue(self._confirm("确认取消", "cancel_appointment"))
        self.assertTrue(self._confirm("确认退号", "cancel_appointment"))
        self.assertTrue(self._confirm("确定取消", "cancel_appointment"))

    def test_vague_words_rejected(self):
        """'好的'/'行'/'OK' should NOT trigger confirmation."""
        for word in ["好的", "行", "OK", "可以", "好", "嗯嗯", "可以的"]:
            with self.subTest(word=word):
                self.assertFalse(self._confirm(word, "appointment"),
                                 f"'{word}' should NOT confirm appointment")
                self.assertFalse(self._confirm(word, "cancel_appointment"),
                                 f"'{word}' should NOT confirm cancel")

    def test_partial_match_rejected(self):
        """'确认一下' without context should not confirm."""
        self.assertFalse(self._confirm("确认", "appointment"),
                         "'确认' alone should not confirm")


class TestMCPSkillCleaned(unittest.TestCase):
    """Verify MCPSkill no longer intercepts appointment keywords."""

    def setUp(self):
        import config
        config.MCP_ENABLED = True
        from mcp_integration.mcp_skill import MCPSkill
        self.skill = MCPSkill()
        self.ctx = {"recent_context": "", "conversation_summary": ""}

    def test_appointment_keywords_not_matched(self):
        """These should NOT trigger MCPSkill."""
        for q in ["挂号", "预约", "帮我预约", "帮我挂", "退号", "帮我取消",
                   "查医生", "查科室", "有没有号", "我要挂号", "取消预约"]:
            with self.subTest(query=q):
                self.assertFalse(self.skill.match(q, context=self.ctx),
                                 f"'{q}' should NOT match MCPSkill anymore")

    def test_mcp_skill_keyword_matches(self):
        """MCPSkill now has L1 keywords — these should match."""
        for q in ["多少钱", "支付", "查库存", "查报告"]:
            with self.subTest(query=q):
                self.assertTrue(self.skill.match(q, context=self.ctx),
                                f"'{q}' should match MCPSkill (L1 keyword)")


class TestMCPBackend(unittest.TestCase):
    """Verify MCPAppointmentBackend factory method."""

    def test_try_create_returns_none_without_mcp_config(self):
        """When MCP is disabled, should return None."""
        import config
        saved = config.MCP_ENABLED
        config.MCP_ENABLED = False
        try:
            from services.mcp_appointment_backend import MCPAppointmentBackend
            result = MCPAppointmentBackend.try_create({})
            self.assertIsNone(result)
        finally:
            config.MCP_ENABLED = saved

    def test_try_create_returns_none_without_user(self):
        """Without a valid user/session, should return None."""
        from services.mcp_appointment_backend import MCPAppointmentBackend
        result = MCPAppointmentBackend.try_create({"thread_id": ""})
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
