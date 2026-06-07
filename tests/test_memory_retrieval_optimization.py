"""Tests for memory retrieval optimizations: rule-based skip + thread cache."""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))


class TestShouldSkipMemoryRetrieval(unittest.TestCase):
    """Test the rule-based pre-filter."""

    def setUp(self):
        from core.chat_interface import ChatInterface
        self._skip = ChatInterface._should_skip_memory_retrieval

    @patch("core.chat_interface.config")
    def test_greeting_skips(self, mock_config):
        mock_config.USER_MEMORY_SKIP_TRIVIAL_INTENT = True
        self.assertTrue(self._skip("你好"))
        self.assertTrue(self._skip("谢谢"))
        self.assertTrue(self._skip("hi"))

    @patch("core.chat_interface.config")
    def test_cancel_intent_skips(self, mock_config):
        mock_config.USER_MEMORY_SKIP_TRIVIAL_INTENT = True
        self.assertTrue(self._skip("取消我刚才的预约"))

    @patch("core.chat_interface.config")
    def test_medical_query_does_not_skip(self, mock_config):
        mock_config.USER_MEMORY_SKIP_TRIVIAL_INTENT = True
        self.assertFalse(self._skip("我头疼应该吃什么药"))
        self.assertFalse(self._skip("有高血压怎么办"))

    @patch("core.chat_interface.config")
    def test_disabled_never_skips(self, mock_config):
        mock_config.USER_MEMORY_SKIP_TRIVIAL_INTENT = False
        self.assertFalse(self._skip("你好"))
        self.assertFalse(self._skip("取消预约"))


class TestMemoryCacheRoundTrip(unittest.TestCase):
    """Test cache get/set/invalidate with a mocked Redis client."""

    def _make_interface(self):
        from core.chat_interface import ChatInterface
        instance = object.__new__(ChatInterface)
        instance.rag_system = MagicMock()
        return instance

    @patch("core.chat_interface.config")
    def test_set_then_get_returns_same_text(self, mock_config):
        mock_config.USER_MEMORY_CACHE_TTL_SECONDS = 300
        mock_config.USER_MEMORY_CACHE_MAX_TURNS = 5

        iface = self._make_interface()
        stored = {}

        client = MagicMock()
        def fake_get(k):
            return stored.get(k)
        def fake_setex(k, _ttl, v):
            stored[k] = v
        def fake_delete(k):
            stored.pop(k, None)
        client.get = fake_get
        client.setex = fake_setex
        client.delete = fake_delete
        iface.rag_system.session_memory._get_client.return_value = client

        iface._set_memory_cache("u1", "t1", "memories_text")
        got = iface._get_memory_cache("u1", "t1")
        self.assertEqual(got, "memories_text")

    @patch("core.chat_interface.config")
    def test_cache_expires_after_max_turns(self, mock_config):
        mock_config.USER_MEMORY_CACHE_TTL_SECONDS = 300
        mock_config.USER_MEMORY_CACHE_MAX_TURNS = 3

        iface = self._make_interface()
        stored = {}

        client = MagicMock()
        client.get = lambda k: stored.get(k)
        def fake_setex(k, _ttl, v):
            stored[k] = v
        client.setex = fake_setex
        def fake_delete(k):
            stored.pop(k, None)
        client.delete = fake_delete
        iface.rag_system.session_memory._get_client.return_value = client

        iface._set_memory_cache("u1", "t1", "data")
        # Hit 1, 2 → still cached; hit 3 → expired
        self.assertEqual(iface._get_memory_cache("u1", "t1"), "data")
        self.assertEqual(iface._get_memory_cache("u1", "t1"), "data")
        self.assertIsNone(iface._get_memory_cache("u1", "t1"))  # max_turns reached

    @patch("core.chat_interface.config")
    def test_invalidate_drops_cache(self, mock_config):
        mock_config.USER_MEMORY_CACHE_TTL_SECONDS = 300
        mock_config.USER_MEMORY_CACHE_MAX_TURNS = 5

        iface = self._make_interface()
        stored = {}

        client = MagicMock()
        client.get = lambda k: stored.get(k)
        def fake_setex(k, _ttl, v):
            stored[k] = v
        client.setex = fake_setex
        def fake_delete(k):
            stored.pop(k, None)
        client.delete = fake_delete
        iface.rag_system.session_memory._get_client.return_value = client

        iface._set_memory_cache("u1", "t1", "data")
        iface._invalidate_memory_cache("u1", "t1")
        self.assertIsNone(iface._get_memory_cache("u1", "t1"))

    def test_no_redis_client_returns_none_safely(self):
        iface = self._make_interface()
        iface.rag_system.session_memory._get_client.return_value = None
        self.assertIsNone(iface._get_memory_cache("u1", "t1"))
        # set/invalidate should not raise
        iface._set_memory_cache("u1", "t1", "data")
        iface._invalidate_memory_cache("u1", "t1")


class TestFetchUserMemoriesIntegration(unittest.TestCase):
    """Test that _fetch_user_memories correctly composes skip + cache + retrieval."""

    def _make_interface(self, retrieve_returns=None):
        from core.chat_interface import ChatInterface
        instance = object.__new__(ChatInterface)
        instance.rag_system = MagicMock()
        instance.rag_system.user_memory_store.retrieve_memories.return_value = retrieve_returns or []
        # No Redis client → cache layer no-ops
        instance.rag_system.session_memory._get_client.return_value = None
        return instance

    @patch("core.chat_interface.config")
    def test_skip_returns_empty_without_db_call(self, mock_config):
        mock_config.USER_MEMORY_SKIP_TRIVIAL_INTENT = True
        mock_config.USER_MEMORY_MAX_RETRIEVED = 5

        iface = self._make_interface()
        result = iface._fetch_user_memories("u1", "你好", "t1")
        self.assertEqual(result, "")
        iface.rag_system.user_memory_store.retrieve_memories.assert_not_called()

    @patch("core.chat_interface.config")
    def test_real_query_calls_retrieval_and_formats(self, mock_config):
        mock_config.USER_MEMORY_SKIP_TRIVIAL_INTENT = True
        mock_config.USER_MEMORY_MAX_RETRIEVED = 5
        mock_config.USER_MEMORY_CACHE_TTL_SECONDS = 300
        mock_config.USER_MEMORY_CACHE_MAX_TURNS = 5

        iface = self._make_interface(retrieve_returns=[
            {"memory_type": "medical", "importance": 10, "content": "对青霉素过敏"},
            {"memory_type": "medical", "importance": 8, "content": "有高血压"},
        ])
        result = iface._fetch_user_memories("u1", "我头疼怎么办", "t1")
        self.assertIn("青霉素过敏", result)
        self.assertIn("高血压", result)
        iface.rag_system.user_memory_store.retrieve_memories.assert_called_once()


if __name__ == "__main__":
    unittest.main()
