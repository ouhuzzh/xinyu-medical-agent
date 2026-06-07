"""Tests for UserMemoryStore — unit tests with mocked psycopg."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

from memory.user_memory_store import UserMemoryStore  # noqa: E402


class TestUserMemoryStoreRecencyScore(unittest.TestCase):
    """Test the static _recency_score method with type-based decay."""

    def test_recent_timestamp_scores_high(self):
        now = datetime.now(timezone.utc)
        score = UserMemoryStore._recency_score(now)
        self.assertGreater(score, 0.95)

    def test_decision_decays_fast(self):
        """Decision memories decay quickly — 1 week = ~43%."""
        one_week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        score = UserMemoryStore._recency_score(one_week_ago, memory_type="decision")
        self.assertLess(score, 0.50)  # 0.005 * 168 = 0.84 decay → ~0.43

    def test_medical_decays_slow(self):
        """Medical memories persist — 1 week still ~92%."""
        one_week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        score = UserMemoryStore._recency_score(one_week_ago, memory_type="medical")
        self.assertGreater(score, 0.85)  # 0.0005 * 168 = 0.084 → ~0.92

    def test_one_month_decision_near_zero(self):
        """Decision after 30 days should be effectively expired."""
        one_month_ago = datetime.now(timezone.utc) - timedelta(days=30)
        score = UserMemoryStore._recency_score(one_month_ago, memory_type="decision")
        self.assertLess(score, 0.10)  # 0.005 * 720 = 3.6 → ~0.027

    def test_none_returns_default(self):
        score = UserMemoryStore._recency_score(None)
        self.assertEqual(score, 0.5)


class TestUserMemoryStoreStatusInfo(unittest.TestCase):
    """Test status_info with various config states."""

    def test_disabled(self):
        with patch("memory.user_memory_store.config") as mock_config:
            mock_config.USER_MEMORY_ENABLED = False
            store = UserMemoryStore()
            info = store.status_info()
            self.assertEqual(info["mode"], "disabled")
            self.assertFalse(info["degraded"])

    def test_enabled_with_embeddings(self):
        with patch("memory.user_memory_store.config") as mock_config:
            mock_config.USER_MEMORY_ENABLED = True
            mock_config.POSTGRES_HOST = "localhost"
            mock_config.POSTGRES_PORT = 5432
            mock_config.POSTGRES_DB = "test"
            mock_config.POSTGRES_USER = "test"
            mock_config.POSTGRES_PASSWORD = ""
            store = UserMemoryStore()
            store._embeddings = MagicMock()
            store._embeddings_checked = True
            info = store.status_info()
            self.assertEqual(info["mode"], "pgvector")
            self.assertFalse(info["degraded"])

    def test_enabled_without_embeddings(self):
        with patch("memory.user_memory_store.config") as mock_config:
            mock_config.USER_MEMORY_ENABLED = True
            mock_config.POSTGRES_HOST = "localhost"
            mock_config.POSTGRES_PORT = 5432
            mock_config.POSTGRES_DB = "test"
            mock_config.POSTGRES_USER = "test"
            mock_config.POSTGRES_PASSWORD = ""
            store = UserMemoryStore()
            store._embeddings = None
            store._embeddings_checked = True
            info = store.status_info()
            self.assertEqual(info["mode"], "importance_only")
            self.assertTrue(info["degraded"])


class TestUserMemoryStoreSaveMemory(unittest.TestCase):
    """Test save_memory with mocked DB and embedding."""

    def test_save_memory_without_embedding(self):
        """When embedding is unavailable, save with NULL embedding."""
        with patch("memory.user_memory_store.config") as mock_config:
            mock_config.USER_MEMORY_ENABLED = True
            mock_config.USER_MEMORY_DEDUP_SIMILARITY = 0.9
            mock_config.POSTGRES_HOST = "localhost"
            mock_config.POSTGRES_PORT = 5432
            mock_config.POSTGRES_DB = "test"
            mock_config.POSTGRES_USER = "test"
            mock_config.POSTGRES_PASSWORD = ""
            store = UserMemoryStore()
            store._embeddings = None
            store._embeddings_checked = True

            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = (42,)
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            with patch.object(store, '_connect', return_value=mock_conn):
                with patch.object(store, '_check_dedup', return_value=None):
                    memory_id = store.save_memory(
                        user_id="user1",
                        memory_type="medical",
                        content="对青霉素过敏",
                        importance=9,
                        source_thread_id="t1",
                    )

            self.assertEqual(memory_id, 42)
            mock_cursor.execute.assert_called_once()
            sql = mock_cursor.execute.call_args[0][0]
            self.assertIn("INSERT INTO user_memories", sql)


class TestUserMemoryStoreRetrieveImportanceOnly(unittest.TestCase):
    """Test importance-only fallback retrieval."""

    def test_retrieve_importance_only(self):
        with patch("memory.user_memory_store.config") as mock_config:
            mock_config.USER_MEMORY_ENABLED = True
            mock_config.USER_MEMORY_MAX_RETRIEVED = 5
            mock_config.POSTGRES_HOST = "localhost"
            mock_config.POSTGRES_PORT = 5432
            mock_config.POSTGRES_DB = "test"
            mock_config.POSTGRES_USER = "test"
            mock_config.POSTGRES_PASSWORD = ""
            store = UserMemoryStore()
            store._embeddings = None
            store._embeddings_checked = True

            now = datetime.now(timezone.utc)
            mock_row = (1, "medical", "对青霉素过敏", 9, "t1", 3, now, now)
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [mock_row]
            mock_conn.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

            with patch.object(store, '_connect', return_value=mock_conn):
                with patch.object(store, '_update_access_stats'):
                    result = store.retrieve_memories("user1", "我过敏了")

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["memory_type"], "medical")
            self.assertEqual(result[0]["content"], "对青霉素过敏")
            self.assertAlmostEqual(result[0]["score"], 0.9)


if __name__ == "__main__":
    unittest.main()
