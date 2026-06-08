"""Tests for the append-only audit log store."""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

from db.audit_log_store import AuditLogStore


class AuditLogStoreTests(unittest.TestCase):

    def setUp(self):
        # Use a mock connection that swallows everything (real DB integration
        # is covered by test_live_db_integration.py).
        self.store = AuditLogStore()

    def test_record_does_not_raise(self):
        with patch.object(self.store, "_connect") as mock_conn:
            mock_conn.return_value.__enter__.return_value = mock_conn.return_value
            self.store.record(action="test_action")
            # Must not raise

    def test_record_swallows_db_errors(self):
        with patch.object(self.store, "_connect") as mock_conn:
            mock_conn.return_value.__enter__.side_effect = Exception("DB gone")
            # Must NOT propagate
            self.store.record(action="test_action")

    def test_list_recent_returns_empty_on_db_error(self):
        with patch.object(self.store, "_connect") as mock_conn:
            mock_conn.return_value.__enter__.side_effect = Exception("DB gone")
            result = self.store.list_recent()
            self.assertEqual(result, [])

    def test_list_recent_with_filters(self):
        with patch.object(self.store, "_connect") as mock_conn:
            mock_conn.return_value.__enter__.return_value = mock_conn.return_value
            mock_cursor = mock_conn.return_value.cursor.return_value.__enter__.return_value
            mock_cursor.fetchall.return_value = []
            r1 = self.store.list_recent(actor_user_id="u1")
            self.assertEqual(r1, [])
            r2 = self.store.list_recent(action="user_login")
            self.assertEqual(r2, [])


if __name__ == "__main__":
    unittest.main()