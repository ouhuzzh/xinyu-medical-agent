"""Tests for MCP server registry reachability check.

The check is best-effort TCP-only — doesn't validate the MCP handshake,
just confirms something is listening.  Good enough to surface "you forgot
to start the mock server" at boot.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))


class FakeListener:
    """Minimal TCP listener — accepts a connection then closes it."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.sock.listen(1)
        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._stop = threading.Event()
        self.thread.start()

    def _accept_loop(self):
        self.sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
                conn.close()
            except socket.timeout:
                continue
            except OSError:
                break

    def close(self):
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass


class MCPReachabilityTests(unittest.TestCase):
    def setUp(self):
        from mcp_integration.mcp_server_registry import MCPServerRegistry
        self.MCPServerRegistry = MCPServerRegistry
        self.registry = MCPServerRegistry()

    def test_reachable_host_is_detected(self):
        listener = FakeListener()
        try:
            fake_hospitals = [
                {"code": "xiehe", "name": "协和",
                 "mcp_url": f"http://127.0.0.1:{listener.port}/mcp",
                 "auth_type": "bearer", "is_active": True,
                 "id": 1, "description": "", "created_at": None},
            ]
            with patch.object(self.registry, "list_active", return_value=fake_hospitals):
                results = self.registry.check_reachability(timeout=1.0)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0]["reachable"])
            self.assertEqual(results[0]["error"], "")
            self.assertEqual(results[0]["code"], "xiehe")
        finally:
            listener.close()

    def test_unreachable_host_is_reported(self):
        # 127.0.0.1:1 is reliably refused on most systems (privileged port, nothing listens)
        fake_hospitals = [
            {"code": "renji", "name": "仁济",
             "mcp_url": "http://127.0.0.1:1/mcp",
             "auth_type": "bearer", "is_active": True,
             "id": 2, "description": "", "created_at": None},
        ]
        with patch.object(self.registry, "list_active", return_value=fake_hospitals):
            results = self.registry.check_reachability(timeout=0.5)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["reachable"])
        self.assertNotEqual(results[0]["error"], "")

    def test_invalid_url_does_not_raise(self):
        fake_hospitals = [
            {"code": "broken", "name": "坏的",
             "mcp_url": "not-a-url",
             "auth_type": "bearer", "is_active": True,
             "id": 3, "description": "", "created_at": None},
        ]
        with patch.object(self.registry, "list_active", return_value=fake_hospitals):
            results = self.registry.check_reachability(timeout=0.5)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["reachable"])

    def test_mixed_results(self):
        """A reachable + an unreachable in one call — both should be reported correctly."""
        listener = FakeListener()
        try:
            fake_hospitals = [
                {"code": "good", "name": "Good",
                 "mcp_url": f"http://127.0.0.1:{listener.port}/mcp",
                 "auth_type": "bearer", "is_active": True,
                 "id": 1, "description": "", "created_at": None},
                {"code": "bad", "name": "Bad",
                 "mcp_url": "http://127.0.0.1:1/mcp",
                 "auth_type": "bearer", "is_active": True,
                 "id": 2, "description": "", "created_at": None},
            ]
            with patch.object(self.registry, "list_active", return_value=fake_hospitals):
                results = self.registry.check_reachability(timeout=0.5)
            self.assertEqual(len(results), 2)
            by_code = {r["code"]: r for r in results}
            self.assertTrue(by_code["good"]["reachable"])
            self.assertFalse(by_code["bad"]["reachable"])
        finally:
            listener.close()

    def test_empty_registry_returns_empty(self):
        with patch.object(self.registry, "list_active", return_value=[]):
            self.assertEqual(self.registry.check_reachability(), [])


if __name__ == "__main__":
    unittest.main()
