import sys
import threading
import time
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from api.session_locks import RedisSessionLock, SessionLockRegistry  # noqa: E402


class FakeRedis:
    def __init__(self):
        self.values = {}
        self._lock = threading.Lock()

    def set(self, key, value, *, nx=False, ex=None):
        with self._lock:
            if nx and key in self.values:
                return False
            self.values[key] = value
            return True

    def eval(self, script, numkeys, key, token):
        with self._lock:
            if self.values.get(key) == token:
                self.values.pop(key, None)
                return 1
            return 0


class FakeSessionMemory:
    def __init__(self, client):
        self.client = client

    def _get_client(self):
        return self.client


class SessionLockTests(unittest.TestCase):
    def test_in_process_lock_blocks_same_thread(self):
        registry = SessionLockRegistry(session_memory=None)
        first = registry.get_lock("thread-1")
        second = registry.get_lock("thread-1")

        self.assertTrue(first.acquire(timeout=0.1))
        try:
            self.assertFalse(second.acquire(timeout=0.05))
        finally:
            first.release()

        self.assertTrue(second.acquire(timeout=0.1))
        second.release()

    def test_redis_lock_blocks_across_registries(self):
        client = FakeRedis()
        registry_a = SessionLockRegistry(FakeSessionMemory(client), lock_ttl_seconds=10)
        registry_b = SessionLockRegistry(FakeSessionMemory(client), lock_ttl_seconds=10)

        first = registry_a.get_lock("thread-1")
        second = registry_b.get_lock("thread-1")

        self.assertEqual(registry_a.backend_name(), "redis")
        self.assertTrue(first.acquire(timeout=0.1))
        try:
            started = time.monotonic()
            self.assertFalse(second.acquire(timeout=0.05))
            self.assertGreaterEqual(time.monotonic() - started, 0.04)
        finally:
            first.release()

        self.assertTrue(second.acquire(timeout=0.1))
        second.release()

    def test_redis_lock_release_is_token_checked(self):
        client = FakeRedis()
        first = RedisSessionLock(client, "lock-key", ttl_seconds=10)
        wrong_token_holder = RedisSessionLock(client, "lock-key", ttl_seconds=10)

        self.assertTrue(first.acquire(timeout=0.1))
        wrong_token_holder._acquired = True
        wrong_token_holder.release()
        self.assertIn("lock-key", client.values)
        first.release()
        self.assertNotIn("lock-key", client.values)


if __name__ == "__main__":
    unittest.main()
