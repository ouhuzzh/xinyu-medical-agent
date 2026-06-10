"""Unit tests for auth-route security hardening (rate limit + lockout).

These tests poke the in-memory trackers directly — no FastAPI client needed.
The HTTP-level integration is covered indirectly via the route signatures.
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

from fastapi import HTTPException


class FakeRequest:
    """Stand-in for FastAPI's Request — only what ``enforce_auth_rate_limit`` reads."""

    def __init__(self, ip: str = "1.2.3.4"):
        self.headers = {}
        self._ip = ip

    @property
    def client(self):
        class _C:
            host = self._ip
        return _C()


class FakeRedis:
    def __init__(self):
        self._strings = {}
        self._zsets = {}
        self._expires = {}

    def _prune_expired(self):
        now = time.time()
        expired = [key for key, deadline in self._expires.items() if deadline <= now]
        for key in expired:
            self._expires.pop(key, None)
            self._strings.pop(key, None)
            self._zsets.pop(key, None)

    def eval(self, script, numkeys, *args):
        self._prune_expired()
        if numkeys == 1:
            key, now_ms, boundary_ms, limit, ttl_seconds, member = args
            entries = self._zsets.setdefault(key, {})
            for existing, score in list(entries.items()):
                if score <= int(boundary_ms):
                    entries.pop(existing, None)
            count = len(entries)
            if count >= int(limit):
                self._expires[key] = time.time() + int(ttl_seconds)
                return [0, count]
            entries[str(member)] = int(now_ms)
            self._expires[key] = time.time() + int(ttl_seconds)
            return [1, count + 1]

        if numkeys == 2:
            failure_key, lock_key, now_ms, boundary_ms, threshold, lockout_seconds, member, ttl_seconds = args
            entries = self._zsets.setdefault(failure_key, {})
            for existing, score in list(entries.items()):
                if score <= int(boundary_ms):
                    entries.pop(existing, None)
            entries[str(member)] = int(now_ms)
            self._expires[failure_key] = time.time() + int(ttl_seconds)
            count = len(entries)
            if count >= int(threshold):
                self._strings[lock_key] = "1"
                self._expires[lock_key] = time.time() + int(lockout_seconds)
                self._zsets.pop(failure_key, None)
                self._expires.pop(failure_key, None)
                return [1, count]
            return [0, count]

        raise NotImplementedError(f"Unsupported numkeys={numkeys}")

    def ttl(self, key):
        self._prune_expired()
        deadline = self._expires.get(key)
        if deadline is None:
            return -2
        return max(int(deadline - time.time()), 0)

    def delete(self, *keys):
        self._prune_expired()
        deleted = 0
        for key in keys:
            existed = key in self._strings or key in self._zsets or key in self._expires
            self._strings.pop(key, None)
            self._zsets.pop(key, None)
            self._expires.pop(key, None)
            deleted += int(existed)
        return deleted


class LoginLockoutTests(unittest.TestCase):
    def setUp(self):
        # Import here so any config patches take effect, and reset module state.
        from api import auth as auth_module
        self.auth = auth_module
        # Replace the singleton with a fresh one for isolation between tests
        self.auth._login_lockout = auth_module.LoginLockoutTracker()

    def test_failures_below_threshold_do_not_lock(self):
        with patch("config.LOGIN_LOCKOUT_MAX_ATTEMPTS", 5):
            for _ in range(4):
                self.auth.record_login_failure("alice")
            # Still unlocked
            self.auth.assert_login_not_locked("alice")

    def test_nth_failure_locks_out(self):
        with patch("config.LOGIN_LOCKOUT_MAX_ATTEMPTS", 3), \
             patch("config.LOGIN_LOCKOUT_SECONDS", 60), \
             patch("config.LOGIN_LOCKOUT_WINDOW_SECONDS", 600):
            for _ in range(3):
                self.auth.record_login_failure("bob")
            with self.assertRaises(HTTPException) as ctx:
                self.auth.assert_login_not_locked("bob")
            self.assertEqual(ctx.exception.status_code, 429)
            self.assertIn("锁定", ctx.exception.detail)

    def test_success_clears_failures(self):
        with patch("config.LOGIN_LOCKOUT_MAX_ATTEMPTS", 3):
            self.auth.record_login_failure("carol")
            self.auth.record_login_failure("carol")
            self.auth.record_login_success("carol")
            # Counter cleared — should take another 3 failures to lock
            self.auth.record_login_failure("carol")
            self.auth.record_login_failure("carol")
            self.auth.assert_login_not_locked("carol")

    def test_lockout_expires(self):
        with patch("config.LOGIN_LOCKOUT_MAX_ATTEMPTS", 2), \
             patch("config.LOGIN_LOCKOUT_SECONDS", 1), \
             patch("config.LOGIN_LOCKOUT_WINDOW_SECONDS", 600):
            self.auth.record_login_failure("dave")
            self.auth.record_login_failure("dave")
            with self.assertRaises(HTTPException):
                self.auth.assert_login_not_locked("dave")
            time.sleep(1.1)
            # Lockout should have expired now
            self.auth.assert_login_not_locked("dave")

    def test_case_insensitive_username(self):
        with patch("config.LOGIN_LOCKOUT_MAX_ATTEMPTS", 2):
            self.auth.record_login_failure("Eve")
            self.auth.record_login_failure("EVE")
            with self.assertRaises(HTTPException):
                self.auth.assert_login_not_locked("eve")

    def test_empty_username_is_noop(self):
        self.auth.record_login_failure("")
        self.auth.record_login_failure("   ")
        # Should never raise — empty username is treated as "no key"
        self.auth.assert_login_not_locked("")


class AuthRateLimitTests(unittest.TestCase):
    def setUp(self):
        from api import auth as auth_module
        self.auth = auth_module
        # Fresh rate limiter for isolation
        self.auth._rate_limiter = auth_module.InMemoryRateLimiter()

    def test_under_limit_passes(self):
        with patch("config.API_RATE_LIMIT_AUTH_PER_MINUTE", 5):
            req = FakeRequest("10.0.0.1")
            for _ in range(5):
                self.auth.enforce_auth_rate_limit(req)

    def test_over_limit_raises_429(self):
        with patch("config.API_RATE_LIMIT_AUTH_PER_MINUTE", 3):
            req = FakeRequest("10.0.0.2")
            for _ in range(3):
                self.auth.enforce_auth_rate_limit(req)
            with self.assertRaises(HTTPException) as ctx:
                self.auth.enforce_auth_rate_limit(req)
            self.assertEqual(ctx.exception.status_code, 429)

    def test_separate_ips_have_independent_quotas(self):
        with patch("config.API_RATE_LIMIT_AUTH_PER_MINUTE", 2):
            r1, r2 = FakeRequest("10.0.0.3"), FakeRequest("10.0.0.4")
            for _ in range(2):
                self.auth.enforce_auth_rate_limit(r1)
                self.auth.enforce_auth_rate_limit(r2)
            # Both at limit — both should now fail
            with self.assertRaises(HTTPException):
                self.auth.enforce_auth_rate_limit(r1)
            with self.assertRaises(HTTPException):
                self.auth.enforce_auth_rate_limit(r2)

    def test_x_forwarded_for_is_honoured(self):
        with patch("config.API_RATE_LIMIT_AUTH_PER_MINUTE", 1):
            req = FakeRequest("internal-proxy")
            req.headers = {"x-forwarded-for": "203.0.113.5, internal-proxy"}
            self.auth.enforce_auth_rate_limit(req)
            with self.assertRaises(HTTPException):
                self.auth.enforce_auth_rate_limit(req)
            # A request from a different real client IP should still pass
            req2 = FakeRequest("internal-proxy")
            req2.headers = {"x-forwarded-for": "203.0.113.99, internal-proxy"}
            self.auth.enforce_auth_rate_limit(req2)


class RedisGuardTests(unittest.TestCase):
    def setUp(self):
        from api import auth as auth_module
        self.auth = auth_module
        self.redis = FakeRedis()

    def test_redis_rate_limiter_enforces_limit(self):
        limiter = self.auth.RedisRateLimiter(self.redis)
        limiter.check(bucket="auth", key="1.1.1.1", limit=2, window_seconds=60)
        limiter.check(bucket="auth", key="1.1.1.1", limit=2, window_seconds=60)
        with self.assertRaises(HTTPException) as ctx:
            limiter.check(bucket="auth", key="1.1.1.1", limit=2, window_seconds=60)
        self.assertEqual(ctx.exception.status_code, 429)

    def test_redis_login_lockout_enforces_and_expires(self):
        tracker = self.auth.RedisLoginLockoutTracker(self.redis)
        with patch("config.LOGIN_LOCKOUT_MAX_ATTEMPTS", 2), \
             patch("config.LOGIN_LOCKOUT_SECONDS", 1), \
             patch("config.LOGIN_LOCKOUT_WINDOW_SECONDS", 600):
            tracker.record_failure("alice")
            tracker.record_failure("alice")
            with self.assertRaises(HTTPException) as ctx:
                tracker.assert_not_locked("alice")
            self.assertEqual(ctx.exception.status_code, 429)
            time.sleep(1.1)
            tracker.assert_not_locked("alice")

    def test_redis_login_success_clears_lock_and_failures(self):
        tracker = self.auth.RedisLoginLockoutTracker(self.redis)
        with patch("config.LOGIN_LOCKOUT_MAX_ATTEMPTS", 2), \
             patch("config.LOGIN_LOCKOUT_SECONDS", 60), \
             patch("config.LOGIN_LOCKOUT_WINDOW_SECONDS", 600):
            tracker.record_failure("Bob")
            tracker.record_failure("BOB")
            with self.assertRaises(HTTPException):
                tracker.assert_not_locked("bob")
            tracker.record_success("bob")
            tracker.assert_not_locked("bob")
            tracker.record_failure("bob")
            tracker.assert_not_locked("bob")


if __name__ == "__main__":
    unittest.main()
