from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Any

from fastapi import HTTPException, status

import config

try:
    import redis
except ImportError:  # pragma: no cover - optional dependency fallback
    redis = None


logger = logging.getLogger(__name__)

_runtime_redis_client = None
_runtime_redis_checked = False
_runtime_redis_lock = threading.Lock()


def _get_runtime_redis_client():
    global _runtime_redis_client, _runtime_redis_checked
    if not config.REDIS_ENABLED or not redis:
        return None
    if _runtime_redis_checked:
        return _runtime_redis_client

    with _runtime_redis_lock:
        if _runtime_redis_checked:
            return _runtime_redis_client
        _runtime_redis_checked = True
        try:
            client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=config.REDIS_DB,
                decode_responses=True,
            )
            client.ping()
            _runtime_redis_client = client
        except Exception as exc:
            logger.warning(
                "Runtime Redis guards unavailable; falling back to in-process auth guards: %s",
                exc,
            )
            _runtime_redis_client = None
        return _runtime_redis_client


class InMemoryRateLimiter:
    def __init__(self):
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, *, bucket: str, key: str, limit: int, window_seconds: int = 60):
        now = time.time()
        boundary = now - window_seconds
        with self._lock:
            events = self._events[(bucket, key)]
            while events and events[0] < boundary:
                events.popleft()
            if len(events) >= limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="请求过于频繁，请稍后再试。",
                )
            events.append(now)


class RedisRateLimiter:
    _CHECK_SCRIPT = """
    redis.call("ZREMRANGEBYSCORE", KEYS[1], "-inf", ARGV[2])
    local count = redis.call("ZCARD", KEYS[1])
    if count >= tonumber(ARGV[3]) then
        redis.call("EXPIRE", KEYS[1], tonumber(ARGV[4]))
        return {0, count}
    end
    redis.call("ZADD", KEYS[1], tonumber(ARGV[1]), ARGV[5])
    redis.call("EXPIRE", KEYS[1], tonumber(ARGV[4]))
    return {1, count + 1}
    """

    def __init__(self, client: Any):
        self._client = client

    def check(self, *, bucket: str, key: str, limit: int, window_seconds: int = 60):
        now_ms = int(time.time() * 1000)
        boundary_ms = now_ms - int(window_seconds * 1000)
        ttl_seconds = max(int(window_seconds), 1) + 1
        member = f"{now_ms}:{uuid.uuid4().hex}"
        allowed, _count = self._client.eval(
            self._CHECK_SCRIPT,
            1,
            self._key(bucket, key),
            now_ms,
            boundary_ms,
            int(limit),
            ttl_seconds,
            member,
        )
        if int(allowed) == 0:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="请求过于频繁，请稍后再试。",
            )

    @staticmethod
    def _key(bucket: str, key: str) -> str:
        return f"guard:ratelimit:{bucket}:{key}"


class LoginLockoutTracker:
    """In-memory sliding-window login lockout tracker."""

    def __init__(self):
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._locked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def _normalize(self, username: str) -> str:
        return (username or "").strip().lower()

    def assert_not_locked(self, username: str):
        key = self._normalize(username)
        if not key:
            return
        now = time.time()
        with self._lock:
            locked_until = self._locked_until.get(key, 0.0)
            if locked_until > now:
                remaining = int(locked_until - now)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"账号已被临时锁定，请在 {remaining} 秒后重试。",
                )
            if locked_until:
                self._locked_until.pop(key, None)
                self._failures.pop(key, None)

    def record_failure(self, username: str):
        key = self._normalize(username)
        if not key:
            return
        now = time.time()
        boundary = now - config.LOGIN_LOCKOUT_WINDOW_SECONDS
        with self._lock:
            events = self._failures[key]
            events.append(now)
            while events and events[0] < boundary:
                events.popleft()
            if len(events) >= config.LOGIN_LOCKOUT_MAX_ATTEMPTS:
                self._locked_until[key] = now + config.LOGIN_LOCKOUT_SECONDS
                events.clear()

    def record_success(self, username: str):
        key = self._normalize(username)
        if not key:
            return
        with self._lock:
            self._failures.pop(key, None)
            self._locked_until.pop(key, None)


class RedisLoginLockoutTracker:
    _FAILURE_SCRIPT = """
    redis.call("ZREMRANGEBYSCORE", KEYS[1], "-inf", ARGV[2])
    redis.call("ZADD", KEYS[1], tonumber(ARGV[1]), ARGV[5])
    redis.call("EXPIRE", KEYS[1], tonumber(ARGV[6]))
    local count = redis.call("ZCARD", KEYS[1])
    if count >= tonumber(ARGV[3]) then
        redis.call("SET", KEYS[2], "1", "EX", tonumber(ARGV[4]))
        redis.call("DEL", KEYS[1])
        return {1, count}
    end
    return {0, count}
    """

    def __init__(self, client: Any):
        self._client = client

    def _normalize(self, username: str) -> str:
        return (username or "").strip().lower()

    def assert_not_locked(self, username: str):
        key = self._normalize(username)
        if not key:
            return
        ttl_seconds = int(self._client.ttl(self._lock_key(key)) or 0)
        if ttl_seconds != -2:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"账号已被临时锁定，请在 {max(ttl_seconds, 1)} 秒后重试。",
            )

    def record_failure(self, username: str):
        key = self._normalize(username)
        if not key:
            return
        now_ms = int(time.time() * 1000)
        boundary_ms = now_ms - int(config.LOGIN_LOCKOUT_WINDOW_SECONDS * 1000)
        member = f"{now_ms}:{uuid.uuid4().hex}"
        self._client.eval(
            self._FAILURE_SCRIPT,
            2,
            self._failure_key(key),
            self._lock_key(key),
            now_ms,
            boundary_ms,
            int(config.LOGIN_LOCKOUT_MAX_ATTEMPTS),
            int(config.LOGIN_LOCKOUT_SECONDS),
            member,
            max(int(config.LOGIN_LOCKOUT_WINDOW_SECONDS), 1) + 1,
        )

    def record_success(self, username: str):
        key = self._normalize(username)
        if not key:
            return
        self._client.delete(self._failure_key(key), self._lock_key(key))

    @staticmethod
    def _failure_key(username: str) -> str:
        return f"guard:login_failures:{username}"

    @staticmethod
    def _lock_key(username: str) -> str:
        return f"guard:login_lock:{username}"


def build_rate_limiter():
    client = _get_runtime_redis_client()
    if client is not None:
        return RedisRateLimiter(client)
    return InMemoryRateLimiter()


def build_login_lockout_tracker():
    client = _get_runtime_redis_client()
    if client is not None:
        return RedisLoginLockoutTracker(client)
    return LoginLockoutTracker()


def get_runtime_guard_backends() -> dict[str, str]:
    backend = "redis" if _get_runtime_redis_client() is not None else "in_process"
    return {
        "rate_limit_backend": backend,
        "login_lockout_backend": backend,
    }


def reset_runtime_guard_clients_for_tests():
    global _runtime_redis_client, _runtime_redis_checked
    with _runtime_redis_lock:
        _runtime_redis_client = None
        _runtime_redis_checked = False
