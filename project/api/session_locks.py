"""Session-level concurrency locks for chat streams.

The public lock object intentionally mirrors ``threading.Lock`` with
``acquire(timeout=...)`` and ``release()`` so API streaming code does not care
whether the backend is local memory or Redis.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

import config


logger = logging.getLogger(__name__)


class InProcessSessionLock:
    def __init__(self, lock: threading.Lock):
        self._lock = lock

    def acquire(self, timeout: float | None = None) -> bool:
        if timeout is None:
            return self._lock.acquire()
        return self._lock.acquire(timeout=max(float(timeout), 0.0))

    def release(self) -> None:
        self._lock.release()


class RedisSessionLock:
    """Small Redis lock with token-checked release.

    This is intentionally simple: it prevents concurrent chat turns for the
    same thread across API workers, but it is not a general distributed lock
    framework. The TTL is bounded to avoid orphaned locks if a worker dies.
    """

    _RELEASE_SCRIPT = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    end
    return 0
    """

    def __init__(self, client: Any, key: str, *, ttl_seconds: int):
        self._client = client
        self._key = key
        self._ttl_seconds = max(int(ttl_seconds), 1)
        self._token = uuid.uuid4().hex
        self._acquired = False

    def acquire(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(float(timeout), 0.0)
        while True:
            acquired = self._client.set(
                self._key,
                self._token,
                nx=True,
                ex=self._ttl_seconds,
            )
            if acquired:
                self._acquired = True
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.1)

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            self._client.eval(self._RELEASE_SCRIPT, 1, self._key, self._token)
        except Exception:
            logger.warning("Failed to release Redis session lock %s", self._key, exc_info=True)
        finally:
            self._acquired = False


class SessionLockRegistry:
    """Factory/cache for chat thread locks.

    Redis is preferred when available because it works across multiple API
    workers. If Redis is disabled or unavailable in development, the registry
    falls back to process-local locks.
    """

    def __init__(self, session_memory=None, *, lock_ttl_seconds: int | None = None):
        self._session_memory = session_memory
        self._lock_ttl_seconds = lock_ttl_seconds or int(config.GRAPH_STREAM_MAX_SECONDS + 60)
        self._local_locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._redis_unavailable_logged = False

    def backend_name(self) -> str:
        return "redis" if self._get_redis_client() is not None else "in_process"

    def get_lock(self, thread_id: str):
        client = self._get_redis_client()
        if client is not None:
            return RedisSessionLock(
                client,
                self._redis_key(thread_id),
                ttl_seconds=self._lock_ttl_seconds,
            )
        return InProcessSessionLock(self._get_local_lock(thread_id))

    def _get_local_lock(self, thread_id: str) -> threading.Lock:
        with self._guard:
            lock = self._local_locks.get(thread_id)
            if lock is None:
                lock = threading.Lock()
                self._local_locks[thread_id] = lock
            return lock

    def _get_redis_client(self):
        if self._session_memory is None:
            return None
        try:
            return self._session_memory._get_client()
        except Exception as exc:
            if not self._redis_unavailable_logged:
                logger.warning(
                    "Redis session lock unavailable; falling back to in-process locks: %s",
                    exc,
                )
                self._redis_unavailable_logged = True
            if config.APP_ENV != "development" and config.REDIS_ENABLED:
                raise
            return None

    @staticmethod
    def _redis_key(thread_id: str) -> str:
        return f"chat:session:{thread_id}:stream_lock"


# Backward-compatible alias for older imports/tests.
ThreadLockRegistry = SessionLockRegistry
