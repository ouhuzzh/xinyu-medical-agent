from __future__ import annotations

import logging
import threading

import config
import psycopg

try:
    from psycopg_pool import ConnectionPool
except ImportError:  # pragma: no cover - optional dependency
    ConnectionPool = None


logger = logging.getLogger(__name__)
_pool = None
_pool_lock = threading.Lock()


class PooledConnectionHandle:
    """Wraps a pooled connection so it can be returned to the pool on close."""

    def __init__(self, pool, conn):
        self._pool = pool
        self._conn = conn
        self._closed = False

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        except Exception:
            # Commit or rollback failed — try rollback as best effort
            try:
                self._conn.rollback()
            except Exception:
                pass
        finally:
            self.close()
        return False

    def close(self):
        if self._closed:
            return
        self._pool.putconn(self._conn)
        self._closed = True


def get_conninfo() -> str:
    return (
        f"host={config.POSTGRES_HOST} "
        f"port={config.POSTGRES_PORT} "
        f"dbname={config.POSTGRES_DB} "
        f"user={config.POSTGRES_USER} "
        f"password={config.POSTGRES_PASSWORD}"
    )


def get_connection_pool():
    global _pool
    if ConnectionPool is None:
        return None
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ConnectionPool(
                    conninfo=get_conninfo(),
                    min_size=config.POSTGRES_POOL_MIN_SIZE,
                    max_size=config.POSTGRES_POOL_MAX_SIZE,
                    timeout=config.POSTGRES_POOL_TIMEOUT_SECONDS,
                    open=False,
                )
                _pool.open(wait=False)
                logger.info("PostgreSQL connection pool initialized.")
    return _pool


def connect():
    pool = get_connection_pool()
    if pool is None:
        return psycopg.connect(get_conninfo())
    return PooledConnectionHandle(pool, pool.getconn())


def close_connection_pool():
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
