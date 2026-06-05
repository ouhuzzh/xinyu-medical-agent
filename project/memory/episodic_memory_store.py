"""Episodic memory store — L4 of the six-layer cognitive memory architecture.

Stores the full conversation timeline per user, enabling temporal and semantic
queries like "what did I ask about last Tuesday?" or "when did we discuss
headaches?".

Each turn is stored with user_message, assistant_message, embedding, and
timestamp.  Retrieval supports both time-range filtering and vector similarity.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

import config
import psycopg

logger = logging.getLogger(__name__)


def _vector_literal(values: List[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


class EpisodicMemoryStore:
    """PostgreSQL-backed episodic memory with pgvector similarity + time-range search."""

    def __init__(self):
        self._conninfo = (
            f"host={config.POSTGRES_HOST} "
            f"port={config.POSTGRES_PORT} "
            f"dbname={config.POSTGRES_DB} "
            f"user={config.POSTGRES_USER} "
            f"password={config.POSTGRES_PASSWORD}"
        )
        self._embeddings = None
        self._embeddings_checked = False
        self._embeddings_lock = threading.Lock()

    def _connect(self):
        return psycopg.connect(self._conninfo)

    def _get_embeddings(self):
        if self._embeddings_checked:
            return self._embeddings
        with self._embeddings_lock:
            if self._embeddings_checked:
                return self._embeddings
            try:
                from model_factory import get_embedding_model
                self._embeddings = get_embedding_model()
                self._embeddings.embed_query("test")
            except Exception:
                logger.warning("Embedding model unavailable for episodic memory.", exc_info=True)
                self._embeddings = None
            self._embeddings_checked = True
        return self._embeddings

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_turn(
        self,
        user_id: str,
        thread_id: str,
        turn_index: int,
        user_message: str,
        assistant_message: str,
    ) -> int:
        """Save a conversation turn to the episodic timeline."""
        embedding = None
        embeddings = self._get_embeddings()
        if embeddings is not None:
            try:
                # Embed the user message for semantic search
                embedding = embeddings.embed_query(user_message)
            except Exception:
                logger.warning("Failed to embed episodic turn.", exc_info=True)

        embedding_sql = "NULL"
        params: list[Any] = [user_id, thread_id, turn_index, user_message, assistant_message]
        if embedding is not None:
            embedding_sql = "CAST(%s AS vector)"
            params.append(_vector_literal(embedding))

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO episodic_memories (user_id, thread_id, turn_index, user_message, assistant_message, embedding)
                    VALUES (%s, %s, %s, %s, %s, {embedding_sql})
                    RETURNING id
                    """,
                    params,
                )
                row_id = cur.fetchone()[0]
            conn.commit()
        return row_id

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve_by_time(
        self,
        user_id: str,
        hours_back: int = 168,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent conversation turns by time range.

        Args:
            user_id: The user to search for.
            hours_back: How many hours back to search (default 7 days).
            limit: Maximum number of turns to return.
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, thread_id, turn_index, user_message, assistant_message, created_at
                    FROM episodic_memories
                    WHERE user_id = %s
                      AND created_at >= NOW() - INTERVAL '%s hours'
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (user_id, hours_back, limit),
                )
                rows = cur.fetchall()

        columns = ["id", "thread_id", "turn_index", "user_message", "assistant_message", "created_at"]
        return [dict(zip(columns, row)) for row in rows]

    def retrieve_by_semantic(
        self,
        user_id: str,
        query: str,
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve conversation turns by semantic similarity to a query."""
        top_k = top_k or config.EPISODIC_MEMORY_MAX_RETRIEVED

        embeddings = self._get_embeddings()
        if embeddings is None:
            # Fallback: return most recent turns
            return self.retrieve_by_time(user_id, hours_back=720, limit=top_k)

        try:
            query_embedding = embeddings.embed_query(query)
        except Exception:
            logger.warning("Query embedding failed for episodic search.", exc_info=True)
            return self.retrieve_by_time(user_id, hours_back=720, limit=top_k)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, thread_id, turn_index, user_message, assistant_message, created_at,
                           1 - (embedding <=> CAST(%s AS vector)) AS similarity
                    FROM episodic_memories
                    WHERE user_id = %s AND embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(%s AS vector)
                    LIMIT %s
                    """,
                    (_vector_literal(query_embedding), user_id, _vector_literal(query_embedding), top_k),
                )
                rows = cur.fetchall()

        columns = ["id", "thread_id", "turn_index", "user_message", "assistant_message", "created_at", "similarity"]
        return [dict(zip(columns, row)) for row in rows]

    def retrieve_by_time_and_semantic(
        self,
        user_id: str,
        query: str,
        hours_back: int = 168,
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve turns within a time range, ranked by semantic similarity."""
        top_k = top_k or config.EPISODIC_MEMORY_MAX_RETRIEVED

        embeddings = self._get_embeddings()
        if embeddings is None:
            return self.retrieve_by_time(user_id, hours_back=hours_back, limit=top_k)

        try:
            query_embedding = embeddings.embed_query(query)
        except Exception:
            return self.retrieve_by_time(user_id, hours_back=hours_back, limit=top_k)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, thread_id, turn_index, user_message, assistant_message, created_at,
                           1 - (embedding <=> CAST(%s AS vector)) AS similarity
                    FROM episodic_memories
                    WHERE user_id = %s
                      AND embedding IS NOT NULL
                      AND created_at >= NOW() - INTERVAL '%s hours'
                    ORDER BY embedding <=> CAST(%s AS vector)
                    LIMIT %s
                    """,
                    (_vector_literal(query_embedding), user_id, hours_back, _vector_literal(query_embedding), top_k),
                )
                rows = cur.fetchall()

        columns = ["id", "thread_id", "turn_index", "user_message", "assistant_message", "created_at", "similarity"]
        return [dict(zip(columns, row)) for row in rows]

    def get_turn_count(self, user_id: str) -> int:
        """Return total number of stored turns for a user."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM episodic_memories WHERE user_id = %s",
                    (user_id,),
                )
                return cur.fetchone()[0]

    def get_recent_turns_for_reflection(
        self, user_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent turns for reflection synthesis (L6)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, thread_id, turn_index, user_message, assistant_message, created_at
                    FROM episodic_memories
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (user_id, limit),
                )
                rows = cur.fetchall()

        columns = ["id", "thread_id", "turn_index", "user_message", "assistant_message", "created_at"]
        return [dict(zip(columns, row)) for row in rows]

    def clear_user_episodes(self, user_id: str):
        """Delete all episodic memories for a user (for testing)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM episodic_memories WHERE user_id = %s", (user_id,))
            conn.commit()

    def status_info(self) -> Dict[str, Any]:
        if not config.EPISODIC_MEMORY_ENABLED:
            return {"component": "episodic_memory", "mode": "disabled", "degraded": False,
                    "message": "Episodic memory is disabled by configuration."}
        embeddings = self._get_embeddings()
        if embeddings is not None:
            return {"component": "episodic_memory", "mode": "pgvector", "degraded": False,
                    "message": "Episodic memory with pgvector search is available."}
        return {"component": "episodic_memory", "mode": "time_only", "degraded": True,
                "message": "Embedding model unavailable; falling back to time-range retrieval."}
