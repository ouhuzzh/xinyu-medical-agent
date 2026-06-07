"""User-level long-term memory store backed by PostgreSQL + pgvector.

Stores structured user memories (preferences, facts, medical history, decisions)
with importance scoring and vector similarity search.  Retrieval uses a
three-factor ranking: recency × importance × relevance.

Gracefully degrades when the embedding model is unavailable — saves memories
with NULL embedding and falls back to importance-only retrieval.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from typing import Any, Dict, List, Optional

import config
import psycopg

logger = logging.getLogger(__name__)


def _vector_literal(values: List[float]) -> str:
    """Format a float list as a pgvector-compatible string literal."""
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


class UserMemoryStore:
    """PostgreSQL-backed user-level memory store with pgvector similarity search."""

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
        """Lazy-init the embedding model.  Returns None on failure (graceful degradation)."""
        if self._embeddings_checked:
            return self._embeddings
        with self._embeddings_lock:
            if self._embeddings_checked:
                return self._embeddings
            try:
                from model_factory import get_embedding_model
                self._embeddings = get_embedding_model()
                # Quick sanity check
                self._embeddings.embed_query("test")
            except Exception:
                logger.warning("Embedding model unavailable; user memory will use importance-only retrieval.", exc_info=True)
                self._embeddings = None
            self._embeddings_checked = True
        return self._embeddings

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_memory(
        self,
        user_id: str,
        memory_type: str,
        content: str,
        importance: int,
        source_thread_id: str = "",
    ) -> int:
        """Save a user memory.  Generates embedding, checks for dedup, merges or inserts.

        Returns the memory row id.
        """
        embedding = None
        embeddings = self._get_embeddings()
        if embeddings is not None:
            try:
                embedding = embeddings.embed_query(content)
            except Exception:
                logger.warning("Failed to embed memory content; storing without vector.", exc_info=True)

        # Dedup check
        if embedding is not None:
            existing_id = self._check_dedup(user_id, embedding)
            if existing_id is not None:
                self._merge_memory(existing_id, content, importance, embedding)
                return existing_id

        # Insert new memory
        embedding_sql = "NULL"
        params: list[Any] = [user_id, memory_type, content, importance]
        if embedding is not None:
            embedding_sql = f"CAST(%s AS vector)"
            params.append(_vector_literal(embedding))

        if source_thread_id:
            params.append(source_thread_id)
            thread_sql = "%s"
        else:
            thread_sql = "NULL"

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO user_memories (user_id, memory_type, content, importance, embedding, source_thread_id)
                    VALUES (%s, %s, %s, %s, {embedding_sql}, {thread_sql})
                    RETURNING id
                    """,
                    params,
                )
                memory_id = cur.fetchone()[0]
            conn.commit()
        return memory_id

    # ------------------------------------------------------------------
    # Retrieve (three-factor ranking)
    # ------------------------------------------------------------------

    def retrieve_memories(
        self,
        user_id: str,
        query: str,
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve user memories ranked by recency × importance × relevance.

        Returns a list of dicts with keys: id, memory_type, content, importance, score.
        """
        top_k = top_k or config.USER_MEMORY_MAX_RETRIEVED

        embeddings = self._get_embeddings()
        if embeddings is None:
            return self._retrieve_importance_only(user_id, top_k)

        try:
            query_embedding = embeddings.embed_query(query)
        except Exception:
            logger.warning("Query embedding failed; falling back to importance-only retrieval.", exc_info=True)
            return self._retrieve_importance_only(user_id, top_k)

        candidates = self._vector_search(user_id, query_embedding, top_k * 3)
        if not candidates:
            return []

        # Three-factor scoring
        scored = []
        for mem in candidates:
            recency = self._recency_score(
                mem.get("last_accessed_at") or mem["created_at"],
                memory_type=mem.get("memory_type", "fact"),
            )
            importance = mem["importance"] / 10.0
            relevance = max(0.0, 1.0 - (mem.get("distance", 1.0)))
            score = (
                config.USER_MEMORY_RECENCY_WEIGHT * recency
                + config.USER_MEMORY_IMPORTANCE_WEIGHT * importance
                + config.USER_MEMORY_RELEVANCE_WEIGHT * relevance
            )
            scored.append({**mem, "score": score})

        scored.sort(key=lambda m: m["score"], reverse=True)
        result = scored[:top_k]

        # Update access stats
        self._update_access_stats([m["id"] for m in result])

        return result

    # ------------------------------------------------------------------
    # Simple queries (testing / admin)
    # ------------------------------------------------------------------

    def get_memories_for_user(
        self, user_id: str, memory_type: str | None = None
    ) -> List[Dict[str, Any]]:
        """Return all memories for a user, optionally filtered by type."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                if memory_type:
                    cur.execute(
                        """
                        SELECT id, user_id, memory_type, content, importance,
                               source_thread_id, access_count, last_accessed_at, created_at, updated_at
                        FROM user_memories
                        WHERE user_id = %s AND memory_type = %s
                        ORDER BY importance DESC, created_at DESC
                        """,
                        (user_id, memory_type),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, user_id, memory_type, content, importance,
                               source_thread_id, access_count, last_accessed_at, created_at, updated_at
                        FROM user_memories
                        WHERE user_id = %s
                        ORDER BY importance DESC, created_at DESC
                        """,
                        (user_id,),
                    )
                rows = cur.fetchall()

        columns = [
            "id", "user_id", "memory_type", "content", "importance",
            "source_thread_id", "access_count", "last_accessed_at", "created_at", "updated_at",
        ]
        return [dict(zip(columns, row)) for row in rows]

    def _update_importance(self, memory_id: int, new_importance: int):
        """Update the importance of a memory (used for contradiction deprecation)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE user_memories SET importance = %s, updated_at = NOW() WHERE id = %s",
                    (new_importance, memory_id),
                )
            conn.commit()

    def clear_user_memories(self, user_id: str):
        """Delete all memories for a user (for testing/cleanup)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM user_memories WHERE user_id = %s", (user_id,))
            conn.commit()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status_info(self) -> Dict[str, Any]:
        """Return component status following RedisSessionMemory.status_info() pattern."""
        if not config.USER_MEMORY_ENABLED:
            return {
                "component": "user_memory",
                "mode": "disabled",
                "degraded": False,
                "message": "User memory is disabled by configuration.",
            }
        embeddings = self._get_embeddings()
        if embeddings is not None:
            return {
                "component": "user_memory",
                "mode": "pgvector",
                "degraded": False,
                "message": "User memory store with pgvector similarity is available.",
            }
        return {
            "component": "user_memory",
            "mode": "importance_only",
            "degraded": True,
            "message": "Embedding model unavailable; falling back to importance-only retrieval.",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_dedup(self, user_id: str, embedding: List[float]) -> Optional[int]:
        """Find an existing memory for the same user with cosine similarity > threshold."""
        threshold = config.USER_MEMORY_DEDUP_SIMILARITY
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, 1 - (embedding <=> CAST(%s AS vector)) AS similarity
                    FROM user_memories
                    WHERE user_id = %s AND embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(%s AS vector)
                    LIMIT 1
                    """,
                    (_vector_literal(embedding), user_id, _vector_literal(embedding)),
                )
                row = cur.fetchone()
        if row and row[1] >= threshold:
            return row[0]
        return None

    def _merge_memory(
        self, existing_id: int, new_content: str, new_importance: int, new_embedding: List[float]
    ):
        """Merge a new memory into an existing one, keeping the higher importance."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                # Read existing
                cur.execute(
                    "SELECT content, importance, merged_from FROM user_memories WHERE id = %s",
                    (existing_id,),
                )
                row = cur.fetchone()
                if not row:
                    return
                old_content, old_importance, old_merged = row

                # Keep the higher-importance content as primary
                if new_importance >= old_importance:
                    primary_content = new_content
                    primary_importance = new_importance
                    merged_list = list(old_merged or []) + [old_content]
                else:
                    primary_content = old_content
                    primary_importance = old_importance
                    merged_list = list(old_merged or []) + [new_content]

                cur.execute(
                    """
                    UPDATE user_memories
                    SET content = %s,
                        importance = %s,
                        embedding = CAST(%s AS vector),
                        merged_from = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (primary_content, primary_importance, _vector_literal(new_embedding),
                     json.dumps(merged_list, ensure_ascii=False), existing_id),
                )
            conn.commit()

    def _vector_search(
        self, user_id: str, query_embedding: List[float], limit: int
    ) -> List[Dict[str, Any]]:
        """Search user memories by vector similarity."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, memory_type, content, importance,
                           source_thread_id, access_count, last_accessed_at,
                           created_at, embedding <=> CAST(%s AS vector) AS distance
                    FROM user_memories
                    WHERE user_id = %s AND embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(%s AS vector)
                    LIMIT %s
                    """,
                    (_vector_literal(query_embedding), user_id, _vector_literal(query_embedding), limit),
                )
                rows = cur.fetchall()

        columns = [
            "id", "memory_type", "content", "importance",
            "source_thread_id", "access_count", "last_accessed_at",
            "created_at", "distance",
        ]
        return [dict(zip(columns, row)) for row in rows]

    def _retrieve_importance_only(self, user_id: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback retrieval when embeddings are unavailable."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, memory_type, content, importance,
                           source_thread_id, access_count, last_accessed_at, created_at
                    FROM user_memories
                    WHERE user_id = %s
                    ORDER BY importance DESC, COALESCE(last_accessed_at, created_at) DESC
                    LIMIT %s
                    """,
                    (user_id, top_k),
                )
                rows = cur.fetchall()

        columns = [
            "id", "memory_type", "content", "importance",
            "source_thread_id", "access_count", "last_accessed_at", "created_at",
        ]
        result = [dict(zip(columns, row)) for row in rows]
        # Assign a simple score based on importance only
        for m in result:
            m["score"] = m["importance"] / 10.0

        self._update_access_stats([m["id"] for m in result])
        return result

    # P1: type-specific decay rates (lower = slower decay = stays relevant longer)
    _TYPE_DECAY_RATES = {
        "medical": 0.0005,     # very slow — allergies/chronic conditions persist
        "preference": 0.002,   # moderate — preferences may change
        "fact": 0.001,         # slow — personal facts are stable
        "decision": 0.005,     # fast — decisions expire quickly
    }

    @staticmethod
    def _recency_score(timestamp, memory_type: str = "fact") -> float:
        """Exponential decay with type-specific rates.

        Lower decay rate = slower decay = stays relevant longer.
        - medical: ~0.90 after 1 week, ~0.67 after 1 month
        - fact: ~0.85 after 1 week, ~0.55 after 1 month
        - preference: ~0.71 after 1 week, ~0.37 after 1 month
        - decision: ~0.43 after 1 week, ~0.10 after 1 month
        """
        if timestamp is None:
            return 0.5
        from datetime import datetime, timezone
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        now = datetime.now(timezone.utc)
        ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        hours = max(0.0, (now - ts).total_seconds() / 3600)
        rate = UserMemoryStore._TYPE_DECAY_RATES.get(memory_type, 0.001)
        return math.exp(-rate * hours)

    def _update_access_stats(self, memory_ids: List[int]):
        """Increment access_count and set last_accessed_at for retrieved memories."""
        if not memory_ids:
            return
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE user_memories
                        SET access_count = access_count + 1,
                            last_accessed_at = NOW()
                        WHERE id = ANY(%s)
                        """,
                        (memory_ids,),
                    )
                conn.commit()
        except Exception:
            logger.warning("Failed to update memory access stats", exc_info=True)
