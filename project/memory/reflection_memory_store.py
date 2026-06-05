"""Reflection memory — L6 of the six-layer cognitive memory architecture.

Periodically synthesizes higher-order abstractions from groups of related
memories.  For example, from three separate observations about headaches,
the reflector might produce: "User has chronic headache concerns, possibly
tension-type, recurring over multiple sessions."

Reflection runs asynchronously after enough episodic turns accumulate.
It reads recent episodic memories + existing user_memories, groups related
ones, and asks the LLM to synthesize a higher-level insight.

Reflections are stored in the reflection_memories table and retrieved
alongside regular user_memories via the same three-factor ranking.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional

import config
import psycopg

logger = logging.getLogger(__name__)


def _vector_literal(values: List[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


class ReflectionMemoryStore:
    """PostgreSQL-backed reflection memory with pgvector similarity search."""

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
        self._llm = None

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
                logger.warning("Embedding model unavailable for reflection memory.", exc_info=True)
                self._embeddings = None
            self._embeddings_checked = True
        return self._embeddings

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        try:
            from model_factory import get_chat_model
            self._llm = get_chat_model().with_config(temperature=0.2)
        except Exception:
            logger.warning("Failed to create LLM for reflection synthesis.", exc_info=True)
            self._llm = None
        return self._llm

    # ------------------------------------------------------------------
    # Reflection synthesis
    # ------------------------------------------------------------------

    def maybe_reflect(self, user_id: str, episodic_store, user_memory_store) -> int:
        """Check if reflection is needed and run it if so.

        Returns the number of new reflections created.
        """
        if not config.REFLECTION_MEMORY_ENABLED:
            return 0

        try:
            # Check if enough turns have accumulated since last reflection
            recent_turns = episodic_store.get_recent_turns_for_reflection(
                user_id, limit=config.REFLECTION_MEMORY_MIN_SOURCE_COUNT * 2
            )
            if len(recent_turns) < config.REFLECTION_MEMORY_MIN_SOURCE_COUNT:
                return 0

            # Get existing user memories to group with
            user_memories = user_memory_store.get_memories_for_user(user_id, memory_type="medical")

            # Group related memories and synthesize reflections
            reflections = self._synthesize_reflections(user_id, recent_turns, user_memories)
            if not reflections:
                return 0

            # Save reflections
            saved = 0
            for reflection in reflections:
                try:
                    self.save_reflection(
                        user_id=user_id,
                        content=reflection["content"],
                        source_memory_ids=reflection.get("source_ids", []),
                        importance=reflection.get("importance", 7),
                    )
                    saved += 1
                except Exception:
                    logger.warning("Failed to save reflection", exc_info=True)

            return saved

        except Exception:
            logger.warning("Reflection synthesis failed for user_id=%s", user_id, exc_info=True)
            return 0

    def _synthesize_reflections(
        self,
        user_id: str,
        recent_turns: List[Dict[str, Any]],
        user_memories: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Ask the LLM to synthesize higher-order reflections from raw memories."""
        llm = self._get_llm()
        if llm is None:
            return []

        # Build context for the LLM
        context_parts = ["Recent conversation turns:"]
        for turn in recent_turns[:10]:
            context_parts.append(f"  [{turn.get('created_at', '')}] 用户: {turn.get('user_message', '')}")
            context_parts.append(f"  [{turn.get('created_at', '')}] 助手: {turn.get('assistant_message', '')[:100]}...")

        if user_memories:
            context_parts.append("\nExisting medical memories:")
            for mem in user_memories[:10]:
                context_parts.append(f"  - {mem.get('content', '')} (importance: {mem.get('importance', '')})")

        context = "\n".join(context_parts)

        prompt = """Analyze the conversation history and existing memories, then synthesize 1-3 higher-level reflections.

A reflection is an abstract insight that connects multiple observations into a deeper understanding.

Examples:
- Three separate mentions of headache → "用户有反复头痛困扰，可能为紧张型头痛，建议进一步检查"
- Multiple questions about the same medication → "用户对XX药物持续关注，可能存在用药顾虑"

Rules:
1. Only synthesize when multiple observations genuinely support a higher-level insight.
2. Each reflection must be a single, specific, actionable insight in Chinese.
3. Rate importance 7-10 (reflections are inherently important).
4. Return a JSON array of objects with keys: content, importance, source_count (how many observations this is based on).
5. Return empty array [] if no meaningful reflections can be made.

Return ONLY the JSON array."""

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            response = llm.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=context),
            ])
            raw = str(response.content or "").strip()
            parsed = self._parse_json_response(raw)
        except Exception:
            logger.warning("LLM call for reflection synthesis failed.", exc_info=True)
            return []

        # Convert to reflection format
        reflections = []
        for item in parsed:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            reflections.append({
                "content": content,
                "importance": int(item.get("importance", 7)),
                "source_ids": [t["id"] for t in recent_turns[:item.get("source_count", 3)] if "id" in t],
            })

        return reflections

    @staticmethod
    def _parse_json_response(raw: str) -> List[dict]:
        if not raw:
            return []
        import re
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []

    # ------------------------------------------------------------------
    # Save / Retrieve
    # ------------------------------------------------------------------

    def save_reflection(
        self,
        user_id: str,
        content: str,
        source_memory_ids: List[int],
        importance: int = 7,
    ) -> int:
        """Save a reflection memory."""
        embedding = None
        embeddings = self._get_embeddings()
        if embeddings is not None:
            try:
                embedding = embeddings.embed_query(content)
            except Exception:
                logger.warning("Failed to embed reflection.", exc_info=True)

        embedding_sql = "NULL"
        params: list[Any] = [user_id, content, json.dumps(source_memory_ids, ensure_ascii=False), importance]
        if embedding is not None:
            embedding_sql = "CAST(%s AS vector)"
            params.append(_vector_literal(embedding))

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO reflection_memories (user_id, content, source_memory_ids, importance, embedding)
                    VALUES (%s, %s, %s, %s, {embedding_sql})
                    RETURNING id
                    """,
                    params,
                )
                row_id = cur.fetchone()[0]
            conn.commit()
        return row_id

    def retrieve_reflections(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Retrieve reflection memories by semantic similarity."""
        embeddings = self._get_embeddings()
        if embeddings is None:
            return self._retrieve_by_importance(user_id, top_k)

        try:
            query_embedding = embeddings.embed_query(query)
        except Exception:
            return self._retrieve_by_importance(user_id, top_k)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, content, source_memory_ids, importance, created_at,
                           1 - (embedding <=> CAST(%s AS vector)) AS similarity
                    FROM reflection_memories
                    WHERE user_id = %s AND embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(%s AS vector)
                    LIMIT %s
                    """,
                    (_vector_literal(query_embedding), user_id, _vector_literal(query_embedding), top_k),
                )
                rows = cur.fetchall()

        columns = ["id", "content", "source_memory_ids", "importance", "created_at", "similarity"]
        return [dict(zip(columns, row)) for row in rows]

    def get_reflections_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Return all reflections for a user."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, content, source_memory_ids, importance, created_at
                    FROM reflection_memories
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    """,
                    (user_id,),
                )
                rows = cur.fetchall()

        columns = ["id", "content", "source_memory_ids", "importance", "created_at"]
        return [dict(zip(columns, row)) for row in rows]

    def clear_user_reflections(self, user_id: str):
        """Delete all reflections for a user (for testing)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM reflection_memories WHERE user_id = %s", (user_id,))
            conn.commit()

    def _retrieve_by_importance(self, user_id: str, top_k: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, content, source_memory_ids, importance, created_at
                    FROM reflection_memories
                    WHERE user_id = %s
                    ORDER BY importance DESC, created_at DESC
                    LIMIT %s
                    """,
                    (user_id, top_k),
                )
                rows = cur.fetchall()
        columns = ["id", "content", "source_memory_ids", "importance", "created_at"]
        return [dict(zip(columns, row)) for row in rows]

    def status_info(self) -> Dict[str, Any]:
        if not config.REFLECTION_MEMORY_ENABLED:
            return {"component": "reflection_memory", "mode": "disabled", "degraded": False,
                    "message": "Reflection memory is disabled by configuration."}
        embeddings = self._get_embeddings()
        if embeddings is not None:
            return {"component": "reflection_memory", "mode": "pgvector", "degraded": False,
                    "message": "Reflection memory with LLM synthesis is available."}
        return {"component": "reflection_memory", "mode": "importance_only", "degraded": True,
                "message": "Embedding model unavailable; reflection retrieval uses importance-only."}
