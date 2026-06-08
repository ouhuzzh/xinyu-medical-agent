"""Memory extractor — extracts structured user memories from conversation turns.

Runs after a conversation turn completes.  Uses an LLM to identify durable,
user-specific information (preferences, facts, medical history, decisions),
then saves them to the UserMemoryStore with importance scoring and dedup.

Optimizations (P0+P3):
  - Pre-filter: skip extraction for greetings/confirmations/thanks (~50% fewer LLM calls)
  - Context-aware: passes existing user memories to the LLM so ambiguous
    references like "那药还能吃吗" are resolved against known medications

All exceptions are caught internally so extraction never disrupts the chat flow.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

import config
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# P0: patterns that indicate no extractable information
_TRIVIAL_USER_MESSAGE_RE = re.compile(
    r"^(好的|好|ok|嗯|哦|谢谢|多谢|知道了|明白|了解了|行|可以|对|是的|没错|对的|"
    r"不是|不对|没有|再见|拜拜|bye|hi|hello|你好|早上好|下午好|晚上好)$",
    re.IGNORECASE,
)

# P0: short messages unlikely to contain memories (len < threshold)
_TRIVIAL_MIN_LENGTH = 6

# P2: negation keywords for contradiction detection
_NEGATION_KEYWORDS = (
    "没有", "不是", "不对", "好了", "正常了", "恢复了", "没再", "不再",
    "好了很多", "已经好了", "痊愈", "康复", "没事了",
)


class MemoryExtractor:
    """Extracts user memories from completed conversation turns."""

    def __init__(self, user_memory_store, chat_session_store):
        self._store = user_memory_store
        self._session_store = chat_session_store
        self._llm = None

    def _get_llm(self):
        """Lazy-init the LLM for extraction."""
        if self._llm is not None:
            return self._llm
        try:
            from model_factory import get_chat_model
            self._llm = get_chat_model().with_config(temperature=0.1)
        except Exception:
            logger.warning("Failed to create LLM for memory extraction.", exc_info=True)
            self._llm = None
        return self._llm

    # ------------------------------------------------------------------
    # P0: Pre-filter — skip trivial messages
    # ------------------------------------------------------------------

    def _should_skip_extraction(self, user_message: str, assistant_message: str) -> bool:
        """Return True if this turn is unlikely to contain extractable memories.

        Skips: greetings, confirmations, thanks, very short messages.
        Saves one LLM call per skipped turn (~300-500ms latency reduction).
        """
        normalized = (user_message or "").strip()
        if not normalized:
            return True

        # Pattern match: single-word responses
        if _TRIVIAL_USER_MESSAGE_RE.match(normalized):
            return True

        # Very short messages without medical/decision keywords
        if len(normalized) < _TRIVIAL_MIN_LENGTH:
            if not any(kw in normalized for kw in (
                "过敏", "病史", "高血压", "糖尿病", "药", "手术", "疼", "痛",
                "挂号", "预约", "科", "医", "症", "病", "血", "检查",
            )):
                return True

        return False

    # ------------------------------------------------------------------
    # Main extraction flow
    # ------------------------------------------------------------------

    def extract_and_save(
        self,
        thread_id: str,
        user_message: str,
        assistant_message: str,
        conversation_summary: str = "",
    ) -> int:
        """Extract memories from a turn and save to the user memory store.

        Returns the number of memories saved.  Never raises — all exceptions
        are caught and logged.
        """
        if not config.USER_MEMORY_ENABLED or not config.USER_MEMORY_EXTRACTION_ENABLED:
            return 0

        # P0: skip trivial turns
        if self._should_skip_extraction(user_message, assistant_message):
            return 0

        try:
            # 1. Resolve user_id from thread_id
            session_info = self._session_store.get_session(thread_id)
            if not session_info:
                return 0
            user_id = (session_info.get("owner_user_id") or "").strip()
            if not user_id:
                return 0

            # 2. Call LLM to extract memories
            llm = self._get_llm()
            if llm is None:
                return 0

            extracted = self._call_llm(user_message, assistant_message, conversation_summary, user_id)
            if not extracted:
                return 0

            # 3. Filter by importance threshold and save
            saved = 0
            for mem in extracted:
                importance = int(mem.get("importance", 5))
                if importance < config.USER_MEMORY_IMPORTANCE_THRESHOLD:
                    continue
                memory_type = str(mem.get("memory_type", "fact")).strip().lower()
                content = str(mem.get("content", "")).strip()
                if not content:
                    continue
                # Validate memory_type
                if memory_type not in ("preference", "fact", "medical", "decision"):
                    memory_type = "fact"

                # P2: check for contradiction before saving
                action = mem.get("action", "add")  # add | update | deprecate
                if action == "deprecate":
                    self._deprecate_contradicting_memory(user_id, content)
                    continue

                try:
                    self._store.save_memory(
                        user_id=user_id,
                        memory_type=memory_type,
                        content=content,
                        importance=importance,
                        source_thread_id=thread_id,
                    )
                    saved += 1
                except Exception:
                    logger.warning("Failed to save memory: %s", content[:80], exc_info=True)

            return saved

        except Exception:
            logger.warning("Memory extraction failed for thread_id=%s", thread_id, exc_info=True)
            return 0

    def _call_llm(
        self,
        user_message: str,
        assistant_message: str,
        conversation_summary: str,
        user_id: str = "",
    ) -> List[dict]:
        """Call the LLM and parse the JSON extraction result.

        P3: Includes existing user memories as context so the LLM can resolve
        ambiguous references (e.g., "那药还能吃吗" → resolves "那药" to known medication).
        """
        from rag_agent.prompts import get_memory_extraction_prompt

        conversation_parts = []

        # P3: include existing memories as context
        existing_context = ""
        if user_id:
            try:
                existing = self._store.get_memories_for_user(user_id)
                if existing:
                    memory_lines = [
                        f"- [{m['memory_type']}|i{m['importance']}] {m['content']}"
                        for m in existing[:10]
                    ]
                    existing_context = "\n".join(memory_lines)
            except Exception:
                pass

        if conversation_summary:
            conversation_parts.append(f"[对话摘要]\n{conversation_summary}")
        if existing_context:
            conversation_parts.append(f"[该用户已有记忆 — 如新信息与已有记忆矛盾请用 action=deprecate]\n{existing_context}")
        conversation_parts.append(f"用户: {user_message}")
        conversation_parts.append(f"助手: {assistant_message}")
        conversation = "\n\n".join(conversation_parts)

        try:
            response = self._get_llm().invoke([
                SystemMessage(content=get_memory_extraction_prompt()),
                HumanMessage(content=conversation),
            ])
            raw = str(response.content or "").strip()
            return self._parse_extraction_response(raw)
        except Exception:
            logger.warning("LLM call for memory extraction failed.", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # P2: Contradiction handling
    # ------------------------------------------------------------------

    def _deprecate_contradicting_memory(self, user_id: str, new_content: str):
        """When LLM says a memory is deprecated, halve the importance of matching old memories.

        Uses embedding cosine similarity (reuses the store's singleton embedding)
        instead of brittle keyword Jaccard overlap.  Only deprecates when the
        semantics genuinely match (similarity > 0.5), avoiding false positives
        from random keyword overlap.
        """
        try:
            existing = self._store.get_memories_for_user(user_id)
            if not existing:
                return

            # Get embeddings for the new content
            embeddings_model = self._store._get_embeddings()
            if embeddings_model is None:
                return
            new_embedding = embeddings_model.embed_query(new_content)

            for mem in existing:
                mem_embedding = self._store._get_memory_embedding(mem["id"])
                if mem_embedding is None:
                    continue
                similarity = self._cosine_similarity(new_embedding, mem_embedding)
                if similarity > 0.5:
                    new_imp = max(1, mem["importance"] // 2)
                    try:
                        self._store._update_importance(mem["id"], new_imp)
                        logger.info("Deprecated memory %d (%s) importance %d→%d (sim=%.2f)",
                                    mem["id"], mem["content"][:40], mem["importance"], new_imp, similarity)
                    except Exception:
                        pass
        except Exception:
            logger.warning("Contradiction deprecation failed", exc_info=True)

    @staticmethod
    def _cosine_similarity(a: list, b: list) -> float:
        dot = sum(x*y for x,y in zip(a,b))
        norm_a = sum(x*x for x in a)**0.5
        norm_b = sum(x*x for x in b)**0.5
        return dot/(norm_a*norm_b) if norm_a and norm_b else 0.0

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_extraction_response(raw: str) -> List[dict]:
        """Parse the LLM response JSON.  Returns empty list on failure."""
        if not raw:
            return []
        # Try to find a JSON array in the response
        try:
            # Direct parse
            result = json.loads(raw)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1).strip())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        # Try finding array brackets
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse memory extraction response: %s", raw[:200])
        return []
