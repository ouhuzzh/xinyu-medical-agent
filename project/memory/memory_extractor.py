"""Memory extractor — extracts structured user memories from conversation turns.

Runs after a conversation turn completes.  Uses an LLM to identify durable,
user-specific information (preferences, facts, medical history, decisions),
then saves them to the UserMemoryStore with importance scoring and dedup.

All exceptions are caught internally so extraction never disrupts the chat flow.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

import config
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


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

            extracted = self._call_llm(user_message, assistant_message, conversation_summary)
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
    ) -> List[dict]:
        """Call the LLM and parse the JSON extraction result.

        Returns a list of dicts, or empty list on any failure.
        """
        from rag_agent.prompts import get_memory_extraction_prompt

        conversation_parts = []
        if conversation_summary:
            conversation_parts.append(f"[对话摘要]\n{conversation_summary}")
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
        import re
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
