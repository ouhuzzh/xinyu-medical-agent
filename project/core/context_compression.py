"""Token-aware context compression for chat session memory.

This module centralises token estimation, message trimming, summarisation,
and the service used for both automatic (per-turn) and user-initiated
context compression.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import config
from rag_agent.prompts import get_conversation_summary_prompt
from utils import estimate_context_tokens


logger = logging.getLogger(__name__)


def _message_content(msg: Any) -> str:
    """Extract string content from a LangChain message or a plain dict."""
    if isinstance(msg, dict):
        return str(msg.get("content", "") or "")
    return str(getattr(msg, "content", "") or "")


def _message_role_label(msg: Any) -> str:
    """Return 'User' or 'Assistant' for summary formatting."""
    if isinstance(msg, HumanMessage):
        return "User"
    if isinstance(msg, AIMessage):
        return "Assistant"
    if isinstance(msg, dict):
        role = msg.get("role", "")
        if role == "user":
            return "User"
        if role == "assistant":
            return "Assistant"
    return "Assistant"


def _serialize_for_summary(messages: list) -> str:
    """Format messages for the summarisation prompt."""
    lines = []
    for msg in messages:
        label = _message_role_label(msg)
        content = _message_content(msg).strip()
        if content:
            lines.append(f"{label}: {content}")
    return "\n".join(lines)


def summarize_messages(messages: list, existing_summary: str, llm) -> str:
    """Merge an existing summary with new messages into an updated summary.

    Args:
        messages: Messages to summarise (older than the preserved recent window).
        existing_summary: Prior summary, if any.
        llm: LangChain chat model bound with desired temperature/config.

    Returns:
        The new summary text, or an empty string if nothing meaningful exists.
    """
    if not messages:
        return existing_summary or ""

    conversation = "Conversation history:\n"
    if existing_summary.strip():
        conversation += f"[Prior conversation summary]\n{existing_summary}\n\n"
    conversation += _serialize_for_summary(messages[-6:])

    if not conversation.strip():
        return existing_summary or ""

    try:
        response = llm.invoke([
            SystemMessage(content=get_conversation_summary_prompt()),
            HumanMessage(content=conversation),
        ])
        return str(response.content or "").strip()
    except Exception:
        logger.exception("summarize_messages failed")
        return existing_summary or ""


def should_compress_context(recent_messages: list, recent_count: int) -> bool:
    """Decide whether recent conversation history should be compressed.

    Priority:
    1. Message-count ceiling (absolute safety limit).
    2. Token threshold (primary signal).
    3. Deprecated fixed-count fallback (when token threshold is disabled).
    """
    if recent_count >= config.SUMMARY_MAX_MESSAGE_CEILING:
        return True
    if config.SUMMARY_TOKEN_THRESHOLD > 0:
        return estimate_context_tokens(recent_messages) >= config.SUMMARY_TOKEN_THRESHOLD
    return recent_count >= config.SUMMARY_REFRESH_THRESHOLD


def trim_messages_to_token_budget(
    messages: list,
    max_tokens: int,
    preserve_recent_turns: int,
) -> list:
    """Drop oldest messages until the total token count is within budget.

    Always preserves at least ``preserve_recent_turns`` complete turns
    (user + assistant pairs) even if that alone exceeds the budget.

    Args:
        messages: List of messages to trim.
        max_tokens: Maximum token budget.
        preserve_recent_turns: Minimum number of recent turns to keep.

    Returns:
        A new list of messages satisfying the budget where possible.
    """
    if estimate_context_tokens(messages) <= max_tokens:
        return list(messages)

    preserve_count = max(1, preserve_recent_turns * 2)
    if len(messages) <= preserve_count:
        return list(messages)

    preserved = messages[-preserve_count:]
    preserved_tokens = estimate_context_tokens(preserved)
    if preserved_tokens >= max_tokens:
        # Even the preserved window alone is too large; keep it anyway
        # so the conversation remains coherent.
        return list(preserved)

    trimmed = list(preserved)
    for msg in reversed(messages[:-preserve_count]):
        test = [msg] + trimmed
        if estimate_context_tokens(test) <= max_tokens:
            trimmed.insert(0, msg)
        else:
            break
    return trimmed


class ContextCompressionService:
    """Service for compressing conversation context."""

    def __init__(self, llm=None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            from model_factory import get_chat_model
            self._llm = get_chat_model().with_config(temperature=0.2)
        return self._llm

    def compress_thread(
        self,
        session_memory,
        summary_store,
        thread_id: str,
        preserve_recent_turns: int | None = None,
    ) -> dict:
        """Compress a thread's recent messages into a summary.

        Args:
            session_memory: Object with ``get_recent_messages(thread_id)`` and
                ``set_recent_messages(thread_id, messages)``.
            summary_store: Object with ``get_summary(thread_id)`` and
                ``save_summary(thread_id, summary, last_message_index)``.
            thread_id: Thread identifier.
            preserve_recent_turns: How many recent turns to keep uncompressed.

        Returns:
            Dict with thread_id, compressed bool, reason, preserved_count,
            summary_length.
        """
        preserve_recent_turns = preserve_recent_turns or config.RECENT_CONTEXT_TURNS
        recent_messages = session_memory.get_recent_messages(thread_id)
        recent_count = len(recent_messages)
        preserve_count = max(1, preserve_recent_turns * 2)

        if recent_count <= preserve_count:
            return {
                "thread_id": thread_id,
                "compressed": False,
                "reason": "below_preserve_threshold",
                "preserved_count": recent_count,
                "summary_length": 0,
            }

        existing_summary = summary_store.get_summary(thread_id) or ""
        older_messages = recent_messages[:-preserve_count]
        preserved_messages = recent_messages[-preserve_count:]

        new_summary = summarize_messages(older_messages, existing_summary, self._get_llm())
        if new_summary:
            summary_store.save_summary(thread_id, new_summary, len(preserved_messages))
            session_memory.set_recent_messages(thread_id, preserved_messages)

        return {
            "thread_id": thread_id,
            "compressed": bool(new_summary),
            "reason": "token_or_ceiling_trigger",
            "preserved_count": len(preserved_messages),
            "summary_length": len(new_summary),
        }

    def hard_trim_messages_for_graph(self, messages: list) -> list:
        """Apply the hard-trim safety net to messages about to enter the graph."""
        max_tokens = max(
            0,
            config.CONTEXT_HARD_TRIM_THRESHOLD - config.CONTEXT_HARD_TRIM_RESERVE,
        )
        return trim_messages_to_token_budget(
            messages,
            max_tokens,
            config.RECENT_CONTEXT_TURNS,
        )
