"""Build LangGraph inputs for one chat turn."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import config
from langchain_core.messages import HumanMessage


logger = logging.getLogger(__name__)


@dataclass
class ChatTurnInput:
    active_thread_id: str
    graph_config: dict
    current_state: object
    user_message: str
    request_id: str
    session_state: dict
    checkpoint_resumed: bool
    user_id: str
    user_memories_text: str
    stream_input: dict | None


class ChatTurnInputService:
    def __init__(
        self,
        rag_system,
        *,
        get_graph_config,
        fetch_user_memories,
        build_state_messages,
        graph_state_from_session,
    ):
        self.rag_system = rag_system
        self._get_graph_config = get_graph_config
        self._fetch_user_memories = fetch_user_memories
        self._build_state_messages = build_state_messages
        self._graph_state_from_session = graph_state_from_session

    def prepare(self, *, message: str, thread_id: str | None = None) -> ChatTurnInput:
        active_thread_id = thread_id or self.rag_system.thread_id
        graph_config = self._get_graph_config(active_thread_id)
        current_state = self.rag_system.agent_graph.get_state(graph_config)
        user_message = message.strip()
        request_id = uuid.uuid4().hex
        session_state = self.rag_system.session_memory.get_state(active_thread_id)
        checkpoint_resumed = bool(current_state.next)

        user_id = self._resolve_user_id(active_thread_id)
        user_memories_text = self._resolve_user_memories(
            user_id=user_id,
            user_message=user_message,
            active_thread_id=active_thread_id,
        )
        stream_input = self._prepare_stream_input(
            active_thread_id=active_thread_id,
            graph_config=graph_config,
            current_state=current_state,
            user_message=user_message,
            request_id=request_id,
            session_state=session_state,
            user_id=user_id,
            user_memories_text=user_memories_text,
        )

        return ChatTurnInput(
            active_thread_id=active_thread_id,
            graph_config=graph_config,
            current_state=current_state,
            user_message=user_message,
            request_id=request_id,
            session_state=session_state,
            checkpoint_resumed=checkpoint_resumed,
            user_id=user_id,
            user_memories_text=user_memories_text,
            stream_input=stream_input,
        )

    def _resolve_user_id(self, active_thread_id: str) -> str:
        if not config.USER_MEMORY_ENABLED:
            return ""
        try:
            chat_sessions = getattr(self.rag_system, "chat_sessions", None)
            if chat_sessions is None:
                return ""
            session_info = chat_sessions.get_session(active_thread_id)
            return (session_info or {}).get("owner_user_id", "") or ""
        except Exception:
            logger.warning("Failed to resolve user_id for memory injection", exc_info=True)
            return ""

    def _resolve_user_memories(self, *, user_id: str, user_message: str, active_thread_id: str) -> str:
        if not (user_id and config.USER_MEMORY_ENABLED and config.USER_MEMORY_INJECTION_ENABLED):
            return ""
        return self._fetch_user_memories(
            user_id=user_id,
            user_message=user_message,
            thread_id=active_thread_id,
        )

    def _prepare_stream_input(
        self,
        *,
        active_thread_id: str,
        graph_config,
        current_state,
        user_message: str,
        request_id: str,
        session_state: dict,
        user_id: str,
        user_memories_text: str,
    ) -> dict | None:
        if current_state.next:
            update_payload = {
                "messages": [HumanMessage(content=user_message)],
                "thread_id": active_thread_id,
                "request_id": request_id,
            }
            if user_memories_text:
                update_payload["user_memories"] = user_memories_text
            self.rag_system.agent_graph.update_state(graph_config, update_payload)
            return None

        stored_messages = self.rag_system.session_memory.get_recent_messages(active_thread_id)
        long_term_summary = self.rag_system.summary_store.get_summary(active_thread_id)
        state_messages = self._build_state_messages(session_state)

        if long_term_summary:
            self.rag_system.agent_graph.update_state(
                graph_config,
                {"conversation_summary": long_term_summary},
            )
        if session_state:
            self.rag_system.agent_graph.update_state(
                graph_config,
                self._graph_state_from_session(active_thread_id, session_state),
            )
        if not session_state:
            self.rag_system.agent_graph.update_state(
                graph_config,
                {"thread_id": active_thread_id, "agent_answers": [{"__reset__": True}]},
            )

        return {
            "messages": [*state_messages, *stored_messages, HumanMessage(content=user_message)],
            "request_id": request_id,
            "user_memories": user_memories_text,
            "_mcp_pool": getattr(self.rag_system, "user_mcp_pool", None),
            "user_id": user_id,
        }
