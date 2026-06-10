"""Post-stream chat turn finalization helpers.

This service keeps ChatInterface focused on streaming/orchestration while
centralizing the work that happens after the graph finishes a turn:
persisting chat history, refreshing summaries, scheduling memory extraction,
updating session state, and recording routing telemetry.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import config
from db.route_log_store import RouteLogStore


logger = logging.getLogger(__name__)


@dataclass
class TurnArtifacts:
    response_messages: list[dict]
    latest_values: dict
    final_assistant: str
    combined_assistant_text: str
    clarification_text: str
    updated_state: dict
    had_pending_state: bool
    route_reason: str
    secondary_turn_executed: bool
    response_messages_changed: bool


class ChatTurnService:
    _memory_executor: ThreadPoolExecutor | None = None

    def __init__(
        self,
        rag_system,
        *,
        route_log_store=None,
        extract_final_assistant_text,
        extract_all_visible_assistant_texts,
        extract_clarification_text,
        extract_latest_state_assistant,
        build_chat_failure_fallback,
        resolved_session_state,
        invalidate_memory_cache,
    ):
        self.rag_system = rag_system
        self.route_log_store = route_log_store or RouteLogStore()
        self._extract_final_assistant_text = extract_final_assistant_text
        self._extract_all_visible_assistant_texts = extract_all_visible_assistant_texts
        self._extract_clarification_text = extract_clarification_text
        self._extract_latest_state_assistant = extract_latest_state_assistant
        self._build_chat_failure_fallback = build_chat_failure_fallback
        self._resolved_session_state = resolved_session_state
        self._invalidate_memory_cache = invalidate_memory_cache

    @staticmethod
    def _make_message(content):
        return {"role": "assistant", "content": content}

    def prepare_turn_artifacts(
        self,
        *,
        active_thread_id: str,
        graph_config,
        response_messages: list[dict],
        user_message: str,
        session_state: dict,
    ) -> TurnArtifacts:
        final_assistant = self._extract_final_assistant_text(response_messages)
        all_visible_assistant_texts = self._extract_all_visible_assistant_texts(response_messages)
        clarification_text = self._extract_clarification_text(response_messages)
        latest_state = self.rag_system.agent_graph.get_state(graph_config)
        latest_values = getattr(latest_state, "values", {}) or {}
        response_messages_changed = False

        if not final_assistant:
            final_from_state = self._extract_latest_state_assistant(latest_values)
            if final_from_state:
                response_messages.append(self._make_message(final_from_state))
                final_assistant = final_from_state
                response_messages_changed = True

        if not final_assistant:
            final_assistant = self._build_chat_failure_fallback(user_message)
            response_messages.append(self._make_message(final_assistant))
            response_messages_changed = True

        combined_assistant_text = "\n\n".join(all_visible_assistant_texts) if all_visible_assistant_texts else final_assistant
        updated_state = self._resolved_session_state(latest_values, session_state, user_message, clarification_text)
        route_reason = latest_values.get("route_reason", updated_state.get("last_route_reason") or "") or ""
        secondary_turn_executed = str(route_reason).startswith("resume_secondary:")
        had_pending_state = bool(
            (session_state or {}).get("pending_action_type")
            or (session_state or {}).get("pending_clarification")
            or (session_state or {}).get("deferred_user_question")
        )

        return TurnArtifacts(
            response_messages=response_messages,
            latest_values=latest_values,
            final_assistant=final_assistant,
            combined_assistant_text=combined_assistant_text,
            clarification_text=clarification_text,
            updated_state=updated_state,
            had_pending_state=had_pending_state,
            route_reason=route_reason,
            secondary_turn_executed=secondary_turn_executed,
            response_messages_changed=response_messages_changed,
        )

    def finalize_turn(
        self,
        *,
        active_thread_id: str,
        request_id: str,
        user_message: str,
        session_state: dict,
        checkpoint_resumed: bool,
        user_id: str,
        artifacts: TurnArtifacts,
    ):
        recent_count = 0
        if artifacts.combined_assistant_text:
            recent_count = self.rag_system.session_memory.append_exchange(
                active_thread_id,
                user_message,
                artifacts.combined_assistant_text,
            )
            if recent_count >= config.SUMMARY_REFRESH_THRESHOLD:
                conversation_summary = artifacts.latest_values.get("conversation_summary", "")
                if conversation_summary:
                    self.rag_system.summary_store.save_summary(active_thread_id, conversation_summary, recent_count)

        self._run_post_chat_summary(
            active_thread_id=active_thread_id,
            latest_values=artifacts.latest_values,
            combined_assistant_text=artifacts.combined_assistant_text,
        )

        self._schedule_memory_extraction(
            active_thread_id=active_thread_id,
            user_id=user_id,
            user_message=user_message,
            combined_assistant_text=artifacts.combined_assistant_text,
            latest_values=artifacts.latest_values,
        )

        if artifacts.updated_state != (session_state or {}):
            self.rag_system.session_memory.set_state(active_thread_id, artifacts.updated_state)

        self._persist_route_log(
            active_thread_id=active_thread_id,
            request_id=request_id,
            user_message=user_message,
            session_state=session_state,
            checkpoint_resumed=checkpoint_resumed,
            artifacts=artifacts,
        )

    def _run_post_chat_summary(self, *, active_thread_id: str, latest_values: dict, combined_assistant_text: str):
        if not combined_assistant_text:
            return
        try:
            from model_factory import get_chat_model
            from rag_agent.routing_nodes import summarize_history

            summary_llm = get_chat_model().with_config(temperature=0.2)
            summary_result = summarize_history(latest_values, summary_llm)
            new_summary = (summary_result or {}).get("conversation_summary", "")
            if new_summary and new_summary != latest_values.get("conversation_summary", ""):
                recent_count = self.rag_system.session_memory.recent_message_count(active_thread_id)
                self.rag_system.summary_store.save_summary(active_thread_id, new_summary, recent_count)
        except Exception:
            logger.warning("Post-chat summarization failed for thread_id=%s", active_thread_id, exc_info=True)

    def _schedule_memory_extraction(
        self,
        *,
        active_thread_id: str,
        user_id: str,
        user_message: str,
        combined_assistant_text: str,
        latest_values: dict,
    ):
        if not (user_id and config.USER_MEMORY_ENABLED and config.USER_MEMORY_EXTRACTION_ENABLED and combined_assistant_text):
            return

        memory_extractor = getattr(self.rag_system, "memory_extractor", None)
        if memory_extractor is None:
            return

        if self.__class__._memory_executor is None:
            self.__class__._memory_executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="memory-extract",
            )

        def _bg_extract():
            try:
                saved = memory_extractor.extract_and_save(
                    thread_id=active_thread_id,
                    user_message=user_message,
                    assistant_message=combined_assistant_text,
                    conversation_summary=latest_values.get("conversation_summary", ""),
                )
                if saved:
                    self._invalidate_memory_cache(user_id, active_thread_id)
            except Exception:
                logger.warning("Memory extraction failed for thread_id=%s", active_thread_id, exc_info=True)

        try:
            self.__class__._memory_executor.submit(_bg_extract)
        except RuntimeError:
            logger.warning(
                "Memory extraction queue full; dropping extraction for thread_id=%s",
                active_thread_id,
            )

    def _persist_route_log(
        self,
        *,
        active_thread_id: str,
        request_id: str,
        user_message: str,
        session_state: dict,
        checkpoint_resumed: bool,
        artifacts: TurnArtifacts,
    ):
        try:
            self.route_log_store.save_log(
                {
                    "thread_id": active_thread_id,
                    "request_id": request_id,
                    "user_query": user_message,
                    "primary_intent": artifacts.latest_values.get("primary_intent", artifacts.updated_state.get("intent")) or "",
                    "secondary_intent": artifacts.updated_state.get("secondary_intent") or "",
                    "decision_source": artifacts.latest_values.get("decision_source", "") or "",
                    "route_reason": artifacts.route_reason,
                    "had_pending_state": artifacts.had_pending_state,
                    "extra_metadata": {
                        "topic_focus": artifacts.updated_state.get("topic_focus") or "",
                        "deferred_user_question": artifacts.updated_state.get("deferred_user_question") or "",
                        "checkpoint_resumed": checkpoint_resumed,
                        "secondary_turn_executed": artifacts.secondary_turn_executed,
                        "pending_action_type": (session_state or {}).get("pending_action_type") or "",
                        "pending_clarification": bool((session_state or {}).get("pending_clarification")),
                    },
                }
            )
        except Exception:
            logger.exception("Failed to persist route log for request_id=%s", request_id)
