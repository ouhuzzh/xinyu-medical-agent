"""ChatInterface — the main facade between FastAPI and the LangGraph agent graph.

Handles:
    - SSE streaming of AI responses and system messages
    - Session state read/write via Redis (with in-memory fallback)
    - Long-term summary persistence to PostgreSQL
    - Thread-level checkpoint resume for interrupted flows
    - Tool call rendering and error fallback messages
"""

import json
import logging
import re

import config
from core.chat_turn_input_service import ChatTurnInputService
from core.chat_turn_service import ChatTurnService
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage, SystemMessage
from rag_agent.node_helpers import _sanitize_final_answer_text
from rag_agent.tools import reset_retrieval_context


logger = logging.getLogger(__name__)

SILENT_NODES = {
    "rewrite_query",
    "intent_router",
    "plan_retrieval_queries",
    "grounded_answer_generation",
    "answer_grounding_check",
}
SYSTEM_NODES = {"summarize_history", "rewrite_query"}
# Classification hints — canonical definitions live in skill modules.
# These aliases import from skills so there is a single source of truth.
from skills.booking_skill import UPDATE_HINTS as APPOINTMENT_UPDATE_HINTS
from skills.cancel_skill import CANCEL_HINTS as CANCEL_UPDATE_HINTS
from skills.medical_rag_skill import KB_HINTS as MEDICAL_KB_HINTS
from skills.medical_rag_skill import KB_QUESTION_HINTS as MEDICAL_KB_QUESTION_HINTS
from skills.medical_rag_skill import FALLBACK_DANGER_HINTS as MEDICAL_FALLBACK_DANGER_HINTS
PENDING_ACK_HINTS = ("可以", "好的", "行", "好", "ok", "okay")

SYSTEM_NODE_CONFIG = {
    "rewrite_query":     {"title": "🔍 Query Analysis & Rewriting"},
    "summarize_history": {"title": "📋 Chat History Summary"},
}

SESSION_STATE_DEFAULTS = {
    "intent": "",
    "risk_level": "normal",
    "pending_clarification": "",
    "clarification_target": "",
    "topic_focus": "",
    "deferred_user_question": "",
    "secondary_intent": "",
    "clarification_attempts": 0,
    "last_route_reason": "",
    "recommended_department": "",
    "appointment_context": {},
    "appointment_skill_mode": "",
    "appointment_candidates": [],
    "selected_doctor": "",
    "selected_schedule_id": "",
    "deferred_confirmation_action": "",
    "skill_last_prompt": "",
    "last_appointment_no": "",
    "pending_action_type": "",
    "pending_action_payload": {},
    "pending_confirmation_id": "",
    "pending_candidates": [],
    "skill_data": {},
    "user_memories": "",
}

# --- Helpers ---

def make_message(content, *, title=None, node=None):
    msg = {"role": "assistant", "content": content}
    if title or node:
        msg["metadata"] = {k: v for k, v in {"title": title, "node": node}.items() if v}
    return msg


def find_msg_idx(messages, node):
    return next(
        (i for i, m in enumerate(messages) if m.get("metadata", {}).get("node") == node),
        None,
    )


def parse_rewrite_json(buffer):
    match = re.search(r"\{.*\}", buffer, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


def format_rewrite_content(buffer):
    data = parse_rewrite_json(buffer)
    if not data:
        return "⏳ Analyzing query..."
    if data.get("is_clear"):
        lines = ["✅ **Query is clear**"]
        if data.get("questions"):
            lines += ["\n**Rewritten queries:**"] + [f"- {q}" for q in data["questions"]]
    else:
        lines = ["❓ **Query is unclear**"]
        clarification = data.get("clarification_needed", "")
        if clarification and clarification.strip().lower() != "no":
            lines.append(f"\nClarification needed: *{clarification}*")
    return "\n".join(lines)

# --- End of Helpers ---

class ChatInterface:

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.input_service = ChatTurnInputService(
            rag_system,
            get_graph_config=self._get_graph_config,
            fetch_user_memories=self._fetch_user_memories,
            build_state_messages=self._build_state_messages,
            graph_state_from_session=self._graph_state_from_session,
        )
        self.turn_service = ChatTurnService(
            rag_system,
            extract_final_assistant_text=self._extract_final_assistant_text,
            extract_all_visible_assistant_texts=self._extract_all_visible_assistant_texts,
            extract_clarification_text=self._extract_clarification_text,
            extract_latest_state_assistant=self._extract_latest_state_assistant,
            build_chat_failure_fallback=self._build_chat_failure_fallback,
            resolved_session_state=self._resolved_session_state,
            invalidate_memory_cache=self._invalidate_memory_cache,
        )

    def _handle_system_node(self, chunk, node, response_messages, system_node_buffer):
        """Update (or create) the collapsible system-node message and surface any clarification."""
        system_node_buffer[node] = system_node_buffer.get(node, "") + chunk.content
        buffer = system_node_buffer[node]
        title  = SYSTEM_NODE_CONFIG[node]["title"]
        content = format_rewrite_content(buffer) if node == "rewrite_query" else buffer

        idx = find_msg_idx(response_messages, node)
        if idx is None:
            response_messages.append(make_message(content, title=title, node=node))
        else:
            response_messages[idx]["content"] = content

        if node == "rewrite_query":
            self._surface_clarification(buffer, response_messages)

    def _surface_clarification(self, buffer, response_messages):
        """If the query is unclear, add/update a plain clarification message."""
        data          = parse_rewrite_json(buffer) or {}
        clarification = data.get("clarification_needed", "")
        if not data.get("is_clear") and clarification.strip().lower() not in ("", "no"):
            cidx = find_msg_idx(response_messages, "clarification")
            if cidx is None:
                response_messages.append(make_message(clarification, node="clarification"))
            else:
                response_messages[cidx]["content"] = clarification

    def _handle_tool_call(self, chunk, response_messages, active_tool_calls):
        """Register new tool calls as collapsible messages."""
        for tc in chunk.tool_calls:
            if tc.get("id") and tc["id"] not in active_tool_calls:
                response_messages.append(
                    make_message(f"Running `{tc['name']}`...", title=f"🛠️ {tc['name']}")
                )
                active_tool_calls[tc["id"]] = len(response_messages) - 1

    def _handle_tool_result(self, chunk, response_messages, active_tool_calls):
        """Fill in the tool result inside the matching collapsible message."""
        idx = active_tool_calls.get(chunk.tool_call_id)
        if idx is not None:
            preview = str(chunk.content)[:300]
            suffix  = "\n..." if len(str(chunk.content)) > 300 else ""
            response_messages[idx]["content"] = f"```\n{preview}{suffix}\n```"

    def _handle_llm_token(self, chunk, node, response_messages):
        """Append streaming LLM tokens to the last plain assistant message."""
        last = response_messages[-1] if response_messages else None
        if not (last and last.get("role") == "assistant" and "metadata" not in last):
            response_messages.append(make_message(""))
        response_messages[-1]["content"] += chunk.content

    @staticmethod
    def _extract_final_assistant_text(response_messages):
        for message in reversed(response_messages):
            if message.get("role") == "assistant" and "metadata" not in message and message.get("content", "").strip():
                return _sanitize_final_answer_text(message["content"].strip())
        return ""

    @staticmethod
    def _extract_all_visible_assistant_texts(response_messages):
        return [
            _sanitize_final_answer_text(message.get("content", "").strip())
            for message in response_messages
            if message.get("role") == "assistant" and "metadata" not in message and message.get("content", "").strip()
        ]

    @staticmethod
    def _extract_latest_state_assistant(latest_values):
        for message in reversed(latest_values.get("messages", []) or []):
            if isinstance(message, AIMessage):
                content = str(message.content or "").strip()
                if content and not getattr(message, "tool_calls", None):
                    return _sanitize_final_answer_text(content)
        return ""

    @staticmethod
    def _extract_clarification_text(response_messages):
        for message in reversed(response_messages):
            if message.get("metadata", {}).get("node") == "clarification":
                content = message.get("content", "").strip()
                if content:
                    return content
        return ""

    @staticmethod
    def _prepare_visible_messages(response_messages, reveal_diagnostics=False):
        if reveal_diagnostics:
            return response_messages

        visible_messages = [message for message in response_messages if "metadata" not in message]
        if visible_messages:
            return visible_messages
        if response_messages:
            return [make_message("正在整理答案，请稍候。")]
        return []

    @staticmethod
    def _looks_like_department_question(query: str) -> bool:
        """Delegate to TriageSkill's keywords for single source of truth."""
        from skills.triage_skill import TriageSkill
        normalized = (query or "").strip().lower()
        # Check skill keywords
        if any(kw in normalized for kw in TriageSkill().keywords):
            return True
        # Extra English patterns not in skill keywords
        for pattern in ("which department", "what department should i visit",
                        "what department should i register for"):
            if pattern in normalized:
                return True
        return False

    @staticmethod
    def _looks_like_schedule_update(query: str) -> bool:
        normalized = (query or "").strip().lower()
        if any(token in normalized for token in APPOINTMENT_UPDATE_HINTS):
            return True
        return bool(re.search(r"\d{1,2}[点时:：]", normalized) or re.search(r"\d{1,2}\s*月|\d{4}-\d{1,2}-\d{1,2}|明天|后天|周[一二三四五六日天]", normalized))

    @staticmethod
    def _should_continue_pending_intent(user_message: str, existing_state: dict) -> bool:
        normalized = (user_message or "").strip().lower()
        pending_action_type = existing_state.get("pending_action_type", "")
        pending_candidates = existing_state.get("pending_candidates") or []
        if pending_action_type:
            if any(word in normalized for word in ("确认预约", "确认挂号", "确认取消", "确认退号", "先不用", "不用了", "算了", "放弃")):
                return True
            if normalized in PENDING_ACK_HINTS:
                return True
            if pending_candidates and (re.search(r"\bapt[a-z0-9]+\b", normalized, re.IGNORECASE) or re.search(r"第\s*[1-9]\d*", normalized)):
                return True
            if pending_action_type == "appointment":
                return ChatInterface._looks_like_schedule_update(user_message)
            if pending_action_type == "cancel_appointment":
                return any(token in normalized for token in CANCEL_UPDATE_HINTS)

        pending_clarification = existing_state.get("pending_clarification")
        current_intent = existing_state.get("intent", "")
        if not pending_clarification or not current_intent:
            return False
        if current_intent == "appointment":
            return ChatInterface._looks_like_schedule_update(user_message) or len(normalized) <= 20
        if current_intent == "cancel_appointment":
            return any(token in normalized for token in CANCEL_UPDATE_HINTS) or len(normalized) <= 20
        if current_intent == "triage":
            return len(normalized) <= 30 and not any(token in normalized for token in ("是什么", "怎么办", "为什么"))
        return False

    @staticmethod
    def _infer_intent(user_message: str, existing_state: dict):
        """Optimistic pre-classification for SSE streaming state hints.

        Tries the skill registry first (single source of truth), then
        falls back to local rule-based classification.
        """
        normalized = (user_message or "").strip().lower()
        if ChatInterface._should_continue_pending_intent(user_message, existing_state or {}):
            return "pending"
        # Try skill registry L1 keyword classification
        try:
            from skills.registry import get_skill_registry
            registry = get_skill_registry()
            if registry.skills:
                kw_match = registry.classify_by_keywords(user_message)
                if kw_match:
                    return kw_match[0]  # (intent_label, skill_name)
        except Exception:
            pass
        # Fallback to local rules
        if ChatInterface._looks_like_department_question(user_message):
            return "triage"
        if ChatInterface._looks_like_explicit_medical_query(user_message):
            return "medical_rag"
        return "unknown"

    @staticmethod
    def _looks_like_explicit_medical_query(user_message: str) -> bool:
        normalized = (user_message or "").strip().lower()
        if not normalized:
            return False
        if ChatInterface._looks_like_department_question(user_message):
            return False
        if any(token in normalized for token in ("挂号前", "预约前")):
            return False
        has_term = any(token in normalized for token in MEDICAL_KB_HINTS)
        has_question = any(token in normalized for token in MEDICAL_KB_QUESTION_HINTS) or normalized.endswith("?") or normalized.endswith("？")
        return has_term and has_question

    @staticmethod
    def _looks_like_health_related_message(user_message: str) -> bool:
        normalized = (user_message or "").strip().lower()
        if not normalized:
            return False
        if ChatInterface._looks_like_department_question(user_message):
            return True
        return any(token in normalized for token in MEDICAL_KB_HINTS)

    @staticmethod
    def _build_chat_failure_fallback(user_message: str) -> str:
        normalized = (user_message or "").strip().lower()
        if ChatInterface._looks_like_health_related_message(user_message):
            danger_note = (
                "如果出现胸痛、呼吸困难、意识模糊、肢体无力、剧烈或突然加重的疼痛、持续高热等情况，请尽快线下就医或急诊处理。"
                if any(token in normalized for token in MEDICAL_FALLBACK_DANGER_HINTS)
                else "如果症状持续不缓解、反复加重，或伴随明显异常表现，建议及时线下就医。"
            )
            return (
                "刚才系统的检索/模型链路有点不稳定，我先给你一个**通用医学信息层面**的建议：\n\n"
                "- 先观察症状的程度、持续时间，以及是否伴随发热、呕吐、呼吸不适、胸痛、肢体无力等危险信号。\n"
                "- 症状轻微时，可以先休息、补水，避免熬夜、饮酒和明显诱发因素；不要自行叠加或加量用药。\n"
                f"- {danger_note}\n\n"
                "这次回答**未充分基于知识库检索结果**，仅供一般健康信息参考，不能替代医生面对面诊断。你也可以再发一次问题，我会继续帮你细化。"
            )
        return "AI 暂时没能给出有效回答，可能是模型接口超时或异常。请稍后再试一次。"

    @staticmethod
    def _infer_risk_level(user_message: str, existing_state: dict):
        """Delegate to the shared risk-level classifier in node_helpers."""
        from rag_agent.node_helpers import _infer_risk_level
        return _infer_risk_level(user_message, existing_state.get("risk_level", "normal"))

    @staticmethod
    def _build_state_messages(session_state: dict):
        if not session_state:
            return []

        parts = []
        if session_state.get("intent"):
            parts.append(f"Active intent: {session_state['intent']}")
        if session_state.get("risk_level"):
            parts.append(f"Risk level: {session_state['risk_level']}")
        if session_state.get("pending_clarification"):
            parts.append(f"Pending clarification: {session_state['pending_clarification']}")
        if session_state.get("clarification_target"):
            parts.append(f"Clarification target: {session_state['clarification_target']}")
        if session_state.get("topic_focus"):
            parts.append(f"Topic focus: {session_state['topic_focus']}")
        if session_state.get("deferred_user_question"):
            parts.append(f"Deferred user question: {session_state['deferred_user_question']}")
        if session_state.get("secondary_intent"):
            parts.append(f"Secondary intent: {session_state['secondary_intent']}")
        if session_state.get("recommended_department"):
            parts.append(f"Recommended department: {session_state['recommended_department']}")
        if session_state.get("appointment_context"):
            parts.append(f"Appointment context: {session_state['appointment_context']}")
        if session_state.get("appointment_skill_mode"):
            parts.append(f"Appointment skill mode: {session_state['appointment_skill_mode']}")
        if session_state.get("appointment_candidates"):
            parts.append(f"Appointment candidates: {session_state['appointment_candidates']}")
        if session_state.get("selected_doctor"):
            parts.append(f"Selected doctor: {session_state['selected_doctor']}")
        if session_state.get("selected_schedule_id"):
            parts.append(f"Selected schedule id: {session_state['selected_schedule_id']}")
        if session_state.get("deferred_confirmation_action"):
            parts.append(f"Deferred confirmation action: {session_state['deferred_confirmation_action']}")
        if session_state.get("skill_last_prompt"):
            parts.append(f"Skill last prompt: {session_state['skill_last_prompt']}")
        if session_state.get("last_appointment_no"):
            parts.append(f"Last appointment number: {session_state['last_appointment_no']}")
        if session_state.get("pending_action_type"):
            parts.append(f"Pending action type: {session_state['pending_action_type']}")
        if session_state.get("pending_action_payload"):
            parts.append(f"Pending action payload: {session_state['pending_action_payload']}")
        if session_state.get("pending_candidates"):
            parts.append(f"Pending candidates: {session_state['pending_candidates']}")

        if not parts:
            return []

        return [SystemMessage(content="Conversation state context:\n" + "\n".join(parts))]

    @staticmethod
    def _graph_state_from_session(thread_id: str, session_state: dict) -> dict:
        update = {"thread_id": thread_id}
        for field, default in SESSION_STATE_DEFAULTS.items():
            value = (session_state or {}).get(field, default)
            if field == "clarification_attempts":
                value = int(value or 0)
            elif value is None:
                value = [] if isinstance(default, list) else {} if isinstance(default, dict) else default
            update[field] = value
        return update

    @staticmethod
    def _resolved_session_state(latest_values: dict, session_state: dict, user_message: str, clarification_text: str) -> dict:
        session_state = session_state or {}
        updated_state = {}
        for field, default in SESSION_STATE_DEFAULTS.items():
            if field in latest_values:
                value = latest_values.get(field)
            else:
                value = session_state.get(field, default)
            if field == "risk_level" and field not in latest_values:
                value = "normal"
            if field == "pending_clarification" and field not in latest_values:
                value = clarification_text or None
            if field == "clarification_attempts":
                value = int(value or 0)
            elif value is None:
                value = [] if isinstance(default, list) else {} if isinstance(default, dict) else None
            updated_state[field] = value
        if "risk_level" not in latest_values:
            updated_state["risk_level"] = ChatInterface._infer_risk_level(user_message, session_state)
        if "pending_clarification" not in latest_values:
            updated_state["pending_clarification"] = clarification_text or None
        return updated_state

    @staticmethod
    def _should_skip_memory_retrieval(user_message: str) -> bool:
        """Rule-based pre-filter: skip vector retrieval for trivial intents.

        Greetings, thanks, and explicit cancel-appointment requests don't benefit
        from long-term memory (allergy/history). Skipping saves ~300-500ms per turn
        on these messages.
        """
        if not config.USER_MEMORY_SKIP_TRIVIAL_INTENT:
            return False
        from rag_agent.node_helpers import (
            _looks_like_greeting,
            _looks_like_explicit_cancel_intent,
        )
        if _looks_like_greeting(user_message):
            return True
        if _looks_like_explicit_cancel_intent(user_message):
            return True
        return False

    def _fetch_user_memories(self, user_id: str, user_message: str, thread_id: str) -> str:
        """Fetch user memories with rule-based skip + thread-level cache.

        Strategy:
          A. Skip retrieval entirely for trivial intents (greetings/cancel).
          B. Cache per-thread for up to USER_MEMORY_CACHE_MAX_TURNS turns within
             USER_MEMORY_CACHE_TTL_SECONDS, so follow-up questions on the same
             topic don't re-embed and re-search.
          Cache is invalidated on extraction (new memories saved → drop cache).
        """
        # A: rule-based skip
        if self._should_skip_memory_retrieval(user_message):
            return ""

        # B: try cache first
        cached = self._get_memory_cache(user_id, thread_id)
        if cached is not None:
            return cached

        # Cache miss → real retrieval
        try:
            memories = self.rag_system.user_memory_store.retrieve_memories(
                user_id, user_message, top_k=config.USER_MEMORY_MAX_RETRIEVED
            )
        except Exception:
            logger.warning("User memory retrieval failed; continuing without memories.", exc_info=True)
            return ""

        text = ""
        if memories:
            lines = ["[以下为用户相关记忆，请参考这些信息回答问题]"]
            for m in memories:
                mem_type = m.get("memory_type", "fact")
                type_label = {"medical": "病史", "preference": "偏好", "fact": "事实", "decision": "决策"}.get(mem_type, mem_type)
                # P4: add type label + priority hint for medical/decision
                if mem_type == "medical":
                    hint = " ← 用药安全相关，必须参考"
                elif mem_type == "decision":
                    hint = " ← 用户的近期操作意向"
                else:
                    hint = ""
                lines.append(f"- [{type_label}|重要性:{m['importance']}] {m['content']}{hint}")
            text = "\n".join(lines)

        self._set_memory_cache(user_id, thread_id, text)
        return text

    def _memory_cache_key(self, user_id: str, thread_id: str) -> str:
        return f"memory_cache:{user_id}:{thread_id}"

    def _get_memory_cache(self, user_id: str, thread_id: str):
        """Return cached memory text, or None if cache miss/expired/maxed out."""
        try:
            client = self.rag_system.session_memory._get_client()
        except Exception:
            return None
        if client is None:
            return None
        try:
            key = self._memory_cache_key(user_id, thread_id)
            raw = client.get(key)
            if raw is None:
                return None
            data = json.loads(raw)
            turns_used = int(data.get("turns_used", 0))
            if turns_used >= config.USER_MEMORY_CACHE_MAX_TURNS:
                client.delete(key)
                return None
            # Bump usage counter
            data["turns_used"] = turns_used + 1
            client.setex(key, config.USER_MEMORY_CACHE_TTL_SECONDS, json.dumps(data, ensure_ascii=False))
            return str(data.get("text", ""))
        except Exception:
            return None

    def _set_memory_cache(self, user_id: str, thread_id: str, text: str):
        try:
            client = self.rag_system.session_memory._get_client()
        except Exception:
            return
        if client is None:
            return
        try:
            key = self._memory_cache_key(user_id, thread_id)
            data = {"text": text, "turns_used": 1}
            client.setex(key, config.USER_MEMORY_CACHE_TTL_SECONDS, json.dumps(data, ensure_ascii=False))
        except Exception:
            pass

    def _invalidate_memory_cache(self, user_id: str, thread_id: str):
        """Drop the cache after new memories are extracted (so next turn re-retrieves)."""
        try:
            client = self.rag_system.session_memory._get_client()
        except Exception:
            return
        if client is None:
            return
        try:
            client.delete(self._memory_cache_key(user_id, thread_id))
        except Exception:
            pass

    def _get_graph_config(self, thread_id: str):
        try:
            return self.rag_system.get_config(thread_id)
        except TypeError:
            return self.rag_system.get_config()

    def chat(self, message, history, reveal_diagnostics=False, thread_id=None):
        """Generator that streams Gradio chat message dicts."""
        if not self.rag_system.agent_graph:
            readiness_getter = getattr(self.rag_system, "get_readiness_message", None)
            if callable(readiness_getter):
                yield readiness_getter()
            else:
                yield "系统正在准备中，请稍后再试。"
            return

        turn_input = None
        try:
            turn_input = self.input_service.prepare(message=message, thread_id=thread_id)

            response_messages  = []
            import time as _time
            _node_timings = {}
            _node_start = {}
            _last_node = None
            _graph_start = _time.time()

            for chunk, metadata in self.rag_system.agent_graph.stream(
                turn_input.stream_input,
                config=turn_input.graph_config,
                stream_mode="messages",
            ):
                # H7: guard against runaway graph execution (e.g. LLM call hangs)
                if _time.time() - _graph_start > config.GRAPH_STREAM_MAX_SECONDS:
                    logger.error("Graph stream timed out after %.1fs, aborting", _time.time() - _graph_start)
                    break

                node = metadata.get("langgraph_node", "")

                # Track node-level timing
                if node and node != _last_node:
                    if _last_node and _last_node in _node_start:
                        _node_timings[_last_node] = _node_timings.get(_last_node, 0) + (_time.time() - _node_start[_last_node])
                    _node_start[node] = _time.time()
                    _last_node = node

                if isinstance(chunk, AIMessageChunk) and chunk.content and node not in SILENT_NODES and node not in SYSTEM_NODES:
                    self._handle_llm_token(chunk, node, response_messages)

                yield self._prepare_visible_messages(response_messages, reveal_diagnostics=reveal_diagnostics)

            # Close out the last node timer
            if _last_node and _last_node in _node_start:
                _node_timings[_last_node] = _node_timings.get(_last_node, 0) + (_time.time() - _node_start[_last_node])
            total = _time.time() - _graph_start
            timing_summary = ", ".join(f"{n}={t:.1f}s" for n, t in sorted(_node_timings.items(), key=lambda x: -x[1]))
            logger.warning("Graph timing: total=%.1fs nodes={%s}", total, timing_summary)

            artifacts = self.turn_service.prepare_turn_artifacts(
                active_thread_id=turn_input.active_thread_id,
                graph_config=turn_input.graph_config,
                response_messages=response_messages,
                user_message=turn_input.user_message,
                session_state=turn_input.session_state,
            )
            if artifacts.response_messages_changed:
                yield self._prepare_visible_messages(
                    artifacts.response_messages,
                    reveal_diagnostics=reveal_diagnostics,
                )
            self.turn_service.finalize_turn(
                active_thread_id=turn_input.active_thread_id,
                request_id=turn_input.request_id,
                user_message=turn_input.user_message,
                session_state=turn_input.session_state,
                checkpoint_resumed=turn_input.checkpoint_resumed,
                user_id=turn_input.user_id,
                artifacts=artifacts,
            )

        except Exception as e:
            request_id = getattr(turn_input, "request_id", "")
            user_message = getattr(turn_input, "user_message", message.strip())
            logger.exception("Chat turn failed for request_id=%s", request_id)
            yield self._build_chat_failure_fallback(user_message)
        finally:
            reset_retrieval_context(locals().get("retrieval_context_token"))

    def clear_session(self, thread_id=None):
        self.rag_system.reset_thread(thread_id)
        # Drop memory cache for this thread (across all users — keys are scoped per user)
        try:
            client = self.rag_system.session_memory._get_client()
            if client and thread_id:
                # Wildcard delete all memory caches tied to this thread
                for key in client.scan_iter(match=f"memory_cache:*:{thread_id}"):
                    client.delete(key)
        except Exception:
            pass
        self.rag_system.observability.flush()
