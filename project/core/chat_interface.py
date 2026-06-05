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
import uuid

import config
from db.route_log_store import RouteLogStore
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage, SystemMessage
from rag_agent.node_helpers import _sanitize_final_answer_text
from rag_agent.tools import reset_retrieval_context, set_retrieval_context


logger = logging.getLogger(__name__)

SILENT_NODES = {
    "rewrite_query",
    "intent_router",
    "plan_retrieval_queries",
    "grounded_answer_generation",
    "answer_grounding_check",
}
SYSTEM_NODES = {"summarize_history", "rewrite_query"}
APPOINTMENT_UPDATE_HINTS = ("改", "换", "预约", "挂号", "医生", "科", "时间", "时段")
CANCEL_UPDATE_HINTS = ("取消", "退号", "预约号", "第", "appointment", "cancel")
PENDING_ACK_HINTS = ("可以", "好的", "行", "好", "ok", "okay")
MEDICAL_KB_HINTS = (
    "高血压", "糖尿病", "感冒", "发烧", "发热", "低烧", "高烧", "头疼", "头痛", "偏头痛",
    "头晕", "眩晕", "咳嗽", "咳痰", "咽痛", "嗓子疼", "喉咙痛", "流鼻涕", "鼻塞",
    "腹痛", "肚子疼", "胃痛", "腹泻", "拉肚子", "便秘", "恶心", "呕吐", "胸闷", "胸痛",
    "心悸", "心慌", "乏力", "呼吸困难", "气短", "肺炎", "哮喘", "鼻炎", "胃炎", "肠胃炎",
    "血压", "血糖", "症状", "治疗", "检查", "药", "用药", "疼", "痛", "不舒服",
    "hypertension", "diabetes", "fever", "cough", "dizziness", "symptom", "treatment",
)
MEDICAL_KB_QUESTION_HINTS = (
    "是什么", "怎么回事", "为什么", "原因", "症状", "表现", "怎么办", "如何", "怎么处理",
    "怎么缓解", "严重吗", "会不会", "会引起", "会导致", "能不能", "可以吗", "要不要",
    "治疗", "预防", "what is", "why", "how to", "symptoms", "treatment",
)
MEDICAL_FALLBACK_DANGER_HINTS = (
    "胸痛", "胸闷", "呼吸困难", "气短", "意识模糊", "抽搐", "晕厥", "剧烈", "突然",
    "持续加重", "大出血", "高热", "肢体无力", "视物模糊",
)

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
        self.route_log_store = RouteLogStore()

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
        normalized = (query or "").strip().lower()
        patterns = [
            "挂什么科",
            "挂哪个科",
            "看什么科",
            "看哪个科",
            "挂哪科",
            "看哪科",
            "咨询什么科",
            "which department",
            "what department should i visit",
            "what department should i register for",
        ]
        return any(pattern in normalized for pattern in patterns)

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
        normalized = (user_message or "").strip().lower()
        if ChatInterface._should_continue_pending_intent(user_message, existing_state or {}):
            return "pending"
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
        return "刚才聊天链路短暂不稳定，但我还在。你可以再发一次问题，我会继续帮你处理。"

    @staticmethod
    def _infer_risk_level(user_message: str, existing_state: dict):
        normalized = (user_message or "").strip().lower()
        if any(keyword.lower() in normalized for keyword in config.HIGH_RISK_KEYWORDS):
            return "high"
        return existing_state.get("risk_level", "normal")

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

        active_thread_id = thread_id or self.rag_system.thread_id
        graph_config  = self._get_graph_config(active_thread_id)
        current_state = self.rag_system.agent_graph.get_state(graph_config)
        user_message  = message.strip()
        request_id    = uuid.uuid4().hex
        session_state = self.rag_system.session_memory.get_state(active_thread_id)
        checkpoint_resumed = bool(current_state.next)

        # Resolve user_id for memory injection/extraction
        user_id = ""
        if config.USER_MEMORY_ENABLED:
            try:
                session_info = self.rag_system.chat_sessions.get_session(active_thread_id)
                user_id = (session_info or {}).get("owner_user_id", "") or ""
            except Exception:
                logger.warning("Failed to resolve user_id for memory injection", exc_info=True)

        # Retrieve user-level memories (L3 semantic)
        user_memories_text = ""
        if user_id and config.USER_MEMORY_ENABLED and config.USER_MEMORY_INJECTION_ENABLED:
            try:
                memories = self.rag_system.user_memory_store.retrieve_memories(
                    user_id, user_message, top_k=config.USER_MEMORY_MAX_RETRIEVED
                )
                if memories:
                    user_memories_text = "\n".join(
                        f"- [{m['memory_type']}|{m['importance']}] {m['content']}" for m in memories
                    )
            except Exception:
                logger.warning("User memory retrieval failed; continuing without memories.", exc_info=True)

        try:
            if current_state.next:
                update_payload = {
                    "messages": [HumanMessage(content=user_message)],
                    "thread_id": active_thread_id,
                    "request_id": request_id,
                }
                if user_memories_text:
                    update_payload["user_memories"] = user_memories_text
                self.rag_system.agent_graph.update_state(
                    graph_config,
                    update_payload,
                )
                stream_input = None
            else:
                stored_messages = []
                stored_messages = self.rag_system.session_memory.get_recent_messages(active_thread_id)
                long_term_summary = self.rag_system.summary_store.get_summary(active_thread_id)
                state_messages = self._build_state_messages(session_state)
                if long_term_summary:
                    self.rag_system.agent_graph.update_state(
                        graph_config,
                        {"conversation_summary": long_term_summary},
                    )
                if session_state:
                    self.rag_system.agent_graph.update_state(graph_config, self._graph_state_from_session(active_thread_id, session_state))
                if not session_state:
                    self.rag_system.agent_graph.update_state(graph_config, {"thread_id": active_thread_id, "agent_answers": [{"__reset__": True}]})
                stream_input = {
                    "messages": [*state_messages, *stored_messages, HumanMessage(content=user_message)],
                    "request_id": request_id,
                    "user_memories": user_memories_text,
                }

            response_messages  = []

            for chunk, metadata in self.rag_system.agent_graph.stream(stream_input, config=graph_config, stream_mode="messages"):
                node = metadata.get("langgraph_node", "")

                if isinstance(chunk, AIMessageChunk) and chunk.content and node not in SILENT_NODES and node not in SYSTEM_NODES:
                    self._handle_llm_token(chunk, node, response_messages)

                yield self._prepare_visible_messages(response_messages, reveal_diagnostics=reveal_diagnostics)

            final_assistant = self._extract_final_assistant_text(response_messages)
            all_visible_assistant_texts = self._extract_all_visible_assistant_texts(response_messages)
            clarification_text = self._extract_clarification_text(response_messages)
            latest_state = self.rag_system.agent_graph.get_state(graph_config)
            latest_values = getattr(latest_state, "values", {}) or {}
            if not final_assistant:
                final_from_state = self._extract_latest_state_assistant(latest_values)
                if final_from_state:
                    response_messages.append(make_message(final_from_state))
                    final_assistant = final_from_state
                    yield self._prepare_visible_messages(response_messages, reveal_diagnostics=reveal_diagnostics)
            if not final_assistant:
                final_assistant = self._build_chat_failure_fallback(user_message)
                response_messages.append(make_message(final_assistant))
                yield self._prepare_visible_messages(response_messages, reveal_diagnostics=reveal_diagnostics)
            combined_assistant_text = "\n\n".join(all_visible_assistant_texts) if all_visible_assistant_texts else final_assistant
            if combined_assistant_text:
                recent_count = self.rag_system.session_memory.append_exchange(active_thread_id, user_message, combined_assistant_text)
                if recent_count >= config.SUMMARY_REFRESH_THRESHOLD:
                    conversation_summary = latest_values.get("conversation_summary", "")
                    if conversation_summary:
                        self.rag_system.summary_store.save_summary(active_thread_id, conversation_summary, recent_count)

            # Run summarize_history as post-chat cleanup (was in graph, now off critical path)
            if combined_assistant_text:
                try:
                    from rag_agent.routing_nodes import summarize_history
                    from model_factory import get_chat_model
                    summary_llm = get_chat_model().with_config(temperature=0.2)
                    summary_result = summarize_history(latest_values, summary_llm)
                    new_summary = (summary_result or {}).get("conversation_summary", "")
                    if new_summary and new_summary != latest_values.get("conversation_summary", ""):
                        recent_count = self.rag_system.session_memory.recent_message_count(active_thread_id)
                        self.rag_system.summary_store.save_summary(active_thread_id, new_summary, recent_count)
                except Exception:
                    logger.warning("Post-chat summarization failed for thread_id=%s", active_thread_id, exc_info=True)

            # Extract user memories (async, fire-and-forget)
            if user_id and config.USER_MEMORY_ENABLED and config.USER_MEMORY_EXTRACTION_ENABLED and combined_assistant_text:
                try:
                    self.rag_system.memory_extractor.extract_and_save(
                        thread_id=active_thread_id,
                        user_message=user_message,
                        assistant_message=combined_assistant_text,
                        conversation_summary=latest_values.get("conversation_summary", ""),
                    )
                except Exception:
                    logger.warning("Memory extraction failed for thread_id=%s", active_thread_id, exc_info=True)

            updated_state = self._resolved_session_state(latest_values, session_state, user_message, clarification_text)
            if updated_state != (session_state or {}):
                self.rag_system.session_memory.set_state(active_thread_id, updated_state)

            had_pending_state = bool(
                (session_state or {}).get("pending_action_type")
                or (session_state or {}).get("pending_clarification")
                or (session_state or {}).get("deferred_user_question")
            )
            try:
                route_reason = latest_values.get("route_reason", updated_state.get("last_route_reason") or "") or ""
                secondary_turn_executed = str(route_reason).startswith("resume_secondary:")
                self.route_log_store.save_log(
                    {
                        "thread_id": active_thread_id,
                        "request_id": request_id,
                        "user_query": user_message,
                        "primary_intent": latest_values.get("primary_intent", updated_state.get("intent")) or "",
                        "secondary_intent": updated_state.get("secondary_intent") or "",
                        "decision_source": latest_values.get("decision_source", "") or "",
                        "route_reason": route_reason,
                        "had_pending_state": had_pending_state,
                        "extra_metadata": {
                            "topic_focus": updated_state.get("topic_focus") or "",
                            "deferred_user_question": updated_state.get("deferred_user_question") or "",
                            "checkpoint_resumed": checkpoint_resumed,
                            "secondary_turn_executed": secondary_turn_executed,
                            "pending_action_type": (session_state or {}).get("pending_action_type") or "",
                            "pending_clarification": bool((session_state or {}).get("pending_clarification")),
                        },
                    }
                )
            except Exception:
                logger.exception("Failed to persist route log for request_id=%s", request_id)

        except Exception as e:
            logger.exception("Chat turn failed for request_id=%s", request_id)
            yield self._build_chat_failure_fallback(user_message)
        finally:
            reset_retrieval_context(locals().get("retrieval_context_token"))

    def clear_session(self, thread_id=None):
        self.rag_system.reset_thread(thread_id)
        self.rag_system.observability.flush()
