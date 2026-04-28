"""Routing and turn-analysis nodes.

Intent classification, compound-request splitting, department recommendation,
and conversation summarisation live here.  Private helpers that are only used
by these public nodes are kept local to this module; cross-cutting helpers are
imported from ``node_helpers``.
"""

import re
import logging

from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage

from .graph_state import State
from .schemas import (
    IntentAnalysis,
    DepartmentRecommendation,
)
from .prompts import *
from .node_helpers import (
    _APPOINTMENT_NO_RE,
    _ORDINAL_RE,
    _build_appointment_context,
    _build_recent_context,
    _clear_pending_action_state,
    _extract_topic_focus,
    _infer_risk_level,
    _is_abort_request,
    _is_explicit_confirmation,
    _looks_like_appointment_discovery_query,
    _looks_like_department_question,
    _looks_like_explicit_appointment_intent,
    _looks_like_explicit_cancel_intent,
    _looks_like_general_non_medical_query,
    _looks_like_greeting,
    _looks_like_medical_follow_up,
    _looks_like_medical_knowledge_question,
    _looks_like_medical_request,
    _looks_like_medication_risk_query,
    _normalize_date,
    _normalize_time_slot,
    _pick_candidate_from_text,
    _reset_pending_action_if_needed,
    _structured_output_llm,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local constants
# ---------------------------------------------------------------------------

_COMPOUND_SPLIT_RE = re.compile(r"(?:，|,)?\s*(另外|然后|然后再|顺便|并且|同时|再帮我|再问一下)\s*")


# ---------------------------------------------------------------------------
# Private helpers (only used by routing nodes)
# ---------------------------------------------------------------------------

def _needs_medication_detail_clarification(query: str) -> bool:
    normalized = (query or "").strip().lower()
    vague_reference = any(token in normalized for token in ("这个药", "这药", "这种药", "它"))
    return vague_reference and _looks_like_medication_risk_query(query)


def _looks_like_department_name_only(user_query: str) -> bool:
    normalized = (user_query or "").strip()
    if not normalized:
        return False
    from .node_helpers import _DEPARTMENT_HINTS
    return any(department in normalized for department in _DEPARTMENT_HINTS)


def _looks_like_clarification_response(user_query: str) -> bool:
    normalized = (user_query or "").strip().lower()
    if not normalized:
        return False
    if _looks_like_greeting(user_query) or _looks_like_department_question(user_query):
        return False
    if _looks_like_explicit_cancel_intent(user_query) or _looks_like_explicit_appointment_intent(user_query):
        return False
    if _looks_like_medical_knowledge_question(user_query):
        return False
    if len(normalized) <= 40:
        return True
    return bool(
        _normalize_date(user_query)
        or _normalize_time_slot(user_query)
        or _APPOINTMENT_NO_RE.search(user_query or "")
        or _ORDINAL_RE.search(user_query or "")
    )


def _intent_for_clarification_target(target: str, current_intent: str) -> str:
    if target == "recommend_department":
        return "triage"
    if target == "handle_appointment_skill":
        return current_intent or "appointment"
    if target == "handle_appointment":
        return "appointment"
    if target == "handle_cancel_appointment":
        return "cancel_appointment"
    if target == "rewrite_query":
        return "medical_rag"
    return current_intent or "medical_rag"


def _classify_query_by_rules(user_query: str, *, conversation_summary: str = "", recent_context: str = "", topic_focus: str = "") -> tuple[str, str]:
    if _looks_like_greeting(user_query):
        return "greeting", "greeting_rule"
    if _looks_like_explicit_cancel_intent(user_query):
        return "cancel_appointment", "explicit_cancel_rule"
    if _looks_like_appointment_discovery_query(user_query):
        return "appointment", "appointment_discovery_rule"
    if _looks_like_explicit_appointment_intent(user_query):
        return "appointment", "explicit_appointment_rule"
    if _looks_like_department_question(user_query):
        return "triage", "department_question_rule"
    if _looks_like_medical_knowledge_question(user_query) or _looks_like_medical_request(
        user_query,
        conversation_summary=conversation_summary,
        recent_context=recent_context,
        topic_focus=topic_focus,
    ) or _looks_like_medical_follow_up(
        user_query,
        "\n".join(part for part in (conversation_summary, topic_focus) if part),
        recent_context,
    ):
        return "medical_rag", "medical_question_rule"
    if _looks_like_general_non_medical_query(user_query):
        return "medical_rag", "general_conversation_rule"
    return "", "rule_inconclusive"


def _split_compound_request(user_query: str) -> list[str]:
    query = (user_query or "").strip()
    if not query:
        return []
    segments = [segment.strip(" ，,。；;") for segment in _COMPOUND_SPLIT_RE.split(query) if segment and segment.strip(" ，,。；;")]
    cleaned = []
    for segment in segments:
        if segment in {"另外", "然后", "然后再", "顺便", "并且", "同时", "再帮我", "再问一下"}:
            continue
        cleaned.append(segment)
    if not cleaned:
        return [query]
    if len(cleaned) == 1:
        return cleaned
    return cleaned[:2]


def _choose_compound_intents(first_intent: str, second_intent: str) -> tuple[str, str]:
    if (first_intent, second_intent) in {
        ("cancel_appointment", "medical_rag"),
        ("appointment", "medical_rag"),
        ("triage", "appointment"),
        ("triage", "medical_rag"),
    }:
        return first_intent, second_intent
    if (first_intent, second_intent) == ("medical_rag", "appointment"):
        return "appointment", "medical_rag"
    return first_intent, ""


def _looks_like_appointment_update(user_query: str) -> bool:
    normalized = (user_query or "").strip().lower()
    if any(keyword in normalized for keyword in ("挂号", "预约", "改到", "换到", "改成", "换成", "医生", "科", "时间", "时段")):
        return True
    return bool(_normalize_date(user_query) or _normalize_time_slot(user_query))


def _looks_like_cancel_update(user_query: str) -> bool:
    normalized = (user_query or "").strip().lower()
    if any(keyword in normalized for keyword in ("取消", "退号", "预约号", "appointment", "cancel")):
        return True
    return bool(_APPOINTMENT_NO_RE.search(user_query or "") or _ORDINAL_RE.search(user_query or ""))


def _should_continue_pending_action(state: State, user_query: str) -> bool:
    pending_action_type = state.get("pending_action_type", "")
    pending_candidates = state.get("pending_candidates", []) or []
    if not pending_action_type and not pending_candidates:
        return False
    if _is_explicit_confirmation(user_query, pending_action_type) or _is_abort_request(user_query):
        return True
    if pending_candidates and _pick_candidate_from_text(user_query, pending_candidates):
        return True
    if pending_action_type == "appointment":
        return _looks_like_appointment_update(user_query)
    if pending_action_type == "cancel_appointment":
        return _looks_like_cancel_update(user_query)
    return False


def _build_history_reset_messages(messages, keep_recent: int = 5):
    non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    keep_ids = {getattr(m, "id", None) for m in non_system_messages[-keep_recent:]}
    delete_messages = []
    for message in non_system_messages:
        message_id = getattr(message, "id", None)
        if message_id and message_id not in keep_ids:
            delete_messages.append(RemoveMessage(id=message_id))
    return delete_messages


# ---------------------------------------------------------------------------
# Public routing nodes
# ---------------------------------------------------------------------------

def analyze_turn(state: State):
    last_message = state["messages"][-1]
    user_query = str(last_message.content).strip()
    recent_context = state.get("recent_context") or _build_recent_context(state.get("messages", []))
    topic_focus = _extract_topic_focus(
        user_query,
        state.get("topic_focus", ""),
        state.get("appointment_context", {}),
        state.get("recommended_department", ""),
    )

    if state.get("pending_action_type") and _should_continue_pending_action(state, user_query):
        primary_intent = state.get("pending_action_type", "")
        return {
            "recent_context": recent_context,
            "topic_focus": topic_focus or state.get("topic_focus", ""),
            "primary_intent": primary_intent,
            "secondary_intent": state.get("secondary_intent", ""),
            "primary_user_query": user_query,
            "secondary_user_query": state.get("secondary_user_query", ""),
            "deferred_user_question": state.get("deferred_user_question", ""),
            "decision_source": "resume",
            "route_reason": "continue_pending_action",
            "last_route_reason": "continue_pending_action",
        }

    if state.get("pending_candidates") and _pick_candidate_from_text(user_query, state.get("pending_candidates") or []):
        return {
            "recent_context": recent_context,
            "topic_focus": topic_focus or state.get("topic_focus", ""),
            "primary_intent": "cancel_appointment",
            "secondary_intent": state.get("secondary_intent", ""),
            "primary_user_query": user_query,
            "secondary_user_query": state.get("secondary_user_query", ""),
            "deferred_user_question": state.get("deferred_user_question", ""),
            "decision_source": "resume",
            "route_reason": "continue_pending_candidates",
            "last_route_reason": "continue_pending_candidates",
        }

    clarification_target = state.get("clarification_target", "")
    if state.get("pending_clarification") and clarification_target and _looks_like_clarification_response(user_query):
        primary_intent = _intent_for_clarification_target(clarification_target, state.get("intent", ""))
        return {
            "recent_context": recent_context,
            "topic_focus": topic_focus or state.get("topic_focus", ""),
            "primary_intent": primary_intent,
            "secondary_intent": state.get("secondary_intent", ""),
            "primary_user_query": user_query,
            "secondary_user_query": state.get("secondary_user_query", ""),
            "deferred_user_question": state.get("deferred_user_question", ""),
            "decision_source": "resume",
            "route_reason": f"continue_{clarification_target}",
            "last_route_reason": f"continue_{clarification_target}",
        }

    if (
        (state.get("intent") == "appointment" or state.get("appointment_skill_mode") in {"discover_department", "clarify", "discover_doctor", "discover_availability"})
        and _looks_like_department_name_only(user_query)
        and not _looks_like_explicit_cancel_intent(user_query)
    ):
        return {
            "recent_context": recent_context,
            "topic_focus": _extract_topic_focus(
                user_query,
                state.get("topic_focus", ""),
                state.get("appointment_context", {}),
                state.get("recommended_department", ""),
            ),
            "primary_intent": "appointment",
            "secondary_intent": "",
            "primary_user_query": user_query,
            "secondary_user_query": "",
            "deferred_user_question": "",
            "decision_source": "resume",
            "route_reason": "continue_department_selection",
            "last_route_reason": "continue_department_selection",
        }

    segments = _split_compound_request(user_query)
    first_segment = segments[0] if segments else user_query
    second_segment = segments[1] if len(segments) > 1 else ""
    first_intent, first_reason = _classify_query_by_rules(
        first_segment,
        conversation_summary=state.get("conversation_summary", ""),
        recent_context=recent_context,
        topic_focus=state.get("topic_focus", ""),
    )
    second_intent = ""
    second_reason = ""
    if second_segment:
        second_intent, second_reason = _classify_query_by_rules(
            second_segment,
            conversation_summary=state.get("conversation_summary", ""),
            recent_context=recent_context,
            topic_focus=state.get("topic_focus", ""),
        )
    primary_intent, secondary_intent = _choose_compound_intents(first_intent, second_intent)
    if primary_intent:
        route_reason = first_reason if not secondary_intent else f"{first_reason}+{second_reason or 'secondary'}"
        return {
            "recent_context": recent_context,
            "topic_focus": topic_focus or state.get("topic_focus", ""),
            "primary_intent": primary_intent,
            "secondary_intent": secondary_intent,
            "primary_user_query": first_segment if secondary_intent else user_query,
            "secondary_user_query": second_segment if secondary_intent else "",
            "deferred_user_question": second_segment if secondary_intent else "",
            "decision_source": "rule",
            "route_reason": route_reason,
            "last_route_reason": route_reason,
        }

    return {
        "recent_context": recent_context,
        "topic_focus": topic_focus or state.get("topic_focus", ""),
        "primary_intent": "",
        "secondary_intent": "",
        "primary_user_query": user_query,
        "secondary_user_query": "",
        "deferred_user_question": "",
        "decision_source": "llm",
        "route_reason": "rule_inconclusive",
        "last_route_reason": "rule_inconclusive",
    }


def summarize_history(state: State, llm):
    existing_summary = state.get("conversation_summary", "")
    if len(state["messages"]) < 4:
        return {"conversation_summary": existing_summary}

    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": existing_summary}

    conversation = "Conversation history:\n"
    if existing_summary.strip():
        conversation += f"[Prior conversation summary]\n{existing_summary}\n\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    summary_response = llm.with_config(temperature=0.2).invoke([SystemMessage(content=get_conversation_summary_prompt()), HumanMessage(content=conversation)])
    return {"conversation_summary": summary_response.content, "agent_answers": [{"__reset__": True}]}


def intent_router(state: State, llm):
    if not state.get("primary_intent"):
        state = {**state, **analyze_turn(state)}
    last_message = state["messages"][-1]
    user_query = str(last_message.content).strip()
    risk_level = _infer_risk_level(user_query, state.get("risk_level", "normal"))
    pending_action_type = state.get("pending_action_type", "")
    pending_candidates = state.get("pending_candidates", []) or []
    recent_context = state.get("recent_context") or _build_recent_context(state.get("messages", []))
    topic_focus = state.get("topic_focus", "")
    primary_intent = state.get("primary_intent", "")
    secondary_intent = state.get("secondary_intent", "")
    primary_user_query = state.get("primary_user_query", "") or user_query
    secondary_user_query = state.get("secondary_user_query", "")
    decision_source = state.get("decision_source", "")
    route_reason = state.get("route_reason", "")

    if _needs_medication_detail_clarification(primary_user_query):
        clarification = "请先告诉我药名、规格或包装上写的剂量信息，我才能更安全地帮你判断怎么用。"
        return {
            "intent": "clarification",
            "primary_intent": "clarification",
            "secondary_intent": "",
            "primary_user_query": primary_user_query,
            "secondary_user_query": "",
            "decision_source": "rule",
            "route_reason": "medication_dose_needs_details",
            "last_route_reason": "medication_dose_needs_details",
            "risk_level": "high",
            "pending_clarification": clarification,
            "clarification_target": "intent_router",
            "recent_context": recent_context,
            "topic_focus": topic_focus or _extract_topic_focus(primary_user_query, topic_focus),
            "deferred_user_question": "",
            "clarification_attempts": 1,
            "recommended_department": state.get("recommended_department", ""),
            "appointment_context": state.get("appointment_context", {}),
            "last_appointment_no": state.get("last_appointment_no", ""),
            **_reset_pending_action_if_needed(state),
            "messages": [AIMessage(content=clarification)],
        }

    if primary_intent == "greeting":
        greeting_response = "你好！我是你的医疗助手，可以帮你：\n- 🏥 推荐就诊科室\n- 📅 预约挂号\n- ❌ 取消预约\n- 💊 解答医疗健康问题\n\n请问有什么可以帮你的？"
        return {
            "intent": "greeting",
            "primary_intent": "greeting",
            "secondary_intent": "",
            "primary_user_query": primary_user_query,
            "secondary_user_query": "",
            "decision_source": decision_source or "rule",
            "route_reason": route_reason or "greeting_rule",
            "last_route_reason": route_reason or "greeting_rule",
            "risk_level": risk_level,
            "pending_clarification": "",
            "clarification_target": "",
            "recent_context": recent_context,
            "topic_focus": topic_focus,
            "deferred_user_question": "",
            "clarification_attempts": 0,
            "recommended_department": state.get("recommended_department", ""),
            "appointment_context": state.get("appointment_context", {}),
            "last_appointment_no": state.get("last_appointment_no", ""),
            **_reset_pending_action_if_needed(state),
            "messages": [AIMessage(content=greeting_response)],
        }

    if primary_intent in {"triage", "appointment", "cancel_appointment", "medical_rag"}:
        if primary_intent == "triage":
            pending_updates = _clear_pending_action_state()
            recommended_department = ""
            appointment_context = {}
            last_appointment_no = ""
        elif primary_intent == "medical_rag":
            pending_updates = _reset_pending_action_if_needed(state)
            recommended_department = state.get("recommended_department", "")
            appointment_context = state.get("appointment_context", {})
            last_appointment_no = state.get("last_appointment_no", "")
        else:
            pending_updates = {
                "pending_action_type": pending_action_type,
                "pending_action_payload": state.get("pending_action_payload", {}),
                "pending_confirmation_id": state.get("pending_confirmation_id", ""),
                "pending_candidates": pending_candidates,
            }
            recommended_department = state.get("recommended_department", "")
            appointment_context = state.get("appointment_context", {})
            last_appointment_no = state.get("last_appointment_no", "")
        return {
            "intent": primary_intent,
            "primary_intent": primary_intent,
            "secondary_intent": secondary_intent,
            "primary_user_query": primary_user_query,
            "secondary_user_query": secondary_user_query,
            "decision_source": decision_source or "rule",
            "route_reason": route_reason or "rule_match",
            "last_route_reason": route_reason or "rule_match",
            "risk_level": risk_level,
            "pending_clarification": "",
            "clarification_target": "",
            "recent_context": recent_context,
            "topic_focus": topic_focus,
            "deferred_user_question": state.get("deferred_user_question", "") or secondary_user_query,
            "clarification_attempts": 0,
            "recommended_department": recommended_department,
            "appointment_context": appointment_context,
            "last_appointment_no": last_appointment_no,
            **pending_updates,
        }

    try:
        llm_with_structure = _structured_output_llm(llm, IntentAnalysis, temperature=0.1)
        response = llm_with_structure.invoke(
            [
                SystemMessage(content=get_intent_router_prompt()),
                HumanMessage(
                    content=(
                        f"Conversation summary:\n{state.get('conversation_summary', '')}\n\n"
                        f"Recent dialogue context:\n{recent_context}\n\n"
                        f"User query:\n{user_query}"
                    )
                ),
            ]
        )
    except Exception:
        logger.exception("Intent router structured output failed; falling back to medical_rag.")
        return {
            "intent": "medical_rag",
            "primary_intent": "medical_rag",
            "secondary_intent": "",
            "primary_user_query": user_query,
            "secondary_user_query": "",
            "decision_source": "llm_error_fallback",
            "route_reason": "intent_router_exception_fallback",
            "last_route_reason": "intent_router_exception_fallback",
            "risk_level": risk_level,
            "pending_clarification": "",
            "clarification_target": "",
            "recent_context": recent_context,
            "topic_focus": topic_focus or _extract_topic_focus(user_query, topic_focus),
            "deferred_user_question": "",
            "clarification_attempts": 0,
            "recommended_department": state.get("recommended_department", ""),
            "appointment_context": state.get("appointment_context", {}),
            "last_appointment_no": state.get("last_appointment_no", ""),
            **_reset_pending_action_if_needed(state),
        }

    if response.is_clear and response.intent in {"medical_rag", "triage", "appointment", "cancel_appointment"}:
        pending_updates = (
            _clear_pending_action_state()
            if response.intent in {"medical_rag", "triage"}
            else {
                "pending_action_type": state.get("pending_action_type", ""),
                "pending_action_payload": state.get("pending_action_payload", {}),
                "pending_confirmation_id": state.get("pending_confirmation_id", ""),
                "pending_candidates": state.get("pending_candidates", []),
            }
        )
        return {
            "intent": response.intent,
            "primary_intent": response.intent,
            "secondary_intent": "",
            "primary_user_query": user_query,
            "secondary_user_query": "",
            "decision_source": "llm",
            "route_reason": f"llm:{response.intent}",
            "last_route_reason": f"llm:{response.intent}",
            "risk_level": risk_level,
            "pending_clarification": "",
            "clarification_target": "",
            "recent_context": recent_context,
            "topic_focus": topic_focus,
            "deferred_user_question": "",
            "clarification_attempts": 0,
            "recommended_department": state.get("recommended_department", ""),
            "appointment_context": state.get("appointment_context", {}),
            "last_appointment_no": state.get("last_appointment_no", ""),
            **pending_updates,
        }

    clarification_attempts = int(state.get("clarification_attempts") or 0) + 1
    if clarification_attempts > 1:
        if _looks_like_medical_request(user_query, conversation_summary=state.get("conversation_summary", ""), recent_context=recent_context, topic_focus=topic_focus):
            fallback_answer = "我先给你一个保守建议：如果你有持续不适、症状加重，建议尽快线下就医；如果你愿意，也可以再补充一句最困扰你的症状，我会继续帮你缩小范围。"
        else:
            fallback_answer = "我先按你现在这句话理解来继续帮你，不再追问太多。如果你愿意，也可以再补充一点背景，我会回答得更贴合。"
        return {
            "intent": "medical_rag",
            "primary_intent": "medical_rag",
            "secondary_intent": "",
            "primary_user_query": user_query,
            "secondary_user_query": "",
            "decision_source": "clarification_budget",
            "route_reason": "clarification_budget_exceeded",
            "last_route_reason": "clarification_budget_exceeded",
            "risk_level": risk_level,
            "pending_clarification": "",
            "clarification_target": "",
            "recent_context": recent_context,
            "topic_focus": topic_focus,
            "deferred_user_question": "",
            "clarification_attempts": clarification_attempts,
            "recommended_department": state.get("recommended_department", ""),
            "appointment_context": state.get("appointment_context", {}),
            "last_appointment_no": state.get("last_appointment_no", ""),
            **_reset_pending_action_if_needed(state),
            "messages": [AIMessage(content=fallback_answer)],
        }

    clarification = response.clarification_needed if response.clarification_needed and len(response.clarification_needed.strip()) > 5 else "可以再具体描述一下你的问题吗？"
    return {
        "intent": "clarification",
        "primary_intent": "clarification",
        "secondary_intent": "",
        "primary_user_query": user_query,
        "secondary_user_query": "",
        "decision_source": "llm",
        "route_reason": "llm:clarification",
        "last_route_reason": "llm:clarification",
        "risk_level": risk_level,
        "pending_clarification": clarification,
        "clarification_target": "intent_router",
        "recent_context": recent_context,
        "topic_focus": topic_focus,
        "deferred_user_question": "",
        "clarification_attempts": clarification_attempts,
        "recommended_department": state.get("recommended_department", ""),
        "appointment_context": state.get("appointment_context", {}),
        "last_appointment_no": state.get("last_appointment_no", ""),
        **_reset_pending_action_if_needed(state),
        "messages": [AIMessage(content=clarification)],
    }


def recommend_department(state: State, llm):
    last_message = state["messages"][-1]
    user_query = state.get("primary_user_query") or str(last_message.content).strip()
    conversation_summary = state.get("conversation_summary", "")
    risk_level = state.get("risk_level", "normal")
    topic_focus = state.get("topic_focus", "")

    try:
        llm_with_structure = _structured_output_llm(llm, DepartmentRecommendation, temperature=0.1)
        response = llm_with_structure.invoke(
            [
                SystemMessage(content=get_department_recommendation_prompt()),
                HumanMessage(content=f"Conversation summary:\n{conversation_summary}\n\nUser query:\n{user_query}"),
            ]
        )
    except Exception:
        logger.exception("Department recommendation structured output failed; returning safe fallback.")
        answer = "我暂时无法稳定判断最合适的科室。若症状较急或明显加重，建议优先急诊；一般不适可以先到全科医学科/普通内科，由医生再进一步分诊。"
        return {
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
            "recommended_department": "全科医学科",
            "topic_focus": topic_focus or "全科医学科",
            "appointment_context": _build_appointment_context(state.get("appointment_context"), {"department": "全科医学科"}),
            **_reset_pending_action_if_needed(state),
            "messages": [AIMessage(content=answer)],
        }

    if response.needs_clarification or not response.department.strip():
        if risk_level == "high":
            answer = "你描述里有较高风险信号，建议优先去 **急诊科** 进一步评估；如果症状明显加重，请立即线下就医。"
            return {
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                "recommended_department": "急诊科",
                "topic_focus": topic_focus or "急诊科",
                "appointment_context": _build_appointment_context(state.get("appointment_context"), {"department": "急诊科"}),
                **_reset_pending_action_if_needed(state),
                "messages": [AIMessage(content=answer)],
            }
        clarification_attempts = int(state.get("clarification_attempts") or 0) + 1
        if clarification_attempts > 1:
            answer = "如果你目前还拿不准具体挂什么科，建议先从 **全科医学科/普通内科** 开始，由医生根据症状再分流；如果出现胸痛、呼吸困难、意识异常等情况，请优先急诊。"
            return {
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": clarification_attempts,
                "recommended_department": "全科医学科",
                "topic_focus": topic_focus or "全科医学科",
                "appointment_context": _build_appointment_context(state.get("appointment_context"), {"department": "全科医学科"}),
                **_reset_pending_action_if_needed(state),
                "messages": [AIMessage(content=answer)],
            }
        clarification = response.clarification_needed if response.clarification_needed and len(response.clarification_needed.strip()) > 5 else "可以再补充一下你的主要症状、持续时间或最不舒服的部位吗？"
        return {
            "pending_clarification": clarification,
            "clarification_target": "recommend_department",
            "clarification_attempts": clarification_attempts,
            "recommended_department": "",
            "topic_focus": topic_focus or _extract_topic_focus(user_query, topic_focus),
            **_reset_pending_action_if_needed(state),
            "messages": [AIMessage(content=clarification)],
        }

    answer = f"建议优先咨询 **{response.department.strip()}**。\n\n原因：{response.reason.strip()}"
    if risk_level == "high":
        answer += "\n\n你当前描述里有较高风险信号，建议尽快线下就医；如果症状明显加重，请优先考虑急诊评估。"

    return {
        "recommended_department": response.department.strip(),
        "pending_clarification": "",
        "clarification_target": "",
        "clarification_attempts": 0,
        "topic_focus": response.department.strip(),
        "appointment_context": _build_appointment_context(state.get("appointment_context"), {"department": response.department.strip()}),
        **_clear_pending_action_state(),
        "messages": [AIMessage(content=answer)],
    }


def request_clarification(state: State):
    return {}


def prepare_secondary_turn(state: State):
    secondary_intent = state.get("secondary_intent", "")
    deferred_question = state.get("deferred_user_question") or state.get("secondary_user_query") or ""
    if not secondary_intent or not deferred_question:
        return {}
    return {
        "intent": secondary_intent,
        "primary_intent": secondary_intent,
        "secondary_intent": "",
        "primary_user_query": deferred_question,
        "secondary_user_query": "",
        "deferred_user_question": "",
        "route_reason": f"resume_secondary:{secondary_intent}",
        "last_route_reason": f"resume_secondary:{secondary_intent}",
        "messages": [HumanMessage(content=deferred_question)],
    }


__all__ = [
    "analyze_turn",
    "intent_router",
    "prepare_secondary_turn",
    "recommend_department",
    "request_clarification",
    "summarize_history",
]
