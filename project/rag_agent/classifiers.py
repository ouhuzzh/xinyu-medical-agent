"""Classification helpers — keyword/regex-based intent and risk detection.

Extracted from node_helpers for focused reusability.
"""

import re
import logging
from datetime import date, timedelta

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage

from .graph_state import State
import config
from config import HIGH_RISK_KEYWORDS

logger = logging.getLogger(__name__)

# Re-export constants used by classification functions
from .node_helpers import (
    _APPOINTMENT_CONFIRM_WORDS,
    _CANCEL_CONFIRM_WORDS,
    _ABORT_WORDS,
    _MEDICAL_FOLLOW_UP_HINTS,
    _MEDICAL_TERMS,
    _MEDICAL_QUESTION_PATTERNS,
    _APPOINTMENT_KEYWORDS,
    _CANCEL_KEYWORDS,
    _EXPLICIT_APPOINTMENT_CUES,
    _EXPLICIT_CANCEL_CUES,
    _GENERAL_CHAT_HINTS,
    _NON_MEDICAL_TOPIC_HINTS,
    _MEDICATION_RISK_HINTS,
    _DEPARTMENT_HINTS,
    _APPOINTMENT_LIST_HINTS,
    _DOCTOR_DISCOVERY_HINTS,
    _ANY_DOCTOR_HINTS,
    _EARLIEST_SLOT_HINTS,
    _RESCHEDULE_HINTS,
    _ORDINAL_RE,
    _APPOINTMENT_NO_RE,
)


def _looks_like_greeting(query: str) -> bool:
    normalized = (query or "").strip().lower()
    greetings = ("你好", "您好", "嗨", "hi", "hello", "hey", "早上好", "下午好",
                "晚上好", "good morning", "good afternoon", "good evening")
    if normalized in greetings:
        return True
    if any(normalized.startswith(g) for g in ("你好", "您好", "hi", "hello")):
        return True
    return False


def _starts_with_polite_decline(query: str) -> bool:
    normalized = (query or "").strip().lower()
    declines = ("不用了", "先不用", "算了", "不需要", "暂时不需要", "不用",
                "no ", "nope", "not now", "not interested", "never mind")
    return any(normalized.startswith(d) for d in declines)


def _looks_like_department_question(query: str) -> bool:
    normalized = (query or "").strip().lower()
    patterns = ("挂什么科", "挂哪个科", "看什么科", "看哪个科", "挂哪科", "看哪科",
                "咨询什么科", "推荐科室", "应该挂什么科", "去哪个科")
    return any(p in normalized for p in patterns)


def _looks_like_medical_knowledge_question(query: str) -> bool:
    normalized = (query or "").strip().lower()
    has_medical_term = any(term in normalized for term in _MEDICAL_TERMS)
    has_question = any(p in normalized for p in _MEDICAL_QUESTION_PATTERNS)
    return has_medical_term and has_question


def _looks_like_medication_risk_query(query: str) -> bool:
    normalized = (query or "").strip().lower()
    return any(hint in normalized for hint in _MEDICATION_RISK_HINTS)


def _context_has_medical_signal(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(term in lower for term in _MEDICAL_TERMS)


def _looks_like_medical_request(query: str, *, conversation_summary: str = "", recent_context: str = "", topic_focus: str = "") -> bool:
    normalized = (query or "").strip().lower()
    if _looks_like_medical_knowledge_question(query):
        return True
    has_term = any(term in normalized for term in _MEDICAL_TERMS)
    has_follow = any(hint in normalized for hint in _MEDICAL_FOLLOW_UP_HINTS)
    if has_term and has_follow:
        return True
    if has_term and _context_has_medical_signal(recent_context):
        return True
    if has_term and _context_has_medical_signal(topic_focus):
        return True
    return False


def _looks_like_medical_follow_up(user_query: str, conversation_summary: str, recent_context: str = "") -> bool:
    normalized = (user_query or "").strip().lower()
    if any(hint in normalized for hint in _MEDICAL_FOLLOW_UP_HINTS):
        if _context_has_medical_signal(recent_context) or _context_has_medical_signal(conversation_summary):
            return True
    return False


def _looks_like_general_non_medical_query(query: str) -> bool:
    normalized = (query or "").strip().lower()
    if any(hint in normalized for hint in _NON_MEDICAL_TOPIC_HINTS):
        return True
    if any(hint in normalized for hint in _GENERAL_CHAT_HINTS):
        return True
    return False


def _needs_strict_medical_safety(query: str, risk_level: str = "normal") -> bool:
    if risk_level == "high":
        return True
    if _looks_like_medication_risk_query(query):
        return True
    return False


def _looks_like_explicit_appointment_intent(user_query: str) -> bool:
    normalized = (user_query or "").strip().lower()
    if any(kw in normalized for kw in _APPOINTMENT_KEYWORDS):
        if any(cue in normalized for cue in _EXPLICIT_APPOINTMENT_CUES):
            return True
        if any(dept in normalized for dept in _DEPARTMENT_HINTS):
            return True
        date_patterns = [r"\d{1,2}\s*月", r"明天", r"后天", r"下周", r"本周"]
        if any(re.search(p, normalized) for p in date_patterns):
            return True
    return False


def _looks_like_appointment_discovery_query(user_query: str) -> bool:
    normalized = (user_query or "").strip().lower()
    return any(hint in normalized for hint in (_APPOINTMENT_LIST_HINTS + _DOCTOR_DISCOVERY_HINTS))


def _looks_like_explicit_cancel_intent(user_query: str) -> bool:
    normalized = (user_query or "").strip().lower()
    if any(kw in normalized for kw in _CANCEL_KEYWORDS):
        return True
    if any(cue in normalized for cue in _EXPLICIT_CANCEL_CUES):
        return True
    if re.search(_APPOINTMENT_NO_RE, normalized):
        return True
    if re.search(_ORDINAL_RE, normalized):
        return any(kw in normalized for kw in _CANCEL_KEYWORDS)
    return False


def _looks_like_department_name_only(user_query: str) -> bool:
    normalized = (user_query or "").strip().lower()
    for dept in _DEPARTMENT_HINTS:
        if normalized == dept.lower():
            return True
    for dept in _DEPARTMENT_HINTS:
        if dept.lower() in normalized and len(normalized) <= len(dept) + 4:
            return True
    return False


def _looks_like_clarification_response(user_query: str) -> bool:
    normalized = (user_query or "").strip().lower()
    if any(word in normalized for word in ("确认预约", "确认挂号", "确认取消", "确认退号")):
        return True
    return len(normalized) <= 25


def _should_continue_pending_action(state: State, user_query: str) -> bool:
    from .node_helpers import _is_explicit_confirmation, _is_abort_request, _pick_candidate_from_text
    normalized = (user_query or "").strip().lower()

    if _is_abort_request(user_query):
        return False

    if _is_explicit_confirmation(user_query, state.get("pending_action_type", "")):
        return True

    pending_candidates = state.get("pending_candidates") or []
    if pending_candidates and _pick_candidate_from_text(user_query, pending_candidates):
        return True

    pending_action_type = state.get("pending_action_type", "")
    if pending_action_type == "appointment":
        from .node_helpers import _looks_like_schedule_update
        return _looks_like_schedule_update(user_query)
    if pending_action_type == "cancel_appointment":
        return any(token in normalized for token in _CANCEL_KEYWORDS)

    return False


def _intent_for_clarification_target(target: str, current_intent: str) -> str:
    mapping = {
        "department": "appointment",
        "doctor": "appointment",
        "time_slot": "appointment",
        "appointment_id": "cancel_appointment",
        "appointment_no": "cancel_appointment",
    }
    return mapping.get(target, current_intent)


def _is_reschedule_intent(user_query: str) -> bool:
    normalized = (user_query or "").strip().lower()
    return any(hint in normalized for hint in _RESCHEDULE_HINTS)


def _wants_any_available_doctor(user_query: str) -> bool:
    normalized = (user_query or "").strip().lower()
    return any(hint in normalized for hint in _ANY_DOCTOR_HINTS)


def _wants_earliest_available_slot(user_query: str) -> bool:
    normalized = (user_query or "").strip().lower()
    return any(hint in normalized for hint in _EARLIEST_SLOT_HINTS)


def _infer_risk_level(user_query: str, existing_risk: str = "normal") -> str:
    normalized = (user_query or "").strip().lower()
    if any(keyword.lower() in normalized for keyword in config.HIGH_RISK_KEYWORDS):
        return "high"
    return existing_risk
