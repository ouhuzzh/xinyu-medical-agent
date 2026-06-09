"""State manipulation helpers — pending action state, context building, etc.

Extracted from node_helpers for focused reusability.
"""

import logging
from datetime import date

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage

from .graph_state import State

logger = logging.getLogger(__name__)


def _clear_pending_action_state() -> dict:
    return {
        "pending_action_type": "",
        "pending_action_payload": {},
        "pending_confirmation_id": "",
        "pending_candidates": [],
    }


def _reset_pending_action_if_needed(state: State) -> dict:
    if not state.get("pending_action_type") and not state.get("pending_candidates"):
        return {}
    return _clear_pending_action_state()


def _build_appointment_context(existing: dict | None, updates: dict) -> dict:
    context = dict(existing or {})
    for key, value in updates.items():
        if value or (key in updates and isinstance(value, list)):
            context[key] = _json_safe_value(value)
    return context


def _json_safe_value(value):
    if isinstance(value, date):
        return value.isoformat()
    return value


def _sanitize_pending_payload(payload: dict | None) -> dict:
    """Ensure pending payload values are JSON-safe (no date objects, etc.)."""
    if not payload:
        return {}
    safe = {}
    for key, value in payload.items():
        safe[key] = _json_safe_value(value)
    return safe


def _extract_topic_focus(user_query: str, existing_topic: str = "", appointment_context: dict | None = None, recommended_department: str = "") -> str:
    """Derive a topic focus string from the user query and existing context."""
    from .classifiers import (
        _looks_like_explicit_appointment_intent,
        _looks_like_explicit_cancel_intent,
        _looks_like_department_question,
    )
    query = (user_query or "").strip()
    if not query:
        return existing_topic

    if _looks_like_explicit_appointment_intent(query):
        return "appointment"
    if _looks_like_explicit_cancel_intent(query):
        return "cancel_appointment"
    if _looks_like_department_question(query):
        return "triage"
    if recommended_department:
        return f"department:{recommended_department}"
    if appointment_context and appointment_context.get("department"):
        return f"appointment:{appointment_context.get('department', '')}"
    return existing_topic


def _build_recent_context(messages, keep_turns: int | None = None, *, exclude_latest_user: bool = True) -> str:
    """Build a condensed recent context string from message history."""
    if not messages:
        return ""
    from .node_helpers import _TOPIC_STOP_WORDS
    relevant = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            content = str(msg.content or "").strip()
            if content and len(content) < 500:
                role = "用户" if isinstance(msg, HumanMessage) else "助手"
                relevant.append(f"{role}: {content}")
    if exclude_latest_user and relevant and "用户:" in relevant[-1]:
        relevant = relevant[:-1]
    if keep_turns:
        relevant = relevant[-keep_turns * 2:]
    return "\n".join(relevant[-6:])


def _is_explicit_confirmation(user_query: str, pending_action_type: str) -> bool:
    from .node_helpers import _APPOINTMENT_CONFIRM_WORDS, _CANCEL_CONFIRM_WORDS
    normalized = (user_query or "").strip().lower()
    if pending_action_type == "appointment":
        return any(w in normalized for w in _APPOINTMENT_CONFIRM_WORDS)
    if pending_action_type == "cancel_appointment":
        return any(w in normalized for w in _CANCEL_CONFIRM_WORDS)
    return any(w in normalized for w in (_APPOINTMENT_CONFIRM_WORDS + _CANCEL_CONFIRM_WORDS))


def _is_abort_request(user_query: str) -> bool:
    from .node_helpers import _ABORT_WORDS
    normalized = (user_query or "").strip().lower()
    return any(w in normalized for w in _ABORT_WORDS)


def _pick_candidate_from_text(user_query: str, pending_candidates: list[dict]) -> dict | None:
    import re
    from .node_helpers import _ORDINAL_RE, _APPOINTMENT_NO_RE
    normalized = (user_query or "").strip().lower()
    # Ordinal: "第2个"
    m = re.search(_ORDINAL_RE, normalized)
    if m and pending_candidates:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(pending_candidates):
            return pending_candidates[idx]
    # Appointment number: "APT123"
    m = re.search(_APPOINTMENT_NO_RE, normalized, re.IGNORECASE)
    if m:
        apt_no = m.group(0).upper()
        for c in pending_candidates:
            if str(c.get("appointment_no", "")).upper() == apt_no:
                return c
    return None


def _should_use_last_appointment(user_query: str) -> bool:
    normalized = (user_query or "").strip().lower()
    return any(w in normalized for w in ("最近", "上次", "上一个", "刚才"))


def _build_history_reset_messages(messages, keep_recent: int = 5):
    """Build a list of RemoveMessage for all but the most recent messages."""
    keep_from = max(0, len(messages) - keep_recent)
    remove_ids = [msg.id for msg in messages[:keep_from] if hasattr(msg, "id") and msg.id]
    return [RemoveMessage(id=mid) for mid in remove_ids]


def _get_user_query(state: State) -> str:
    last = state["messages"][-1] if state.get("messages") else None
    if last:
        return str(last.content).strip()
    return ""


def _get_appointment_context(state: State) -> dict:
    return state.get("appointment_context") or {}


def _get_pending_payload(state: State) -> dict:
    return state.get("pending_action_payload") or {}


def _next_clarification_attempt(state: State) -> int:
    return state.get("clarification_attempts", 0) + 1


def _looks_like_schedule_update(user_query: str) -> bool:
    import re
    from .node_helpers import _APPOINTMENT_UPDATE_HINTS
    normalized = (user_query or "").strip().lower()
    if any(token in normalized for token in _APPOINTMENT_UPDATE_HINTS):
        return True
    return bool(re.search(r"\d{1,2}[点时:：]", normalized) or re.search(r"\d{1,2}\s*月|\d{4}-\d{1,2}-\d{1,2}|明天|后天|周[一二三四五六日天]", normalized))


def _split_compound_request(user_query: str) -> list[str]:
    """Split a compound request like '我感冒了，帮我挂个号' into segments."""
    import re
    segments = re.split(r"[,，;；\n]|(?:并且|同时|还要|另外|而且|还有)", user_query)
    return [s.strip() for s in segments if s.strip()]
