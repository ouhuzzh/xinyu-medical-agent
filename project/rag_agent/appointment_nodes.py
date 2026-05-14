from __future__ import annotations
"""Appointment skill graph nodes and compatibility wrappers.

Booking, cancellation, rescheduling, and the appointment-skill discovery /
planning / action workflow live here.  Private helpers that are only used by
these public nodes are kept local; cross-cutting helpers are imported from
``node_helpers``.
"""

import uuid
import logging
from datetime import date

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .graph_state import State
from .schemas import (
    AppointmentActionCall,
    CancelActionCall,
    AppointmentSkillRequest,
)
from .prompts import get_appointment_request_prompt, get_cancel_appointment_prompt, get_appointment_skill_prompt
from db.appointment_skill_log_store import AppointmentSkillLogStore
from services.appointment_skill import AppointmentSkill

from .node_helpers import (
    _APPOINTMENT_NO_RE,
    _ORDINAL_RE,
    _RESCHEDULE_HINTS,
    _build_appointment_context,
    _clear_pending_action_state,
    _get_appointment_context,
    _get_pending_payload,
    _get_user_query,
    _is_abort_request,
    _is_explicit_confirmation,
    _json_safe_value,
    _looks_like_appointment_discovery_query,
    _next_clarification_attempt,
    _normalize_date,
    _normalize_time_slot,
    _pick_candidate_from_text,
    _sanitize_pending_payload,
    _should_use_last_appointment,
    _structured_output_llm,
    _wants_any_available_doctor,
    _wants_earliest_available_slot,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Appointment-specific module-level state
# ---------------------------------------------------------------------------

_APPOINTMENT_SKILL_LOG_STORE = None


# ---------------------------------------------------------------------------
# Appointment-specific private helpers
# ---------------------------------------------------------------------------

def _get_appointment_skill_log_store():
    global _APPOINTMENT_SKILL_LOG_STORE
    if _APPOINTMENT_SKILL_LOG_STORE is None:
        _APPOINTMENT_SKILL_LOG_STORE = AppointmentSkillLogStore()
    return _APPOINTMENT_SKILL_LOG_STORE


def _pick_doctor_name_from_text(user_query: str, doctor_options: list[dict] | None) -> str:
    normalized = (user_query or "").strip().lower()
    for item in doctor_options or []:
        doctor_name = str(item.get("doctor_name") or "").strip()
        if doctor_name and doctor_name.lower() in normalized:
            return doctor_name
    return ""


def _sort_schedule_options(options: list[dict]) -> list[dict]:
    return sorted(
        list(options or []),
        key=lambda item: (
            str(item.get("schedule_date") or ""),
            str(item.get("time_slot") or ""),
            str(item.get("doctor_name") or ""),
            int(item.get("schedule_id") or 0),
        ),
    )


def _find_matching_doctor_options(options: list[dict], doctor_name: str) -> list[dict]:
    doctor_name_normalized = str(doctor_name or "").strip().lower()
    if not doctor_name_normalized:
        return []
    return [
        item
        for item in (options or [])
        if doctor_name_normalized in str(item.get("doctor_name") or "").strip().lower()
    ]


def _schedule_to_preview_payload(schedule: dict, *, action: str = "book") -> dict:
    return {
        "department": schedule.get("department_name") or schedule.get("department") or "",
        "date": str(schedule.get("schedule_date") or ""),
        "time_slot": schedule.get("time_slot") or "",
        "doctor_name": schedule.get("doctor_name") or "",
        "action": action,
    }


def _format_doctor_slot_selection_message(department: str, doctor_name: str, options: list[dict]) -> str:
    lines = [
        f"{idx}. **{item.get('schedule_date')} {item.get('time_slot')}**（剩余号源 {item.get('quota_available', 0)}）"
        for idx, item in enumerate(_sort_schedule_options(options)[:8], start=1)
    ]
    return (
        f"我找到 **{department}** 的 **{doctor_name}** 可预约时段：\n\n"
        + "\n".join(lines)
        + "\n\n你可以直接回复具体日期和时段，例如“2026-04-18 下午”；如果你希望我直接优先选最早可用时段，也可以回复 **最早可用时段**。"
    )


def _format_doctor_options(department: str, normalized_date: str, time_slot: str, doctor_options: list[dict]) -> str:
    options = "\n".join(
        f"{idx}. **{item['doctor_name']}**（剩余号源 {item.get('quota_available', 0)}）"
        for idx, item in enumerate(doctor_options[:8], start=1)
    )
    return (
        f"目前 **{department}** 在 {normalized_date} {time_slot} 可预约的医生有：\n\n"
        f"{options}\n\n"
        "请直接回复医生姓名；如果你不挑医生，也可以回复 **任一可用医生**，我会为你自动安排。"
    )


def _parse_tool_call(response, expected_name: str) -> dict:
    tool_calls = getattr(response, "tool_calls", None) or []
    for tool_call in tool_calls:
        if tool_call.get("name") == expected_name:
            return tool_call.get("args") or {}
    return {}


def _build_pending_confirmation(action_type: str, payload: dict) -> dict:
    return {
        "pending_action_type": action_type,
        "pending_action_payload": _sanitize_pending_payload(payload),
        "pending_confirmation_id": uuid.uuid4().hex,
        "pending_candidates": [],
    }


def _format_booking_preview(payload: dict) -> str:
    doctor_name = payload.get("doctor_name") or "不限"
    return (
        "我已经整理好预约信息，请回复 **确认预约** 来正式提交：\n\n"
        f"- 科室：**{payload['department']}**\n"
        f"- 日期：**{payload['date']}**\n"
        f"- 时段：**{payload['time_slot']}**\n"
        f"- 医生：**{doctor_name}**\n\n"
        "如果你想改日期、时段、科室或医生，直接告诉我新的要求即可。"
    )


def _format_cancel_preview(payload: dict) -> str:
    return (
        "我已找到要取消的预约，请回复 **确认取消** 来正式提交：\n\n"
        f"- 预约号：**{payload['appointment_no']}**\n"
        f"- 科室：**{payload['department']}**\n"
        f"- 日期：**{payload['date']}**\n"
        f"- 时段：**{payload['time_slot']}**\n\n"
        "如果你想换一条预约取消，也可以直接告诉我新的预约号或条件。"
    )


def _format_reschedule_confirmation_preview(payload: dict) -> str:
    previous_doctor = payload.get("previous_doctor_name") or "未指定"
    next_doctor = payload.get("doctor_name") or "未指定"
    return (
        "我已整理好改约信息，请回复 **确认预约** 来正式提交改约：\n\n"
        f"- 原预约：**{payload['previous_department']}**，**{payload['previous_date']}**，**{payload['previous_time_slot']}**，医生：**{previous_doctor}**\n"
        f"- 新预约：**{payload['department']}**，**{payload['date']}**，**{payload['time_slot']}**，医生：**{next_doctor}**\n\n"
        "如果你想再换一个日期、时段或医生，直接告诉我新的要求即可。"
    )


# ---------------------------------------------------------------------------
# Legacy compatibility wrappers
# ---------------------------------------------------------------------------

def _handle_appointment_legacy(state: State, llm, appointment_service):
    user_query = _get_user_query(state)
    appointment_context = _get_appointment_context(state)
    pending_action_type = state.get("pending_action_type", "")
    pending_payload = _get_pending_payload(state)

    if pending_action_type == "appointment":
        if _is_abort_request(user_query):
            return {
                "intent": "appointment",
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                "topic_focus": appointment_context.get("department", state.get("topic_focus", "")),
                "appointment_context": appointment_context,
                **_clear_pending_action_state(),
                "messages": [AIMessage(content="好的，这次预约我先不提交了。你如果想改时间、科室或重新预约，直接告诉我即可。")],
            }

        if _is_explicit_confirmation(user_query, "appointment"):
            booking = appointment_service.create_appointment(
                thread_id=state["thread_id"],
                department=pending_payload["department"],
                schedule_date=date.fromisoformat(pending_payload["date"]),
                time_slot=pending_payload["time_slot"],
                doctor_name=pending_payload.get("doctor_name") or None,
            )
            merged_context = _build_appointment_context(appointment_context, pending_payload)
            if not booking:
                answer = (
                    f"刚刚确认时，**{pending_payload['department']}** 在 {pending_payload['date']} "
                    f"{pending_payload['time_slot']} 的号源已经不可用了。你可以换个日期、时段，或让我继续帮你改约。"
                )
                return {
                    "intent": "appointment",
                    "pending_clarification": "",
                    "clarification_target": "",
                    "clarification_attempts": 0,
                    "topic_focus": merged_context.get("department", state.get("topic_focus", "")),
                    "appointment_context": merged_context,
                    **_clear_pending_action_state(),
                    "messages": [AIMessage(content=answer)],
                }

            answer = (
                f"已为你预约成功：\n\n"
                f"- 科室：**{booking['department']}**\n"
                f"- 日期：**{booking['date']}**\n"
                f"- 时段：**{booking['time_slot']}**\n"
                f"- 医生：**{booking['doctor_name']}**\n"
                f"- 预约号：**{booking['appointment_no']}**"
            )
            return {
                "intent": "appointment",
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                "topic_focus": merged_context.get("department", state.get("topic_focus", "")),
                "appointment_context": merged_context,
                "last_appointment_no": booking["appointment_no"],
                **_clear_pending_action_state(),
                "messages": [AIMessage(content=answer)],
            }

    llm_with_tools = llm.with_config(temperature=0.1).bind_tools([AppointmentActionCall])
    response = llm_with_tools.invoke(
        [
            SystemMessage(content=get_appointment_request_prompt()),
            HumanMessage(
                content=(
                    f"Conversation summary:\n{state.get('conversation_summary', '')}\n\n"
                    f"Recommended department:\n{state.get('recommended_department', '')}\n\n"
                    f"Existing appointment context:\n{appointment_context}\n\n"
                    f"User query:\n{user_query}"
                )
            ),
        ]
    )
    call_args = _parse_tool_call(response, "AppointmentActionCall")

    department = (call_args.get("department") or "").strip() or state.get("recommended_department", "") or appointment_context.get("department", "")
    normalized_date = _normalize_date(call_args.get("date") or appointment_context.get("date", "") or user_query)
    time_slot = _normalize_time_slot(call_args.get("time_slot") or appointment_context.get("time_slot", "") or user_query)
    available_doctors = appointment_context.get("available_doctors") or []
    doctor_name = (
        (call_args.get("doctor_name") or "").strip()
        or _pick_doctor_name_from_text(user_query, available_doctors)
        or appointment_context.get("doctor_name", "")
    )
    wants_any_doctor = _wants_any_available_doctor(user_query)

    merged_context = _build_appointment_context(
        appointment_context,
        {
            "department": department,
            "date": normalized_date,
            "time_slot": time_slot,
            "doctor_name": doctor_name,
            "available_doctors": available_doctors,
        },
    )

    missing_fields = []
    if not department:
        missing_fields.append("科室")
    if not normalized_date:
        missing_fields.append("日期")
    if not time_slot:
        missing_fields.append("时间段")

    if call_args.get("action") == "clarify" or missing_fields:
        clarification = (call_args.get("clarification") or "").strip() or f"请补充要预约的{'、'.join(missing_fields)}。"
        return {
            "intent": "appointment",
            "pending_clarification": clarification,
            "clarification_target": "handle_appointment",
            "clarification_attempts": _next_clarification_attempt(state),
            "topic_focus": department or state.get("topic_focus", ""),
            "appointment_context": merged_context,
            **_clear_pending_action_state(),
            "messages": [AIMessage(content=clarification)],
        }

    schedule_date_value = date.fromisoformat(normalized_date)
    doctor_options = appointment_service.list_available_doctors(
        department=department,
        schedule_date=schedule_date_value,
        time_slot=time_slot,
    )
    if not doctor_options:
        answer = f"暂时没有找到 **{department}** 在 {normalized_date} {time_slot} 的可预约号源。你可以换一个日期、时间段，或继续让我帮你改约。"
        return {
            "intent": "appointment",
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
            "topic_focus": department or state.get("topic_focus", ""),
            "appointment_context": _build_appointment_context(merged_context, {"available_doctors": []}),
            **_clear_pending_action_state(),
            "messages": [AIMessage(content=answer)],
        }

    if not doctor_name and len(doctor_options) > 1 and not wants_any_doctor:
        clarification = _format_doctor_options(department, normalized_date, time_slot, doctor_options)
        return {
            "intent": "appointment",
            "pending_clarification": clarification,
            "clarification_target": "handle_appointment",
            "clarification_attempts": _next_clarification_attempt(state),
            "topic_focus": department or state.get("topic_focus", ""),
            "appointment_context": _build_appointment_context(merged_context, {"available_doctors": doctor_options, "doctor_name": ""}),
            **_clear_pending_action_state(),
            "messages": [AIMessage(content=clarification)],
        }

    schedule = appointment_service.find_available_schedule(
        department=department,
        schedule_date=schedule_date_value,
        time_slot=time_slot,
        doctor_name=doctor_name or None,
    )
    if not schedule:
        if doctor_name and doctor_options:
            doctor_hint = _format_doctor_options(department, normalized_date, time_slot, doctor_options)
            answer = f"没有找到 **{doctor_name}** 在该时段的可预约号源。\n\n{doctor_hint}"
            return {
                "intent": "appointment",
                "pending_clarification": answer,
                "clarification_target": "handle_appointment",
                "clarification_attempts": _next_clarification_attempt(state),
                "topic_focus": department or state.get("topic_focus", ""),
                "appointment_context": _build_appointment_context(merged_context, {"available_doctors": doctor_options, "doctor_name": ""}),
                **_clear_pending_action_state(),
                "messages": [AIMessage(content=answer)],
            }
        answer = f"暂时没有找到 **{department}** 在 {normalized_date} {time_slot} 的可预约号源。你可以换一个日期、时间段，或继续让我帮你改约。"
        return {
            "intent": "appointment",
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
            "topic_focus": department or state.get("topic_focus", ""),
            "appointment_context": _build_appointment_context(merged_context, {"available_doctors": doctor_options}),
            **_clear_pending_action_state(),
            "messages": [AIMessage(content=answer)],
        }

    preview_payload = {
        "department": schedule["department_name"],
        "date": schedule["schedule_date"].isoformat(),
        "time_slot": schedule["time_slot"],
        "doctor_name": schedule["doctor_name"],
        "action": "book",
    }
    return {
        "intent": "appointment",
        "pending_clarification": "",
        "clarification_target": "",
        "clarification_attempts": 0,
        "topic_focus": preview_payload["department"],
        "appointment_context": _build_appointment_context(merged_context, {"available_doctors": doctor_options, "doctor_name": schedule["doctor_name"]}),
        **_build_pending_confirmation("appointment", preview_payload),
        "messages": [AIMessage(content=_format_booking_preview(preview_payload))],
    }


def _handle_cancel_appointment_legacy(state: State, llm, appointment_service):
    user_query = _get_user_query(state)
    appointment_context = _get_appointment_context(state)
    last_appointment_no = state.get("last_appointment_no", "")
    pending_action_type = state.get("pending_action_type", "")
    pending_payload = _get_pending_payload(state)
    pending_candidates = state.get("pending_candidates", []) or []

    if pending_action_type == "cancel_appointment":
        if _is_abort_request(user_query):
            return {
                "intent": "cancel_appointment",
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_clear_pending_action_state(),
                "messages": [AIMessage(content="好的，这次取消我先不提交了。如果你想改成别的预约，直接告诉我新的预约号或条件即可。")],
            }

        if _is_explicit_confirmation(user_query, "cancel_appointment"):
            cancelled = appointment_service.cancel_appointment(state["thread_id"], int(pending_payload["appointment_id"]))
            if not cancelled:
                return {
                    "intent": "cancel_appointment",
                    "pending_clarification": "",
                    "clarification_target": "",
                    "clarification_attempts": 0,
                    **_clear_pending_action_state(),
                    "messages": [AIMessage(content="这条预约当前无法取消，可能已经被处理过了。你可以再给我新的预约号或条件。")],
                }

            answer = (
                f"已为你取消预约：\n\n"
                f"- 预约号：**{cancelled['appointment_no']}**\n"
                f"- 日期：**{cancelled['date']}**\n"
                f"- 时段：**{cancelled['time_slot']}**"
            )
            return {
                "intent": "cancel_appointment",
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                "last_appointment_no": "",
                **_clear_pending_action_state(),
                "messages": [AIMessage(content=answer)],
            }

    if pending_candidates:
        if _is_abort_request(user_query):
            return {
                "intent": "cancel_appointment",
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_clear_pending_action_state(),
                "messages": [AIMessage(content="好的，我先不取消了。如果你还想取消其他预约，可以继续告诉我预约号或条件。")],
            }

        selected = _pick_candidate_from_text(user_query, pending_candidates)
        if selected:
            preview_payload = {
                "appointment_id": str(selected["appointment_id"]),
                "appointment_no": selected["appointment_no"],
                "department": selected["department"],
                "date": selected["appointment_date"].isoformat(),
                "time_slot": selected["time_slot"],
                "doctor_name": selected.get("doctor_name") or "",
                "action": "cancel",
            }
            return {
                "intent": "cancel_appointment",
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_build_pending_confirmation("cancel_appointment", preview_payload),
                "messages": [AIMessage(content=_format_cancel_preview(preview_payload))],
            }

    llm_with_tools = llm.with_config(temperature=0.1).bind_tools([CancelActionCall])
    response = llm_with_tools.invoke(
        [
            SystemMessage(content=get_cancel_appointment_prompt()),
            HumanMessage(
                content=(
                    f"Conversation summary:\n{state.get('conversation_summary', '')}\n\n"
                    f"Last appointment number:\n{last_appointment_no}\n\n"
                    f"Existing appointment context:\n{appointment_context}\n\n"
                    f"User query:\n{user_query}"
                )
            ),
        ]
    )
    call_args = _parse_tool_call(response, "CancelActionCall")

    appointment_no = (call_args.get("appointment_no") or "").strip()
    if not appointment_no and _should_use_last_appointment(user_query):
        appointment_no = last_appointment_no
    department = (call_args.get("department") or "").strip() or appointment_context.get("department", "")
    normalized_date = _normalize_date(call_args.get("date") or appointment_context.get("date", "") or user_query)

    if call_args.get("action") == "clarify" or (not appointment_no and not (department and normalized_date)):
        clarification = (call_args.get("clarification") or "").strip() or "请告诉我要取消的预约号，或者提供科室和日期。"
        return {
            "intent": "cancel_appointment",
            "pending_clarification": clarification,
            "clarification_target": "handle_cancel_appointment",
            "clarification_attempts": _next_clarification_attempt(state),
            **_clear_pending_action_state(),
            "messages": [AIMessage(content=clarification)],
        }

    candidates = appointment_service.find_candidate_appointments(
        thread_id=state["thread_id"],
        appointment_no=appointment_no or None,
        department=department or None,
        schedule_date=date.fromisoformat(normalized_date) if normalized_date else None,
    )
    if not candidates:
        return {
            "intent": "cancel_appointment",
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
            **_clear_pending_action_state(),
            "messages": [AIMessage(content="我没有找到符合条件的可取消预约。你可以再提供预约号，或者补充科室和日期。")],
        }
    if len(candidates) > 1:
        options = "\n".join(
            f"{idx}. 预约号：{item['appointment_no']}，{item['department']}，{item['appointment_date'].isoformat()} {item['time_slot']}"
            for idx, item in enumerate(candidates[:5], start=1)
        )
        clarification = (
            "我找到了多条可取消预约，请回复具体预约号，或直接说“第 1 个 / 第 2 个”：\n"
            f"{options}"
        )
        return {
            "intent": "cancel_appointment",
            "pending_clarification": clarification,
            "clarification_target": "handle_cancel_appointment",
            "clarification_attempts": _next_clarification_attempt(state),
            "pending_action_type": "",
            "pending_action_payload": {},
            "pending_confirmation_id": "",
            "pending_candidates": candidates[:5],
            "messages": [AIMessage(content=clarification)],
        }

    selected = candidates[0]
    preview_payload = {
        "appointment_id": str(selected["appointment_id"]),
        "appointment_no": selected["appointment_no"],
        "department": selected["department"],
        "date": selected["appointment_date"].isoformat(),
        "time_slot": selected["time_slot"],
        "doctor_name": selected.get("doctor_name") or "",
        "action": "cancel",
    }
    return {
        "intent": "cancel_appointment",
        "pending_clarification": "",
        "clarification_target": "",
        "clarification_attempts": 0,
        **_build_pending_confirmation("cancel_appointment", preview_payload),
        "messages": [AIMessage(content=_format_cancel_preview(preview_payload))],
    }


# ---------------------------------------------------------------------------
# Appointment skill internals
# ---------------------------------------------------------------------------

def _log_appointment_skill_event(
    state: State,
    *,
    skill_mode: str,
    request_type: str,
    selected_candidate_count: int = 0,
    required_confirmation: bool = False,
    final_action: str = "",
    extra_metadata: dict | None = None,
):
    try:
        _get_appointment_skill_log_store().save_log(
            {
                "thread_id": state.get("thread_id") or "",
                "skill_mode": skill_mode,
                "request_type": request_type,
                "selected_candidate_count": selected_candidate_count,
                "required_confirmation": required_confirmation,
                "final_action": final_action,
                "extra_metadata": extra_metadata or {},
            }
        )
    except Exception:
        logger.warning("Failed to persist appointment skill log", exc_info=True)
        pass


def _invoke_appointment_skill_request(llm, state: State, user_query: str) -> dict:
    appointment_context = _get_appointment_context(state)
    llm_with_tools = llm.with_config(temperature=0.1).bind_tools([AppointmentSkillRequest])
    response = llm_with_tools.invoke(
        [
            SystemMessage(content=get_appointment_skill_prompt()),
            HumanMessage(
                content=(
                    f"Conversation summary:\n{state.get('conversation_summary', '')}\n\n"
                    f"Current intent:\n{state.get('intent') or state.get('primary_intent', '')}\n\n"
                    f"Recommended department:\n{state.get('recommended_department', '')}\n\n"
                    f"Existing appointment context:\n{appointment_context}\n\n"
                    f"Pending action type:\n{state.get('pending_action_type', '')}\n\n"
                    f"Last appointment number:\n{state.get('last_appointment_no', '')}\n\n"
                    f"User query:\n{user_query}"
                )
            ),
        ]
    )
    skill_call = _parse_tool_call(response, "AppointmentSkillRequest")
    if skill_call:
        return skill_call

    legacy_booking = _parse_tool_call(response, "AppointmentActionCall")
    if legacy_booking:
        return {
            "action": "clarify" if legacy_booking.get("action") == "clarify" else "prepare_appointment",
            "department": legacy_booking.get("department", ""),
            "date": legacy_booking.get("date", ""),
            "time_slot": legacy_booking.get("time_slot", ""),
            "doctor_name": legacy_booking.get("doctor_name", ""),
            "clarification": legacy_booking.get("clarification", ""),
        }

    legacy_cancel = _parse_tool_call(response, "CancelActionCall")
    if legacy_cancel:
        return {
            "action": "clarify" if legacy_cancel.get("action") == "clarify" else "prepare_cancellation",
            "appointment_no": legacy_cancel.get("appointment_no", ""),
            "department": legacy_cancel.get("department", ""),
            "date": legacy_cancel.get("date", ""),
            "clarification": legacy_cancel.get("clarification", ""),
        }

    return {}


def _base_skill_state_update(
    state: State,
    *,
    intent: str,
    skill_mode: str,
    topic_focus: str = "",
    appointment_context: dict | None = None,
    candidates: list[dict] | None = None,
    skill_last_prompt: str = "",
) -> dict:
    return {
        "intent": intent,
        "appointment_skill_mode": skill_mode,
        "topic_focus": topic_focus or state.get("topic_focus", ""),
        "appointment_context": _json_safe_value(appointment_context if appointment_context is not None else dict(state.get("appointment_context") or {})),
        "appointment_candidates": _json_safe_value(list(candidates or [])),
        "skill_last_prompt": skill_last_prompt or "",
    }


# ---------------------------------------------------------------------------
# Public appointment nodes
# ---------------------------------------------------------------------------

def handle_appointment_skill(state: State, llm, appointment_service):
    user_query = _get_user_query(state)
    appointment_context = _get_appointment_context(state)
    pending_action_type = state.get("pending_action_type", "")
    pending_payload = _get_pending_payload(state)
    pending_candidates = state.get("pending_candidates", []) or []
    active_intent = state.get("intent") or state.get("primary_intent") or "appointment"
    skill = AppointmentSkill(appointment_service)

    if pending_action_type == "appointment":
        if _is_abort_request(user_query):
            _log_appointment_skill_event(state, skill_mode="action", request_type="abort_booking", final_action="abort")
            return {
                **_base_skill_state_update(state, intent="appointment", skill_mode="idle", topic_focus=appointment_context.get("department", state.get("topic_focus", "")), appointment_context=appointment_context),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_clear_pending_action_state(),
                "messages": [AIMessage(content="好的，这次预约我先不提交了。你如果想改时间、科室或重新预约，直接告诉我即可。")],
            }
        if _is_explicit_confirmation(user_query, "appointment"):
            booking = skill.confirm_appointment(state["thread_id"], pending_payload)
            merged_context = _build_appointment_context(appointment_context, pending_payload)
            _log_appointment_skill_event(state, skill_mode="action", request_type="confirm_appointment", required_confirmation=True, final_action="confirm_appointment")
            if not booking:
                answer = (
                    f"刚刚确认时，**{pending_payload['department']}** 在 {pending_payload['date']} "
                    f"{pending_payload['time_slot']} 的号源已经不可用了。你可以换个日期、时段，或让我继续帮你改约。"
                )
                return {
                    **_base_skill_state_update(state, intent="appointment", skill_mode="planning", topic_focus=merged_context.get("department", state.get("topic_focus", "")), appointment_context=merged_context),
                    "pending_clarification": "",
                    "clarification_target": "",
                    "clarification_attempts": 0,
                    **_clear_pending_action_state(),
                    "messages": [AIMessage(content=answer)],
                }
            answer = (
                f"已为你预约成功：\n\n"
                f"- 科室：**{booking['department']}**\n"
                f"- 日期：**{booking['date']}**\n"
                f"- 时段：**{booking['time_slot']}**\n"
                f"- 医生：**{booking['doctor_name']}**\n"
                f"- 预约号：**{booking['appointment_no']}**"
            )
            return {
                **_base_skill_state_update(state, intent="appointment", skill_mode="completed", topic_focus=merged_context.get("department", state.get("topic_focus", "")), appointment_context=merged_context),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                "last_appointment_no": booking["appointment_no"],
                **_clear_pending_action_state(),
                "messages": [AIMessage(content=answer)],
            }
        if not _looks_like_appointment_discovery_query(user_query):
            return {
                **_base_skill_state_update(
                    state,
                    intent="appointment",
                    skill_mode="prepare_appointment",
                    topic_focus=appointment_context.get("department", state.get("topic_focus", "")),
                    appointment_context=appointment_context,
                ),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_build_pending_confirmation("appointment", pending_payload),
                "messages": [AIMessage(content="如果你确认这条预约，请直接回复 **确认预约**；如果想改时间、医生或科室，也可以直接告诉我。")],
            }

    if pending_action_type == "cancel_appointment":
        if _is_abort_request(user_query):
            _log_appointment_skill_event(state, skill_mode="action", request_type="abort_cancellation", final_action="abort")
            return {
                **_base_skill_state_update(state, intent="cancel_appointment", skill_mode="idle"),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_clear_pending_action_state(),
                "messages": [AIMessage(content="好的，这次取消我先不提交了。如果你想取消其他预约，直接告诉我预约号或条件即可。")],
            }
        if _is_explicit_confirmation(user_query, "cancel_appointment"):
            cancelled = skill.confirm_cancellation(state["thread_id"], pending_payload)
            _log_appointment_skill_event(state, skill_mode="action", request_type="confirm_cancellation", required_confirmation=True, final_action="confirm_cancellation")
            if not cancelled:
                return {
                    **_base_skill_state_update(state, intent="cancel_appointment", skill_mode="planning"),
                    "pending_clarification": "",
                    "clarification_target": "",
                    "clarification_attempts": 0,
                    **_clear_pending_action_state(),
                    "messages": [AIMessage(content="这条预约当前无法取消，可能已经被处理过了。你可以再给我新的预约号或条件。")],
                }
            answer = (
                f"已为你取消预约：\n\n"
                f"- 预约号：**{cancelled['appointment_no']}**\n"
                f"- 日期：**{cancelled['date']}**\n"
                f"- 时段：**{cancelled['time_slot']}**"
            )
            return {
                **_base_skill_state_update(state, intent="cancel_appointment", skill_mode="completed"),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                "last_appointment_no": "",
                **_clear_pending_action_state(),
                "messages": [AIMessage(content=answer)],
            }
        if not _should_use_last_appointment(user_query):
            return {
                **_base_skill_state_update(state, intent="cancel_appointment", skill_mode="prepare_cancellation"),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_build_pending_confirmation("cancel_appointment", pending_payload),
                "messages": [AIMessage(content="如果你确认取消这条预约，请直接回复 **确认取消**；如果想取消别的预约，也可以直接告诉我预约号或说“第 1 个 / 第 2 个”。")],
            }

    if pending_action_type == "reschedule_appointment":
        if _is_abort_request(user_query):
            _log_appointment_skill_event(state, skill_mode="action", request_type="abort_reschedule", final_action="abort")
            return {
                **_base_skill_state_update(state, intent="appointment", skill_mode="idle", topic_focus=appointment_context.get("department", state.get("topic_focus", "")), appointment_context=appointment_context),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_clear_pending_action_state(),
                "messages": [AIMessage(content="好的，这次改约我先不提交了。你如果想继续改时间、时段或医生，直接告诉我即可。")],
            }
        if _is_explicit_confirmation(user_query, "reschedule_appointment"):
            rescheduled = skill.confirm_reschedule(state["thread_id"], pending_payload)
            _log_appointment_skill_event(state, skill_mode="action", request_type="confirm_reschedule", required_confirmation=True, final_action="confirm_reschedule")
            if not rescheduled:
                return {
                    **_base_skill_state_update(state, intent="appointment", skill_mode="planning", topic_focus=appointment_context.get("department", state.get("topic_focus", "")), appointment_context=appointment_context),
                    "pending_clarification": "",
                    "clarification_target": "",
                    "clarification_attempts": 0,
                    **_clear_pending_action_state(),
                    "messages": [AIMessage(content="刚刚确认改约时，新时段已经不可用了。你可以换一个日期、时段，或者让我重新帮你找可改约的医生。")],
                }
            return {
                **_base_skill_state_update(state, intent="appointment", skill_mode="completed", topic_focus=rescheduled.get("department", state.get("topic_focus", "")), appointment_context=_build_appointment_context(appointment_context, {"department": rescheduled.get("department", ""), "date": rescheduled.get("date", ""), "time_slot": rescheduled.get("time_slot", ""), "doctor_name": rescheduled.get("doctor_name", "")})),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                "last_appointment_no": rescheduled["appointment_no"],
                **_clear_pending_action_state(),
                "messages": [
                    AIMessage(
                        content=(
                            "已为你改约成功：\n\n"
                            f"- 预约号：**{rescheduled['appointment_no']}**\n"
                            f"- 原预约：**{rescheduled['previous_department']}**，**{rescheduled['previous_date']}**，**{rescheduled['previous_time_slot']}**\n"
                            f"- 新预约：**{rescheduled['department']}**，**{rescheduled['date']}**，**{rescheduled['time_slot']}**\n"
                            f"- 医生：**{rescheduled['doctor_name']}**"
                        )
                    )
                ],
            }
        return {
            **_base_skill_state_update(state, intent="appointment", skill_mode="prepare_reschedule", topic_focus=appointment_context.get("department", state.get("topic_focus", "")), appointment_context=appointment_context),
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
            **_build_pending_confirmation("reschedule_appointment", pending_payload),
            "messages": [AIMessage(content="如果你确认这次改约，请直接回复 **确认预约**；如果想换成别的日期、时段或医生，也可以直接告诉我。")],
        }

    if pending_candidates and active_intent == "cancel_appointment":
        selected = _pick_candidate_from_text(user_query, pending_candidates)
        if selected:
            preview_payload = {
                "appointment_id": str(selected["appointment_id"]),
                "appointment_no": selected["appointment_no"],
                "department": selected["department"],
                "date": selected["appointment_date"].isoformat(),
                "time_slot": selected["time_slot"],
                "doctor_name": selected.get("doctor_name") or "",
                "action": "cancel",
            }
            _log_appointment_skill_event(state, skill_mode="planning", request_type="select_cancellation_candidate", selected_candidate_count=len(pending_candidates), required_confirmation=True, final_action="prepare_cancellation")
            return {
                **_base_skill_state_update(state, intent="cancel_appointment", skill_mode="planning", candidates=[], skill_last_prompt=_format_cancel_preview(preview_payload)),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_build_pending_confirmation("cancel_appointment", preview_payload),
                "messages": [AIMessage(content=_format_cancel_preview(preview_payload))],
            }
        return {
            **_base_skill_state_update(state, intent="cancel_appointment", skill_mode="list_my_appointments", candidates=pending_candidates),
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
            "pending_candidates": pending_candidates,
            "messages": [AIMessage(content="我还没确定你要取消哪一条。你可以直接回复预约号，或者说“第 1 个 / 第 2 个”。")],
        }

    available_doctors = list(appointment_context.get("available_doctors") or [])
    selected_doctor_name = _pick_doctor_name_from_text(user_query, available_doctors) or appointment_context.get("doctor_name", "")
    if active_intent == "appointment" and available_doctors:
        if _wants_any_available_doctor(user_query):
            chosen_schedule = _sort_schedule_options(available_doctors)[0]
            payload = _schedule_to_preview_payload(chosen_schedule)
            preview_message = _format_booking_preview(payload)
            _log_appointment_skill_event(
                state,
                skill_mode="planning",
                request_type="prepare_appointment",
                selected_candidate_count=len(available_doctors),
                required_confirmation=True,
                final_action="prepare_any_available_doctor",
            )
            return {
                **_base_skill_state_update(
                    state,
                    intent="appointment",
                    skill_mode="prepare_appointment",
                    topic_focus=payload["department"],
                    appointment_context=_build_appointment_context(
                        appointment_context,
                        {
                            "department": payload["department"],
                            "date": payload["date"],
                            "time_slot": payload["time_slot"],
                            "doctor_name": payload["doctor_name"],
                            "available_doctors": available_doctors,
                        },
                    ),
                    candidates=available_doctors,
                    skill_last_prompt=preview_message,
                ),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_build_pending_confirmation("appointment", payload),
                "messages": [AIMessage(content=preview_message)],
            }
        matching_doctor_options = _find_matching_doctor_options(available_doctors, selected_doctor_name)
        if matching_doctor_options:
            if len(matching_doctor_options) == 1 or _wants_earliest_available_slot(user_query):
                chosen_schedule = _sort_schedule_options(matching_doctor_options)[0]
                payload = _schedule_to_preview_payload(chosen_schedule)
                preview_message = _format_booking_preview(payload)
                _log_appointment_skill_event(
                    state,
                    skill_mode="planning",
                    request_type="prepare_appointment",
                    selected_candidate_count=len(matching_doctor_options),
                    required_confirmation=True,
                    final_action="prepare_selected_doctor",
                )
                return {
                    **_base_skill_state_update(
                        state,
                        intent="appointment",
                        skill_mode="prepare_appointment",
                        topic_focus=payload["department"],
                        appointment_context=_build_appointment_context(
                            appointment_context,
                            {
                                "department": payload["department"],
                                "date": payload["date"],
                                "time_slot": payload["time_slot"],
                                "doctor_name": payload["doctor_name"],
                                "available_doctors": matching_doctor_options,
                            },
                        ),
                        candidates=matching_doctor_options,
                        skill_last_prompt=preview_message,
                    ),
                    "pending_clarification": "",
                    "clarification_target": "",
                    "clarification_attempts": 0,
                    **_build_pending_confirmation("appointment", payload),
                    "messages": [AIMessage(content=preview_message)],
                }
            selection_message = _format_doctor_slot_selection_message(
                appointment_context.get("department", "") or matching_doctor_options[0].get("department_name", ""),
                selected_doctor_name,
                matching_doctor_options,
            )
            _log_appointment_skill_event(
                state,
                skill_mode="discovery",
                request_type="discover_availability",
                selected_candidate_count=len(matching_doctor_options),
                final_action="discover_selected_doctor_slots",
            )
            return {
                **_base_skill_state_update(
                    state,
                    intent="appointment",
                    skill_mode="discover_availability",
                    topic_focus=appointment_context.get("department", "") or selected_doctor_name,
                    appointment_context=_build_appointment_context(
                        appointment_context,
                        {"available_doctors": matching_doctor_options, "doctor_name": selected_doctor_name},
                    ),
                    candidates=matching_doctor_options,
                    skill_last_prompt=selection_message,
                ),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_clear_pending_action_state(),
                "messages": [AIMessage(content=selection_message)],
            }

    call_args = _invoke_appointment_skill_request(llm, state, user_query)
    department = (call_args.get("department") or "").strip() or state.get("recommended_department", "") or appointment_context.get("department", "")
    normalized_date = _normalize_date(call_args.get("date") or appointment_context.get("date", "") or user_query)
    time_slot = _normalize_time_slot(call_args.get("time_slot") or appointment_context.get("time_slot", "") or user_query)
    appointment_no = (call_args.get("appointment_no") or "").strip()
    doctor_name = (
        (call_args.get("doctor_name") or "").strip()
        or _pick_doctor_name_from_text(user_query, appointment_context.get("available_doctors") or [])
        or appointment_context.get("doctor_name", "")
    )
    skill_action = (call_args.get("action") or "").strip() or ("prepare_cancellation" if active_intent == "cancel_appointment" else "prepare_appointment")
    wants_any_doctor = _wants_any_available_doctor(user_query)
    merged_context = _build_appointment_context(
        appointment_context,
        {"department": department, "date": normalized_date, "time_slot": time_slot, "doctor_name": doctor_name},
    )
    available_doctors = list(appointment_context.get("available_doctors") or [])
    matching_doctor_options = _find_matching_doctor_options(available_doctors, doctor_name)

    if active_intent == "appointment" and available_doctors and not normalized_date and not time_slot:
        if wants_any_doctor:
            chosen_schedule = _sort_schedule_options(available_doctors)[0]
            payload = _schedule_to_preview_payload(chosen_schedule)
            preview_message = _format_booking_preview(payload)
            _log_appointment_skill_event(
                state,
                skill_mode="planning",
                request_type="prepare_appointment",
                selected_candidate_count=len(available_doctors),
                required_confirmation=True,
                final_action="prepare_any_available_doctor",
            )
            return {
                **_base_skill_state_update(
                    state,
                    intent="appointment",
                    skill_mode="prepare_appointment",
                    topic_focus=payload["department"],
                    appointment_context=_build_appointment_context(
                        merged_context,
                        {
                            "department": payload["department"],
                            "date": payload["date"],
                            "time_slot": payload["time_slot"],
                            "doctor_name": payload["doctor_name"],
                            "available_doctors": available_doctors,
                        },
                    ),
                    candidates=available_doctors,
                    skill_last_prompt=preview_message,
                ),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_build_pending_confirmation("appointment", payload),
                "messages": [AIMessage(content=preview_message)],
            }
        if matching_doctor_options:
            if len(matching_doctor_options) == 1 or _wants_earliest_available_slot(user_query):
                chosen_schedule = _sort_schedule_options(matching_doctor_options)[0]
                payload = _schedule_to_preview_payload(chosen_schedule)
                preview_message = _format_booking_preview(payload)
                _log_appointment_skill_event(
                    state,
                    skill_mode="planning",
                    request_type="prepare_appointment",
                    selected_candidate_count=len(matching_doctor_options),
                    required_confirmation=True,
                    final_action="prepare_selected_doctor",
                )
                return {
                    **_base_skill_state_update(
                        state,
                        intent="appointment",
                        skill_mode="prepare_appointment",
                        topic_focus=payload["department"],
                        appointment_context=_build_appointment_context(
                            merged_context,
                            {
                                "department": payload["department"],
                                "date": payload["date"],
                                "time_slot": payload["time_slot"],
                                "doctor_name": payload["doctor_name"],
                                "available_doctors": matching_doctor_options,
                            },
                        ),
                        candidates=matching_doctor_options,
                        skill_last_prompt=preview_message,
                    ),
                    "pending_clarification": "",
                    "clarification_target": "",
                    "clarification_attempts": 0,
                    **_build_pending_confirmation("appointment", payload),
                    "messages": [AIMessage(content=preview_message)],
                }
            selection_message = _format_doctor_slot_selection_message(
                department or matching_doctor_options[0].get("department_name", ""),
                doctor_name,
                matching_doctor_options,
            )
            _log_appointment_skill_event(
                state,
                skill_mode="discovery",
                request_type="discover_availability",
                selected_candidate_count=len(matching_doctor_options),
                final_action="discover_selected_doctor_slots",
            )
            return {
                **_base_skill_state_update(
                    state,
                    intent="appointment",
                    skill_mode="discover_availability",
                    topic_focus=department or doctor_name,
                    appointment_context=_build_appointment_context(
                        merged_context,
                        {"available_doctors": matching_doctor_options, "doctor_name": doctor_name},
                    ),
                    candidates=matching_doctor_options,
                    skill_last_prompt=selection_message,
                ),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_clear_pending_action_state(),
                "messages": [AIMessage(content=selection_message)],
            }

    if skill_action == "clarify":
        clarification = (call_args.get("clarification") or "").strip() or "你可以再补充一下要处理的预约信息。"
        _log_appointment_skill_event(state, skill_mode="clarify", request_type=active_intent, final_action="clarify")
        return {
            **_base_skill_state_update(state, intent=active_intent, skill_mode="clarify", topic_focus=department or state.get("topic_focus", ""), appointment_context=merged_context, skill_last_prompt=clarification),
            "pending_clarification": clarification,
            "clarification_target": "handle_appointment_skill",
            "clarification_attempts": _next_clarification_attempt(state),
            **_clear_pending_action_state(),
            "messages": [AIMessage(content=clarification)],
        }

    if skill_action == "discover_department":
        message = skill.discover_departments(department or user_query)
        _log_appointment_skill_event(state, skill_mode="discovery", request_type="discover_department", final_action="discover_department")
        return {
            **_base_skill_state_update(state, intent="appointment", skill_mode="discover_department", appointment_context=merged_context, skill_last_prompt=message),
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
            **_clear_pending_action_state(),
            "messages": [AIMessage(content=message)],
        }

    if skill_action == "list_my_appointments" or (active_intent == "cancel_appointment" and not appointment_no and not department and not normalized_date):
        message, appointments = skill.list_my_appointments(state["thread_id"])
        _log_appointment_skill_event(state, skill_mode="discovery", request_type="list_my_appointments", selected_candidate_count=len(appointments), final_action="list_my_appointments")
        return {
            **_base_skill_state_update(state, intent=active_intent, skill_mode="list_my_appointments", candidates=appointments, skill_last_prompt=message),
            "pending_clarification": message if active_intent == "cancel_appointment" and appointments else "",
            "clarification_target": "handle_appointment_skill" if active_intent == "cancel_appointment" and appointments else "",
            "clarification_attempts": int(state.get("clarification_attempts") or 0) + (1 if active_intent == "cancel_appointment" and appointments else 0),
            "pending_candidates": appointments[:8] if active_intent == "cancel_appointment" else [],
            "messages": [AIMessage(content=message)],
        }

    if skill_action == "discover_doctor":
        if not department:
            clarification = "你想先看哪个科室的医生？如果还不确定，我也可以先根据症状帮你推荐科室。"
            return {
                **_base_skill_state_update(state, intent="appointment", skill_mode="clarify", appointment_context=merged_context, skill_last_prompt=clarification),
                "pending_clarification": clarification,
                "clarification_target": "handle_appointment_skill",
                "clarification_attempts": _next_clarification_attempt(state),
                "messages": [AIMessage(content=clarification)],
            }
        schedule_date_value = date.fromisoformat(normalized_date) if normalized_date and time_slot else None
        message, doctor_options = skill.discover_doctors(department, schedule_date=schedule_date_value, time_slot=time_slot)
        _log_appointment_skill_event(state, skill_mode="discovery", request_type="discover_doctor", selected_candidate_count=len(doctor_options), final_action="discover_doctor")
        return {
            **_base_skill_state_update(state, intent="appointment", skill_mode="discover_doctor", topic_focus=department, appointment_context=_build_appointment_context(merged_context, {"available_doctors": doctor_options}), candidates=doctor_options, skill_last_prompt=message),
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
            **_clear_pending_action_state(),
            "messages": [AIMessage(content=message)],
        }

    if skill_action == "discover_availability":
        if doctor_name:
            schedule_date_value = date.fromisoformat(normalized_date) if normalized_date else None
            message, availability = skill.discover_doctor_availability(
                doctor_name,
                department=department,
                schedule_date=schedule_date_value,
                time_slot=time_slot,
            )
            _log_appointment_skill_event(state, skill_mode="discovery", request_type="discover_availability", selected_candidate_count=len(availability), final_action="discover_doctor_availability")
            return {
                **_base_skill_state_update(state, intent="appointment", skill_mode="discover_availability", topic_focus=department or doctor_name, appointment_context=_build_appointment_context(merged_context, {"available_doctors": availability}), candidates=availability, skill_last_prompt=message),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_clear_pending_action_state(),
                "messages": [AIMessage(content=message)],
            }
        if department:
            message, upcoming = skill.discover_department_availability(department)
            _log_appointment_skill_event(state, skill_mode="discovery", request_type="discover_availability", selected_candidate_count=len(upcoming), final_action="discover_department_availability")
            return {
                **_base_skill_state_update(state, intent="appointment", skill_mode="discover_availability", topic_focus=department, appointment_context=_build_appointment_context(merged_context, {"available_doctors": upcoming}), candidates=upcoming, skill_last_prompt=message),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_clear_pending_action_state(),
                "messages": [AIMessage(content=message)],
            }

    if skill_action == "prepare_reschedule" or any(token in (user_query or "").lower() for token in _RESCHEDULE_HINTS):
        current_items = appointment_service.find_candidate_appointments(
            thread_id=state["thread_id"],
            appointment_no=appointment_no or (state.get("last_appointment_no", "") if _should_use_last_appointment(user_query) else "") or None,
            department=department or None,
            schedule_date=date.fromisoformat(normalized_date) if normalized_date else None,
        )
        if not current_items:
            message = "我暂时没锁定要改约的那条预约。你可以先告诉我预约号，或者说“改最近那个预约”。"
            return {
                **_base_skill_state_update(state, intent="appointment", skill_mode="clarify", appointment_context=merged_context, skill_last_prompt=message),
                "pending_clarification": message,
                "clarification_target": "handle_appointment_skill",
                "clarification_attempts": _next_clarification_attempt(state),
                "messages": [AIMessage(content=message)],
            }
        if not normalized_date or not time_slot:
            message = skill.prepare_reschedule(
                state["thread_id"],
                current_items[0],
                target_date=date.fromisoformat(normalized_date) if normalized_date else None,
                time_slot=time_slot,
            )
            _log_appointment_skill_event(state, skill_mode="planning", request_type="prepare_reschedule", selected_candidate_count=1, final_action="prepare_reschedule_options")
            return {
                **_base_skill_state_update(state, intent="appointment", skill_mode="prepare_reschedule", topic_focus=current_items[0]["department"], appointment_context=merged_context, candidates=current_items, skill_last_prompt=message),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_clear_pending_action_state(),
                "messages": [AIMessage(content=message)],
            }
        preview, doctor_options, alternatives = skill.prepare_reschedule_preview(
            candidate=current_items[0],
            target_date=date.fromisoformat(normalized_date),
            time_slot=time_slot,
            doctor_name=doctor_name,
            allow_any_doctor=wants_any_doctor,
        )
        if preview:
            payload = preview.__dict__
            _log_appointment_skill_event(state, skill_mode="planning", request_type="prepare_reschedule", selected_candidate_count=len(doctor_options), required_confirmation=True, final_action="prepare_reschedule")
            return {
                **_base_skill_state_update(state, intent="appointment", skill_mode="prepare_reschedule", topic_focus=payload["department"], appointment_context=_build_appointment_context(merged_context, {"department": payload["department"], "date": payload["date"], "time_slot": payload["time_slot"], "doctor_name": payload.get("doctor_name", "")}), candidates=doctor_options, skill_last_prompt=_format_reschedule_confirmation_preview(payload)),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_build_pending_confirmation("reschedule_appointment", payload),
                "messages": [AIMessage(content=_format_reschedule_confirmation_preview(payload))],
            }
        if doctor_options:
            message, doctor_options = skill.discover_doctors(current_items[0]["department"], schedule_date=date.fromisoformat(normalized_date), time_slot=time_slot)
            _log_appointment_skill_event(state, skill_mode="discovery", request_type="discover_doctor", selected_candidate_count=len(doctor_options), final_action="discover_reschedule_doctor")
            return {
                **_base_skill_state_update(state, intent="appointment", skill_mode="discover_doctor", topic_focus=current_items[0]["department"], appointment_context=_build_appointment_context(merged_context, {"available_doctors": doctor_options, "doctor_name": ""}), candidates=doctor_options, skill_last_prompt=message),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_clear_pending_action_state(),
                "messages": [AIMessage(content=message)],
            }
        if alternatives:
            message = "当前目标时段没有合适的可改约号源，我找到这些替代选择：\n\n" + "\n".join(
                f"- **{item['doctor_name']}**：{item['schedule_date']} {item['time_slot']}（剩余号源 {item.get('quota_available', 0)}）"
                for item in alternatives[:6]
            )
            _log_appointment_skill_event(state, skill_mode="discovery", request_type="prepare_reschedule", selected_candidate_count=len(alternatives), final_action="discover_reschedule_alternatives")
            return {
                **_base_skill_state_update(state, intent="appointment", skill_mode="discover_availability", topic_focus=current_items[0]["department"], appointment_context=merged_context, candidates=alternatives, skill_last_prompt=message),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_clear_pending_action_state(),
                "messages": [AIMessage(content=message)],
            }
        message = "暂时没有找到可改约的新号源。你可以换一个日期、时段，或者让我继续找其他医生。"
        return {
            **_base_skill_state_update(state, intent="appointment", skill_mode="discover_availability", topic_focus=current_items[0]["department"], appointment_context=merged_context, skill_last_prompt=message),
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
            **_clear_pending_action_state(),
            "messages": [AIMessage(content=message)],
        }

    if active_intent == "cancel_appointment" or skill_action in {"prepare_cancellation", "confirm_cancellation"}:
        if not appointment_no and _should_use_last_appointment(user_query):
            appointment_no = state.get("last_appointment_no", "")
        preview, candidates = skill.prepare_cancellation(
            state["thread_id"],
            appointment_no=appointment_no,
            department=department,
            schedule_date=date.fromisoformat(normalized_date) if normalized_date else None,
        )
        if preview:
            payload = preview.__dict__
            _log_appointment_skill_event(state, skill_mode="planning", request_type="prepare_cancellation", required_confirmation=True, final_action="prepare_cancellation")
            return {
                **_base_skill_state_update(state, intent="cancel_appointment", skill_mode="prepare_cancellation", topic_focus=payload["department"], appointment_context=merged_context, skill_last_prompt=_format_cancel_preview(payload)),
                "pending_clarification": "",
                "clarification_target": "",
                "clarification_attempts": 0,
                **_build_pending_confirmation("cancel_appointment", payload),
                "messages": [AIMessage(content=_format_cancel_preview(payload))],
            }
        message = "我没有找到符合条件的可取消预约。你可以再提供预约号，或者补充科室和日期。"
        if candidates:
            message = "我找到了多条可取消预约，请回复具体预约号，或直接说“第 1 个 / 第 2 个”：\n" + "\n".join(
                f"{idx}. 预约号：{item['appointment_no']}，{item['department']}，{item['appointment_date'].isoformat()} {item['time_slot']}"
                for idx, item in enumerate(candidates[:8], start=1)
            )
        _log_appointment_skill_event(state, skill_mode="discovery", request_type="prepare_cancellation", selected_candidate_count=len(candidates), final_action="list_cancellation_candidates")
        return {
            **_base_skill_state_update(state, intent="cancel_appointment", skill_mode="list_my_appointments", candidates=candidates, skill_last_prompt=message),
            "pending_clarification": message if candidates else "",
            "clarification_target": "handle_appointment_skill" if candidates else "",
            "clarification_attempts": int(state.get("clarification_attempts") or 0) + (1 if candidates else 0),
            **_clear_pending_action_state(),
            "pending_candidates": candidates[:8],
            "messages": [AIMessage(content=message)],
        }

    if not department:
        clarification = "你想挂哪个科室？如果还不确定，我也可以先根据症状帮你推荐挂什么科。"
        return {
            **_base_skill_state_update(state, intent="appointment", skill_mode="clarify", appointment_context=merged_context, skill_last_prompt=clarification),
            "pending_clarification": clarification,
            "clarification_target": "handle_appointment_skill",
            "clarification_attempts": _next_clarification_attempt(state),
            **_clear_pending_action_state(),
            "messages": [AIMessage(content=clarification)],
        }

    if not normalized_date or not time_slot:
        message, upcoming = skill.discover_department_availability(department)
        _log_appointment_skill_event(state, skill_mode="discovery", request_type="discover_department_availability", selected_candidate_count=len(upcoming), final_action="discover_department_availability")
        return {
            **_base_skill_state_update(state, intent="appointment", skill_mode="discover_availability", topic_focus=department, appointment_context=_build_appointment_context(merged_context, {"available_doctors": upcoming}), candidates=upcoming, skill_last_prompt=message),
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
            **_clear_pending_action_state(),
            "messages": [AIMessage(content=message)],
        }

    preview, doctor_options, alternatives = skill.prepare_appointment(
        department=department,
        schedule_date=date.fromisoformat(normalized_date),
        time_slot=time_slot,
        doctor_name=doctor_name,
        allow_any_doctor=wants_any_doctor,
    )
    if preview:
        payload = preview.__dict__
        _log_appointment_skill_event(state, skill_mode="planning", request_type="prepare_appointment", selected_candidate_count=len(doctor_options), required_confirmation=True, final_action="prepare_appointment")
        return {
            **_base_skill_state_update(state, intent="appointment", skill_mode="prepare_appointment", topic_focus=payload["department"], appointment_context=_build_appointment_context(merged_context, {"available_doctors": doctor_options, "doctor_name": payload.get("doctor_name", "")}), candidates=doctor_options, skill_last_prompt=_format_booking_preview(payload)),
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
            **_build_pending_confirmation("appointment", payload),
            "messages": [AIMessage(content=_format_booking_preview(payload))],
        }
    if doctor_options:
        message, doctor_options = skill.discover_doctors(department, schedule_date=date.fromisoformat(normalized_date), time_slot=time_slot)
        _log_appointment_skill_event(state, skill_mode="discovery", request_type="discover_doctor", selected_candidate_count=len(doctor_options), final_action="discover_doctor")
        return {
            **_base_skill_state_update(state, intent="appointment", skill_mode="discover_doctor", topic_focus=department, appointment_context=_build_appointment_context(merged_context, {"available_doctors": doctor_options, "doctor_name": ""}), candidates=doctor_options, skill_last_prompt=message),
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
            **_clear_pending_action_state(),
            "messages": [AIMessage(content=message)],
        }
    if alternatives:
        message = "当前指定医生或时段没有可用号源，我找到这些替代选择：\n\n" + "\n".join(
            f"- **{item['doctor_name']}**：{item['schedule_date']} {item['time_slot']}（剩余号源 {item.get('quota_available', 0)}）"
            for item in alternatives[:6]
        )
        _log_appointment_skill_event(state, skill_mode="discovery", request_type="discover_alternatives", selected_candidate_count=len(alternatives), final_action="discover_alternatives")
        return {
            **_base_skill_state_update(state, intent="appointment", skill_mode="discover_availability", topic_focus=department, appointment_context=merged_context, candidates=alternatives, skill_last_prompt=message),
            "pending_clarification": "",
            "clarification_target": "",
            "clarification_attempts": 0,
            **_clear_pending_action_state(),
            "messages": [AIMessage(content=message)],
        }

    message = f"暂时没有找到 **{department}** 在 {normalized_date} {time_slot} 的可预约号源。你可以换一个日期、时间段，或继续让我帮你找其他医生。"
    _log_appointment_skill_event(state, skill_mode="discovery", request_type="prepare_appointment", final_action="no_availability")
    return {
        **_base_skill_state_update(state, intent="appointment", skill_mode="discover_availability", topic_focus=department, appointment_context=merged_context, skill_last_prompt=message),
        "pending_clarification": "",
        "clarification_target": "",
        "clarification_attempts": 0,
        **_clear_pending_action_state(),
        "messages": [AIMessage(content=message)],
    }


def handle_appointment(state: State, llm, appointment_service):
    merged_state = dict(state)
    merged_state.setdefault("intent", "appointment")
    merged_state.setdefault("primary_intent", "appointment")
    return handle_appointment_skill(merged_state, llm, appointment_service)


def handle_cancel_appointment(state: State, llm, appointment_service):
    merged_state = dict(state)
    merged_state.setdefault("intent", "cancel_appointment")
    merged_state.setdefault("primary_intent", "cancel_appointment")
    return handle_appointment_skill(merged_state, llm, appointment_service)


__all__ = [
    "handle_appointment",
    "handle_appointment_skill",
    "handle_cancel_appointment",
    "_handle_appointment_legacy",
    "_handle_cancel_appointment_legacy",
    "_log_appointment_skill_event",
    "_invoke_appointment_skill_request",
    "_base_skill_state_update",
    "_build_pending_confirmation",
    "_format_booking_preview",
    "_format_cancel_preview",
    "_format_reschedule_confirmation_preview",
    "_format_doctor_options",
    "_format_doctor_slot_selection_message",
    "_parse_tool_call",
    "_pick_doctor_name_from_text",
    "_sort_schedule_options",
    "_find_matching_doctor_options",
    "_schedule_to_preview_payload",
    "_get_appointment_skill_log_store",
]
