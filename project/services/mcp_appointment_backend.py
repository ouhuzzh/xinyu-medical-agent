"""MCP Appointment Backend — plugs real hospital systems into the appointment state machine.

Replaces local mock data in handle_appointment_skill with MCP tool calls.
The state machine (discover → preview → confirm → abort) is unchanged;
only the DATA SOURCE is swapped from local PostgreSQL to MCP hospital tools.

Usage:
    backend = MCPAppointmentBackend.try_create(state)
    if backend:
        message, doctors = backend.discover_doctors("心内科", date, slot)
    else:
        ... show "请绑定医院服务" message ...
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from mcp_integration.tool_mapping import (
    DEFAULT_APPOINTMENT_TOOL_ALIASES,
    MCPAppointmentToolMapper,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP tool name patterns — try each synonym until one is found in the user's
# tool list.  Individual hospital MCP servers may name these differently.
# ---------------------------------------------------------------------------
_TOOL_ALIASES: Dict[str, List[str]] = DEFAULT_APPOINTMENT_TOOL_ALIASES


class MCPAppointmentBackend:
    """Thin adapter over MCP user tools for appointment data operations.

    Each method returns (result, error_message).  Exactly one of the pair
    is non-None on success.  This lets callers branch cleanly:

        doctors, err = backend.discover_doctors(department)
        if err:
            return error_response(err)
        ... use doctors ...
    """

    def __init__(
        self,
        pool,
        user_id: str,
        *,
        tool_mapper: MCPAppointmentToolMapper | None = None,
        preferred_hospital_code: str = "",
    ):
        self._pool = pool
        self._uid = user_id
        self._available = pool is not None and bool(user_id)
        self._tool_mapper = tool_mapper or MCPAppointmentToolMapper()
        self._preferred_hospital_code = str(preferred_hospital_code or "").strip().lower()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def try_create(
        cls,
        state: dict,
        *,
        pool=None,
        user_id: str = "",
        tool_mapper: MCPAppointmentToolMapper | None = None,
        preferred_hospital_code: str = "",
    ) -> Optional["MCPAppointmentBackend"]:
        """Try to build an MCP backend from graph state.  Returns None if
        MCP is not available for this user (not enabled, not logged in,
        no credentials bound, or no matching tools).

        Prefer passing pool and user_id explicitly.  Falls back to
        state-injected values for backward compatibility.
        """
        try:
            import config
            if not config.MCP_ENABLED:
                return None
        except Exception:
            return None

        # Explicit injection first, then state-based fallback
        if pool is None:
            pool = state.get("_mcp_pool")
        if not user_id:
            user_id = (state.get("user_id") or "").strip()
        if pool is None or not user_id:
            return None

        # Verify at least one appointment-relevant tool exists
        try:
            tools = pool.get_tools_for_user(user_id)
            mapper = tool_mapper or MCPAppointmentToolMapper()
            if not mapper.supports_any_action(
                tools,
                preferred_hospital_code=preferred_hospital_code,
            ):
                return None
        except Exception:
            return None

        return cls(
            pool,
            user_id,
            tool_mapper=mapper,
            preferred_hospital_code=preferred_hospital_code,
        )

    # ------------------------------------------------------------------
    # Public data operations
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self._available

    def discover_doctors(
        self,
        department: str,
        schedule_date=None,
        time_slot: str = "",
    ) -> Tuple[Optional[List[dict]], Optional[str]]:
        """Search for doctors in a department.  Returns (doctors, error)."""
        if not self._available:
            return (None, "请先在设置中绑定医院服务，才能查询医生。")
        params = {"department": department}
        if schedule_date:
            params["date"] = str(schedule_date)
        if time_slot:
            params["time_slot"] = time_slot
        return self._call("search_doctors", params)

    def discover_schedules(
        self,
        department: str = "",
        schedule_date=None,
        time_slot: str = "",
        doctor_name: str = "",
    ) -> Tuple[Optional[List[dict]], Optional[str]]:
        """Search for available time slots.  Returns (schedules, error)."""
        if not self._available:
            return (None, "请先在设置中绑定医院服务，才能查询号源。")
        params: Dict[str, Any] = {}
        if department:
            params["department"] = department
        if schedule_date:
            params["date"] = str(schedule_date)
        if time_slot:
            params["time_slot"] = time_slot
        if doctor_name:
            params["doctor_name"] = doctor_name
        return self._call("search_schedules", params)

    def book_appointment(self, payload: dict) -> Tuple[Optional[dict], Optional[str]]:
        """Execute a real booking.  payload keys: department, date, time_slot,
        doctor_name.  Returns (booking_result, error)."""
        if not self._available:
            return (None, "无法执行预约，请检查医院服务连接。")
        params = {
            "department": payload.get("department", ""),
            "date": payload.get("date", ""),
            "time_slot": payload.get("time_slot", ""),
            "doctor_name": payload.get("doctor_name", ""),
        }
        return self._call("book_appointment", params)

    def list_appointments(self) -> Tuple[Optional[List[dict]], Optional[str]]:
        """List the user's existing appointments."""
        if not self._available:
            return (None, "请先在设置中绑定医院服务，才能查看预约。")
        return self._call("list_appointments", {})

    def cancel_appointment(self, payload: dict) -> Tuple[Optional[dict], Optional[str]]:
        """Cancel an existing appointment.  payload keys: appointment_id,
        appointment_no.  Returns (cancellation_result, error)."""
        if not self._available:
            return (None, "无法执行取消，请检查医院服务连接。")
        params = {
            "appointment_id": str(payload.get("appointment_id", "")),
            "appointment_no": str(payload.get("appointment_no", "")),
        }
        return self._call("cancel_appointment", params)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(
        self, action: str, params: Dict[str, Any]
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Call the first matching MCP tool for the given action.

        Returns (result, None) on success, (None, error_message) on failure.
        Retries up to 3 times with exponential backoff on transient failures.
        """
        try:
            tools = self._pool.get_tools_for_user(self._uid)
        except Exception as e:
            logger.warning("Failed to get MCP tools: %s", e)
            return (None, "医院服务暂时不可用，请稍后重试。")

        resolution = self._tool_mapper.find_tool(
            tools,
            action,
            preferred_hospital_code=self._preferred_hospital_code,
        )
        tool = resolution.tool if resolution else None

        if tool is None:
            return (None, f"该医院暂不支持{action}操作，请确认服务配置。")
        logger.debug(
            "Resolved MCP appointment action %s to tool %s via %s",
            action,
            resolution.tool_name,
            resolution.source,
        )

        max_retries = 3
        retry_delays = [1.0, 2.0, 4.0]
        last_err = None

        for attempt in range(max_retries):
            try:
                raw = tool.invoke(params)
                # Normalise to dict or list-of-dicts
                if isinstance(raw, str):
                    import json
                    try:
                        raw = json.loads(raw)
                    except json.JSONDecodeError:
                        return ({"message": raw}, None)
                if isinstance(raw, dict):
                    return (raw, None)
                if isinstance(raw, list):
                    return (raw, None)
                return ({"raw": raw}, None)
            except (ConnectionError, TimeoutError, OSError) as e:
                # Transient errors — retry
                last_err = e
                if attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    logger.warning(
                        "MCP tool %s attempt %d/%d failed: %s — retrying in %.1fs",
                        action, attempt + 1, max_retries, str(e)[:200], delay,
                    )
                    time.sleep(delay)
                else:
                    err_msg = str(e)[:200] or type(e).__name__
                    logger.warning("MCP tool %s failed after %d attempts: %s", action, max_retries, err_msg)
                    return (None, f"医院服务调用失败，请稍后重试（{err_msg}）。")
            except Exception as e:
                # Non-transient errors — fail immediately
                err_msg = str(e)[:200] or type(e).__name__
                logger.warning("MCP tool %s failed: %s", action, err_msg)
                return (None, f"医院服务调用失败，请稍后重试（{err_msg}）。")

        # Should not reach here, but safety net
        return (None, f"医院服务调用失败，请稍后重试。")

    # ------------------------------------------------------------------
    # Formatting helpers (mirror the native skill's output format)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_schedule_list(raw: Any) -> List[dict]:
        """Normalise MCP response into a list of schedule dicts with standard keys."""
        items: List[dict] = []
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            # Maybe wrapped in a key like 'schedules' or 'data'
            for key in ("schedules", "data", "results", "items"):
                if key in raw and isinstance(raw[key], list):
                    items = raw[key]
                    break
            if not items:
                items = [raw]
        return items
