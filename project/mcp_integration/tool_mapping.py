"""Resolve standard appointment actions to concrete MCP tool names.

The appointment workflow should call stable platform actions such as
``search_schedules`` and ``book_appointment``.  Individual hospitals may expose
different raw MCP tool names, so this module resolves in three steps:

1. Per-hospital explicit mapping from ``MCP_APPOINTMENT_TOOL_MAPPING``.
2. Exact match on the standard action name.
3. Exact match on legacy aliases.

The final alias step is intentionally exact on the canonical raw tool name.  It
does not use substring matching, so ``cancel`` will not accidentally match an
unrelated ``cancelled_invoice_lookup`` tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import config


DEFAULT_APPOINTMENT_TOOL_ALIASES: dict[str, list[str]] = {
    "search_doctors": [
        "search_doctors",
        "list_doctors",
        "query_doctor",
        "get_doctors",
        "find_doctor",
        "doctor_search",
    ],
    "search_schedules": [
        "search_schedules",
        "query_schedule",
        "get_schedules",
        "get_availability",
        "check_availability",
        "find_slot",
    ],
    "book_appointment": [
        "book_appointment",
        "create_appointment",
        "make_booking",
        "book",
        "schedule_appointment",
    ],
    "list_appointments": [
        "list_appointments",
        "my_appointments",
        "get_bookings",
        "get_appointments",
        "query_appointments",
    ],
    "cancel_appointment": [
        "cancel_appointment",
        "cancel_booking",
        "delete_booking",
        "cancel",
        "revoke_appointment",
    ],
}


@dataclass(frozen=True)
class ToolResolution:
    action: str
    tool: Any
    tool_name: str
    hospital_code: str
    raw_tool_name: str
    source: str


class MCPAppointmentToolMapper:
    """Find concrete MCP tools for appointment actions."""

    def __init__(
        self,
        *,
        mapping: Mapping[str, Any] | None = None,
        aliases: Mapping[str, list[str]] | None = None,
        namespace_sep: str | None = None,
    ):
        self._mapping = _normalize_mapping(
            mapping if mapping is not None else getattr(config, "MCP_APPOINTMENT_TOOL_MAPPING", {})
        )
        self._aliases = {
            _canonical_name(action): [_canonical_name(alias) for alias in values]
            for action, values in (aliases or DEFAULT_APPOINTMENT_TOOL_ALIASES).items()
        }
        self._namespace_sep = namespace_sep or config.MCP_TOOL_NAMESPACE_SEPARATOR

    def find_tool(
        self,
        tools: list[Any],
        action: str,
        *,
        preferred_hospital_code: str = "",
    ) -> ToolResolution | None:
        action_key = _canonical_name(action)
        indexed = self._index_tools(tools)
        hospital_code = _canonical_code(preferred_hospital_code)

        mapped = self._find_mapped_tool(indexed, action_key, hospital_code)
        if mapped:
            return mapped

        standard = self._find_by_raw_names(indexed, action_key, [action_key], "standard_name", hospital_code)
        if standard:
            return standard

        aliases = self._aliases.get(action_key, [])
        return self._find_by_raw_names(indexed, action_key, aliases, "alias", hospital_code)

    def supports_any_action(self, tools: list[Any], *, preferred_hospital_code: str = "") -> bool:
        return any(
            self.find_tool(tools, action, preferred_hospital_code=preferred_hospital_code) is not None
            for action in self._aliases
        )

    def _find_mapped_tool(
        self,
        indexed: "_IndexedTools",
        action: str,
        preferred_hospital_code: str = "",
    ) -> ToolResolution | None:
        for hospital_code, hospital_mapping in self._mapping.items():
            if preferred_hospital_code and hospital_code != preferred_hospital_code:
                continue
            mapped_name = hospital_mapping.get(action)
            if not mapped_name:
                continue

            full_name = self._full_tool_name(hospital_code, mapped_name)
            tool = indexed.by_full_name.get(full_name)
            if tool is not None:
                return _resolution(action, tool, "mapping", self._namespace_sep)

            raw_name = _canonical_name(_strip_namespace(str(mapped_name), self._namespace_sep)[1])
            tool = indexed.by_hospital_and_raw.get((hospital_code, raw_name))
            if tool is not None:
                return _resolution(action, tool, "mapping", self._namespace_sep)

        return None

    def _find_by_raw_names(
        self,
        indexed: "_IndexedTools",
        action: str,
        raw_names: list[str],
        source: str,
        preferred_hospital_code: str = "",
    ) -> ToolResolution | None:
        raw_name_set = {_canonical_name(name) for name in raw_names}
        for tool in indexed.ordered_tools:
            hospital_code, raw_tool_name = _strip_namespace(
                str(getattr(tool, "name", "") or ""), self._namespace_sep
            )
            if preferred_hospital_code and hospital_code != preferred_hospital_code:
                continue
            if _canonical_name(raw_tool_name) in raw_name_set:
                return ToolResolution(
                    action=action,
                    tool=tool,
                    tool_name=str(getattr(tool, "name", "") or ""),
                    hospital_code=hospital_code,
                    raw_tool_name=_canonical_name(raw_tool_name),
                    source=source,
                )
        return None

    def _full_tool_name(self, hospital_code: str, raw_or_full_name: str) -> str:
        name = str(raw_or_full_name or "").strip()
        if self._namespace_sep in name:
            return name
        return f"{hospital_code}{self._namespace_sep}{name}"

    def _index_tools(self, tools: list[Any]) -> "_IndexedTools":
        by_full_name: dict[str, Any] = {}
        by_hospital_and_raw: dict[tuple[str, str], Any] = {}
        ordered_tools = list(tools or [])
        for tool in ordered_tools:
            full_name = str(getattr(tool, "name", "") or "")
            hospital_code, raw_tool_name = _strip_namespace(full_name, self._namespace_sep)
            raw_key = _canonical_name(raw_tool_name)
            by_full_name.setdefault(full_name, tool)
            by_hospital_and_raw.setdefault((hospital_code, raw_key), tool)
        return _IndexedTools(ordered_tools, by_full_name, by_hospital_and_raw)


@dataclass(frozen=True)
class _IndexedTools:
    ordered_tools: list[Any]
    by_full_name: dict[str, Any]
    by_hospital_and_raw: dict[tuple[str, str], Any]


def _resolution(action: str, tool: Any, source: str, namespace_sep: str) -> ToolResolution:
    tool_name = str(getattr(tool, "name", "") or "")
    hospital_code, raw_tool_name = _strip_namespace(tool_name, namespace_sep)
    return ToolResolution(
        action=action,
        tool=tool,
        tool_name=tool_name,
        hospital_code=hospital_code,
        raw_tool_name=_canonical_name(raw_tool_name),
        source=source,
    )


def _normalize_mapping(raw_mapping: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    if not isinstance(raw_mapping, Mapping):
        return normalized
    for hospital_code, value in raw_mapping.items():
        if not isinstance(value, Mapping):
            continue
        hospital_key = str(hospital_code or "").strip().lower()
        if not hospital_key:
            continue
        action_map: dict[str, str] = {}
        for action, tool_name in value.items():
            action_key = _canonical_name(str(action or ""))
            mapped_tool = str(tool_name or "").strip()
            if action_key and mapped_tool:
                action_map[action_key] = mapped_tool
        if action_map:
            normalized[hospital_key] = action_map
    return normalized


def _strip_namespace(tool_name: str, namespace_sep: str) -> tuple[str, str]:
    name = str(tool_name or "").strip()
    if namespace_sep and namespace_sep in name:
        hospital_code, raw_name = name.split(namespace_sep, 1)
        return hospital_code.strip().lower(), raw_name.strip()
    return "", name


def _canonical_name(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _canonical_code(value: str) -> str:
    return str(value or "").strip().lower()
