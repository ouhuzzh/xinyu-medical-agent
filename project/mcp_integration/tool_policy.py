"""Safety policy for externally supplied MCP tools.

MCP tools are discovered at runtime from hospitals and other external
services.  Tool names are useful, but they are not a strong enough contract for
deciding whether a generic LLM tool-calling path may execute a tool.  This
module centralizes the policy:

* Prefer explicit metadata/annotations supplied by the MCP server.
* Fall back to conservative name-based rules for older servers.
* Keep appointment mutations reserved for the appointment confirmation state
  machine instead of the generic MCP skill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import config


APPOINTMENT_MUTATION_TOOL_NAMES = {
    "book",
    "book_appointment",
    "create_appointment",
    "make_booking",
    "schedule_appointment",
    "cancel",
    "cancel_appointment",
    "cancel_booking",
    "delete_booking",
    "revoke_appointment",
    "reschedule_appointment",
    "update_appointment",
}

WRITE_EFFECTS = {"write", "mutation", "mutating", "create", "update", "delete", "destructive"}
APPOINTMENT_DOMAINS = {"appointment", "appointments", "registration", "booking", "visit_booking"}


@dataclass(frozen=True)
class MCPToolDecision:
    """Result of evaluating a tool against the generic-MCP execution policy."""

    allowed: bool
    reason: str
    tool_name: str
    canonical_name: str
    requires_confirmation: bool = False
    effect: str = ""
    domain: str = ""


def canonical_tool_name(tool_name: str) -> str:
    """Return the raw tool name without the project-added hospital namespace."""

    base = str(tool_name or "").strip().split(config.MCP_TOOL_NAMESPACE_SEPARATOR)[-1]
    return base.lower().replace("-", "_")


def is_appointment_mutation_name(tool_name: str) -> bool:
    return canonical_tool_name(tool_name) in APPOINTMENT_MUTATION_TOOL_NAMES


def evaluate_generic_tool(tool: Any) -> MCPToolDecision:
    """Decide whether a tool may be executed by the generic MCP skill.

    Appointment booking/cancel/reschedule tools are still usable through
    ``MCPAppointmentBackend`` after the appointment state machine has collected
    explicit confirmation.  This decision only gates the generic tool-calling
    path.
    """

    tool_name = str(getattr(tool, "name", "") or "")
    canonical_name = canonical_tool_name(tool_name)
    metadata = _collect_policy_metadata(tool)

    requires_confirmation = _metadata_bool(metadata, "requires_confirmation", "requiresConfirmation")
    destructive = _metadata_bool(metadata, "destructiveHint", "destructive", "is_destructive")
    read_only = _metadata_bool(metadata, "readOnlyHint", "read_only", "readonly")
    effect = _metadata_str(metadata, "effect", "operation_effect", "operationEffect", "side_effect")
    domain = _metadata_str(metadata, "domain", "category", "tool_domain", "business_domain")

    effect_is_write = effect in WRITE_EFFECTS
    domain_is_appointment = domain in APPOINTMENT_DOMAINS

    if requires_confirmation:
        return MCPToolDecision(
            allowed=False,
            reason="requires_confirmation",
            tool_name=tool_name,
            canonical_name=canonical_name,
            requires_confirmation=True,
            effect=effect,
            domain=domain,
        )

    if destructive:
        return MCPToolDecision(
            allowed=False,
            reason="destructive",
            tool_name=tool_name,
            canonical_name=canonical_name,
            effect=effect,
            domain=domain,
        )

    if read_only is False and (effect_is_write or domain_is_appointment):
        return MCPToolDecision(
            allowed=False,
            reason="non_read_only_mutation",
            tool_name=tool_name,
            canonical_name=canonical_name,
            effect=effect,
            domain=domain,
        )

    if effect_is_write and domain_is_appointment:
        return MCPToolDecision(
            allowed=False,
            reason="appointment_write_metadata",
            tool_name=tool_name,
            canonical_name=canonical_name,
            effect=effect,
            domain=domain,
        )

    if is_appointment_mutation_name(tool_name):
        return MCPToolDecision(
            allowed=False,
            reason="appointment_mutation_name",
            tool_name=tool_name,
            canonical_name=canonical_name,
            effect=effect,
            domain=domain,
        )

    return MCPToolDecision(
        allowed=True,
        reason="allowed",
        tool_name=tool_name,
        canonical_name=canonical_name,
        effect=effect,
        domain=domain,
    )


def filter_generic_mcp_tools(tools: list[Any]) -> tuple[list[Any], list[MCPToolDecision]]:
    allowed: list[Any] = []
    blocked: list[MCPToolDecision] = []
    for tool in tools or []:
        decision = evaluate_generic_tool(tool)
        if decision.allowed:
            allowed.append(tool)
        else:
            blocked.append(decision)
    return allowed, blocked


def _collect_policy_metadata(tool: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    for attr in ("annotations", "metadata", "extra", "response_metadata"):
        value = _as_mapping(getattr(tool, attr, None))
        if value:
            _merge_metadata(metadata, value)

    # Some adapters keep MCP annotations under metadata["annotations"].
    for nested_key in ("annotations", "mcp_annotations", "mcpAnnotations", "x-mcp"):
        nested = _as_mapping(metadata.get(nested_key))
        if nested:
            _merge_metadata(metadata, nested)

    return metadata


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        dumped = to_dict()
        if isinstance(dumped, Mapping):
            return dumped
    return None


def _merge_metadata(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        target[str(key)] = value


def _metadata_bool(metadata: Mapping[str, Any], *keys: str) -> bool | None:
    for key in keys:
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1"}:
                return True
            if normalized in {"false", "no", "0"}:
                return False
    return None


def _metadata_str(metadata: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower().replace("-", "_")
    return ""
