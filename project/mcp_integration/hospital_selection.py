"""Deterministic hospital selection for MCP appointment workflows."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping

import config


@dataclass(frozen=True)
class HospitalCandidate:
    code: str
    name: str


@dataclass(frozen=True)
class HospitalSelection:
    selected_code: str = ""
    selected_name: str = ""
    needs_clarification: bool = False
    needs_confirmation: bool = False
    candidates: tuple[HospitalCandidate, ...] = ()
    reason: str = ""


class MCPHospitalSelectionPolicy:
    """Choose a hospital without letting the LLM make the final decision."""

    def __init__(
        self,
        *,
        aliases: Mapping[str, Any] | None = None,
        hospital_lookup: Callable[[str], Mapping[str, Any] | None] | None = None,
    ):
        self._aliases = _normalize_aliases(
            aliases if aliases is not None else getattr(config, "MCP_HOSPITAL_ALIASES", {})
        )
        self._hospital_lookup = hospital_lookup

    def select(
        self,
        *,
        user_query: str,
        appointment_context: Mapping[str, Any] | None,
        connected_hospital_codes: list[str],
    ) -> HospitalSelection:
        connected = [_normalize_code(code) for code in connected_hospital_codes if _normalize_code(code)]
        connected = list(dict.fromkeys(connected))
        candidates = tuple(self._candidate_for(code) for code in connected)

        if not connected:
            return HospitalSelection(reason="no_connected_hospitals")

        context_code = _normalize_code((appointment_context or {}).get("hospital_code", ""))
        if context_code:
            if context_code in connected:
                candidate = self._candidate_for(context_code)
                return HospitalSelection(candidate.code, candidate.name, False, False, candidates, "context")
            return HospitalSelection(
                selected_code=context_code,
                selected_name=str((appointment_context or {}).get("hospital_name") or context_code),
                needs_clarification=False,
                needs_confirmation=False,
                candidates=candidates,
                reason="context_unavailable",
            )

        strong_matches, confirmation_matches = self._match_query(user_query, connected)
        if len(strong_matches) == 1:
            candidate = self._candidate_for(strong_matches[0])
            return HospitalSelection(candidate.code, candidate.name, False, False, candidates, "query")
        if len(strong_matches) > 1:
            return HospitalSelection(
                needs_clarification=True,
                candidates=tuple(self._candidate_for(c) for c in strong_matches),
                reason="ambiguous_query",
            )

        if len(confirmation_matches) == 1:
            candidate = self._candidate_for(confirmation_matches[0])
            return HospitalSelection(
                selected_code=candidate.code,
                selected_name=candidate.name,
                needs_confirmation=True,
                candidates=(candidate,),
                reason="query_needs_confirmation",
            )
        if len(confirmation_matches) > 1:
            return HospitalSelection(
                needs_clarification=True,
                candidates=tuple(self._candidate_for(c) for c in confirmation_matches),
                reason="ambiguous_query",
            )

        if len(connected) == 1:
            candidate = self._candidate_for(connected[0])
            return HospitalSelection(candidate.code, candidate.name, False, False, candidates, "single_connected")

        return HospitalSelection(needs_clarification=True, candidates=candidates, reason="multiple_connected")

    def _match_query(self, user_query: str, connected: list[str]) -> tuple[list[str], list[str]]:
        normalized_query = _normalize_text(user_query)
        raw_query = str(user_query or "").lower()
        if not normalized_query and not raw_query.strip():
            return [], []

        strong_matches: list[str] = []
        confirmation_matches: list[str] = []
        for code in connected:
            match_strength = self._query_mentions_hospital(
                code=code,
                normalized_query=normalized_query,
                raw_query=raw_query,
            )
            if match_strength == "strong":
                strong_matches.append(code)
            elif match_strength == "confirmation":
                confirmation_matches.append(code)
        return strong_matches, confirmation_matches

    def _query_mentions_hospital(self, *, code: str, normalized_query: str, raw_query: str) -> str:
        code_token = _normalize_code(code)
        if code_token and _contains_ascii_token(raw_query, code_token):
            return "strong"

        info = self._lookup(code) or {}
        name = _normalize_text(str(info.get("name") or ""))
        if name and _contains_precise_phrase(normalized_query, name):
            return "strong"

        for alias in self._aliases.get(code, []):
            alias_text = _normalize_text(alias)
            if not alias_text:
                continue
            if _is_ascii_token(alias_text):
                if _contains_ascii_token(raw_query, alias_text):
                    return "strong"
                continue
            if _is_precise_cjk_phrase(alias_text):
                if _contains_precise_phrase(normalized_query, alias_text):
                    return "strong"
                continue
            if normalized_query == alias_text:
                return "strong"
            if alias_text in normalized_query:
                return "confirmation"

        return ""

    def _candidate_for(self, code: str) -> HospitalCandidate:
        info = self._lookup(code)
        name = str((info or {}).get("name") or code).strip() or code
        return HospitalCandidate(code=code, name=name)

    def _lookup(self, code: str) -> Mapping[str, Any] | None:
        if self._hospital_lookup is None:
            return None
        try:
            return self._hospital_lookup(code)
        except Exception:
            return None


def format_hospital_clarification(selection: HospitalSelection) -> str:
    names = [candidate.name for candidate in selection.candidates if candidate.name]
    if not names:
        return "请先选择要预约的医院。"
    return "你绑定了多家医院，请选择要预约哪家：" + " / ".join(names)


def format_hospital_confirmation(selection: HospitalSelection) -> str:
    name = selection.selected_name or selection.selected_code
    if not name:
        return "请先确认要预约的医院。"
    return f"你说的医院我理解为：{name}。为避免挂错医院，请回复“确认医院”后我再继续。"


def _normalize_aliases(raw_aliases: Mapping[str, Any] | None) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    if not isinstance(raw_aliases, Mapping):
        return normalized
    for code, aliases in raw_aliases.items():
        code_key = _normalize_code(code)
        if not code_key:
            continue
        if isinstance(aliases, str):
            values = [aliases]
        elif isinstance(aliases, list):
            values = [str(item) for item in aliases]
        else:
            continue
        normalized[code_key] = values
    return normalized


def _normalize_code(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "")


def _is_ascii_token(value: str) -> bool:
    return bool(value) and all(ord(char) < 128 for char in value)


def _contains_ascii_token(raw_query: str, token: str) -> bool:
    if not token:
        return False
    pattern = rf"(?<![a-z0-9_]){re.escape(token.lower())}(?![a-z0-9_])"
    return re.search(pattern, raw_query.lower()) is not None


def _contains_precise_phrase(normalized_query: str, phrase: str) -> bool:
    if not phrase:
        return False
    if phrase == normalized_query:
        return True
    if not _is_precise_cjk_phrase(phrase):
        return False
    return phrase in normalized_query


def _is_precise_cjk_phrase(value: str) -> bool:
    if _is_ascii_token(value):
        return False
    if any(marker in value for marker in ("医院", "院区", "门诊部", "医学中心")):
        return True
    cjk_chars = sum(1 for char in value if "\u4e00" <= char <= "\u9fff")
    return cjk_chars >= 4
