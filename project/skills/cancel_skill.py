"""Cancellation skill — '取消预约', '退号'."""

from __future__ import annotations

from typing import List, Tuple

from .base_skill import BaseSkill


# Canonical cancel update hints — shared with ChatInterface for
# optimistic pre-classification in the SSE streaming path.
CANCEL_HINTS: Tuple[str, ...] = ("取消", "退号", "预约号", "第", "appointment", "cancel")


class CancelSkill(BaseSkill):
    """Explicit cancellation requests."""

    @property
    def name(self) -> str:
        return "cancel_appointment"

    @property
    def priority(self) -> int:
        return 22

    @property
    def intent_label(self) -> str:
        return "cancel_appointment"

    @property
    def keywords(self) -> Tuple[str, ...]:
        return ("取消预约", "取消挂号", "退号")

    @property
    def allow_l1_substring(self) -> bool:
        return True  # "取消预约 APT123" contains "取消预约"

    @property
    def utterances(self) -> List[str]:
        return [
            "取消预约", "取消挂号", "退号", "帮我取消预约",
            "取消刚才的挂号", "取消那个预约", "取消最近那个号",
            "我要取消预约", "帮我退号", "取消今天的号",
            "cancel my appointment", "cancel booking",
        ]

    @property
    def llm_hint(self) -> str:
        return 'cancellation requests ("取消预约 APT123", "取消刚才的挂号")'

    def get_route_targets(self):
        return {"cancel_appointment": "handle_appointment_skill"}
