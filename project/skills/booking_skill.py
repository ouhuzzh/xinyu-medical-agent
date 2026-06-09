"""Appointment booking skill — handle_appointment_skill MCP backend."""

from __future__ import annotations

from typing import List, Tuple

from .base_skill import BaseSkill


# Canonical appointment update hints — shared with ChatInterface for
# optimistic pre-classification in the SSE streaming path.
UPDATE_HINTS: Tuple[str, ...] = ("改", "换", "预约", "挂号", "医生", "科", "时间", "时段")


class AppointmentSkill(BaseSkill):
    """Booking requests ('帮我预约心内科', '我要挂号明天')."""

    @property
    def name(self) -> str:
        return "appointment"

    @property
    def priority(self) -> int:
        return 20

    @property
    def intent_label(self) -> str:
        return "appointment"

    # L1: high-confidence exact action + entity patterns only
    @property
    def keywords(self) -> Tuple[str, ...]:
        return (
            "帮我预约心内科", "帮我预约内科", "帮我预约外科",
            "我要挂号", "帮我预约", "帮我挂号",
        )

    @property
    def allow_l1_substring(self) -> bool:
        return True  # "帮我预约心内科明天" contains "帮我预约"

    # L2: semantic variants
    @property
    def utterances(self) -> List[str]:
        return [
            "我要挂号", "帮我预约", "我想预约看病", "帮我挂个号",
            "预约心内科", "挂号内科", "帮我安排个医生",
            "我要预约明天的号", "帮我挂个专家号",
            "我想挂号", "帮我挂个心内科的号", "我要预约门诊",
            "book an appointment", "register for a doctor",
        ]

    @property
    def llm_hint(self) -> str:
        return 'booking requests ("我要挂号", "帮我预约")'

    def get_route_targets(self):
        return {"appointment": "handle_appointment_skill"}
