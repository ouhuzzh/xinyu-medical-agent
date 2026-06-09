"""Triage skill — '挂什么科', '看哪个科'."""

from __future__ import annotations

from typing import List, Tuple

from .base_skill import BaseSkill


class TriageSkill(BaseSkill):
    """Department recommendation questions."""

    @property
    def name(self) -> str:
        return "triage"

    @property
    def priority(self) -> int:
        return 15

    @property
    def intent_label(self) -> str:
        return "triage"

    @property
    def keywords(self) -> Tuple[str, ...]:
        return ("挂什么科", "挂哪个科", "看什么科", "看哪个科",
                "挂哪科", "看哪科", "该挂什么科")

    @property
    def allow_l1_substring(self) -> bool:
        return True  # "胸痛挂什么科" contains "挂什么科"

    @property
    def utterances(self) -> List[str]:
        return [
            "挂什么科", "看哪个科", "应该看什么科室", "我该挂哪个科",
            "这个症状看什么科", "推荐个科室", "应该挂哪个科室",
            "该挂什么科", "去哪个科室", "看什么科室",
            "which department should I visit", "what department",
        ]

    @property
    def llm_hint(self) -> str:
        return '"挂什么科/看什么科/去哪个科室" department-recommendation questions'

    def get_route_targets(self):
        return {"triage": "recommend_department"}
