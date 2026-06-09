"""Greeting skill — handles polite greetings and farewells.

Uses L1 exact keyword matching for short, unambiguous greetings,
and L2 semantic matching for polite variants.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from langchain_core.messages import AIMessage

from .base_skill import BaseSkill

_GREETING_RESPONSE = (
    "你好！我是你的医疗助手，可以帮你：\n"
    "- 🏥 推荐就诊科室\n"
    "- 📅 预约挂号\n"
    "- ❌ 取消预约\n"
    "- 💊 解答医疗健康问题\n\n"
    "请问有什么可以帮你的？"
)


class GreetingSkill(BaseSkill):
    """Handles greeting intents."""

    @property
    def name(self) -> str:
        return "greeting"

    @property
    def priority(self) -> int:
        return 10

    @property
    def intent_label(self) -> str:
        return "greeting"

    # L1: extremely short, exact-match greetings only
    @property
    def keywords(self) -> Tuple[str, ...]:
        return (
            "你好", "您好", "hi", "hello", "hey", "嗨",
            "谢谢", "感谢", "thanks", "thank you", "多谢",
            "再见", "拜拜", "bye", "goodbye",
            "早上好", "下午好", "晚上好",
            "good morning", "good afternoon", "good evening",
        )

    # L2: longer / polite variants for semantic matching
    @property
    def utterances(self) -> List[str]:
        return [
            "你好", "您好", "早上好", "下午好", "晚上好",
            "谢谢", "感谢你", "再见", "拜拜",
            "hello", "hi", "hey", "good morning", "thank you", "bye",
        ]

    # L3: hint for LLM intent classifier
    @property
    def llm_hint(self) -> str:
        return 'polite greetings/declines ("你好", "谢谢", "再见", "谢谢我不用了")'

    def get_state_schema(self) -> Dict[str, Any]:
        return {}

    def register_nodes(
        self, graph_builder, *, llm_router=None, tools_list=None, services=None
    ) -> Dict[str, Callable]:
        def greeting_handler(state: dict) -> dict:
            return {
                "intent": "greeting",
                "messages": [AIMessage(content=_GREETING_RESPONSE)],
                "route_reason": "skill:greeting",
                "decision_source": "skill",
            }

        return {"greeting_handler": greeting_handler}

    def get_route_targets(self) -> Dict[str, str]:
        return {"greeting": "greeting_handler"}
