"""Greeting skill — proof of concept for the skill plugin framework.

Migrates the "greeting" intent from the hardcoded if-elif chain in
``_classify_query_by_rules()`` into a pluggable skill.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from langchain_core.messages import AIMessage

from .base_skill import BaseSkill

# Reuse the existing rule function from node_helpers
_GREETING_RESPONSE = (
    "你好！我是你的医疗助手，可以帮你解答医学问题、推荐科室或预约挂号。"
    "请问有什么我可以帮你的？"
)


class GreetingSkill(BaseSkill):
    """Handles greeting intents (你好、hello、早上好, etc.)."""

    @property
    def name(self) -> str:
        return "greeting"

    @property
    def priority(self) -> int:
        return 10  # Check first — greetings are unambiguous

    @property
    def intent_label(self) -> str:
        return "greeting"

    def match(self, query: str, *, context: Dict[str, Any]) -> bool:
        from rag_agent.node_helpers import _looks_like_greeting
        return _looks_like_greeting(query)

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
        # Greeting goes to its own handler, which then goes to END
        return {"greeting": "greeting_handler"}
