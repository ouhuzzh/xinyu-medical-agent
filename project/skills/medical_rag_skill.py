"""Medical RAG skill — proof of concept for the skill plugin framework.

Migrates the "medical_rag" intent detection from the hardcoded if-elif chain
into a pluggable skill.  This skill does NOT register its own handler node —
it reuses the existing ``rewrite_query → plan_retrieval_queries → agent`` chain.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from .base_skill import BaseSkill


class MedicalRagSkill(BaseSkill):
    """Handles medical knowledge question intents."""

    @property
    def name(self) -> str:
        return "medical_rag"

    @property
    def priority(self) -> int:
        return 60  # Lower priority than greeting/cancel/appointment/triage

    @property
    def intent_label(self) -> str:
        return "medical_rag"

    def match(self, query: str, *, context: Dict[str, Any]) -> bool:
        """Match medical knowledge questions.

        Reuses the existing rule functions from node_helpers.
        """
        from rag_agent.node_helpers import (
            _looks_like_medical_knowledge_question,
            _looks_like_medical_request,
            _looks_like_medical_follow_up,
        )
        conversation_summary = context.get("conversation_summary", "")
        recent_context = context.get("recent_context", "")
        topic_focus = context.get("topic_focus", "")
        return (
            _looks_like_medical_knowledge_question(query)
            or _looks_like_medical_request(
                query,
                conversation_summary=conversation_summary,
                recent_context=recent_context,
                topic_focus=topic_focus,
            )
            or _looks_like_medical_follow_up(
                query,
                "\n".join(part for part in (conversation_summary, topic_focus) if part),
                recent_context,
            )
        )

    def get_state_schema(self) -> Dict[str, Any]:
        return {}

    def register_nodes(
        self, graph_builder, *, llm_router=None, tools_list=None, services=None
    ) -> Dict[str, Callable]:
        # No custom handler — reuse the existing rewrite_query → agent chain
        return {}

    def get_route_targets(self) -> Dict[str, str]:
        # Route medical_rag to the existing rewrite_query node
        return {"medical_rag": "rewrite_query"}
