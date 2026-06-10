"""Factory for compiling the LangGraph agent runtime."""

from __future__ import annotations

import logging
from typing import Callable

import config
from rag_agent.graph import create_agent_graph
from rag_agent.tools import ToolFactory


logger = logging.getLogger(__name__)


class AgentGraphFactory:
    def __init__(
        self,
        *,
        vector_db,
        appointment_service,
        user_mcp_pool,
        chat_sessions,
        tool_factory_cls=ToolFactory,
        graph_builder: Callable = create_agent_graph,
        llm_router_factory: Callable | None = None,
        skill_registrar: Callable | None = None,
    ):
        self.vector_db = vector_db
        self.appointment_service = appointment_service
        self.user_mcp_pool = user_mcp_pool
        self.chat_sessions = chat_sessions
        self._tool_factory_cls = tool_factory_cls
        self._graph_builder = graph_builder
        self._llm_router_factory = llm_router_factory
        self._skill_registrar = skill_registrar

    def create_llm_runtime(self):
        if self._llm_router_factory is not None:
            llm_router = self._llm_router_factory()
        else:
            from llm_tiered_router import TieredLLMRouter

            llm_router = TieredLLMRouter.from_env()
        return llm_router, llm_router.get_llm("default")

    def build_graph(self, *, collection_name: str, llm_router, llm):
        collection = self.vector_db.get_collection(collection_name)
        tools = self._tool_factory_cls(collection).create_tools()
        self.register_skills()
        return self._graph_builder(
            llm,
            tools,
            appointment_service=self.appointment_service,
            llm_router=llm_router,
            extra_services={
                "user_mcp_pool": self.user_mcp_pool,
                "chat_sessions": self.chat_sessions,
            },
        )

    def register_skills(self) -> int:
        if not getattr(config, "SKILLS_ENABLED", False):
            return 0
        if self._skill_registrar is not None:
            return int(self._skill_registrar() or 0)

        from mcp_integration.mcp_skill import MCPSkill
        from skills.booking_skill import AppointmentSkill as BookingIntentSkill
        from skills.cancel_skill import CancelSkill
        from skills.greeting_skill import GreetingSkill
        from skills.medical_rag_skill import MedicalRagSkill
        from skills.registry import get_skill_registry
        from skills.triage_skill import TriageSkill

        registry = get_skill_registry()
        registry.register(GreetingSkill())
        registry.register(TriageSkill())
        registry.register(BookingIntentSkill())
        registry.register(CancelSkill())
        registry.register(MedicalRagSkill())
        registry.register(MCPSkill())
        logger.info("Skill plugin framework enabled: %d skills registered", len(registry.skills))
        return len(registry.skills)
