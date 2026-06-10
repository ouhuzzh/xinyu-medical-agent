"""Factory for compiling the LangGraph agent runtime."""

from __future__ import annotations

from typing import Callable

from core.skill_bootstrapper import SkillBootstrapper
from rag_agent.graph import create_agent_graph
from rag_agent.tools import ToolFactory

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
        skill_bootstrapper: SkillBootstrapper | None = None,
    ):
        self.vector_db = vector_db
        self.appointment_service = appointment_service
        self.user_mcp_pool = user_mcp_pool
        self.chat_sessions = chat_sessions
        self._tool_factory_cls = tool_factory_cls
        self._graph_builder = graph_builder
        self._llm_router_factory = llm_router_factory
        self._skill_bootstrapper = skill_bootstrapper or SkillBootstrapper()

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
        return self._skill_bootstrapper.bootstrap()
