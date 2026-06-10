"""Service assembly for the RAG system facade."""

from __future__ import annotations

from typing import Callable

from core.agent_graph_factory import AgentGraphFactory
from core.container import ServiceContainer
from core.document_chunker import DocumentChuncker
from core.knowledge_base_supervisor import KnowledgeBaseSupervisor
from core.observability import Observability
from core.skill_bootstrapper import SkillBootstrapper
from db.chat_session_store import ChatSessionStore
from db.import_task_store import ImportTaskStore
from db.parent_store_manager import ParentStoreManager
from db.vector_db_manager import VectorDbManager
from mcp_integration.mcp_server_registry import MCPServerRegistry
from mcp_integration.user_mcp_credential_store import UserMCPCredentialStore
from mcp_integration.user_mcp_pool import UserMCPPool
from memory.memory_extractor import MemoryExtractor
from memory.redis_memory import RedisSessionMemory
from memory.summary_store import SummaryStore
from memory.user_memory_store import UserMemoryStore
from services.appointment_service import AppointmentService


class ServiceBootstrapper:
    """Create and register the subsystem singletons owned by RAGSystem."""

    def __init__(self, factories: dict[str, Callable] | None = None):
        self._factories = dict(factories or {})

    def bootstrap(self, rag_system) -> ServiceContainer:
        services = self._create_services(rag_system)
        services["service_bootstrapper"] = self

        for name, service in services.items():
            setattr(rag_system, name, service)

        rag_system._knowledge_base_status = services["knowledge_base_supervisor"].status

        container = ServiceContainer()
        for name, service in services.items():
            container.register(name, service)
        return container

    def _create_services(self, rag_system) -> dict:
        vector_db = self._create("vector_db")
        parent_store = self._create("parent_store")
        import_task_store = self._create("import_task_store")
        chunker = self._create("chunker")
        session_memory = self._create("session_memory")
        summary_store = self._create("summary_store")
        user_memory_store = self._create("user_memory_store")
        chat_sessions = self._create("chat_sessions")
        memory_extractor = self._create("memory_extractor", user_memory_store, chat_sessions)
        mcp_server_registry = self._create("mcp_server_registry")
        user_mcp_credential_store = self._create("user_mcp_credential_store")
        user_mcp_pool = self._create("user_mcp_pool", mcp_server_registry, user_mcp_credential_store)
        appointment_service = self._create("appointment_service")
        observability = self._create("observability")
        skill_bootstrapper = self._create("skill_bootstrapper")
        agent_graph_factory = self._create(
            "agent_graph_factory",
            vector_db=vector_db,
            appointment_service=appointment_service,
            user_mcp_pool=user_mcp_pool,
            chat_sessions=chat_sessions,
            skill_bootstrapper=skill_bootstrapper,
        )

        services = {
            "vector_db": vector_db,
            "parent_store": parent_store,
            "import_task_store": import_task_store,
            "chunker": chunker,
            "session_memory": session_memory,
            "summary_store": summary_store,
            "user_memory_store": user_memory_store,
            "chat_sessions": chat_sessions,
            "memory_extractor": memory_extractor,
            "mcp_server_registry": mcp_server_registry,
            "user_mcp_credential_store": user_mcp_credential_store,
            "user_mcp_pool": user_mcp_pool,
            "appointment_service": appointment_service,
            "observability": observability,
            "skill_bootstrapper": skill_bootstrapper,
            "agent_graph_factory": agent_graph_factory,
        }

        for name, service in services.items():
            setattr(rag_system, name, service)
        services["knowledge_base_supervisor"] = self._create("knowledge_base_supervisor", rag_system)
        return services

    def _create(self, name: str, *args, **kwargs):
        factory = self._factories.get(name) or DEFAULT_FACTORIES[name]
        return factory(*args, **kwargs)


DEFAULT_FACTORIES: dict[str, Callable] = {
    "vector_db": VectorDbManager,
    "parent_store": ParentStoreManager,
    "import_task_store": ImportTaskStore,
    "chunker": DocumentChuncker,
    "session_memory": RedisSessionMemory,
    "summary_store": SummaryStore,
    "user_memory_store": UserMemoryStore,
    "chat_sessions": ChatSessionStore,
    "memory_extractor": MemoryExtractor,
    "mcp_server_registry": MCPServerRegistry,
    "user_mcp_credential_store": UserMCPCredentialStore,
    "user_mcp_pool": UserMCPPool,
    "appointment_service": AppointmentService,
    "observability": Observability,
    "skill_bootstrapper": SkillBootstrapper,
    "agent_graph_factory": AgentGraphFactory,
    "knowledge_base_supervisor": KnowledgeBaseSupervisor,
}
