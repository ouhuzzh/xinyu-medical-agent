import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from core.service_bootstrapper import ServiceBootstrapper  # noqa: E402


class FakeFactory:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        return {"factory": self.name, "args": args, "kwargs": kwargs}


class FakeKnowledgeBaseSupervisor:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.status = {"status": "not_checked", "stats": {"recent_imports": []}}


class ServiceBootstrapperTests(unittest.TestCase):
    def test_bootstrap_binds_services_and_registers_container(self):
        factories = {
            "vector_db": FakeFactory("vector_db"),
            "parent_store": FakeFactory("parent_store"),
            "import_task_store": FakeFactory("import_task_store"),
            "chunker": FakeFactory("chunker"),
            "session_memory": FakeFactory("session_memory"),
            "summary_store": FakeFactory("summary_store"),
            "user_memory_store": FakeFactory("user_memory_store"),
            "chat_sessions": FakeFactory("chat_sessions"),
            "memory_extractor": FakeFactory("memory_extractor"),
            "mcp_server_registry": FakeFactory("mcp_server_registry"),
            "user_mcp_credential_store": FakeFactory("user_mcp_credential_store"),
            "user_mcp_pool": FakeFactory("user_mcp_pool"),
            "appointment_service": FakeFactory("appointment_service"),
            "observability": FakeFactory("observability"),
            "skill_bootstrapper": FakeFactory("skill_bootstrapper"),
            "agent_graph_factory": FakeFactory("agent_graph_factory"),
            "knowledge_base_supervisor": FakeKnowledgeBaseSupervisor,
        }
        rag_system = type("FakeRagSystem", (), {})()

        container = ServiceBootstrapper(factories=factories).bootstrap(rag_system)

        self.assertIs(container.vector_db, rag_system.vector_db)
        self.assertIs(container.service_bootstrapper, rag_system.service_bootstrapper)
        self.assertIs(container.knowledge_base_supervisor, rag_system.knowledge_base_supervisor)
        self.assertIs(rag_system.knowledge_base_supervisor.rag_system, rag_system)
        self.assertIs(rag_system._knowledge_base_status, rag_system.knowledge_base_supervisor.status)
        self.assertIn("agent_graph_factory", container.service_names)
        self.assertIs(
            rag_system.agent_graph_factory["kwargs"]["skill_bootstrapper"],
            rag_system.skill_bootstrapper,
        )
        self.assertIs(
            rag_system.user_mcp_pool["args"][0],
            rag_system.mcp_server_registry,
        )


if __name__ == "__main__":
    unittest.main()
