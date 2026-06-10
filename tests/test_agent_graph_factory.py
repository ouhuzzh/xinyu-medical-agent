import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from core.agent_graph_factory import AgentGraphFactory  # noqa: E402


class FakeVectorDb:
    def get_collection(self, collection_name):
        return {"collection_name": collection_name}


class FakeToolFactory:
    def __init__(self, collection):
        self.collection = collection

    def create_tools(self):
        return [{"tool_collection": self.collection["collection_name"]}]


class FakeRouter:
    def __init__(self):
        self.calls = []

    def get_llm(self, tier):
        self.calls.append(tier)
        return f"llm:{tier}"


class FakeSkillBootstrapper:
    def __init__(self):
        self.calls = 0

    def bootstrap(self):
        self.calls += 1
        return 3


class AgentGraphFactoryTests(unittest.TestCase):
    def test_create_llm_runtime_uses_router_default_tier(self):
        router = FakeRouter()
        factory = AgentGraphFactory(
            vector_db=FakeVectorDb(),
            appointment_service=object(),
            user_mcp_pool=object(),
            chat_sessions=object(),
            llm_router_factory=lambda: router,
        )

        llm_router, llm = factory.create_llm_runtime()

        self.assertIs(llm_router, router)
        self.assertEqual(llm, "llm:default")
        self.assertEqual(router.calls, ["default"])

    def test_build_graph_wires_tools_and_extra_services(self):
        captured = {}
        appointment_service = object()
        user_mcp_pool = object()
        chat_sessions = object()
        skill_bootstrapper = FakeSkillBootstrapper()

        def graph_builder(llm, tools, **kwargs):
            captured["llm"] = llm
            captured["tools"] = tools
            captured["kwargs"] = kwargs
            return "graph"

        factory = AgentGraphFactory(
            vector_db=FakeVectorDb(),
            appointment_service=appointment_service,
            user_mcp_pool=user_mcp_pool,
            chat_sessions=chat_sessions,
            tool_factory_cls=FakeToolFactory,
            graph_builder=graph_builder,
            skill_bootstrapper=skill_bootstrapper,
        )

        graph = factory.build_graph(collection_name="child_chunks", llm_router="router", llm="llm")

        self.assertEqual(graph, "graph")
        self.assertEqual(skill_bootstrapper.calls, 1)
        self.assertEqual(captured["tools"], [{"tool_collection": "child_chunks"}])
        self.assertIs(captured["kwargs"]["appointment_service"], appointment_service)
        self.assertEqual(captured["kwargs"]["llm_router"], "router")
        self.assertIs(captured["kwargs"]["extra_services"]["user_mcp_pool"], user_mcp_pool)
        self.assertIs(captured["kwargs"]["extra_services"]["chat_sessions"], chat_sessions)


if __name__ == "__main__":
    unittest.main()
