import sys
import unittest

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from core.chat_turn_input_service import ChatTurnInputService  # noqa: E402


class FakeGraphState:
    def __init__(self, *, next_value=False):
        self.next = next_value


class FakeGraph:
    def __init__(self, *, next_value=False):
        self.next_value = next_value
        self.updates = []

    def get_state(self, config):
        return FakeGraphState(next_value=self.next_value)

    def update_state(self, config, updates):
        self.updates.append((config, updates))


class FakeSessionMemory:
    def __init__(self, state=None):
        self.state = state or {}
        self.recent_messages = [
            HumanMessage(content="之前的问题"),
            AIMessage(content="之前的回答"),
        ]

    def get_state(self, thread_id):
        return dict(self.state)

    def get_recent_messages(self, thread_id):
        return list(self.recent_messages)


class FakeSummaryStore:
    def __init__(self, summary=""):
        self.summary = summary

    def get_summary(self, thread_id):
        return self.summary


class FakeChatSessions:
    def get_session(self, thread_id):
        return {"owner_user_id": "user-1"}


class FakeRagSystem:
    def __init__(self, *, next_value=False, state=None, summary="", with_optionals=True):
        self.thread_id = "default-thread"
        self.agent_graph = FakeGraph(next_value=next_value)
        self.session_memory = FakeSessionMemory(state=state)
        self.summary_store = FakeSummaryStore(summary=summary)
        if with_optionals:
            self.chat_sessions = FakeChatSessions()
            self.user_mcp_pool = object()

    def get_config(self, thread_id):
        return {"configurable": {"thread_id": thread_id}}


class ChatTurnInputServiceTests(unittest.TestCase):
    def _service(self, rag_system, fetched_memories=""):
        return ChatTurnInputService(
            rag_system,
            get_graph_config=rag_system.get_config,
            fetch_user_memories=lambda **kwargs: fetched_memories,
            build_state_messages=lambda session_state: [SystemMessage(content="state")] if session_state else [],
            graph_state_from_session=lambda thread_id, session_state: {"thread_id": thread_id, **session_state},
        )

    def test_prepare_builds_regular_stream_input(self):
        rag_system = FakeRagSystem(state={"intent": "appointment"}, summary="旧摘要")
        service = self._service(rag_system, fetched_memories="用户记忆")

        turn_input = service.prepare(message="  帮我挂号  ", thread_id="thread-1")

        self.assertEqual(turn_input.active_thread_id, "thread-1")
        self.assertEqual(turn_input.user_message, "帮我挂号")
        self.assertFalse(turn_input.checkpoint_resumed)
        self.assertEqual(turn_input.user_id, "user-1")
        self.assertEqual(turn_input.user_memories_text, "用户记忆")
        self.assertIsNotNone(turn_input.stream_input)
        self.assertEqual(turn_input.stream_input["user_id"], "user-1")
        self.assertEqual(turn_input.stream_input["user_memories"], "用户记忆")
        self.assertEqual(turn_input.stream_input["messages"][-1].content, "帮我挂号")
        self.assertEqual(rag_system.agent_graph.updates[0][1], {"conversation_summary": "旧摘要"})
        self.assertEqual(rag_system.agent_graph.updates[1][1]["intent"], "appointment")

    def test_prepare_checkpoint_resume_updates_state_and_streams_none(self):
        rag_system = FakeRagSystem(next_value=True)
        service = self._service(rag_system, fetched_memories="用户记忆")

        turn_input = service.prepare(message="继续", thread_id="thread-2")

        self.assertTrue(turn_input.checkpoint_resumed)
        self.assertIsNone(turn_input.stream_input)
        self.assertEqual(rag_system.agent_graph.updates[0][1]["thread_id"], "thread-2")
        self.assertEqual(rag_system.agent_graph.updates[0][1]["messages"][0].content, "继续")
        self.assertEqual(rag_system.agent_graph.updates[0][1]["user_memories"], "用户记忆")

    def test_prepare_tolerates_missing_optional_dependencies(self):
        rag_system = FakeRagSystem(with_optionals=False)
        service = self._service(rag_system, fetched_memories="should-not-load")

        turn_input = service.prepare(message="你好")

        self.assertEqual(turn_input.active_thread_id, "default-thread")
        self.assertEqual(turn_input.user_id, "")
        self.assertEqual(turn_input.user_memories_text, "")
        self.assertIsNone(turn_input.stream_input["_mcp_pool"])


if __name__ == "__main__":
    unittest.main()
