import sys
import unittest

from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, r"D:\nageoffer\agentic-rag-for-dummies\project")

from core.chat_interface import ChatInterface  # noqa: E402
from core.chat_turn_service import ChatTurnService, TurnArtifacts  # noqa: E402
from core.context_compression import ContextCompressionService  # noqa: E402


class FakeGraphState:
    def __init__(self, values=None):
        self.values = values or {}


class FakeGraph:
    def __init__(self, final_values=None):
        self.final_values = final_values or {}

    def get_state(self, config):
        return FakeGraphState(values=self.final_values)


class FakeSessionMemory:
    def __init__(self):
        self.appended = []
        self.saved_states = []
        self._recent_messages = []

    def append_exchange(self, thread_id, user_message, assistant_message):
        self.appended.append((thread_id, user_message, assistant_message))
        self._recent_messages.extend([
            HumanMessage(content=user_message),
            AIMessage(content=assistant_message),
        ])
        return len(self._recent_messages)

    def get_recent_messages(self, thread_id):
        return list(self._recent_messages)

    def set_recent_messages(self, thread_id, messages):
        self._recent_messages = list(messages)

    def set_state(self, thread_id, state):
        self.saved_states.append((thread_id, dict(state)))

    def recent_message_count(self, thread_id):
        return len(self.appended)


class FakeSummaryStore:
    def __init__(self):
        self.saved = []

    def save_summary(self, thread_id, summary, recent_count):
        self.saved.append((thread_id, summary, recent_count))


class FakeRouteLogStore:
    def __init__(self):
        self.logs = []

    def save_log(self, payload):
        self.logs.append(dict(payload))


class FakeRagSystem:
    def __init__(self, final_values=None):
        self.agent_graph = FakeGraph(final_values=final_values)
        self.session_memory = FakeSessionMemory()
        self.summary_store = FakeSummaryStore()


class ChatTurnServiceTests(unittest.TestCase):
    def _build_service(self, rag_system, route_log_store):
        return ChatTurnService(
            rag_system,
            route_log_store=route_log_store,
            extract_final_assistant_text=ChatInterface._extract_final_assistant_text,
            extract_all_visible_assistant_texts=ChatInterface._extract_all_visible_assistant_texts,
            extract_clarification_text=ChatInterface._extract_clarification_text,
            extract_latest_state_assistant=ChatInterface._extract_latest_state_assistant,
            build_chat_failure_fallback=lambda user_message: f"fallback:{user_message}",
            resolved_session_state=lambda latest_values, session_state, user_message, clarification_text: {
                "intent": latest_values.get("primary_intent", ""),
                "last_route_reason": latest_values.get("route_reason", ""),
            },
            invalidate_memory_cache=lambda user_id, thread_id: None,
        )

    def test_prepare_turn_artifacts_appends_latest_state_answer(self):
        rag_system = FakeRagSystem(
            final_values={
                "primary_intent": "medical_rag",
                "messages": [AIMessage(content="这是图状态里的最终回答。")],
            }
        )
        route_log_store = FakeRouteLogStore()
        service = self._build_service(rag_system, route_log_store)

        artifacts = service.prepare_turn_artifacts(
            active_thread_id="thread-1",
            graph_config={},
            response_messages=[],
            user_message="头疼怎么办",
            session_state={},
        )

        self.assertTrue(artifacts.response_messages_changed)
        self.assertEqual(artifacts.final_assistant, "这是图状态里的最终回答。")
        self.assertEqual(artifacts.response_messages[-1]["content"], "这是图状态里的最终回答。")
        self.assertEqual(artifacts.combined_assistant_text, "这是图状态里的最终回答。")

    def test_finalize_turn_persists_exchange_state_and_route_log(self):
        rag_system = FakeRagSystem()
        route_log_store = FakeRouteLogStore()
        service = self._build_service(rag_system, route_log_store)
        service._schedule_memory_extraction = lambda **kwargs: None

        artifacts = TurnArtifacts(
            response_messages=[{"role": "assistant", "content": "请确认预约。"}],
            latest_values={
                "primary_intent": "appointment",
                "decision_source": "skill_registry",
                "route_reason": "pending:appointment",
            },
            final_assistant="请确认预约。",
            combined_assistant_text="请确认预约。",
            clarification_text="",
            updated_state={"intent": "appointment", "last_route_reason": "pending:appointment"},
            had_pending_state=True,
            route_reason="pending:appointment",
            secondary_turn_executed=False,
            response_messages_changed=False,
        )

        service.finalize_turn(
            active_thread_id="thread-2",
            request_id="req-2",
            user_message="帮我挂号",
            session_state={"pending_action_type": "appointment"},
            checkpoint_resumed=False,
            user_id="",
            artifacts=artifacts,
        )

        self.assertEqual(
            rag_system.session_memory.appended,
            [("thread-2", "帮我挂号", "请确认预约。")],
        )
        self.assertEqual(
            rag_system.session_memory.saved_states,
            [("thread-2", {"intent": "appointment", "last_route_reason": "pending:appointment"})],
        )
        self.assertEqual(len(route_log_store.logs), 1)
        self.assertEqual(route_log_store.logs[0]["route_reason"], "pending:appointment")
        self.assertTrue(route_log_store.logs[0]["had_pending_state"])

    def test_finalize_turn_triggers_compression_based_on_token_threshold(self):
        from unittest import mock
        import config

        rag_system = FakeRagSystem()
        route_log_store = FakeRouteLogStore()
        service = self._build_service(rag_system, route_log_store)
        service._schedule_memory_extraction = lambda **kwargs: None

        compressed_result = {
            "thread_id": "thread-tok",
            "compressed": True,
            "reason": "token",
            "preserved_count": 2,
            "summary_length": 10,
        }
        fake_service = mock.MagicMock(spec=ContextCompressionService)
        fake_service.compress_thread.return_value = compressed_result
        service._compression_service = fake_service

        artifacts = TurnArtifacts(
            response_messages=[{"role": "assistant", "content": "answer"}],
            latest_values={"primary_intent": "medical_rag"},
            final_assistant="answer",
            combined_assistant_text="answer",
            clarification_text="",
            updated_state={},
            had_pending_state=False,
            route_reason="medical_rag",
            secondary_turn_executed=False,
            response_messages_changed=False,
        )

        with mock.patch.object(config, "SUMMARY_TOKEN_THRESHOLD", 1):
            service.finalize_turn(
                active_thread_id="thread-tok",
                request_id="req-tok",
                user_message="question",
                session_state={},
                checkpoint_resumed=False,
                user_id="",
                artifacts=artifacts,
            )

        fake_service.compress_thread.assert_called_once()
        call_kwargs = fake_service.compress_thread.call_args.kwargs
        self.assertEqual(call_kwargs["thread_id"], "thread-tok")
        self.assertEqual(call_kwargs["preserve_recent_turns"], config.RECENT_CONTEXT_TURNS)

    def test_finalize_turn_skips_compression_when_under_threshold(self):
        from unittest import mock
        import config

        rag_system = FakeRagSystem()
        route_log_store = FakeRouteLogStore()
        service = self._build_service(rag_system, route_log_store)
        service._schedule_memory_extraction = lambda **kwargs: None

        fake_service = mock.MagicMock(spec=ContextCompressionService)
        service._compression_service = fake_service

        artifacts = TurnArtifacts(
            response_messages=[{"role": "assistant", "content": "answer"}],
            latest_values={"primary_intent": "medical_rag"},
            final_assistant="answer",
            combined_assistant_text="answer",
            clarification_text="",
            updated_state={},
            had_pending_state=False,
            route_reason="medical_rag",
            secondary_turn_executed=False,
            response_messages_changed=False,
        )

        with mock.patch.object(config, "SUMMARY_TOKEN_THRESHOLD", 100000), \
             mock.patch.object(config, "SUMMARY_MAX_MESSAGE_CEILING", 1000):
            service.finalize_turn(
                active_thread_id="thread-no",
                request_id="req-no",
                user_message="question",
                session_state={},
                checkpoint_resumed=False,
                user_id="",
                artifacts=artifacts,
            )

        fake_service.compress_thread.assert_not_called()


if __name__ == "__main__":
    unittest.main()
