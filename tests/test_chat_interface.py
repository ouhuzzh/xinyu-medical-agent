import sys
import contextvars
import unittest

from langchain_core.messages import AIMessage

sys.path.insert(0, r"D:\nageoffer\agentic-rag-for-dummies\project")

from core.chat_interface import ChatInterface, SILENT_NODES  # noqa: E402
from rag_agent.tools import reset_retrieval_context, set_retrieval_context  # noqa: E402


class FakeGraphState:
    def __init__(self, *, next_value=False, values=None):
        self.next = next_value
        self.values = values or {}


class FakeGraph:
    def __init__(self, final_values=None):
        self.final_values = final_values or {}
        self._calls = 0

    def get_state(self, config):
        self._calls += 1
        if self._calls == 1:
            return FakeGraphState()
        return FakeGraphState(values=self.final_values)

    def update_state(self, config, updates):
        return None

    def stream(self, stream_input, config=None, stream_mode=None):
        return iter(())


class FailingGraph(FakeGraph):
    def stream(self, stream_input, config=None, stream_mode=None):
        raise RuntimeError("graph failed")


class FakeVectorDb:
    def __init__(self, has_documents=False):
        self._has_documents = has_documents

    def has_documents(self):
        return self._has_documents


class FakeMemory:
    def __init__(self):
        self._state = {}

    def get_state(self, thread_id):
        return dict(self._state)

    def get_recent_messages(self, thread_id):
        return []

    def append_exchange(self, thread_id, user_message, assistant_message):
        return 1

    def set_state(self, thread_id, state):
        self._state = dict(state)


class FakeSummaryStore:
    def get_summary(self, thread_id):
        return ""

    def save_summary(self, thread_id, summary, recent_count):
        return None


class FakeRagSystem:
    def __init__(self, *, has_documents=False, final_values=None):
        self.agent_graph = FakeGraph(final_values=final_values)
        self.thread_id = "thread-chat"
        self.session_memory = FakeMemory()
        self.vector_db = FakeVectorDb(has_documents=has_documents)
        self.summary_store = FakeSummaryStore()
        self.observability = type("Observability", (), {"flush": staticmethod(lambda: None)})()

    def get_config(self):
        return {}


class FailingRagSystem(FakeRagSystem):
    def __init__(self):
        super().__init__()
        self.agent_graph = FailingGraph()


class ChatInterfaceTests(unittest.TestCase):
    def test_chat_allows_medical_question_to_continue_when_knowledge_base_is_empty(self):
        interface = ChatInterface(
            FakeRagSystem(
                has_documents=False,
                final_values={
                    "intent": "medical_rag",
                    "messages": [
                        AIMessage(
                            content=(
                                "回答模式：通用医学信息回答（本次未充分基于知识库检索结果）\n\n"
                                "感冒发烧时可以先注意休息、补充水分。"
                            )
                        )
                    ],
                },
            )
        )

        responses = list(interface.chat("感冒发烧怎么办？", []))

        self.assertEqual(len(responses), 1)
        self.assertIn("通用医学信息回答", responses[0][0]["content"])

    def test_chat_does_not_block_appointment_when_knowledge_base_is_empty(self):
        interface = ChatInterface(
            FakeRagSystem(
                has_documents=False,
                final_values={
                    "intent": "appointment",
                    "messages": [AIMessage(content="我已经整理好预约信息，请回复确认预约。")],
                },
            )
        )

        responses = list(interface.chat("帮我挂呼吸内科，明天下午", []))

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0][0]["content"], "我已经整理好预约信息，请回复确认预约。")

    def test_infer_intent_does_not_let_pending_appointment_hijack_unrelated_medical_question(self):
        intent = ChatInterface._infer_intent(
            "高血压会引起头晕吗",
            {"pending_action_type": "appointment", "pending_action_payload": {"department": "呼吸内科"}},
        )

        self.assertEqual(intent, "medical_rag")

    def test_infer_intent_keeps_pending_appointment_for_short_acknowledgement(self):
        intent = ChatInterface._infer_intent(
            "可以",
            {"pending_action_type": "appointment", "pending_action_payload": {"department": "呼吸内科"}},
        )

        self.assertEqual(intent, "pending")

    def test_infer_intent_downgrades_pending_clarification_to_ui_hint(self):
        intent = ChatInterface._infer_intent(
            "明天下午",
            {"intent": "appointment", "pending_clarification": "请补充时间"},
        )

        self.assertEqual(intent, "pending")

    def test_prepare_visible_messages_hides_diagnostics_in_user_mode(self):
        response_messages = [
            {"role": "assistant", "content": "分析中", "metadata": {"node": "rewrite_query"}},
            {"role": "assistant", "content": "这是最终回答"},
        ]

        visible = ChatInterface._prepare_visible_messages(response_messages, reveal_diagnostics=False)

        self.assertEqual(visible, [{"role": "assistant", "content": "这是最终回答"}])

    def test_prepare_visible_messages_shows_placeholder_when_only_diagnostics_exist(self):
        response_messages = [
            {"role": "assistant", "content": "调用工具中", "metadata": {"title": "tool"}},
        ]

        visible = ChatInterface._prepare_visible_messages(response_messages, reveal_diagnostics=False)

        self.assertEqual(len(visible), 1)
        self.assertIn("正在整理答案", visible[0]["content"])

    def test_extract_latest_state_assistant_sanitizes_query_plan_prefix(self):
        latest_values = {
            "messages": [
                AIMessage(content='{"queries": ["高血压患者应该注意哪些事项"]}高血压患者要注意低盐饮食。')
            ]
        }

        extracted = ChatInterface._extract_latest_state_assistant(latest_values)

        self.assertEqual(extracted, "高血压患者要注意低盐饮食。")

    def test_final_answer_nodes_are_silent_in_streaming_ui(self):
        self.assertIn("grounded_answer_generation", SILENT_NODES)
        self.assertIn("answer_grounding_check", SILENT_NODES)
        self.assertIn("decompose_tasks", SILENT_NODES)
        self.assertIn("supervise", SILENT_NODES)
        self.assertIn("reset_supervisor_state", SILENT_NODES)
        self.assertIn("self_eval", SILENT_NODES)

    def test_reset_retrieval_context_tolerates_cross_context_token(self):
        ctx = contextvars.Context()
        token = ctx.run(set_retrieval_context, thread_id="thread-x", original_query="头疼怎么办")

        reset_retrieval_context(token)

    def test_chat_returns_friendly_medical_fallback_when_graph_fails(self):
        interface = ChatInterface(FailingRagSystem())

        responses = list(interface.chat("头疼怎么处理", []))

        self.assertTrue(responses)
        self.assertIn("通用医学信息", responses[-1])
        self.assertIn("未充分基于知识库检索结果", responses[-1])
        self.assertNotIn("❌ Error", responses[-1])


if __name__ == "__main__":
    unittest.main()
