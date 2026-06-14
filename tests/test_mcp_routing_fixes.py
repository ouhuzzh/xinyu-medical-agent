"""Tests for MCP backend integration and routing fixes."""

import os
import sys
import unittest

from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))


class FakeMCPTool:
    def __init__(self, name, result="ok", metadata=None, annotations=None):
        self.name = name
        self.description = f"{name} description"
        self.result = result
        self.called = False
        self.metadata = metadata or {}
        self.annotations = annotations or {}

    def invoke(self, _args):
        self.called = True
        return self.result


class FakeChatSessions:
    def get_session(self, _thread_id):
        return {"owner_user_id": "user-1"}


class FakeRegistry:
    def __init__(self, hospitals=None):
        self.hospitals = hospitals or {
            "xiehe": {"code": "xiehe", "name": "协和医院", "description": ""},
            "ruijin": {"code": "ruijin", "name": "瑞金医院", "description": ""},
        }

    def get_by_code(self, code):
        return self.hospitals.get(str(code or "").strip().lower())


class FakeMCPPool:
    def __init__(self, tools, connected=None, hospitals=None):
        self.tools = tools
        self.connected = connected or ["xiehe"]
        self._registry = FakeRegistry(hospitals)

    def get_tools_for_user(self, _user_id):
        return list(self.tools)

    def get_connected_hospitals(self, _user_id):
        return list(self.connected)

    def get_failed_hospitals(self, _user_id):
        return {}


class FakeRouter:
    has_tiers = True

    def __init__(self, llm):
        self.llm = llm

    def get_llm(self, _tier):
        return self.llm


class FakeMCPLLM:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls
        self.bound_tool_names = []
        self.invocation_count = 0

    def bind_tools(self, tools):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def invoke(self, _messages):
        self.invocation_count += 1
        if self.invocation_count == 1:
            return AIMessage(content="", tool_calls=self.tool_calls)
        return AIMessage(content="调用已处理")


class TestStrictConfirmation(unittest.TestCase):
    """Confirm _is_explicit_confirmation rejects vague words."""

    def setUp(self):
        from rag_agent.node_helpers import _is_explicit_confirmation
        self._confirm = _is_explicit_confirmation

    def test_explicit_booking_confirmed(self):
        self.assertTrue(self._confirm("确认预约", "appointment"))
        self.assertTrue(self._confirm("确认挂号", "appointment"))

    def test_explicit_cancel_confirmed(self):
        self.assertTrue(self._confirm("确认取消", "cancel_appointment"))
        self.assertTrue(self._confirm("确认退号", "cancel_appointment"))
        self.assertTrue(self._confirm("确定取消", "cancel_appointment"))

    def test_vague_words_rejected(self):
        """'好的'/'行'/'OK' should NOT trigger confirmation."""
        for word in ["好的", "行", "OK", "可以", "好", "嗯嗯", "可以的"]:
            with self.subTest(word=word):
                self.assertFalse(self._confirm(word, "appointment"),
                                 f"'{word}' should NOT confirm appointment")
                self.assertFalse(self._confirm(word, "cancel_appointment"),
                                 f"'{word}' should NOT confirm cancel")

    def test_partial_match_rejected(self):
        """'确认一下' without context should not confirm."""
        self.assertFalse(self._confirm("确认", "appointment"),
                         "'确认' alone should not confirm")


class TestMCPSkillCleaned(unittest.TestCase):
    """Verify MCPSkill no longer intercepts appointment keywords."""

    def setUp(self):
        import config
        self._saved_mcp_enabled = config.MCP_ENABLED
        config.MCP_ENABLED = True
        from mcp_integration.mcp_skill import MCPSkill
        self.skill = MCPSkill()
        self.ctx = {"recent_context": "", "conversation_summary": ""}

    def tearDown(self):
        import config
        config.MCP_ENABLED = self._saved_mcp_enabled

    def test_appointment_keywords_not_matched(self):
        """These should NOT trigger MCPSkill."""
        for q in ["挂号", "预约", "帮我预约", "帮我挂", "退号", "帮我取消",
                   "查医生", "查科室", "有没有号", "我要挂号", "取消预约"]:
            with self.subTest(query=q):
                self.assertFalse(self.skill.match(q, context=self.ctx),
                                 f"'{q}' should NOT match MCPSkill anymore")

    def test_mcp_skill_keyword_matches(self):
        """MCPSkill now has L1 keywords — these should match."""
        for q in ["多少钱", "支付", "查库存", "查报告"]:
            with self.subTest(query=q):
                self.assertTrue(self.skill.match(q, context=self.ctx),
                                f"'{q}' should match MCPSkill (L1 keyword)")

    def test_appointment_mutation_tool_names_are_reserved(self):
        from mcp_integration.tool_policy import is_appointment_mutation_name

        for name in [
            "xiehe__book_appointment",
            "xiehe__create_appointment",
            "xiehe__cancel_appointment",
            "xiehe__reschedule_appointment",
        ]:
            with self.subTest(name=name):
                self.assertTrue(is_appointment_mutation_name(name))

        self.assertFalse(is_appointment_mutation_name("xiehe__check_stock"))

    def test_generic_policy_blocks_metadata_required_confirmation(self):
        from mcp_integration.tool_policy import evaluate_generic_tool

        decision = evaluate_generic_tool(
            FakeMCPTool(
                "xiehe__submit_visit_order",
                metadata={
                    "domain": "appointment",
                    "effect": "write",
                    "requires_confirmation": True,
                },
            )
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "requires_confirmation")

    def test_generic_policy_allows_read_only_metadata(self):
        from mcp_integration.tool_policy import evaluate_generic_tool

        decision = evaluate_generic_tool(
            FakeMCPTool(
                "xiehe__query_schedule",
                annotations={
                    "readOnlyHint": True,
                    "domain": "appointment",
                    "effect": "read",
                },
            )
        )

        self.assertTrue(decision.allowed)

    def test_generic_mcp_handler_filters_and_blocks_appointment_mutation_tools(self):
        from mcp_integration.mcp_skill import MCPSkill

        booking_tool = FakeMCPTool("xiehe__book_appointment", {"appointment_no": "APT001"})
        stock_tool = FakeMCPTool("xiehe__check_stock", {"count": 3})
        llm = FakeMCPLLM(
            [
                {
                    "id": "call-book",
                    "type": "tool_call",
                    "name": "xiehe__book_appointment",
                    "args": {"department": "心内科"},
                },
                {
                    "id": "call-stock",
                    "type": "tool_call",
                    "name": "xiehe__check_stock",
                    "args": {"drug": "布洛芬"},
                },
            ]
        )
        handler = MCPSkill().register_nodes(
            None,
            llm_router=FakeRouter(llm),
            services={
                "user_mcp_pool": FakeMCPPool([booking_tool, stock_tool]),
                "chat_sessions": FakeChatSessions(),
            },
        )["mcp_services_handler"]

        result = handler({"messages": [HumanMessage(content="查库存")], "thread_id": "t-1"})

        self.assertNotIn("xiehe__book_appointment", llm.bound_tool_names)
        self.assertIn("xiehe__check_stock", llm.bound_tool_names)
        self.assertFalse(booking_tool.called)
        self.assertTrue(stock_tool.called)
        self.assertEqual(result["route_reason"], "skill:mcp_tool_executed")

    def test_generic_mcp_handler_does_not_run_when_only_appointment_mutation_tools_exist(self):
        from mcp_integration.mcp_skill import MCPSkill

        booking_tool = FakeMCPTool("xiehe__book_appointment", {"appointment_no": "APT001"})
        llm = FakeMCPLLM([])
        handler = MCPSkill().register_nodes(
            None,
            llm_router=FakeRouter(llm),
            services={
                "user_mcp_pool": FakeMCPPool([booking_tool]),
                "chat_sessions": FakeChatSessions(),
            },
        )["mcp_services_handler"]

        result = handler({"messages": [HumanMessage(content="多少钱")], "thread_id": "t-1"})

        self.assertFalse(booking_tool.called)
        self.assertEqual(llm.invocation_count, 0)
        self.assertEqual(result["route_reason"], "skill:mcp_reserved_tools_blocked")

    def test_generic_mcp_handler_blocks_metadata_declared_write_tool(self):
        from mcp_integration.mcp_skill import MCPSkill

        write_tool = FakeMCPTool(
            "xiehe__submit_visit_order",
            {"appointment_no": "APT001"},
            metadata={
                "domain": "appointment",
                "effect": "write",
                "requires_confirmation": True,
            },
        )
        llm = FakeMCPLLM([])
        handler = MCPSkill().register_nodes(
            None,
            llm_router=FakeRouter(llm),
            services={
                "user_mcp_pool": FakeMCPPool([write_tool]),
                "chat_sessions": FakeChatSessions(),
            },
        )["mcp_services_handler"]

        result = handler({"messages": [HumanMessage(content="多少钱")], "thread_id": "t-1"})

        self.assertFalse(write_tool.called)
        self.assertEqual(llm.invocation_count, 0)
        self.assertEqual(result["route_reason"], "skill:mcp_reserved_tools_blocked")


class TestMCPBackend(unittest.TestCase):
    """Verify MCPAppointmentBackend factory method."""

    def test_try_create_returns_none_without_mcp_config(self):
        """When MCP is disabled, should return None."""
        import config
        saved = config.MCP_ENABLED
        config.MCP_ENABLED = False
        try:
            from services.mcp_appointment_backend import MCPAppointmentBackend
            result = MCPAppointmentBackend.try_create({})
            self.assertIsNone(result)
        finally:
            config.MCP_ENABLED = saved

    def test_try_create_returns_none_without_user(self):
        """Without a valid user/session, should return None."""
        from services.mcp_appointment_backend import MCPAppointmentBackend
        result = MCPAppointmentBackend.try_create({"thread_id": ""})
        self.assertIsNone(result)

    def test_tool_mapper_prefers_hospital_mapping_for_nonstandard_name(self):
        from mcp_integration.tool_mapping import MCPAppointmentToolMapper

        tool = FakeMCPTool("xiehe__create_registration_order", {"appointment_no": "APT001"})
        mapper = MCPAppointmentToolMapper(
            mapping={"xiehe": {"book_appointment": "create_registration_order"}}
        )

        resolution = mapper.find_tool([tool], "book_appointment")

        self.assertIsNotNone(resolution)
        self.assertEqual(resolution.tool, tool)
        self.assertEqual(resolution.source, "mapping")

    def test_tool_mapper_uses_exact_alias_not_substring(self):
        from mcp_integration.tool_mapping import MCPAppointmentToolMapper

        misleading = FakeMCPTool("xiehe__cancelled_invoice_lookup")
        mapper = MCPAppointmentToolMapper()

        self.assertIsNone(mapper.find_tool([misleading], "cancel_appointment"))

    def test_tool_mapper_keeps_legacy_alias_as_exact_fallback(self):
        from mcp_integration.tool_mapping import MCPAppointmentToolMapper

        tool = FakeMCPTool("xiehe__get_availability", [{"schedule_id": 1}])
        mapper = MCPAppointmentToolMapper()

        resolution = mapper.find_tool([tool], "search_schedules")

        self.assertIsNotNone(resolution)
        self.assertEqual(resolution.tool, tool)
        self.assertEqual(resolution.source, "alias")

    def test_tool_mapper_respects_preferred_hospital_code(self):
        from mcp_integration.tool_mapping import MCPAppointmentToolMapper

        xiehe_tool = FakeMCPTool("xiehe__book_appointment", {"appointment_no": "XH001"})
        ruijin_tool = FakeMCPTool("ruijin__book_appointment", {"appointment_no": "RJ001"})
        mapper = MCPAppointmentToolMapper()

        resolution = mapper.find_tool(
            [xiehe_tool, ruijin_tool],
            "book_appointment",
            preferred_hospital_code="ruijin",
        )

        self.assertIsNotNone(resolution)
        self.assertEqual(resolution.tool, ruijin_tool)

    def test_tool_mapper_does_not_fallback_to_other_hospital_when_preferred_missing(self):
        from mcp_integration.tool_mapping import MCPAppointmentToolMapper

        ruijin_tool = FakeMCPTool("ruijin__book_appointment", {"appointment_no": "RJ001"})
        mapper = MCPAppointmentToolMapper()

        resolution = mapper.find_tool(
            [ruijin_tool],
            "book_appointment",
            preferred_hospital_code="xiehe",
        )

        self.assertIsNone(resolution)

    def test_backend_calls_mapped_nonstandard_tool(self):
        from mcp_integration.tool_mapping import MCPAppointmentToolMapper
        from services.mcp_appointment_backend import MCPAppointmentBackend

        tool = FakeMCPTool("xiehe__create_registration_order", {"appointment_no": "APT001"})
        backend = MCPAppointmentBackend(
            FakeMCPPool([tool]),
            "user-1",
            tool_mapper=MCPAppointmentToolMapper(
                mapping={"xiehe": {"book_appointment": "create_registration_order"}}
            ),
        )

        result, error = backend.book_appointment(
            {"department": "心内科", "date": "2026-06-11", "time_slot": "morning"}
        )

        self.assertIsNone(error)
        self.assertEqual(result["appointment_no"], "APT001")
        self.assertTrue(tool.called)

    def test_backend_calls_only_selected_hospital_tool(self):
        from services.mcp_appointment_backend import MCPAppointmentBackend

        xiehe_tool = FakeMCPTool("xiehe__book_appointment", {"appointment_no": "XH001"})
        ruijin_tool = FakeMCPTool("ruijin__book_appointment", {"appointment_no": "RJ001"})
        backend = MCPAppointmentBackend(
            FakeMCPPool([xiehe_tool, ruijin_tool], connected=["xiehe", "ruijin"]),
            "user-1",
            preferred_hospital_code="ruijin",
        )

        result, error = backend.book_appointment(
            {"department": "心内科", "date": "2026-06-11", "time_slot": "morning"}
        )

        self.assertIsNone(error)
        self.assertEqual(result["appointment_no"], "RJ001")
        self.assertFalse(xiehe_tool.called)
        self.assertTrue(ruijin_tool.called)

    def test_backend_does_not_fallback_when_selected_hospital_lacks_tool(self):
        from services.mcp_appointment_backend import MCPAppointmentBackend

        ruijin_tool = FakeMCPTool("ruijin__book_appointment", {"appointment_no": "RJ001"})
        backend = MCPAppointmentBackend(
            FakeMCPPool([ruijin_tool], connected=["xiehe", "ruijin"]),
            "user-1",
            preferred_hospital_code="xiehe",
        )

        result, error = backend.book_appointment(
            {"department": "心内科", "date": "2026-06-11", "time_slot": "morning"}
        )

        self.assertIsNone(result)
        self.assertIn("不支持", error)
        self.assertFalse(ruijin_tool.called)

    def test_try_create_accepts_nonstandard_tool_when_mapping_exists(self):
        from mcp_integration.tool_mapping import MCPAppointmentToolMapper
        from services.mcp_appointment_backend import MCPAppointmentBackend

        tool = FakeMCPTool("xiehe__create_registration_order", {"appointment_no": "APT001"})
        mapper = MCPAppointmentToolMapper(
            mapping={"xiehe": {"book_appointment": "create_registration_order"}}
        )

        result = MCPAppointmentBackend.try_create(
            {},
            pool=FakeMCPPool([tool]),
            user_id="user-1",
            tool_mapper=mapper,
        )

        self.assertIsNotNone(result)


class TestMCPHospitalSelection(unittest.TestCase):
    def test_single_connected_hospital_auto_selected(self):
        from mcp_integration.hospital_selection import MCPHospitalSelectionPolicy

        policy = MCPHospitalSelectionPolicy(hospital_lookup=FakeRegistry().get_by_code)
        selection = policy.select(
            user_query="帮我挂明天上午心内科",
            appointment_context={},
            connected_hospital_codes=["xiehe"],
        )

        self.assertFalse(selection.needs_clarification)
        self.assertEqual(selection.selected_code, "xiehe")

    def test_query_alias_selects_hospital(self):
        from mcp_integration.hospital_selection import MCPHospitalSelectionPolicy

        policy = MCPHospitalSelectionPolicy(
            aliases={"xiehe": ["北京协和", "PUMCH"]},
            hospital_lookup=FakeRegistry().get_by_code,
        )
        selection = policy.select(
            user_query="帮我挂北京协和明天上午心内科",
            appointment_context={},
            connected_hospital_codes=["xiehe", "ruijin"],
        )

        self.assertFalse(selection.needs_clarification)
        self.assertEqual(selection.selected_code, "xiehe")

    def test_short_alias_only_selects_when_query_is_exact_reply(self):
        from mcp_integration.hospital_selection import MCPHospitalSelectionPolicy

        policy = MCPHospitalSelectionPolicy(
            aliases={"xiehe": ["协和"]},
            hospital_lookup=FakeRegistry().get_by_code,
        )

        sentence = policy.select(
            user_query="帮我挂协和明天上午心内科",
            appointment_context={},
            connected_hospital_codes=["xiehe", "ruijin"],
        )
        exact_reply = policy.select(
            user_query="协和",
            appointment_context={},
            connected_hospital_codes=["xiehe", "ruijin"],
        )

        self.assertFalse(sentence.needs_clarification)
        self.assertTrue(sentence.needs_confirmation)
        self.assertEqual(sentence.selected_code, "xiehe")
        self.assertFalse(exact_reply.needs_clarification)
        self.assertFalse(exact_reply.needs_confirmation)
        self.assertEqual(exact_reply.selected_code, "xiehe")

    def test_full_hospital_name_selects_hospital(self):
        from mcp_integration.hospital_selection import MCPHospitalSelectionPolicy

        policy = MCPHospitalSelectionPolicy(hospital_lookup=FakeRegistry().get_by_code)
        selection = policy.select(
            user_query="帮我挂协和医院明天上午心内科",
            appointment_context={},
            connected_hospital_codes=["xiehe", "ruijin"],
        )

        self.assertFalse(selection.needs_clarification)
        self.assertEqual(selection.selected_code, "xiehe")

    def test_ascii_code_requires_token_boundary(self):
        from mcp_integration.hospital_selection import MCPHospitalSelectionPolicy

        policy = MCPHospitalSelectionPolicy(hospital_lookup=FakeRegistry().get_by_code)
        no_match = policy.select(
            user_query="帮我挂 xiehe2 明天上午心内科",
            appointment_context={},
            connected_hospital_codes=["xiehe", "ruijin"],
        )
        matched = policy.select(
            user_query="帮我挂 xiehe 明天上午心内科",
            appointment_context={},
            connected_hospital_codes=["xiehe", "ruijin"],
        )

        self.assertTrue(no_match.needs_clarification)
        self.assertFalse(matched.needs_clarification)
        self.assertEqual(matched.selected_code, "xiehe")

    def test_multiple_hospital_mentions_need_clarification(self):
        from mcp_integration.hospital_selection import MCPHospitalSelectionPolicy

        policy = MCPHospitalSelectionPolicy(hospital_lookup=FakeRegistry().get_by_code)
        selection = policy.select(
            user_query="协和医院和瑞金医院哪个都有号",
            appointment_context={},
            connected_hospital_codes=["xiehe", "ruijin"],
        )

        self.assertTrue(selection.needs_clarification)
        self.assertEqual([item.code for item in selection.candidates], ["xiehe", "ruijin"])

    def test_multiple_connected_without_query_match_needs_clarification(self):
        from mcp_integration.hospital_selection import MCPHospitalSelectionPolicy

        policy = MCPHospitalSelectionPolicy(hospital_lookup=FakeRegistry().get_by_code)
        selection = policy.select(
            user_query="帮我挂明天上午心内科",
            appointment_context={},
            connected_hospital_codes=["xiehe", "ruijin"],
        )

        self.assertTrue(selection.needs_clarification)
        self.assertEqual([item.code for item in selection.candidates], ["xiehe", "ruijin"])

    def test_existing_context_hospital_is_reused(self):
        from mcp_integration.hospital_selection import MCPHospitalSelectionPolicy

        policy = MCPHospitalSelectionPolicy(hospital_lookup=FakeRegistry().get_by_code)
        selection = policy.select(
            user_query="确认预约",
            appointment_context={"hospital_code": "ruijin", "hospital_name": "瑞金医院"},
            connected_hospital_codes=["xiehe", "ruijin"],
        )

        self.assertFalse(selection.needs_clarification)
        self.assertEqual(selection.selected_code, "ruijin")

    def test_appointment_node_clarifies_when_multiple_hospitals_unselected(self):
        from rag_agent.appointment_nodes import handle_appointment_skill

        class NoCallLLM:
            def invoke(self, _messages):
                raise AssertionError("LLM should not run before hospital selection")

            def with_config(self, **_kwargs):
                return self

            def bind_tools(self, _tools):
                return self

        result = handle_appointment_skill(
            {
                "messages": [HumanMessage(content="帮我挂明天上午心内科")],
                "thread_id": "t-1",
                "user_id": "user-1",
                "_mcp_pool": FakeMCPPool(
                    [
                        FakeMCPTool("xiehe__search_schedules"),
                        FakeMCPTool("ruijin__search_schedules"),
                    ],
                    connected=["xiehe", "ruijin"],
                ),
                "appointment_context": {},
                "pending_action_type": "",
                "pending_candidates": [],
            },
            NoCallLLM(),
            object(),
        )

        self.assertEqual(result["appointment_skill_mode"], "select_hospital")
        self.assertIn("协和医院", result["messages"][0].content)
        self.assertIn("瑞金医院", result["messages"][0].content)

    def test_appointment_node_confirms_short_hospital_alias_before_calling_llm(self):
        import config
        from rag_agent.appointment_nodes import handle_appointment_skill

        class NoCallLLM:
            def invoke(self, _messages):
                raise AssertionError("LLM should not run before hospital confirmation")

            def with_config(self, **_kwargs):
                return self

            def bind_tools(self, _tools):
                return self

        saved_aliases = config.MCP_HOSPITAL_ALIASES
        config.MCP_HOSPITAL_ALIASES = {"xiehe": ["协和"]}
        try:
            result = handle_appointment_skill(
                {
                    "messages": [HumanMessage(content="帮我挂协和明天上午心内科")],
                    "thread_id": "t-1",
                    "user_id": "user-1",
                    "_mcp_pool": FakeMCPPool(
                        [
                            FakeMCPTool("xiehe__search_schedules"),
                            FakeMCPTool("ruijin__search_schedules"),
                        ],
                        connected=["xiehe", "ruijin"],
                    ),
                    "appointment_context": {},
                    "pending_action_type": "",
                    "pending_candidates": [],
                },
                NoCallLLM(),
                object(),
            )
        finally:
            config.MCP_HOSPITAL_ALIASES = saved_aliases

        self.assertEqual(result["appointment_skill_mode"], "confirm_hospital")
        self.assertEqual(result["appointment_context"]["pending_hospital_code"], "xiehe")
        self.assertNotIn("hospital_code", result["appointment_context"])
        self.assertIn("确认医院", result["messages"][0].content)

    def test_hospital_confirmation_promotes_pending_hospital_to_selected(self):
        from rag_agent.appointment_nodes import _resolve_hospital_selection

        context, selection, message = _resolve_hospital_selection(
            {
                "user_id": "user-1",
                "_mcp_pool": FakeMCPPool([], connected=["xiehe", "ruijin"]),
            },
            "确认医院",
            {
                "pending_hospital_code": "xiehe",
                "pending_hospital_name": "协和医院",
            },
            {},
        )

        self.assertEqual(message, "")
        self.assertIsNone(selection)
        self.assertEqual(context["hospital_code"], "xiehe")
        self.assertEqual(context["hospital_name"], "协和医院")
        self.assertNotIn("pending_hospital_code", context)


if __name__ == "__main__":
    unittest.main()
