"""Tests for P1 agentic retrieval loop: evaluate_evidence + route_after_evidence."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from unittest.mock import MagicMock, patch  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from project.rag_agent.graph_state import AgentState  # noqa: E402
from project.rag_agent.rag_nodes import evaluate_evidence, orchestrator  # noqa: E402
from project.rag_agent.edges import route_after_evidence  # noqa: E402


class TestAgentStateFields(unittest.TestCase):
    def test_evidence_reflection_fields_exist(self):
        """AgentState must carry the four evidence-reflection fields."""
        defaults = AgentState.__annotations__
        for field in ("evidence_rounds", "evidence_critique", "last_refined_query", "refined_queries"):
            self.assertIn(field, defaults, f"AgentState missing field: {field}")

    def test_refined_queries_is_accumulating(self):
        """refined_queries accumulates across rounds (operator.add reducer)."""
        from project.rag_agent.graph_state import AgentState
        import typing
        hints = typing.get_type_hints(AgentState, include_extras=True)
        meta = typing.get_args(hints["refined_queries"])
        import operator
        self.assertIn(operator.add, meta, "refined_queries must use operator.add reducer")


def _make_state(messages, **extra):
    base = {
        "messages": messages,
        "question": "高血压合并痛风吃什么药安全",
        "query_plan": [],
        "evidence_rounds": 0,
        "evidence_critique": "",
        "last_refined_query": "",
        "refined_queries": [],
    }
    base.update(extra)
    return base


class TestEvaluateEvidenceFastPath(unittest.TestCase):
    def test_sufficient_evidence_skips_llm(self):
        """Strong graded evidence → check_sufficiency returns sufficient=True WITHOUT calling the LLM.

        This is the REAL fast path: the tool message carries genuine
        `Score:` / `Relevance Grade:` lines (the format tools.py:654-672 emits),
        the node parses them into graded Documents, and the rule check fires.
        No mocking of check_sufficiency — it must actually return sufficient.
        """
        tool_msg = ToolMessage(
            content=(
                "Parent ID: p_001\n"
                "File Name: hypertension_gout.md\n"
                "Source Title: 高血压合并痛风用药指南\n"
                "Source Type: medlineplus\n"
                "Score: 0.9100\n"
                "Relevance Grade: high\n"
                "Confidence Bucket: high\n"
                "Content: 高血压合并痛风患者应避免使用噻嗪类利尿剂，因可能加重高尿酸血症。"
            ),
            tool_call_id="1",
        )
        state = _make_state([tool_msg])

        llm = MagicMock()
        with patch("project.rag_agent.rag_nodes._structured_output_llm") as mock_so:
            result = evaluate_evidence(state, llm)
            mock_so.assert_not_called()  # fast path skips LLM

        self.assertTrue(result["evidence_sufficient"])
        self.assertEqual(result["evidence_critique"], "direct_evidence")
        self.assertEqual(result["evidence_rounds"], 0)  # no round counted on fast-path success
        self.assertEqual(result["last_refined_query"], "")


class TestEvaluateEvidenceReflection(unittest.TestCase):
    def test_insufficient_triggers_llm_and_records_refined_query(self):
        """Rule says insufficient → LLM reflects → refined_query recorded, round counted."""
        tool_msg = ToolMessage(content="[DOC1] 高血压常规用药包括ACEI。", tool_call_id="1")
        state = _make_state([tool_msg])

        llm = MagicMock()
        parser = MagicMock()
        from project.rag_agent.schemas import EvidenceSufficiency
        parser.invoke.return_value = EvidenceSufficiency(
            is_sufficient=False,
            reason="证据只覆盖高血压用药，未涉及与痛风的交互",
            retry_query="高血压 合并痛风 药物 相互作用 禁忌",
        )

        with patch("project.rag_agent.rag_nodes.check_sufficiency") as mock_check, \
             patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser) as mock_so:
            mock_check.return_value = {"is_sufficient": False, "reason": "weak", "retry_query": "x"}
            result = evaluate_evidence(state, llm)

            mock_so.assert_called_once()
        self.assertFalse(result["evidence_sufficient"])
        self.assertEqual(result["evidence_rounds"], 1)
        self.assertEqual(result["last_refined_query"], "高血压 合并痛风 药物 相互作用 禁忌")
        self.assertEqual(result["refined_queries"], ["高血压 合并痛风 药物 相互作用 禁忌"])
        self.assertIn("痛风", result["evidence_critique"])

    def test_llm_says_sufficient_reflection_path(self):
        """Loop-terminating reflection: rule says insufficient but LLM says sufficient.

        When the LLM verdict is sufficient, the node must NOT increment
        evidence_rounds, must NOT record a refined query, and must signal
        `evidence_sufficient=True` so the downstream edge terminates the loop.
        """
        tool_msg = ToolMessage(content="[DOC1] 高血压常规用药包括ACEI。", tool_call_id="1")
        state = _make_state([tool_msg], evidence_rounds=2)

        llm = MagicMock()
        parser = MagicMock()
        from project.rag_agent.schemas import EvidenceSufficiency
        parser.invoke.return_value = EvidenceSufficiency(
            is_sufficient=True,
            reason="证据充分",
            retry_query="",
        )

        with patch("project.rag_agent.rag_nodes.check_sufficiency") as mock_check, \
             patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser) as mock_so:
            mock_check.return_value = {"is_sufficient": False, "reason": "weak", "retry_query": "x"}
            result = evaluate_evidence(state, llm)

            mock_so.assert_called_once()

        self.assertTrue(result["evidence_sufficient"])
        self.assertEqual(result["evidence_rounds"], 2)  # unchanged, not incremented
        self.assertEqual(result["last_refined_query"], "")
        # No refined query recorded when sufficient.
        self.assertNotIn("refined_queries", result)
        self.assertEqual(result["evidence_critique"], "证据充分")


class TestEvaluateEvidenceLLMFailureFallback(unittest.TestCase):
    def test_llm_failure_falls_back_to_rule_retry_query(self):
        """If the LLM reflection returns insufficient with empty retry_query (parse failure default), fall back to rule's retry_query."""
        tool_msg = ToolMessage(content="[DOC1] 略", tool_call_id="1")
        state = _make_state([tool_msg])

        llm = MagicMock()
        parser = MagicMock()
        from project.rag_agent.schemas import EvidenceSufficiency
        parser.invoke.return_value = EvidenceSufficiency(is_sufficient=False, reason="", retry_query="")

        with patch("project.rag_agent.rag_nodes.check_sufficiency") as mock_check, \
             patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser):
            mock_check.return_value = {"is_sufficient": False, "reason": "weak", "retry_query": "高血压 痛风 医学资料"}
            result = evaluate_evidence(state, llm)

        self.assertEqual(result["last_refined_query"], "高血压 痛风 医学资料")
        self.assertFalse(result["evidence_sufficient"])


class TestRouteAfterEvidence(unittest.TestCase):
    def test_sufficient_routes_to_compress(self):
        state = _make_state([], evidence_sufficient=True, evidence_rounds=0)
        self.assertEqual(route_after_evidence(state), "should_compress_context")

    def test_insufficient_with_budget_routes_to_compress(self):
        """Insufficient but under round limit and novel query → loop back via compress."""
        import config
        state = _make_state(
            [],
            evidence_sufficient=False,
            evidence_rounds=config.MAX_EVIDENCE_ROUNDS - 1,
            last_refined_query="新检索式A",
            refined_queries=["新检索式A"],
        )
        self.assertEqual(route_after_evidence(state), "should_compress_context")

    def test_round_limit_reached_routes_to_fallback(self):
        import config
        state = _make_state(
            [],
            evidence_sufficient=False,
            evidence_rounds=config.MAX_EVIDENCE_ROUNDS,
            last_refined_query="q",
            refined_queries=["q"],
        )
        self.assertEqual(route_after_evidence(state), "fallback_response")

    def test_repeated_refined_query_routes_to_fallback(self):
        """Refined query repeats a prior one (no progress) → fallback."""
        state = _make_state(
            [],
            evidence_sufficient=False,
            evidence_rounds=1,
            last_refined_query="重复检索式",
            refined_queries=["重复检索式", "别的", "重复检索式"],
        )
        self.assertEqual(route_after_evidence(state), "fallback_response")

    def test_no_progress_when_refined_query_empty_routes_to_fallback(self):
        state = _make_state(
            [],
            evidence_sufficient=False,
            evidence_rounds=1,
            last_refined_query="",
            refined_queries=[],
        )
        self.assertEqual(route_after_evidence(state), "fallback_response")

    def test_repeated_refined_query_at_round_limit_routes_to_fallback(self):
        """Realistic round-2 scenario: refined query repeats an earlier one → fallback (no-progress guard now reachable before budget)."""
        import config
        state = _make_state(
            [],
            evidence_sufficient=False,
            evidence_rounds=config.MAX_EVIDENCE_ROUNDS,
            last_refined_query="查询A",
            refined_queries=["查询A", "查询B", "查询A"],  # latest "查询A" repeats the first
        )
        self.assertEqual(route_after_evidence(state), "fallback_response")


class TestOrchestratorRefinedQueryInjection(unittest.TestCase):
    def test_refined_query_is_injected_as_hint(self):
        """When last_refined_query is set, orchestrator appends a re-search hint."""
        from langchain_core.messages import HumanMessage
        from project.rag_agent.rag_nodes import orchestrator

        state = {
            "question": "高血压合并痛风吃什么药安全",
            "query_plan": ["高血压 痛风"],
            "last_refined_query": "高血压 合并痛风 药物 相互作用 禁忌",
            "evidence_critique": "证据未涉及与痛风的交互",
            "messages": [],
            "context_summary": "",
            "recent_context": "",
            "topic_focus": "",
            "user_memories": "",
        }

        llm_with_tools = MagicMock()
        response = MagicMock()
        response.tool_calls = [{"name": "search_child_chunks", "args": {"query": "高血压 合并痛风 药物 相互作用 禁忌"}, "id": "1"}]
        response.content = ""
        llm_with_tools.invoke.return_value = response

        result = orchestrator(state, llm_with_tools)

        # The injected hint should mention the critique and refined query.
        invoked_messages = llm_with_tools.invoke.call_args[0][0]
        joined = "\n".join(str(getattr(m, "content", "")) for m in invoked_messages)
        self.assertIn("高血压 合并痛风 药物 相互作用 禁忌", joined)
        self.assertIn("证据未涉及与痛风的交互", joined)
        # last_refined_query is cleared after injection.
        self.assertEqual(result.get("last_refined_query", ""), "")

    def test_no_injection_when_refined_query_absent(self):
        from project.rag_agent.rag_nodes import orchestrator

        state = {
            "question": "普通问题",
            "query_plan": [],
            "last_refined_query": "",
            "messages": [],
            "context_summary": "",
            "recent_context": "",
            "topic_focus": "",
            "user_memories": "",
        }
        llm_with_tools = MagicMock()
        response = MagicMock()
        response.tool_calls = []
        response.content = "answer"
        llm_with_tools.invoke.return_value = response

        result = orchestrator(state, llm_with_tools)
        joined = "\n".join(str(getattr(m, "content", "")) for m in llm_with_tools.invoke.call_args[0][0])
        self.assertNotIn("上一次检索证据不足", joined)
        self.assertEqual(result.get("last_refined_query", ""), "")

    def test_refined_query_injected_in_reuse_branch(self):
        """When messages already exist (multi-turn loop), the reuse branch also injects the hint and clears last_refined_query."""
        from langchain_core.messages import HumanMessage, AIMessage
        from project.rag_agent.rag_nodes import orchestrator

        state = {
            "question": "高血压合并痛风吃什么药安全",
            "query_plan": ["高血压 痛风"],
            "last_refined_query": "高血压 合并痛风 药物 禁忌",
            "evidence_critique": "证据偏离问题",
            "messages": [HumanMessage(content="高血压合并痛风吃什么药安全"), AIMessage(content="")],
            "context_summary": "",
            "recent_context": "",
            "topic_focus": "",
            "user_memories": "",
        }
        llm_with_tools = MagicMock()
        response = MagicMock()
        response.tool_calls = []
        response.content = "answer"
        llm_with_tools.invoke.return_value = response

        result = orchestrator(state, llm_with_tools)
        joined = "\n".join(str(getattr(m, "content", "")) for m in llm_with_tools.invoke.call_args[0][0])
        self.assertIn("高血压 合并痛风 药物 禁忌", joined)
        self.assertIn("证据偏离问题", joined)
        self.assertEqual(result.get("last_refined_query"), "")


class TestAgenticRetrievalLoopIntegration(unittest.TestCase):
    """End-to-end-ish: weak first retrieval → reflection → refined re-search → grounded answer.

    Manually threads state through evaluate_evidence + route_after_evidence to
    simulate the graph loop without a live LLM.  The REAL check_sufficiency
    rule is exercised in both rounds (no patching of the rule check), so the
    insufficient→sufficient transition is honest:
      - Round 1: tool message has no graded block → empty docs → real rule
        returns insufficient (no_relevant_documents) → LLM reflection (mocked)
        produces a refined retry query → edge loops back via compress.
      - Round 2: tool message carries a genuine graded block (Score 0.91 +
        high grade) → real rule returns sufficient via the fast path, with no
        LLM call → edge routes to compress → collect_answer.
    """

    def test_weak_first_then_refined_second_completes(self):
        from project.rag_agent.schemas import EvidenceSufficiency

        # Round 1: weak evidence (no graded block → empty docs → real
        # check_sufficiency returns insufficient), triggering LLM reflection
        # that produces a refined retry query.
        state1 = _make_state(
            [ToolMessage(content="[DOC1] 高血压常规用药ACEI。", tool_call_id="1")],
            evidence_rounds=0,
        )
        parser = MagicMock()
        parser.invoke.return_value = EvidenceSufficiency(
            is_sufficient=False,
            reason="未覆盖痛风交互",
            retry_query="高血压 合并痛风 药物 相互作用 禁忌",
        )
        with patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser) as mock_so1:
            delta1 = evaluate_evidence(state1, MagicMock())
            mock_so1.assert_called_once()  # reflection path invoked the LLM

        state1.update(delta1)
        # Insufficient + novel query + under budget → loop back via compress.
        self.assertFalse(state1["evidence_sufficient"])
        self.assertEqual(state1["evidence_rounds"], 1)
        self.assertEqual(state1["last_refined_query"], "高血压 合并痛风 药物 相互作用 禁忌")
        self.assertEqual(route_after_evidence(state1), "should_compress_context")

        # Round 2: refined retrieval yields strong evidence.  We do NOT patch
        # check_sufficiency — the tool message carries a genuine graded block
        # (Score 0.91 >= RAG_DIRECT_EVIDENCE_SCORE 0.84 + high grade), so the
        # REAL rule check must return sufficient via the fast path, without
        # any LLM call.  This honestly verifies the fast path.
        state2 = dict(state1)
        state2["messages"] = [ToolMessage(
            content="Score: 0.9100\nRelevance Grade: high\nContent: 高血压合并痛风者禁用噻嗪类利尿剂，首选CCB；避免非甾体抗炎药。",
            tool_call_id="2",
        )]
        llm2 = MagicMock()
        with patch("project.rag_agent.rag_nodes._structured_output_llm") as mock_so2:
            delta2 = evaluate_evidence(state2, llm2)
            mock_so2.assert_not_called()  # fast path skips LLM
        state2.update(delta2)
        self.assertTrue(state2["evidence_sufficient"])
        self.assertEqual(state2["evidence_critique"], "direct_evidence")
        self.assertEqual(state2["last_refined_query"], "")
        # sufficient → compress → collect
        self.assertEqual(route_after_evidence(state2), "should_compress_context")

    def test_loop_terminates_on_round_limit(self):
        """Two insufficient rounds with distinct queries hit MAX_EVIDENCE_ROUNDS → fallback."""
        import config

        state = _make_state(
            [ToolMessage(content="[DOC] 略", tool_call_id="1")],
            evidence_rounds=config.MAX_EVIDENCE_ROUNDS,  # already at limit
            last_refined_query="某检索式",
            refined_queries=["别的", "某检索式"],
            evidence_sufficient=False,
        )
        self.assertEqual(route_after_evidence(state), "fallback_response")


class TestCompiledGraphStateHandoff(unittest.TestCase):
    """Verify the evaluate_evidence -> route_after_evidence handoff survives LangGraph's real state machinery.

    The node-level tests simulate the loop via state.update(delta); this compiles an
    actual StateGraph and invokes it, proving the declared `evidence_sufficient` field
    (and friends) persist from node to edge through LangGraph's channels/reducers.
    """
    def _build_graph(self, llm):
        from langgraph.graph import StateGraph, START, END
        from project.rag_agent.graph_state import AgentState
        from project.rag_agent.rag_nodes import evaluate_evidence
        from project.rag_agent.edges import route_after_evidence
        from functools import partial

        builder = StateGraph(AgentState)
        builder.add_node("evaluate_evidence", partial(evaluate_evidence, llm=llm))
        # Sink nodes record which branch the edge routed to.
        sink = {"hit": None}

        def _sink_compress(state):
            sink["hit"] = "should_compress_context"
            return {}

        def _sink_fallback(state):
            sink["hit"] = "fallback_response"
            return {}

        builder.add_node("should_compress_context", _sink_compress)
        builder.add_node("fallback_response", _sink_fallback)
        builder.add_edge(START, "evaluate_evidence")
        builder.add_conditional_edges(
            "evaluate_evidence",
            route_after_evidence,
            {"should_compress_context": "should_compress_context", "fallback_response": "fallback_response"},
        )
        builder.add_edge("should_compress_context", END)
        builder.add_edge("fallback_response", END)
        return builder.compile(), sink

    def test_strong_evidence_routes_to_compress_through_real_graph(self):
        """Fast path: a strong-evidence tool message → evaluate_evidence writes evidence_sufficient=True
        → route_after_evidence reads it via real LangGraph state → routes to should_compress_context."""
        from langchain_core.messages import ToolMessage
        from unittest.mock import MagicMock

        graph, sink = self._build_graph(MagicMock())
        state = _make_state([
            ToolMessage(content="Score: 0.9100\nRelevance Grade: high\nContent: 高血压合并痛风者禁用噻嗪类利尿剂。", tool_call_id="1"),
        ])
        # No LLM mock needed: fast path skips _structured_output_llm entirely.
        graph.invoke(state, {"recursion_limit": 10})
        self.assertEqual(sink["hit"], "should_compress_context")

    def test_insufficient_with_refined_query_routes_to_compress_through_real_graph(self):
        """Reflection path: weak evidence → LLM says insufficient + gives refined query
        → route_after_evidence reads evidence_sufficient=False + last_refined_query → routes to should_compress_context."""
        from langchain_core.messages import ToolMessage
        from unittest.mock import MagicMock, patch
        from project.rag_agent.schemas import EvidenceSufficiency

        parser = MagicMock()
        parser.invoke.return_value = EvidenceSufficiency(
            is_sufficient=False,
            reason="未覆盖痛风交互",
            retry_query="高血压 合并痛风 药物 禁忌",
        )
        with patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser):
            graph, sink = self._build_graph(MagicMock())
            state = _make_state([
                ToolMessage(content="[DOC1] 高血压常规用药ACEI。", tool_call_id="1"),
            ], evidence_rounds=0)
            graph.invoke(state, {"recursion_limit": 10})
        self.assertEqual(sink["hit"], "should_compress_context")

    def test_round_limit_routes_to_fallback_through_real_graph(self):
        """Termination: at the round limit with insufficient evidence → route_after_evidence → fallback_response."""
        import config
        from langchain_core.messages import ToolMessage
        from unittest.mock import MagicMock, patch
        from project.rag_agent.schemas import EvidenceSufficiency

        parser = MagicMock()
        parser.invoke.return_value = EvidenceSufficiency(
            is_sufficient=False, reason="仍不足", retry_query="又一次查询"
        )
        with patch("project.rag_agent.rag_nodes._structured_output_llm", return_value=parser):
            graph, sink = self._build_graph(MagicMock())
            state = _make_state([
                ToolMessage(content="[DOC] 略", tool_call_id="1"),
            ], evidence_rounds=config.MAX_EVIDENCE_ROUNDS, refined_queries=["前一个查询"])
            graph.invoke(state, {"recursion_limit": 10})
        self.assertEqual(sink["hit"], "fallback_response")


if __name__ == "__main__":
    unittest.main()
