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


if __name__ == "__main__":
    unittest.main()
