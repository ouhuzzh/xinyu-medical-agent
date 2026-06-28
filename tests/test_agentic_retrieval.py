"""Tests for P1 agentic retrieval loop: evaluate_evidence + route_after_evidence."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from unittest.mock import MagicMock, patch  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from project.rag_agent.graph_state import AgentState  # noqa: E402
from project.rag_agent.rag_nodes import evaluate_evidence  # noqa: E402


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
        """Rule path returns sufficient=True WITHOUT calling the LLM."""
        tool_msg = ToolMessage(
            content="[DOC1] 高血压合并痛风患者应避免使用噻嗪类利尿剂，因可能加重高尿酸血症。",
            tool_call_id="1",
        )
        state = _make_state([tool_msg])

        llm = MagicMock()
        with patch("project.rag_agent.rag_nodes.check_sufficiency") as mock_check:
            mock_check.return_value = {"is_sufficient": True, "reason": "direct_evidence", "retry_query": ""}
            with patch("project.rag_agent.rag_nodes._structured_output_llm") as mock_so:
                result = evaluate_evidence(state, llm)
                mock_so.assert_not_called()  # fast path skips LLM

        self.assertEqual(result["evidence_critique"], "direct_evidence")
        self.assertEqual(result["evidence_rounds"], 0)  # no round counted on fast-path success


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
        self.assertFalse(result["_evidence_sufficient"])
        self.assertEqual(result["evidence_rounds"], 1)
        self.assertEqual(result["last_refined_query"], "高血压 合并痛风 药物 相互作用 禁忌")
        self.assertEqual(result["refined_queries"], ["高血压 合并痛风 药物 相互作用 禁忌"])
        self.assertIn("痛风", result["evidence_critique"])


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
        self.assertFalse(result["_evidence_sufficient"])


if __name__ == "__main__":
    unittest.main()
