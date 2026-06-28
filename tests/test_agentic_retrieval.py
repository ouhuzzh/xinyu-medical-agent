"""Tests for P1 agentic retrieval loop: evaluate_evidence + route_after_evidence."""

import unittest
from project.rag_agent.graph_state import AgentState


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


if __name__ == "__main__":
    unittest.main()
