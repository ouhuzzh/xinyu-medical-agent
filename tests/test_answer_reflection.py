"""Tests for P2 answer reflection loop: revise_answer + route_after_grounding."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from unittest.mock import MagicMock, patch  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage  # noqa: E402
from langchain_core.documents import Document  # noqa: E402


class TestConfigFields(unittest.TestCase):
    def test_grounding_config_fields_exist(self):
        import config
        self.assertTrue(hasattr(config, "MAX_GROUNDING_ROUNDS"))
        self.assertEqual(config.MAX_GROUNDING_ROUNDS, 1)
        self.assertTrue(hasattr(config, "ENABLE_ANSWER_REFLECTION"))
        self.assertTrue(config.ENABLE_ANSWER_REFLECTION)


def _make_main_state(messages, **extra):
    base = {
        "messages": messages,
        "originalQuery": "高血压合并痛风吃什么药安全",
        "agent_answers": [],
        "grounding_evidence_score": None,
        "grounding_rounds": 0,
        "grounding_critique": "",
        "grounding_passed": False,
    }
    base.update(extra)
    return base


if __name__ == "__main__":
    unittest.main()
