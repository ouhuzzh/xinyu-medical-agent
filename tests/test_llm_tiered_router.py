"""Tests for TieredLLMRouter and CircuitBreaker."""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is on sys.path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

from llm_tiered_router import CircuitBreaker, LLMTierConfig, TieredLLMRouter


class TestCircuitBreaker(unittest.TestCase):
    """Unit tests for the CircuitBreaker state machine."""

    def test_starts_closed(self):
        cb = CircuitBreaker()
        self.assertEqual(cb.state, "closed")
        self.assertTrue(cb.allow_request())

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        self.assertEqual(cb.state, "open")
        self.assertFalse(cb.allow_request())

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Should not open after one more failure (count was reset)
        cb.record_failure()
        self.assertEqual(cb.state, "closed")

    def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        cb.record_failure()
        cb.record_failure()
        self.assertEqual(cb.state, "open")

        # Wait for recovery timeout
        import time
        time.sleep(0.02)

        # Accessing state auto-transitions to half_open
        self.assertEqual(cb.state, "half_open")
        self.assertTrue(cb.allow_request())

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        cb.record_failure()
        cb.record_failure()

        import time
        time.sleep(0.02)
        _ = cb.state  # trigger half_open

        cb.record_success()
        self.assertEqual(cb.state, "closed")

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        cb.record_failure()
        cb.record_failure()

        import time
        time.sleep(0.02)
        _ = cb.state  # trigger half_open

        cb.record_failure()
        self.assertEqual(cb.state, "open")


class TestTieredLLMRouter(unittest.TestCase):
    """Unit tests for TieredLLMRouter."""

    def _make_router(self, tiers=None, fallback=None):
        if tiers is None:
            tiers = {
                "default": LLMTierConfig(
                    name="default", provider="deepseek",
                    model="Qwen/Qwen3-32B",
                ),
            }
        return TieredLLMRouter(tiers=tiers, fallback_provider=fallback)

    def test_single_tier_has_tiers_false(self):
        router = self._make_router()
        self.assertFalse(router.has_tiers)

    def test_multi_tier_has_tiers_true(self):
        router = self._make_router(tiers={
            "light": LLMTierConfig(name="light", provider="deepseek", model="small"),
            "strong": LLMTierConfig(name="strong", provider="deepseek", model="big"),
            "default": LLMTierConfig(name="default", provider="deepseek", model="big"),
        })
        self.assertTrue(router.has_tiers)

    @patch("model_factory.get_chat_model_for_tier")
    def test_get_llm_returns_model(self, mock_factory):
        mock_llm = MagicMock()
        mock_factory.return_value = mock_llm
        router = self._make_router()
        llm = router.get_llm("default")
        # May be wrapped in _CircuitBreakerWrapper, check invoke works
        self.assertIsNotNone(llm)

    @patch("config.LLM_TIERS_JSON", "")
    @patch("config.LLM_FALLBACK_PROVIDER", "")
    @patch("config.ACTIVE_LLM_PROVIDER", "deepseek")
    @patch("config.LLM_MODEL", "Qwen/Qwen3-32B")
    @patch("config.LLM_TEMPERATURE", 0.0)
    @patch("config.LLM_MAX_TOKENS", 2048)
    @patch("config.LLM_TIMEOUT_SECONDS", 45.0)
    def test_from_env_empty_json_single_tier(self):
        router = TieredLLMRouter.from_env()
        self.assertFalse(router.has_tiers)
        self.assertIn("default", router._tiers)

    @patch("config.LLM_TIERS_JSON", json.dumps([
        {"name": "light", "provider": "deepseek", "model": "Qwen/Qwen3-8B"},
        {"name": "strong", "provider": "deepseek", "model": "Qwen/Qwen3-32B"},
    ]))
    @patch("config.LLM_FALLBACK_PROVIDER", "openai")
    @patch("config.ACTIVE_LLM_PROVIDER", "deepseek")
    @patch("config.LLM_MODEL", "Qwen/Qwen3-32B")
    @patch("config.LLM_TEMPERATURE", 0.0)
    @patch("config.LLM_MAX_TOKENS", 2048)
    @patch("config.LLM_TIMEOUT_SECONDS", 45.0)
    def test_from_env_with_tiers_json(self):
        router = TieredLLMRouter.from_env()
        self.assertTrue(router.has_tiers)
        self.assertIn("light", router._tiers)
        self.assertIn("strong", router._tiers)
        self.assertIn("default", router._tiers)  # auto-added
        self.assertEqual(router._fallback_provider, "openai")

    @patch("config.LLM_TIERS_JSON", "not valid json{{{")
    @patch("config.LLM_FALLBACK_PROVIDER", "")
    @patch("config.ACTIVE_LLM_PROVIDER", "deepseek")
    @patch("config.LLM_MODEL", "Qwen/Qwen3-32B")
    @patch("config.LLM_TEMPERATURE", 0.0)
    @patch("config.LLM_MAX_TOKENS", 2048)
    @patch("config.LLM_TIMEOUT_SECONDS", 45.0)
    def test_from_env_invalid_json_falls_back(self):
        router = TieredLLMRouter.from_env()
        self.assertFalse(router.has_tiers)

    def test_get_status(self):
        router = self._make_router(fallback="openai")
        status = router.get_status()
        self.assertIn("tiers", status)
        self.assertIn("circuit_breakers", status)
        self.assertEqual(status["fallback_provider"], "openai")


if __name__ == "__main__":
    unittest.main()
