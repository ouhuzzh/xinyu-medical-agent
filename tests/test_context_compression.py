"""Tests for token-aware context compression."""

from __future__ import annotations

import sys
import unittest
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "project"))

import config
from core.context_compression import (
    ContextCompressionService,
    should_compress_context,
    summarize_messages,
    trim_messages_to_token_budget,
)


class FakeLLM:
    def __init__(self, response_text="summary"):
        self._response = response_text

    def invoke(self, messages):
        return AIMessage(content=self._response)


class FakeSessionMemory:
    def __init__(self, messages):
        self._messages = list(messages)
        self.set_calls = []

    def get_recent_messages(self, thread_id):
        return list(self._messages)

    def set_recent_messages(self, thread_id, messages):
        self._messages = list(messages)
        self.set_calls.append((thread_id, len(messages)))


class FakeSummaryStore:
    def __init__(self):
        self.saved = []
        self._summary = ""

    def get_summary(self, thread_id):
        return self._summary

    def save_summary(self, thread_id, summary, last_message_index):
        self._summary = summary
        self.saved.append((thread_id, summary, last_message_index))


class TrimMessagesTests(unittest.TestCase):
    def test_trim_preserves_recent_turns_when_budget_too_small(self):
        messages = [
            HumanMessage(content="older question"),
            AIMessage(content="older answer"),
            HumanMessage(content="recent question"),
            AIMessage(content="recent answer"),
        ]
        result = trim_messages_to_token_budget(messages, max_tokens=1, preserve_recent_turns=1)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].content, "recent question")
        self.assertEqual(result[1].content, "recent answer")

    def test_no_trim_when_under_budget(self):
        messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
        result = trim_messages_to_token_budget(messages, max_tokens=1000, preserve_recent_turns=1)
        self.assertEqual(len(result), len(messages))
        self.assertEqual(result[0].content, "hi")

    def test_trim_keeps_as_many_messages_as_budget_allows(self):
        messages = [
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            HumanMessage(content="q2"),
            AIMessage(content="a2"),
        ]
        # Large enough for exactly two messages.
        result = trim_messages_to_token_budget(messages, max_tokens=2, preserve_recent_turns=1)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].content, "q2")
        self.assertEqual(result[1].content, "a2")


class ShouldCompressTests(unittest.TestCase):
    def test_message_ceiling_triggers_compression(self):
        with mock.patch.object(config, "SUMMARY_MAX_MESSAGE_CEILING", 4):
            self.assertTrue(should_compress_context([], recent_count=4))

    def test_token_threshold_triggers_compression(self):
        big = [HumanMessage(content="word " * 2000)]
        with mock.patch.object(config, "SUMMARY_TOKEN_THRESHOLD", 1):
            self.assertTrue(should_compress_context(big, recent_count=1))

    def test_under_threshold_returns_false(self):
        with mock.patch.object(config, "SUMMARY_TOKEN_THRESHOLD", 10000), \
             mock.patch.object(config, "SUMMARY_MAX_MESSAGE_CEILING", 100):
            self.assertFalse(should_compress_context([HumanMessage(content="hi")], recent_count=1))

    def test_deprecated_count_fallback(self):
        with mock.patch.object(config, "SUMMARY_TOKEN_THRESHOLD", 0), \
             mock.patch.object(config, "SUMMARY_REFRESH_THRESHOLD", 3):
            self.assertTrue(should_compress_context([], recent_count=3))


class SummarizeMessagesTests(unittest.TestCase):
    def test_returns_existing_summary_when_no_messages(self):
        result = summarize_messages([], "existing summary", FakeLLM())
        self.assertEqual(result, "existing summary")

    def test_uses_llm_when_messages_present(self):
        messages = [HumanMessage(content="hello")]
        result = summarize_messages(messages, "", FakeLLM("new summary"))
        self.assertEqual(result, "new summary")

    def test_llm_failure_returns_existing_summary(self):
        class BadLLM:
            def invoke(self, messages):
                raise RuntimeError("llm error")

        result = summarize_messages([HumanMessage(content="hello")], "old summary", BadLLM())
        self.assertEqual(result, "old summary")


class CompressionServiceIntegrationTests(unittest.TestCase):
    def test_compress_thread_summarizes_older_and_preserves_recent(self):
        messages = [
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            HumanMessage(content="q2"),
            AIMessage(content="a2"),
            HumanMessage(content="q3"),
            AIMessage(content="a3"),
        ]
        memory = FakeSessionMemory(messages)
        store = FakeSummaryStore()
        service = ContextCompressionService(llm=FakeLLM("new summary"))

        result = service.compress_thread(memory, store, "t1", preserve_recent_turns=1)

        self.assertTrue(result["compressed"])
        self.assertEqual(result["preserved_count"], 2)
        self.assertEqual(memory._messages[-1].content, "a3")
        self.assertEqual(len(store.saved), 1)
        self.assertEqual(store.saved[0][1], "new summary")

    def test_compress_thread_no_op_when_below_preserve_threshold(self):
        messages = [HumanMessage(content="q1"), AIMessage(content="a1")]
        memory = FakeSessionMemory(messages)
        store = FakeSummaryStore()
        service = ContextCompressionService(llm=FakeLLM("new summary"))

        result = service.compress_thread(memory, store, "t1", preserve_recent_turns=3)

        self.assertFalse(result["compressed"])
        self.assertEqual(result["reason"], "below_preserve_threshold")
        self.assertEqual(len(store.saved), 0)

    def test_hard_trim_keeps_recent_turns(self):
        messages = [
            HumanMessage(content="old " * 1000),
            AIMessage(content="old answer"),
            HumanMessage(content="recent question"),
            AIMessage(content="recent answer"),
        ]
        service = ContextCompressionService(llm=FakeLLM())
        with mock.patch.object(config, "CONTEXT_HARD_TRIM_THRESHOLD", 20), \
             mock.patch.object(config, "CONTEXT_HARD_TRIM_RESERVE", 5), \
             mock.patch.object(config, "RECENT_CONTEXT_TURNS", 1):
            result = service.hard_trim_messages_for_graph(messages)

        self.assertGreaterEqual(len(result), 2)
        self.assertEqual(result[-1].content, "recent answer")


if __name__ == "__main__":
    unittest.main()
