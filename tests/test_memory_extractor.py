"""Tests for MemoryExtractor — unit tests with mocked LLM and store."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))

from memory.memory_extractor import MemoryExtractor  # noqa: E402


class TestMemoryExtractorParseResponse(unittest.TestCase):
    """Test the JSON parsing logic."""

    def test_parse_valid_json_array(self):
        result = MemoryExtractor._parse_extraction_response(
            '[{"memory_type": "medical", "content": "对青霉素过敏", "importance": 9}]'
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["memory_type"], "medical")

    def test_parse_empty_array(self):
        result = MemoryExtractor._parse_extraction_response("[]")
        self.assertEqual(result, [])

    def test_parse_json_in_markdown_code_block(self):
        result = MemoryExtractor._parse_extraction_response(
            '```json\n[{"memory_type": "fact", "content": "用户是教师", "importance": 5}]\n```'
        )
        self.assertEqual(len(result), 1)

    def test_parse_json_with_surrounding_text(self):
        result = MemoryExtractor._parse_extraction_response(
            'Here are the memories:\n[{"memory_type": "preference", "content": "偏好简洁", "importance": 4}]'
        )
        self.assertEqual(len(result), 1)

    def test_parse_empty_string(self):
        result = MemoryExtractor._parse_extraction_response("")
        self.assertEqual(result, [])

    def test_parse_invalid_json(self):
        result = MemoryExtractor._parse_extraction_response("not json at all")
        self.assertEqual(result, [])


class TestMemoryExtractorExtractAndSave(unittest.TestCase):
    """Test the extract_and_save flow."""

    def _make_extractor(self, user_id="user1"):
        mock_store = MagicMock()
        mock_session_store = MagicMock()
        mock_session_store.get_session.return_value = {"owner_user_id": user_id}
        extractor = MemoryExtractor(mock_store, mock_session_store)
        return extractor, mock_store, mock_session_store

    def test_skips_when_disabled(self):
        with patch("memory.memory_extractor.config") as mock_config:
            mock_config.USER_MEMORY_ENABLED = False
            extractor, mock_store, _ = self._make_extractor()
            result = extractor.extract_and_save("t1", "hi", "hello")
            self.assertEqual(result, 0)
            mock_store.save_memory.assert_not_called()

    def test_skips_when_no_user_id(self):
        with patch("memory.memory_extractor.config") as mock_config:
            mock_config.USER_MEMORY_ENABLED = True
            mock_config.USER_MEMORY_EXTRACTION_ENABLED = True
            extractor, mock_store, _ = self._make_extractor(user_id="")
            result = extractor.extract_and_save("t1", "hi", "hello")
            self.assertEqual(result, 0)

    def test_handles_llm_failure_gracefully(self):
        with patch("memory.memory_extractor.config") as mock_config:
            mock_config.USER_MEMORY_ENABLED = True
            mock_config.USER_MEMORY_EXTRACTION_ENABLED = True
            mock_config.USER_MEMORY_IMPORTANCE_THRESHOLD = 4
            extractor, mock_store, _ = self._make_extractor()

            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("LLM unavailable")
            extractor._llm = mock_llm

            result = extractor.extract_and_save("t1", "我有高血压", "建议...")
            self.assertEqual(result, 0)

    def test_handles_empty_extraction_result(self):
        with patch("memory.memory_extractor.config") as mock_config:
            mock_config.USER_MEMORY_ENABLED = True
            mock_config.USER_MEMORY_EXTRACTION_ENABLED = True
            mock_config.USER_MEMORY_IMPORTANCE_THRESHOLD = 4
            extractor, mock_store, _ = self._make_extractor()

            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "[]"
            mock_llm.invoke.return_value = mock_response
            extractor._llm = mock_llm

            result = extractor.extract_and_save("t1", "好的", "收到")
            self.assertEqual(result, 0)
            mock_store.save_memory.assert_not_called()

    def test_saves_extracted_memories(self):
        with patch("memory.memory_extractor.config") as mock_config:
            mock_config.USER_MEMORY_ENABLED = True
            mock_config.USER_MEMORY_EXTRACTION_ENABLED = True
            mock_config.USER_MEMORY_IMPORTANCE_THRESHOLD = 4
            extractor, mock_store, _ = self._make_extractor()

            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = '[{"memory_type": "medical", "content": "有高血压", "importance": 8}]'
            mock_llm.invoke.return_value = mock_response
            extractor._llm = mock_llm

            result = extractor.extract_and_save("t1", "我有高血压", "建议...")
            self.assertEqual(result, 1)
            mock_store.save_memory.assert_called_once()
            call_kwargs = mock_store.save_memory.call_args[1]
            self.assertEqual(call_kwargs["memory_type"], "medical")
            self.assertEqual(call_kwargs["content"], "有高血压")
            self.assertEqual(call_kwargs["importance"], 8)

    def test_filters_low_importance(self):
        with patch("memory.memory_extractor.config") as mock_config:
            mock_config.USER_MEMORY_ENABLED = True
            mock_config.USER_MEMORY_EXTRACTION_ENABLED = True
            mock_config.USER_MEMORY_IMPORTANCE_THRESHOLD = 4
            extractor, mock_store, _ = self._make_extractor()

            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = '[{"memory_type": "preference", "content": "偏好短回答", "importance": 3}]'
            mock_llm.invoke.return_value = mock_response
            extractor._llm = mock_llm

            result = extractor.extract_and_save("t1", "简短点", "好的")
            self.assertEqual(result, 0)
            mock_store.save_memory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
