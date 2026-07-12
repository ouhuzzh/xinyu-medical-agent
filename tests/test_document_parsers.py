import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "project"))

from core.document_parsers import supported_upload_extensions, unstructured_to_markdown  # noqa: E402


class DocumentParserTests(unittest.TestCase):
    def setUp(self):
        temp_root = PROJECT_ROOT / "runtime" / "test_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = str(temp_root / f"doc-parser-{uuid.uuid4().hex}")
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_supported_upload_extensions_include_common_unstructured_formats(self):
        extensions = supported_upload_extensions()

        self.assertIn(".pdf", extensions)
        self.assertIn(".md", extensions)
        self.assertIn(".docx", extensions)
        self.assertIn(".pptx", extensions)
        self.assertIn(".html", extensions)
        self.assertIn(".txt", extensions)

    def test_txt_falls_back_when_unstructured_is_not_installed(self):
        source = Path(self.temp_dir) / "care.txt"
        source.write_text("高血压患者应注意低盐饮食。", encoding="utf-8")

        with mock.patch.dict("sys.modules", {"unstructured.partition.auto": None}):
            result = unstructured_to_markdown(source, self.temp_dir)

        self.assertEqual(result.method_used, "plain_text_fallback")
        self.assertTrue(result.output_path.exists())
        self.assertIn("# care", result.output_path.read_text(encoding="utf-8"))
        self.assertIn("高血压患者", result.output_path.read_text(encoding="utf-8"))

    def test_html_falls_back_when_unstructured_is_not_installed(self):
        source = Path(self.temp_dir) / "care.html"
        source.write_text("<html><body><h1>护理建议</h1><p>规律监测血压。</p></body></html>", encoding="utf-8")

        with mock.patch.dict("sys.modules", {"unstructured.partition.auto": None}):
            result = unstructured_to_markdown(source, self.temp_dir)

        self.assertEqual(result.method_used, "html_fallback")
        self.assertTrue(result.output_path.exists())
        self.assertIn("规律监测血压", result.output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
