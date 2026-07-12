import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "project"))

from core.medical_source_ingest import NhcPdfWhitelistImporter  # noqa: E402

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"


class FakeNhcImporter(NhcPdfWhitelistImporter):
    def __init__(self, manifest_path):
        super().__init__(manifest_path=manifest_path)
        self.downloaded_urls = []

    def _download_pdf_bytes(self, entry: dict) -> bytes:
        self.downloaded_urls.append(entry["pdf_url"])
        return b"%PDF-1.4 fake"

    def _convert_pdf_bytes_to_markdown(self, pdf_bytes: bytes, stem: str):
        return (
            "第一段内容\n\n第二段内容",
            SimpleNamespace(method_used="plain_text_fallback", extracted_char_count=12, scan_like=False, warnings=["fallback-used"]),
        )


class NhcPdfImporterTests(unittest.TestCase):
    def test_load_manifest_reads_entries(self):
        importer = NhcPdfWhitelistImporter(manifest_path=FIXTURES_DIR / "nhc_sample_manifest.json")

        entries = importer.load_manifest()

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["id"], "nhc-sample-guideline")
        self.assertTrue(entries[0]["pdf_url"].startswith("https://"))

    def test_import_whitelist_writes_wrapped_markdown(self):
        importer = FakeNhcImporter(manifest_path=FIXTURES_DIR / "nhc_sample_manifest.json")

        with tempfile.TemporaryDirectory(prefix="nhc-import-") as temp_dir:
            result = importer.import_whitelist(temp_dir, limit=1, overwrite=False)
            output_path = Path(temp_dir) / "nhc-sample-guideline.md"
            content = output_path.read_text(encoding="utf-8")

        self.assertEqual(result.downloaded, 1)
        self.assertEqual(result.written, 1)
        self.assertEqual(result.failed, 0)
        self.assertEqual(result.skipped, 0)
        self.assertEqual(len(result.conversion_details), 1)
        self.assertIn("plain_text_fallback", result.conversion_details[0])
        self.assertIn("Source: 国家卫生健康委员会", content)
        self.assertIn("样例诊疗指南", content)
        self.assertIn("Published At:", content)
        self.assertIn("Fetched At:", content)
        self.assertIn("Freshness Bucket:", content)
        self.assertIn("第一段内容", content)
        self.assertEqual(importer.downloaded_urls, ["https://www.nhc.gov.cn/sample/files/sample.pdf"])


if __name__ == "__main__":
    unittest.main()
