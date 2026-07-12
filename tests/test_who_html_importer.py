import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "project"))

from core.medical_source_ingest import WhoHtmlWhitelistImporter  # noqa: E402

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"


class FakeWhoImporter(WhoHtmlWhitelistImporter):
    def __init__(self, manifest_path, html_text):
        super().__init__(manifest_path=manifest_path)
        self.html_text = html_text
        self.downloaded_urls = []

    def _download_html(self, entry: dict) -> str:
        self.downloaded_urls.append(entry["url"])
        return self.html_text


class WhoHtmlImporterTests(unittest.TestCase):
    def test_load_manifest_reads_entries(self):
        importer = WhoHtmlWhitelistImporter(manifest_path=FIXTURES_DIR / "who_sample_manifest.json")

        entries = importer.load_manifest()

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["id"], "who-sample-topic")
        self.assertTrue(entries[0]["url"].startswith("https://"))

    def test_import_whitelist_writes_markdown_from_article(self):
        html_text = (FIXTURES_DIR / "who_sample_article.html").read_text(encoding="utf-8")
        importer = FakeWhoImporter(FIXTURES_DIR / "who_sample_manifest.json", html_text)

        with tempfile.TemporaryDirectory(prefix="who-import-") as temp_dir:
            result = importer.import_whitelist(temp_dir, limit=1, overwrite=False)
            output_path = Path(temp_dir) / "who-sample-topic.md"
            content = output_path.read_text(encoding="utf-8")

        self.assertEqual(result.downloaded, 1)
        self.assertEqual(result.written, 1)
        self.assertEqual(result.failed, 0)
        self.assertIn("Source: World Health Organization", content)
        self.assertIn("Sample WHO topic", content)
        self.assertIn("Published At:", content)
        self.assertIn("Fetched At:", content)
        self.assertIn("Freshness Bucket:", content)
        self.assertIn("Symptom one", content)
        self.assertEqual(importer.downloaded_urls, ["https://www.who.int/news-room/fact-sheets/detail/sample-topic"])


if __name__ == "__main__":
    unittest.main()
