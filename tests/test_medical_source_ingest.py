import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "project"))

from core.medical_source_ingest import MedlinePlusXmlImporter, html_to_markdown  # noqa: E402

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"


class FakeResponse:
    def __init__(self, text="", content=b""):
        self.text = text
        self.content = content

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, timeout=30):
        self.calls.append(url)
        return self.responses[url]


class MedicalSourceIngestTests(unittest.TestCase):
    def test_html_to_markdown_keeps_basic_structure(self):
        markdown = html_to_markdown("<p>Hello</p><ul><li>One</li><li>Two</li></ul>")

        self.assertIn("Hello", markdown)
        self.assertIn("- One", markdown)
        self.assertIn("- Two", markdown)

    def test_parse_topics_extracts_structured_documents(self):
        xml_text = (FIXTURES_DIR / "medlineplus_sample.xml").read_text(encoding="utf-8")
        importer = MedlinePlusXmlImporter()

        topics = importer.parse_topics(xml_text, limit=1)

        self.assertEqual(len(topics), 1)
        self.assertEqual(topics[0].title, "Asthma")
        self.assertIn("Respiratory Diseases", topics[0].categories)
        self.assertIn("Reactive airway disease", topics[0].related_terms)
        self.assertIn("wheezing", topics[0].body_markdown)

    def test_import_latest_writes_markdown_from_discovered_zip(self):
        xml_text = (FIXTURES_DIR / "medlineplus_sample.xml").read_text(encoding="utf-8")
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("health_topics.xml", xml_text)

        index_html = """
        <html><body>
            <a href="/xml/medlineplus-health-topics.zip">Download XML</a>
        </body></html>
        """
        session = FakeSession(
            {
                "https://medlineplus.gov/xml.html": FakeResponse(text=index_html),
                "https://medlineplus.gov/xml/medlineplus-health-topics.zip": FakeResponse(content=archive_bytes.getvalue()),
            }
        )
        importer = MedlinePlusXmlImporter(session=session)

        with tempfile.TemporaryDirectory(prefix="medical-import-") as temp_dir:
            result = importer.import_latest(temp_dir, limit=2, overwrite=False)
            files = sorted(Path(temp_dir).glob("*.md"))

        self.assertEqual(result.downloaded, 2)
        self.assertEqual(result.written, 2)
        self.assertEqual(result.skipped, 0)
        self.assertEqual(len(files), 2)
        self.assertTrue(any("medlineplus-asthma.md" in str(path) for path in files))


if __name__ == "__main__":
    unittest.main()
