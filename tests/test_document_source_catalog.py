import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "project"))

from core.document_source_catalog import OFFICIAL_MEDICAL_SOURCES, export_catalog  # noqa: E402


class DocumentSourceCatalogTests(unittest.TestCase):
    def test_catalog_entries_have_required_fields(self):
        required_fields = {
            "id",
            "title",
            "provider",
            "language",
            "content_type",
            "format",
            "url",
            "reuse_notes",
            "recommended",
        }

        ids = set()
        for entry in OFFICIAL_MEDICAL_SOURCES:
            self.assertTrue(required_fields.issubset(entry.keys()))
            self.assertNotIn(entry["id"], ids)
            self.assertTrue(entry["url"].startswith("https://"))
            ids.add(entry["id"])

    def test_export_catalog_writes_valid_json(self):
        with tempfile.TemporaryDirectory(prefix="catalog-export-") as temp_dir:
            output_path = Path(temp_dir) / "medical_sources.json"

            saved_path = export_catalog(output_path)

            self.assertEqual(saved_path, output_path)
            data = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(data), len(OFFICIAL_MEDICAL_SOURCES))
            self.assertEqual(data[0]["id"], OFFICIAL_MEDICAL_SOURCES[0]["id"])


if __name__ == "__main__":
    unittest.main()
