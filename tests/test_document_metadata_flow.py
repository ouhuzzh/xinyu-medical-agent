import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "project"))

from core.document_chunker import DocumentChuncker  # noqa: E402


class DocumentMetadataFlowTests(unittest.TestCase):
    def test_front_matter_metadata_is_attached_to_chunks(self):
        markdown = """Source: World Health Organization
Source type: public_health
Language: en
File type: html
Title: Hypertension
Original URL: https://www.who.int/news-room/fact-sheets/detail/hypertension

# Hypertension

Hypertension is a major risk factor for cardiovascular disease.
""" + ("\nBlood pressure monitoring matters.\n" * 250)

        with tempfile.TemporaryDirectory(prefix="chunk-meta-") as temp_dir:
            markdown_path = Path(temp_dir) / "who-hypertension.md"
            markdown_path.write_text(markdown, encoding="utf-8")

            parent_chunks, child_chunks = DocumentChuncker().create_chunks_single(markdown_path)

        self.assertTrue(parent_chunks)
        self.assertTrue(child_chunks)
        parent_metadata = parent_chunks[0][1].metadata
        child_metadata = child_chunks[0].metadata
        self.assertEqual(parent_metadata["source_type"], "public_health")
        self.assertEqual(parent_metadata["language"], "en")
        self.assertEqual(parent_metadata["title"], "Hypertension")
        self.assertEqual(child_metadata["source_type"], "public_health")
        self.assertEqual(child_metadata["original_url"], "https://www.who.int/news-room/fact-sheets/detail/hypertension")


if __name__ == "__main__":
    unittest.main()
