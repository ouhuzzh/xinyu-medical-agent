import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "project"))

from core.document_manager import DocumentManager  # noqa: E402


class FakeCollection:
    def __init__(self):
        self.added_batches = []

    def add_documents(self, documents):
        self.added_batches.append(list(documents))


class FakeVectorDb:
    def __init__(self, indexed_document_nos=None):
        self.indexed_document_nos = set(indexed_document_nos or [])
        self.create_calls = 0
        self.collection = FakeCollection()

    def create_collection(self, collection_name):
        self.create_calls += 1

    def get_collection(self, collection_name):
        return self.collection

    def get_indexed_document_nos(self):
        return set(self.indexed_document_nos)


class FakeParentStore:
    def __init__(self):
        self.saved_batches = []

    def save_many(self, parents):
        self.saved_batches.append(list(parents))


class FakeChunker:
    def create_chunks_single(self, md_path):
        source_name = f"{Path(md_path).stem}.pdf"
        parent_id = f"{Path(md_path).stem}_parent_0"
        parent_doc = Document(
            page_content="parent content",
            metadata={"source": source_name, "parent_id": parent_id},
        )
        child_doc = Document(
            page_content="child content",
            metadata={"source": source_name, "parent_id": parent_id, "chunk_id": f"{parent_id}_child_0"},
        )
        return [(parent_id, parent_doc)], [child_doc]


class FakeRagSystem:
    def __init__(self, indexed_document_nos=None):
        self.collection_name = "test_collection"
        self.vector_db = FakeVectorDb(indexed_document_nos=indexed_document_nos)
        self.parent_store = FakeParentStore()
        self.chunker = FakeChunker()


class DocumentManagerTests(unittest.TestCase):
    def setUp(self):
        temp_root = PROJECT_ROOT / "runtime" / "test_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = str(temp_root / f"doc-manager-{uuid.uuid4().hex}")
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_markdown(self, filename: str, content: str = "# Title\n\ncontent"):
        path = Path(self.temp_dir) / filename
        path.write_text(content, encoding="utf-8")
        return path

    def test_index_existing_markdowns_skips_docs_already_in_database(self):
        rag_system = FakeRagSystem(indexed_document_nos={"already-indexed"})
        manager = DocumentManager(rag_system)
        manager.markdown_dir = Path(self.temp_dir)
        self._write_markdown("already-indexed.md")
        self._write_markdown("needs-index.md")

        result = manager.index_existing_markdowns(skip_existing=True)

        self.assertEqual(result["processed"], 2)
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(rag_system.vector_db.create_calls, 1)
        self.assertEqual(len(rag_system.parent_store.saved_batches), 1)
        self.assertEqual(len(rag_system.vector_db.collection.added_batches), 1)

    def test_local_document_stats_reflect_markdown_directory(self):
        rag_system = FakeRagSystem()
        manager = DocumentManager(rag_system)
        manager.markdown_dir = Path(self.temp_dir)
        self._write_markdown("alpha.md")
        self._write_markdown("beta.md")

        stats = manager.get_local_document_stats()

        self.assertEqual(stats["local_markdown_files"], 2)
        self.assertEqual(stats["local_markdown_names"], ["alpha.md", "beta.md"])

    def test_get_markdown_files_preserves_real_extension(self):
        rag_system = FakeRagSystem()
        manager = DocumentManager(rag_system)
        manager.markdown_dir = Path(self.temp_dir)
        self._write_markdown("care-plan.md")

        files = manager.get_markdown_files()

        self.assertEqual(files, ["care-plan.md"])

    def test_get_document_inventory_exposes_user_facing_metadata(self):
        rag_system = FakeRagSystem()
        manager = DocumentManager(rag_system)
        manager.markdown_dir = Path(self.temp_dir)
        self._write_markdown(
            "who-headache.md",
            "\n".join(
                [
                    "Source: World Health Organization",
                    "Source Key: official:who:who-headache",
                    "Source type: public_health",
                    "Title: Headache disorders",
                    "Original URL: https://www.who.int/example",
                    "Freshness Bucket: current",
                    "Sync Status: active",
                    "",
                    "# Headache disorders",
                    "",
                    "content",
                ]
            ),
        )

        inventory = manager.get_document_inventory()

        self.assertEqual(inventory[0]["title"], "Headache disorders")
        self.assertEqual(inventory[0]["source_key"], "official:who:who-headache")
        self.assertEqual(inventory[0]["source_type"], "public_health")
        self.assertEqual(inventory[0]["freshness_bucket"], "current")
        self.assertEqual(inventory[0]["original_url"], "https://www.who.int/example")

    def test_add_documents_with_report_explains_duplicate_markdown(self):
        rag_system = FakeRagSystem()
        manager = DocumentManager(rag_system)
        manager.markdown_dir = Path(self.temp_dir)
        existing = self._write_markdown("duplicate.md", "# Existing\n\nhello")
        duplicate_source_dir = Path(self.temp_dir) / "source"
        duplicate_source_dir.mkdir(parents=True, exist_ok=True)
        duplicate_source = duplicate_source_dir / "duplicate.md"
        duplicate_source.write_text("# Updated\n\nworld", encoding="utf-8")

        fake_sync_result = mock.Mock(
            added=0,
            updated=1,
            unchanged=0,
            deactivated=0,
            skipped=0,
            failure_details=[],
            conversion_details=[],
        )
        fake_sync_result.to_event.return_value = {"source": "local", "written": 0, "updated": 1, "deactivated": 0, "unchanged": 0}
        with mock.patch("core.document_manager.KnowledgeBaseSyncService") as sync_cls:
            sync_cls.return_value.sync_local_documents.return_value = fake_sync_result
            report = manager.add_documents_with_report([str(duplicate_source)])

        self.assertEqual(report["added"], 0)
        self.assertEqual(report["updated"], 1)
        self.assertEqual(report["unchanged"], 0)
        self.assertEqual(existing.read_text(encoding="utf-8"), "# Updated\n\nworld")

    def test_add_documents_with_report_accepts_txt_via_multiformat_parser(self):
        rag_system = FakeRagSystem()
        manager = DocumentManager(rag_system)
        manager.markdown_dir = Path(self.temp_dir)
        text_source = Path(self.temp_dir) / "source.txt"
        text_source.write_text("高血压患者应规律监测血压。", encoding="utf-8")

        fake_sync_result = mock.Mock(
            added=1,
            updated=0,
            unchanged=0,
            deactivated=0,
            skipped=0,
            failure_details=[],
            conversion_details=[],
        )
        fake_sync_result.to_event.return_value = {"source": "local", "written": 1, "updated": 0, "deactivated": 0, "unchanged": 0}
        with mock.patch("core.document_manager.KnowledgeBaseSyncService") as sync_cls:
            sync_cls.return_value.sync_local_documents.return_value = fake_sync_result
            report = manager.add_documents_with_report([str(text_source)])

        self.assertEqual(report["added"], 1)
        self.assertTrue((Path(self.temp_dir) / "source.md").exists())
        self.assertIn("plain_text_fallback", report["conversion_details"][0])


if __name__ == "__main__":
    unittest.main()
