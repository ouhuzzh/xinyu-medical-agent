import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "project"))

from core.document_manager import DocumentManager  # noqa: E402


class FakeRagSystem:
    def __init__(self):
        self.collection_name = "test"
        self.vector_db = object()
        self.parent_store = object()
        self.chunker = object()
        self.refreshed = False

    def refresh_knowledge_base_status(self):
        self.refreshed = True


class OfficialImportEntrypointTests(unittest.TestCase):
    def test_import_official_source_calls_sync_service(self):
        rag_system = FakeRagSystem()
        manager = DocumentManager(rag_system)
        manager.markdown_dir = PROJECT_ROOT / "markdown_docs"

        fake_result = mock.Mock(downloaded=3, written=2, updated=1, unchanged=0, deactivated=0, failed=0, index_added=3, index_skipped=0)
        with mock.patch("core.document_manager.KnowledgeBaseSyncService") as sync_cls:
            sync_cls.return_value.sync_official_source.return_value = fake_result
            result = manager.import_official_source("medlineplus", limit=3, overwrite=False, index_after_import=True)

        self.assertEqual(result, fake_result)
        sync_cls.return_value.sync_official_source.assert_called_once_with(
            source="medlineplus",
            limit=3,
            trigger_type="manual",
            progress_callback=None,
            soft_delete_missing=True,
        )


if __name__ == "__main__":
    unittest.main()
