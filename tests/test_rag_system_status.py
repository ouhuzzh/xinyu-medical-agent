import sys
import unittest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "project"))

from core.rag_system import RAGSystem  # noqa: E402
from core import document_manager as document_manager_module  # noqa: E402


class FakeVectorDb:
    def __init__(self, stats):
        self._stats = stats

    def get_collection_stats(self):
        return dict(self._stats)


class RAGSystemStatusTests(unittest.TestCase):
    def setUp(self):
        self.original_document_manager = document_manager_module.DocumentManager

    def tearDown(self):
        document_manager_module.DocumentManager = self.original_document_manager

    def _make_rag(self, vector_stats):
        rag = object.__new__(RAGSystem)
        rag.vector_db = FakeVectorDb(vector_stats)
        rag._knowledge_base_status = {
            "status": "not_checked",
            "message": "",
            "last_error": "",
            "stats": {"last_bootstrap_result": ""},
        }
        rag._startup_status = {"state": "not_started", "message": "", "last_error": "", "steps": {}}
        return rag

    def test_refresh_knowledge_base_status_marks_no_documents(self):
        class FakeDocumentManager:
            def __init__(self, rag_system):
                pass

            def get_local_document_stats(self):
                return {"local_markdown_files": 0, "local_markdown_names": []}

        document_manager_module.DocumentManager = FakeDocumentManager
        rag = self._make_rag({"documents": 0, "parent_chunks": 0, "child_chunks": 0})

        status = RAGSystem.refresh_knowledge_base_status(rag)

        self.assertEqual(status["status"], "no_documents")
        self.assertIn("尚无可索引文档", status["message"])

    def test_refresh_knowledge_base_status_marks_pending_rebuild_for_missing_index(self):
        class FakeDocumentManager:
            def __init__(self, rag_system):
                pass

            def get_local_document_stats(self):
                return {"local_markdown_files": 2, "local_markdown_names": ["a.md", "b.md"]}

        document_manager_module.DocumentManager = FakeDocumentManager
        rag = self._make_rag({"documents": 1, "parent_chunks": 1, "child_chunks": 1})

        status = RAGSystem.refresh_knowledge_base_status(rag)

        self.assertEqual(status["status"], "pending_rebuild")
        self.assertIn("未完成索引", status["message"])


if __name__ == "__main__":
    unittest.main()
