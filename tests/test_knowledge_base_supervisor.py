import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from core.knowledge_base_supervisor import KnowledgeBaseSupervisor  # noqa: E402


class FakeVectorDb:
    def __init__(self, stats):
        self._stats = stats

    def get_collection_stats(self):
        return dict(self._stats)


class FakeImportTaskStore:
    def __init__(self):
        self.saved = []

    def save_event(self, payload):
        self.saved.append(dict(payload))

    def list_recent(self, limit):
        return list(reversed(self.saved))[:limit]


class FakeDocumentManager:
    def __init__(self, local_stats, index_result=None):
        self.local_stats = local_stats
        self.index_result = index_result or {"processed": 0, "added": 0, "skipped": 0}
        self.index_calls = []

    def get_local_document_stats(self):
        return dict(self.local_stats)

    def index_existing_markdowns(self, skip_existing=True):
        self.index_calls.append({"skip_existing": skip_existing})
        return dict(self.index_result)


class FakeRagSystem:
    def __init__(self, *, local_stats=None, vector_stats=None, index_result=None):
        self.vector_db = FakeVectorDb(vector_stats or {"documents": 0, "parent_chunks": 0, "child_chunks": 0})
        self.import_task_store = FakeImportTaskStore()
        self.document_manager = FakeDocumentManager(
            local_stats or {"local_markdown_files": 0, "local_markdown_names": []},
            index_result=index_result,
        )
        self.startup_steps = []
        self.ready = True

    def _set_startup_step(self, key, state, message):
        self.startup_steps.append((key, state, message))

    def is_ready(self):
        return self.ready


class KnowledgeBaseSupervisorTests(unittest.TestCase):
    def test_refresh_status_marks_ready_when_chunks_exist(self):
        rag = FakeRagSystem(
            local_stats={"local_markdown_files": 1, "local_markdown_names": ["a.md"]},
            vector_stats={"documents": 1, "parent_chunks": 1, "child_chunks": 3},
        )
        supervisor = KnowledgeBaseSupervisor(rag)

        status = supervisor.refresh_status()

        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["stats"]["child_chunks"], 3)

    def test_record_import_event_persists_and_keeps_recent_history(self):
        rag = FakeRagSystem()
        supervisor = KnowledgeBaseSupervisor(rag)

        for index in range(10):
            supervisor.record_import_event({"source": f"source-{index}", "written": index})

        status = supervisor.get_status()

        self.assertEqual(len(status["stats"]["recent_imports"]), 8)
        self.assertEqual(status["stats"]["recent_imports"][0]["source"], "source-9")
        self.assertEqual(rag.import_task_store.saved[-1]["source"], "source-9")
        self.assertIn("timestamp", status["stats"]["recent_imports"][0])

    def test_bootstrap_no_documents_updates_status_and_startup_step(self):
        rag = FakeRagSystem(
            local_stats={"local_markdown_files": 0, "local_markdown_names": []},
            vector_stats={"documents": 0, "parent_chunks": 0, "child_chunks": 0},
        )
        supervisor = KnowledgeBaseSupervisor(rag)

        supervisor._bootstrap_knowledge_base()

        self.assertEqual(supervisor.get_status()["status"], "no_documents")
        self.assertIn(("knowledge_base_bootstrap", "completed", "当前没有本地文档，无需补建。"), rag.startup_steps)


if __name__ == "__main__":
    unittest.main()
