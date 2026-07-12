import sys
import unittest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "project"))

from core.rag_system import RAGSystem  # noqa: E402


class RagSystemImportHistoryTests(unittest.TestCase):
    def test_record_import_event_keeps_recent_history(self):
        rag = RAGSystem()

        for index in range(10):
            rag.record_import_event(
                {
                    "source": f"source-{index}",
                    "written": index,
                    "skipped": 0,
                    "failed": 0,
                }
            )

        history = rag.get_knowledge_base_status()["stats"]["recent_imports"]

        self.assertEqual(len(history), 8)
        self.assertEqual(history[0]["source"], "source-9")
        self.assertEqual(history[-1]["source"], "source-2")
        self.assertIn("timestamp", history[0])


if __name__ == "__main__":
    unittest.main()
