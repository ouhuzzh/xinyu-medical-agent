import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from rag_agent.intent_embedder import IntentEmbedder  # noqa: E402


class MissingEmbeddingIntentEmbedder(IntentEmbedder):
    def _load_embedding_model(self):
        return None


class IntentEmbedderTests(unittest.TestCase):
    def test_missing_embedding_model_disables_l2_without_exception(self):
        embedder = MissingEmbeddingIntentEmbedder()

        self.assertIsNone(embedder.classify("高血压怎么控制"))
        self.assertTrue(embedder._initialized)
        self.assertEqual(embedder._centroids, {})


if __name__ == "__main__":
    unittest.main()
