import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class WorkerDeploymentTests(unittest.TestCase):
    def _load(self, filename):
        with (ROOT / filename).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def test_compose_files_define_dedicated_worker(self):
        for filename in ("docker-compose.yml", "docker-compose.prod.yml"):
            with self.subTest(filename=filename):
                services = self._load(filename)["services"]
                worker = services["worker"]
                self.assertEqual(worker["command"], ["python", "worker.py"])
                self.assertIn("postgres", worker["depends_on"])
                self.assertEqual(
                    services["api"]["environment"]["ENABLE_KB_SYNC_SCHEDULER"],
                    "false",
                )
                self.assertEqual(
                    services["api"]["environment"]["AUTO_BOOTSTRAP_KNOWLEDGE_BASE"],
                    "false",
                )

    def test_production_worker_owns_writable_markdown_volume(self):
        services = self._load("docker-compose.prod.yml")["services"]

        self.assertIn("./markdown_docs:/app/markdown_docs:ro", services["api"]["volumes"])
        self.assertIn("./markdown_docs:/app/markdown_docs", services["worker"]["volumes"])


if __name__ == "__main__":
    unittest.main()
